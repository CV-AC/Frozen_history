from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice
_A = -0.75
FP_FUSION = False


def _f32(x) -> float:
    return float(np.float32(x))


def _inv(x) -> float:
    return float(np.float32(1.0) / np.float32(x))


@triton.jit
def _warp_coords(Yf, Xf, mhy, mhx, inv_hm1, inv_wm1, Hm1f, Wm1f):
    ny = 2.0 * (Yf + mhy) * inv_hm1 - 1.0
    nx = 2.0 * (Xf + mhx) * inv_wm1 - 1.0
    inside = (tl.abs(nx) <= 1.0) & (tl.abs(ny) <= 1.0)
    iy = (ny + 1.0) * 0.5 * Hm1f
    ix = (nx + 1.0) * 0.5 * Wm1f
    iy = tl.minimum(tl.maximum(iy, 0.0), Hm1f)
    ix = tl.minimum(tl.maximum(ix, 0.0), Wm1f)
    return (iy, ix, inside)


@triton.jit
def _bilinear_setup(iy, ix, H, W, mask):
    fy0 = tl.floor(iy)
    fx0 = tl.floor(ix)
    y0 = fy0.to(tl.int32)
    x0 = fx0.to(tl.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    fy1 = fy0 + 1.0
    fx1 = fx0 + 1.0
    nw = (fx1 - ix) * (fy1 - iy)
    ne = (ix - fx0) * (fy1 - iy)
    sw = (fx1 - ix) * (iy - fy0)
    se = (ix - fx0) * (iy - fy0)
    m_nw = mask & (y0 >= 0) & (y0 < H) & (x0 >= 0) & (x0 < W)
    m_ne = mask & (y0 >= 0) & (y0 < H) & (x1 >= 0) & (x1 < W)
    m_sw = mask & (y1 >= 0) & (y1 < H) & (x0 >= 0) & (x0 < W)
    m_se = mask & (y1 >= 0) & (y1 < H) & (x1 >= 0) & (x1 < W)
    o_nw = y0 * W + x0
    o_ne = y0 * W + x1
    o_sw = y1 * W + x0
    o_se = y1 * W + x1
    return (nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se)


@triton.jit
def _bilinear_plane(
    ptr, nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se
):
    v = tl.load(ptr + o_nw, mask=m_nw, other=0.0)
    acc = v * nw
    v = tl.load(ptr + o_ne, mask=m_ne, other=0.0)
    acc = tl.fma(v, ne, acc)
    v = tl.load(ptr + o_sw, mask=m_sw, other=0.0)
    acc = tl.fma(v, sw, acc)
    v = tl.load(ptr + o_se, mask=m_se, other=0.0)
    acc = tl.fma(v, se, acc)
    return acc


@triton.jit
def _cc1(x):
    return tl.fma(tl.fma(x, 1.25, -2.25) * x, x, 1.0)


@triton.jit
def _cc2(x):
    return tl.fma(tl.fma(tl.fma(x, -0.75, 3.75), x, -6.0), x, 3.0)


@triton.jit
def _interp4(x0, x1, x2, x3, c0, c1, c2, c3):
    return tl.fma(x3, c3, tl.fma(x2, c2, tl.fma(x1, c1, x0 * c0)))


@triton.jit
def _bicubic_row(ptr, row, cx0, cx1, cx2, cx3, c0, c1, c2, c3, m, r_sw):
    v0 = tl.load(ptr + row + cx0 * r_sw, mask=m, other=0.0)
    v1 = tl.load(ptr + row + cx1 * r_sw, mask=m, other=0.0)
    v2 = tl.load(ptr + row + cx2 * r_sw, mask=m, other=0.0)
    v3 = tl.load(ptr + row + cx3 * r_sw, mask=m, other=0.0)
    return _interp4(v0, v1, v2, v3, c0, c1, c2, c3)


@triton.jit
def _bicubic_skip(
    rgb_ptr,
    r_sc,
    r_sh,
    r_sw,
    Yf,
    Xf,
    offy,
    offx,
    centre,
    inv_s,
    inv_lhm1,
    inv_lwm1,
    lhm1f,
    lwm1f,
    m,
):
    u = (Yf - centre) * inv_s - offy
    v = (Xf - centre) * inv_s - offx
    ny = 2.0 * u * inv_lhm1 - 1.0
    nx = 2.0 * v * inv_lwm1 - 1.0
    iy = (ny + 1.0) * 0.5 * lhm1f
    ix = (nx + 1.0) * 0.5 * lwm1f
    fy = tl.floor(iy)
    fx = tl.floor(ix)
    ty = iy - fy
    tx = ix - fx
    cy0 = _cc2(ty + 1.0)
    cy1 = _cc1(ty)
    cy2 = _cc1(1.0 - ty)
    cy3 = _cc2(1.0 - ty + 1.0)
    cx0 = _cc2(tx + 1.0)
    cx1 = _cc1(tx)
    cx2 = _cc1(1.0 - tx)
    cx3 = _cc2(1.0 - tx + 1.0)
    ry0 = tl.minimum(tl.maximum(fy - 1.0, 0.0), lhm1f).to(tl.int32) * r_sh
    ry1 = tl.minimum(tl.maximum(fy, 0.0), lhm1f).to(tl.int32) * r_sh
    ry2 = tl.minimum(tl.maximum(fy + 1.0, 0.0), lhm1f).to(tl.int32) * r_sh
    ry3 = tl.minimum(tl.maximum(fy + 2.0, 0.0), lhm1f).to(tl.int32) * r_sh
    kx0 = tl.minimum(tl.maximum(fx - 1.0, 0.0), lwm1f).to(tl.int32)
    kx1 = tl.minimum(tl.maximum(fx, 0.0), lwm1f).to(tl.int32)
    kx2 = tl.minimum(tl.maximum(fx + 1.0, 0.0), lwm1f).to(tl.int32)
    kx3 = tl.minimum(tl.maximum(fx + 2.0, 0.0), lwm1f).to(tl.int32)
    p0 = rgb_ptr
    r0 = _interp4(
        _bicubic_row(p0, ry0, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p0, ry1, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p0, ry2, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p0, ry3, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        cy0,
        cy1,
        cy2,
        cy3,
    )
    p1 = rgb_ptr + r_sc
    r1 = _interp4(
        _bicubic_row(p1, ry0, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p1, ry1, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p1, ry2, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p1, ry3, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        cy0,
        cy1,
        cy2,
        cy3,
    )
    p2 = rgb_ptr + 2 * r_sc
    r2 = _interp4(
        _bicubic_row(p2, ry0, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p2, ry1, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p2, ry2, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        _bicubic_row(p2, ry3, kx0, kx1, kx2, kx3, cx0, cx1, cx2, cx3, m, r_sw),
        cy0,
        cy1,
        cy2,
        cy3,
    )
    r0 = tl.minimum(tl.maximum(r0, 0.0), 1.0)
    r1 = tl.minimum(tl.maximum(r1, 0.0), 1.0)
    r2 = tl.minimum(tl.maximum(r2, 0.0), 1.0)
    return (r0, r1, r2)


@triton.jit
def _sigmoid(x):
    return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _k1_hr_pre(
    prev_ptr,
    mot_ptr,
    wh_ptr,
    bh_ptr,
    wc_ptr,
    bc_ptr,
    wr_ptr,
    br_ptr,
    out_ptr,
    h,
    w,
    H,
    W,
    m_sc,
    m_sh,
    m_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    S: tl.constexpr,
    BP: tl.constexpr,
):
    S2: tl.constexpr = S * S
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    dyi = tl.arange(0, S)[:, None, None]
    pi = tl.arange(0, BP)[None, :, None]
    dxi = tl.arange(0, S)[None, None, :]
    j = j0 + pi
    pm = j < w
    mask = pm & (dyi >= 0) & (dxi >= 0)
    Y = i * S + dyi
    X = j * S + dxi
    my = tl.load(mot_ptr + i * m_sh + j * m_sw, mask=pm, other=0.0)
    mx = tl.load(mot_ptr + m_sc + i * m_sh + j * m_sw, mask=pm, other=0.0)
    (iy, ix, inside) = _warp_coords(
        Y.to(tl.float32),
        X.to(tl.float32),
        my * sf,
        mx * sf,
        inv_hm1,
        inv_wm1,
        Hm1f,
        Wm1f,
    )
    (nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se) = _bilinear_setup(
        iy, ix, H, W, mask
    )
    HW = H * W
    ins = inside.to(tl.float32)
    c0 = _bilinear_plane(
        prev_ptr, nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se
    )
    c1 = _bilinear_plane(
        prev_ptr + HW, nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se
    )
    c2 = _bilinear_plane(
        prev_ptr + 2 * HW,
        nw,
        ne,
        sw,
        se,
        m_nw,
        m_ne,
        m_sw,
        m_se,
        o_nw,
        o_ne,
        o_sw,
        o_se,
    )
    cw = (
        _bilinear_plane(
            prev_ptr + 3 * HW,
            nw,
            ne,
            sw,
            se,
            m_nw,
            m_ne,
            m_sw,
            m_se,
            o_nw,
            o_ne,
            o_sw,
            o_se,
        )
        * ins
    )
    rw = _bilinear_plane(
        prev_ptr + 4 * HW,
        nw,
        ne,
        sw,
        se,
        m_nw,
        m_ne,
        m_sw,
        m_se,
        o_nw,
        o_ne,
        o_sw,
        o_se,
    )
    rw = rw * ins + (1.0 - ins)
    jj = j0 + tl.arange(0, BP)
    om = jj < w
    hw = h * w
    base = out_ptr + i * w + jj
    q = dyi * S + dxi
    for o in tl.static_range(16):
        wv0 = tl.load(wh_ptr + o * 3 * S2 + q)
        wv1 = tl.load(wh_ptr + o * 3 * S2 + S2 + q)
        wv2 = tl.load(wh_ptr + o * 3 * S2 + 2 * S2 + q)
        t = c0 * wv0 + c1 * wv1 + c2 * wv2
        acc = tl.sum(tl.sum(t, axis=2), axis=0) + tl.load(bh_ptr + o)
        tl.store(base + o * hw, acc, mask=om)
    for o in tl.static_range(4):
        wv = tl.load(wc_ptr + o * S2 + q)
        acc = tl.sum(tl.sum(cw * wv, axis=2), axis=0) + tl.load(bc_ptr + o)
        tl.store(base + (16 + o) * hw, acc, mask=om)
    for o in tl.static_range(4):
        wv = tl.load(wr_ptr + o * S2 + q)
        acc = tl.sum(tl.sum(rw * wv, axis=2), axis=0) + tl.load(br_ptr + o)
        tl.store(base + (20 + o) * hw, acc, mask=om)


@triton.jit
def _k2_hr_post(
    prev_ptr,
    mot_ptr,
    col_ptr,
    stat_ptr,
    feat_ptr,
    rgb_ptr,
    jit_ptr,
    out_ptr,
    new_ptr,
    flag_ptr,
    h,
    w,
    H,
    W,
    m_sc,
    m_sh,
    m_sw,
    c_sc,
    c_sh,
    c_sw,
    t_sc,
    t_sh,
    t_sw,
    f_sc,
    f_sh,
    f_sw,
    r_sc,
    r_sh,
    r_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    centre,
    inv_s,
    inv_lhm1,
    inv_lwm1,
    lhm1f,
    lwm1f,
    half,
    width_k,
    inv_cap,
    g_lo,
    g_hi,
    f_box,
    S: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
):
    S2: tl.constexpr = S * S
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    dyi = tl.arange(0, S)[:, None, None]
    pi = tl.arange(0, BP)[None, :, None]
    dxi = tl.arange(0, S)[None, None, :]
    j = j0 + pi
    pm = j < w
    mask = pm & (dyi >= 0) & (dxi >= 0)
    Y = i * S + dyi
    X = j * S + dxi
    Yf = Y.to(tl.float32)
    Xf = X.to(tl.float32)
    my = tl.load(mot_ptr + i * m_sh + j * m_sw, mask=pm, other=0.0)
    mx = tl.load(mot_ptr + m_sc + i * m_sh + j * m_sw, mask=pm, other=0.0)
    (iy, ix, inside) = _warp_coords(
        Yf, Xf, my * sf, mx * sf, inv_hm1, inv_wm1, Hm1f, Wm1f
    )
    (nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se) = _bilinear_setup(
        iy, ix, H, W, mask
    )
    HW = H * W
    valid = inside.to(tl.float32)
    h0 = _bilinear_plane(
        prev_ptr, nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se
    )
    h1 = _bilinear_plane(
        prev_ptr + HW, nw, ne, sw, se, m_nw, m_ne, m_sw, m_se, o_nw, o_ne, o_sw, o_se
    )
    h2 = _bilinear_plane(
        prev_ptr + 2 * HW,
        nw,
        ne,
        sw,
        se,
        m_nw,
        m_ne,
        m_sw,
        m_se,
        o_nw,
        o_ne,
        o_sw,
        o_se,
    )
    conf = (
        _bilinear_plane(
            prev_ptr + 3 * HW,
            nw,
            ne,
            sw,
            se,
            m_nw,
            m_ne,
            m_sw,
            m_se,
            o_nw,
            o_ne,
            o_sw,
            o_se,
        )
        * valid
    )
    jy = tl.load(jit_ptr)
    jx = tl.load(jit_ptr + 1)
    (s0, s1, s2) = _bicubic_skip(
        rgb_ptr,
        r_sc,
        r_sh,
        r_sw,
        Yf,
        Xf,
        -jy,
        jx,
        centre,
        inv_s,
        inv_lhm1,
        inv_lwm1,
        lhm1f,
        lwm1f,
        mask,
    )
    q = dyi * S + dxi
    lr_c = i * c_sh + j * c_sw + q * c_sc
    k0 = tl.load(col_ptr + lr_c, mask=mask, other=0.0).to(tl.float32) + s0
    k1 = tl.load(col_ptr + lr_c + S2 * c_sc, mask=mask, other=0.0).to(tl.float32) + s1
    k2 = (
        tl.load(col_ptr + lr_c + 2 * S2 * c_sc, mask=mask, other=0.0).to(tl.float32)
        + s2
    )
    lr_t = i * t_sh + j * t_sw + q * t_sc
    rel = _sigmoid(tl.load(stat_ptr + lr_t, mask=mask, other=0.0))
    ret = _sigmoid(tl.load(stat_ptr + lr_t + S2 * t_sc, mask=mask, other=0.0))
    py = centre + -jy * sf - dyi.to(tl.float32)
    px = centre + jx * sf - dxi.to(tl.float32)
    py = libdevice.fmod(py + half, sf)
    py = tl.where((py != 0.0) & (py < 0.0), py + sf, py) - half
    px = libdevice.fmod(px + half, sf)
    px = tl.where((px != 0.0) & (px < 0.0), px + sf, px) - half
    radius = tl.sqrt_rn(py * py + px * px) * inv_s
    rk = tl.math.div_rn(radius, width_k)
    weight = libdevice.exp(-(rk * rk) * 0.5) * rel
    keep = ret * valid
    sn0 = keep * (h0 * conf) + weight * k0
    sn1 = keep * (h1 * conf) + weight * k1
    sn2 = keep * (h2 * conf) + weight * k2
    wn = tl.maximum(keep * conf + weight, 0.0001, propagate_nan=tl.PropagateNan.ALL)
    excess = tl.maximum(wn * inv_cap, 1.0, propagate_nan=tl.PropagateNan.ALL)
    sn0 = tl.math.div_rn(sn0, excess)
    sn1 = tl.math.div_rn(sn1, excess)
    sn2 = tl.math.div_rn(sn2, excess)
    wn = tl.math.div_rn(wn, excess)
    o0 = tl.math.div_rn(sn0, wn)
    o1 = tl.math.div_rn(sn1, wn)
    o2 = tl.math.div_rn(sn2, wn)
    hr = Y * W + X
    tl.store(out_ptr + hr, o0, mask=mask)
    tl.store(out_ptr + HW + hr, o1, mask=mask)
    tl.store(out_ptr + 2 * HW + hr, o2, mask=mask)
    tl.store(new_ptr + hr, o0, mask=mask)
    tl.store(new_ptr + HW + hr, o1, mask=mask)
    tl.store(new_ptr + 2 * HW + hr, o2, mask=mask)
    tl.store(new_ptr + 3 * HW + hr, wn, mask=mask)
    radius_b = radius + tl.zeros_like(o0)
    tl.store(new_ptr + 4 * HW + hr, radius_b, mask=mask)
    ok = (
        (o0 >= g_lo)
        & (o0 <= g_hi)
        & (o1 >= g_lo)
        & (o1 <= g_hi)
        & (o2 >= g_lo)
        & (o2 <= g_hi)
        & (wn == wn)
        & (tl.abs(wn) < float("inf"))
    )
    bad = (mask & (ok == 0)).to(tl.int32)
    bad_blk = tl.max(tl.max(bad, axis=2), axis=0)
    jj = j0 + tl.arange(0, BP)
    om = jj < w
    fc = tl.arange(0, FD)
    fv = tl.load(
        feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
        mask=om[:, None],
        other=0.0,
    ).to(tl.float32)
    fok = tl.abs(fv) <= f_box
    bad_f = tl.max((fok == 0).to(tl.int32), axis=1)
    flag = tl.maximum(bad_blk, bad_f)
    tl.store(flag_ptr + i * w + jj, flag.to(tl.int8), mask=om)


@triton.jit
def _k3_guard(
    flag_ptr,
    new_ptr,
    feat_ptr,
    rgb_ptr,
    jit_ptr,
    h,
    w,
    H,
    W,
    f_sc,
    f_sh,
    f_sw,
    r_sc,
    r_sh,
    r_sw,
    centre,
    inv_s,
    inv_lhm1,
    inv_lwm1,
    lhm1f,
    lwm1f,
    S: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
):
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    jj = j0 + tl.arange(0, BP)
    om = jj < w
    bad = tl.zeros([BP], dtype=tl.int32)
    for di in tl.static_range(3):
        for dj in tl.static_range(3):
            ii = i + di - 1
            jn = jj + dj - 1
            m = om & (ii >= 0) & (ii < h) & (jn >= 0) & (jn < w)
            bad = tl.maximum(
                bad, tl.load(flag_ptr + ii * w + jn, mask=m, other=0).to(tl.int32)
            )
    if tl.max(bad, axis=0) > 0:
        badm = (bad > 0) & om
        fc = tl.arange(0, FD)
        fz = tl.zeros([BP, FD], dtype=tl.float32)
        tl.store(
            feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
            fz.to(feat_ptr.dtype.element_ty),
            mask=badm[:, None],
        )
        dyi = tl.arange(0, S)[:, None, None]
        pi = tl.arange(0, BP)[None, :, None]
        dxi = tl.arange(0, S)[None, None, :]
        j = j0 + pi
        b3 = (bad > 0)[None, :, None] & (j < w) & (dyi >= 0) & (dxi >= 0)
        Y = i * S + dyi
        X = j * S + dxi
        jy = tl.load(jit_ptr)
        jx = tl.load(jit_ptr + 1)
        (s0, s1, s2) = _bicubic_skip(
            rgb_ptr,
            r_sc,
            r_sh,
            r_sw,
            Y.to(tl.float32),
            X.to(tl.float32),
            -jy,
            jx,
            centre,
            inv_s,
            inv_lhm1,
            inv_lwm1,
            lhm1f,
            lwm1f,
            b3,
        )
        HW = H * W
        hr = Y * W + X
        tl.store(new_ptr + hr, s0, mask=b3)
        tl.store(new_ptr + HW + hr, s1, mask=b3)
        tl.store(new_ptr + 2 * HW + hr, s2, mask=b3)
        tl.store(new_ptr + 3 * HW + hr, tl.zeros_like(s0), mask=b3)
        tl.store(new_ptr + 4 * HW + hr, tl.zeros_like(s0) + 1.0, mask=b3)


def _bp(scale):
    return 32 if scale == 4 else 64


def _consts(h, w, scale):
    (H, W) = (h * scale, w * scale)
    return dict(
        inv_hm1=_inv(max(H - 1, 1)),
        inv_wm1=_inv(max(W - 1, 1)),
        Hm1f=_f32(H - 1),
        Wm1f=_f32(W - 1),
        sf=_f32(scale),
        centre=_f32(0.5 * (scale - 1)),
        inv_s=_inv(scale),
        inv_lhm1=_inv(max(h - 1, 1)),
        inv_lwm1=_inv(max(w - 1, 1)),
        lhm1f=_f32(h - 1),
        lwm1f=_f32(w - 1),
    )


@triton_op("rrta_fast::hr_pre", mutates_args={})
def hr_pre(
    prev: torch.Tensor,
    motion: torch.Tensor,
    wh: torch.Tensor,
    bh: torch.Tensor,
    wc: torch.Tensor,
    bc: torch.Tensor,
    wr: torch.Tensor,
    br: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    (h, w) = motion.shape[-2:]
    (H, W) = (h * scale, w * scale)
    out = torch.empty((1, 24, h, w), device=prev.device, dtype=torch.float32)
    c = _consts(h, w, scale)
    bp = _bp(scale)
    wrap_triton(_k1_hr_pre)[triton.cdiv(w, bp), h](
        prev,
        motion,
        wh,
        bh,
        wc,
        bc,
        wr,
        br,
        out,
        h,
        w,
        H,
        W,
        motion.stride(1),
        motion.stride(2),
        motion.stride(3),
        c["inv_hm1"],
        c["inv_wm1"],
        c["Hm1f"],
        c["Wm1f"],
        c["sf"],
        S=scale,
        BP=bp,
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )
    return out


@triton_op("rrta_fast::hr_post", mutates_args={})
def hr_post(
    prev: torch.Tensor,
    motion: torch.Tensor,
    colour: torch.Tensor,
    stat: torch.Tensor,
    feat: torch.Tensor,
    rgb: torch.Tensor,
    jitter: torch.Tensor,
    scale: int,
    width_k: float,
    inv_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (h, w) = motion.shape[-2:]
    (H, W) = (h * scale, w * scale)
    out = torch.empty((1, 3, H, W), device=prev.device, dtype=torch.float32)
    new = torch.empty((1, 5, H, W), device=prev.device, dtype=torch.float32)
    flags = torch.empty((h, w), device=prev.device, dtype=torch.int8)
    c = _consts(h, w, scale)
    bp = _bp(scale)
    wrap_triton(_k2_hr_post)[triton.cdiv(w, bp), h](
        prev,
        motion,
        colour,
        stat,
        feat,
        rgb,
        jitter,
        out,
        new,
        flags,
        h,
        w,
        H,
        W,
        motion.stride(1),
        motion.stride(2),
        motion.stride(3),
        colour.stride(1),
        colour.stride(2),
        colour.stride(3),
        stat.stride(1),
        stat.stride(2),
        stat.stride(3),
        feat.stride(1),
        feat.stride(2),
        feat.stride(3),
        rgb.stride(1),
        rgb.stride(2),
        rgb.stride(3),
        c["inv_hm1"],
        c["inv_wm1"],
        c["Hm1f"],
        c["Wm1f"],
        c["sf"],
        c["centre"],
        c["inv_s"],
        c["inv_lhm1"],
        c["inv_lwm1"],
        c["lhm1f"],
        c["lwm1f"],
        _f32(0.5 * scale),
        width_k,
        inv_cap,
        -2.0,
        3.0,
        32.0,
        S=scale,
        BP=bp,
        FD=feat.shape[1],
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )
    return (out, new, flags)


@triton_op("rrta_fast::guard", mutates_args={"new", "feat"})
def guard(
    flags: torch.Tensor,
    new: torch.Tensor,
    feat: torch.Tensor,
    rgb: torch.Tensor,
    jitter: torch.Tensor,
    scale: int,
) -> None:
    (h, w) = flags.shape
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    bp = _bp(scale)
    wrap_triton(_k3_guard)[triton.cdiv(w, bp), h](
        flags,
        new,
        feat,
        rgb,
        jitter,
        h,
        w,
        H,
        W,
        feat.stride(1),
        feat.stride(2),
        feat.stride(3),
        rgb.stride(1),
        rgb.stride(2),
        rgb.stride(3),
        c["centre"],
        c["inv_s"],
        c["inv_lhm1"],
        c["inv_lwm1"],
        c["lhm1f"],
        c["lwm1f"],
        S=scale,
        BP=bp,
        FD=feat.shape[1],
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )


def packed_from_state(state) -> torch.Tensor:
    (c, conf, reach) = (state[0], state[1], state[4])
    b = c._base
    if (
        b is not None
        and b.dim() == 4
        and (b.shape[1] == 5)
        and (b.dtype == torch.float32)
        and b.is_contiguous()
        and (conf._base is b)
        and (reach._base is b)
        and (c.data_ptr() == b.data_ptr())
        and (conf.data_ptr() == b[:, 3:4].data_ptr())
        and (reach.data_ptr() == b[:, 4:5].data_ptr())
    ):
        return b
    return torch.cat((c.float(), conf.float(), reach.float()), dim=1).contiguous()


class FusedResolve:
    def __init__(
        self, model, compile: bool = True, mode: str = "default", fp_fusion: bool = True
    ):
        from .lean_step import check_config

        check_config(model)
        if model.scale not in (2, 4):
            raise ValueError("fused stage supports x2 and x4")
        if model.feature_dim & model.feature_dim - 1:
            raise ValueError("feature_dim must be a power of two")
        self.m = model
        with torch.no_grad():
            s2 = model.scale**2
            self.wh = (
                model.history_proj.weight.detach()
                .float()
                .reshape(16, 3 * s2)
                .contiguous()
            )
            self.bh = model.history_proj.bias.detach().float().contiguous()
            self.wc = (
                model.confidence_proj.weight.detach()
                .float()
                .reshape(4, s2)
                .contiguous()
            )
            self.bc = model.confidence_proj.bias.detach().float().contiguous()
            self.wr = (
                model.reach_proj.weight.detach().float().reshape(4, s2).contiguous()
            )
            self.br = model.reach_proj.bias.detach().float().contiguous()
            self.width_k = float(
                (F.softplus(model.kernel_width.detach()) + 0.25).item()
            )
        self.inv_cap = _inv(1.0 / model.rectifier.alpha_min)
        self.compiled = compile
        self.inner = torch.compile(self._inner, mode=mode) if compile else self._inner

    def _inner(self, rgb, depth, motion, jitter, prev, prev_depth, prev_feat, prev_jit):
        import diffsr.rrta as R

        m = self.m
        scale = m.scale
        motion = R.dilate_motion(motion, depth)
        motion_lr = m._lr_motion(motion, jitter, prev_jit)
        lr24 = hr_pre(
            prev, motion, self.wh, self.bh, self.wc, self.bc, self.wr, self.br, scale
        )
        (warped_depth, depth_inside) = R.warp(prev_depth, motion)
        (warped_feat, feat_inside) = R.warp(prev_feat, motion_lr)
        feature_hist = warped_feat * feat_inside
        disocclusion = (depth - warped_depth).abs() * depth_inside
        stem_in = torch.cat(
            [
                rgb,
                depth,
                motion,
                disocclusion,
                lr24,
                feature_hist,
                R.depth_edges(depth),
            ],
            dim=1,
        )
        x = m.body(m.stem(stem_in, jitter))
        colour = m.head_colour(x)
        stat = m._head(m.head_weight, x)
        feat = m.head_feature(x)
        (out, new, flags) = hr_post(
            prev,
            motion,
            colour,
            stat,
            feat,
            rgb,
            jitter.float(),
            scale,
            self.width_k,
            self.inv_cap,
        )
        guard(flags, new, feat, rgb, jitter.float(), scale)
        return (out, new, feat)

    def step(self, rgb, depth, motion, jitter, state, init_hr=None, init_mask=None):
        if state is None:
            return type(self.m).step(
                self.m,
                rgb,
                depth,
                motion,
                jitter,
                None,
                init_hr=init_hr,
                init_mask=init_mask,
            )
        prev = packed_from_state(state)
        (out, new, feat) = self.inner(
            rgb, depth, motion, jitter, prev, state[2], state[3], state[9]
        )
        colour = new[:, 0:3]
        empty = new[:, :0]
        new_state = (
            colour,
            new[:, 3:4],
            depth,
            feat,
            new[:, 4:5],
            None,
            None,
            colour,
            empty,
            jitter.float(),
            empty,
        )
        return (out, new_state, {})


def install(model, **kw) -> FusedResolve:
    runner = FusedResolve(model, **kw)
    model.step = runner.step
    return runner
