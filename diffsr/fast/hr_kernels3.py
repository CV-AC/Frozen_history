from __future__ import annotations
import torch
import triton
import triton.language as tl
from .resolve_triton import (
    FP_FUSION,
    _bicubic_skip,
    _cc1,
    _cc2,
    _consts,
    _f32,
    _sigmoid,
    _warp_coords,
    libdevice,
)
from .resolve_triton2 import _bil_idx


def alloc4(H, W, device):
    s4 = torch.empty((1, H, W, 4), device=device, dtype=torch.float32)
    r = torch.empty((1, 1, H, W), device=device, dtype=torch.float32)
    return (s4, r)


def views4(s4, r):
    v = s4.permute(0, 3, 1, 2)
    return (v[:, 0:3], v[:, 3:4], r)


def is_ours4(state):
    (c, conf, reach) = (state[0], state[1], state[4])
    if c is None or conf is None or reach is None:
        return None
    if c.dim() != 4 or c.shape[1] != 3 or c.dtype != torch.float32:
        return None
    (H, W) = c.shape[-2:]
    if c.stride() != (4 * H * W, 1, 4 * W, 4):
        return None
    if (
        conf.stride() != (4 * H * W, 1, 4 * W, 4)
        or conf.data_ptr() != c.data_ptr() + 12
    ):
        return None
    if not reach.is_contiguous() or reach.dtype != torch.float32:
        return None
    s4 = torch.as_strided(c, (1, H, W, 4), (4 * H * W, 4 * W, 4, 1))
    return (s4, reach)


def pack4(state):
    got = is_ours4(state)
    if got is not None:
        return got
    (c, conf, reach) = (state[0], state[1], state[4])
    s4 = torch.cat((c.float(), conf.float()), dim=1).permute(0, 2, 3, 1).contiguous()
    return (s4, reach.float().contiguous())


@triton.jit
def _k1c_hr_pre(
    s4_ptr,
    r_ptr,
    motd_ptr,
    wa_ptr,
    wr_ptr,
    ba_ptr,
    stem_ptr,
    hist_ptr,
    rgb_ptr,
    jit_ptr,
    cand_ptr,
    h,
    w,
    H,
    W,
    s_sc,
    s_sh,
    s_sw,
    r_sc,
    r_sh,
    r_sw,
    c_sh,
    c_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    centre,
    inv_s,
    S: tl.constexpr,
    BP: tl.constexpr,
    PREC: tl.constexpr,
    HIST: tl.constexpr = False,
    ABL: tl.constexpr = 0,
    OFS: tl.constexpr = 7,
    SPLITDOT: tl.constexpr = False,
    SKIP: tl.constexpr = False,
    SKIP_PREC: tl.constexpr = "ieee",
):
    S2: tl.constexpr = S * S
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < w
    q = tl.arange(0, S2)
    ch = tl.arange(0, 4)
    Y = i * S + (q // S)[None, :]
    X = j[:, None] * S + (q % S)[None, :]
    mask = pm[:, None] & (q >= 0)[None, :]
    my = tl.load(motd_ptr + i * w + j, mask=pm, other=0.0)
    mx = tl.load(motd_ptr + h * w + i * w + j, mask=pm, other=0.0)
    (iy, ix, inside) = _warp_coords(
        Y.to(tl.float32),
        X.to(tl.float32),
        (my * sf)[:, None],
        (mx * sf)[:, None],
        inv_hm1,
        inv_wm1,
        Hm1f,
        Wm1f,
    )
    fy0 = tl.floor(iy)
    fx0 = tl.floor(ix)
    y0 = fy0.to(tl.int32)
    x0 = fx0.to(tl.int32)
    wy1 = iy - fy0
    wx1 = ix - fx0
    wy0 = fy0 + 1.0 - iy
    wx0 = fx0 + 1.0 - ix
    nw = wx0 * wy0
    ne = wx1 * wy0
    sw = wx0 * wy1
    se = wx1 * wy1
    my1 = mask & (y0 + 1 < H)
    mx1 = mask & (x0 + 1 < W)
    o00 = y0 * W + x0
    o01 = o00 + 1
    o10 = o00 + W
    o11 = o10 + 1
    c4 = ch[None, None, :]
    if ABL & 1 == 0:
        v = (
            tl.load(s4_ptr + o00[:, :, None] * 4 + c4, mask=mask[:, :, None], other=0.0)
            * nw[:, :, None]
        )
        v = tl.fma(
            tl.load(s4_ptr + o01[:, :, None] * 4 + c4, mask=mx1[:, :, None], other=0.0),
            ne[:, :, None],
            v,
        )
        v = tl.fma(
            tl.load(s4_ptr + o10[:, :, None] * 4 + c4, mask=my1[:, :, None], other=0.0),
            sw[:, :, None],
            v,
        )
        v = tl.fma(
            tl.load(
                s4_ptr + o11[:, :, None] * 4 + c4,
                mask=(my1 & mx1)[:, :, None],
                other=0.0,
            ),
            se[:, :, None],
            v,
        )
    else:
        v = (nw + ne + sw + se)[:, :, None] + c4.to(tl.float32)
    if ABL & 2 == 0:
        rw = tl.load(r_ptr + o00, mask=mask, other=0.0) * nw
        rw = tl.fma(tl.load(r_ptr + o01, mask=mx1, other=0.0), ne, rw)
        rw = tl.fma(tl.load(r_ptr + o10, mask=my1, other=0.0), sw, rw)
        rw = tl.fma(tl.load(r_ptr + o11, mask=my1 & mx1, other=0.0), se, rw)
    else:
        rw = nw * 0.5 + se
    ins = inside.to(tl.float32)
    v = tl.where(c4 == 3, v * ins[:, :, None], v)
    rw = tl.where(mask, rw * ins + (1.0 - ins), 0.0)
    if HIST:
        tl.store(hist_ptr + (Y * W + X)[:, :, None] * 4 + c4, v, mask=mask[:, :, None])
    n = tl.arange(0, 32)
    k64 = tl.arange(0, S2 * 4)
    if SPLITDOT:
        (a02, a13) = tl.split(tl.reshape(v, (BP, S2, 2, 2)))
        (v0, v2) = tl.split(a02)
        (v1, v3) = tl.split(a13)
        wq = (q * 4)[:, None] * 32 + n[None, :]
        acc = tl.dot(v0, tl.load(wa_ptr + wq), input_precision=PREC)
        acc = tl.dot(v1, tl.load(wa_ptr + wq + 32), acc, input_precision=PREC)
        acc = tl.dot(v2, tl.load(wa_ptr + wq + 64), acc, input_precision=PREC)
        acc = tl.dot(v3, tl.load(wa_ptr + wq + 96), acc, input_precision=PREC)
        acc = tl.dot(
            rw,
            tl.load(wr_ptr + q[:, None] * 32 + n[None, :]),
            acc,
            input_precision=PREC,
        )
    elif PREC == "bf16":
        x4 = tl.reshape(v, (BP, S2 * 4)).to(tl.bfloat16)
        wa = tl.load(wa_ptr + k64[:, None] * 32 + n[None, :]).to(tl.bfloat16)
        wr = tl.load(wr_ptr + q[:, None] * 32 + n[None, :]).to(tl.bfloat16)
        acc = tl.dot(x4, wa)
        acc = tl.dot(rw.to(tl.bfloat16), wr, acc)
        acc = acc.to(tl.bfloat16).to(tl.float32)
    elif ABL & 4 == 0:
        x4 = tl.reshape(v, (BP, S2 * 4))
        wa = tl.load(wa_ptr + k64[:, None] * 32 + n[None, :])
        wr = tl.load(wr_ptr + q[:, None] * 32 + n[None, :])
        acc = tl.dot(x4, wa, input_precision=PREC)
        acc = tl.dot(rw, wr, acc, input_precision=PREC)
    else:
        x4 = tl.reshape(v, (BP, S2 * 4))
        acc = (
            tl.reshape(tl.sum(tl.reshape(x4, (BP, 32, 2)), axis=2), (BP, 32))
            + tl.sum(rw, axis=1)[:, None]
        )
    acc = acc + tl.load(ba_ptr + n)[None, :]
    dt = stem_ptr.dtype.element_ty
    if ABL & 8 == 0:
        tl.store(
            stem_ptr + i * s_sh + j[:, None] * s_sw + (OFS + n)[None, :] * s_sc,
            acc.to(dt),
            mask=pm[:, None] & (n < (24 if OFS == 7 else 32))[None, :],
        )
    else:
        tl.store(
            stem_ptr + i * s_sh + j * s_sw + 7 * s_sc,
            tl.sum(acc, axis=1).to(dt),
            mask=pm,
        )
    if SKIP:
        jy = tl.load(jit_ptr)
        jx = tl.load(jit_ptr + 1)
        kk = tl.arange(0, 32)
        wsk = _skip_weights(q, kk, centre, inv_s, -jy, S, 0) * _skip_weights(
            q, kk, centre, inv_s, jx, S, 1
        )
        prow = tl.minimum(tl.maximum(i - 2 + kk // 5, 0), h - 1)
        pcol = tl.minimum(tl.maximum(j[:, None] - 2 + (kk % 5)[None, :], 0), w - 1)
        pmask = pm[:, None] & (kk < 25)[None, :]
        pofs = prow[None, :] * r_sh + pcol * r_sw
        cb = cand_ptr + i * c_sh + j[:, None] * c_sw + q[None, :]
        for c in tl.static_range(3):
            sk = tl.dot(
                tl.load(rgb_ptr + c * r_sc + pofs, mask=pmask, other=0.0),
                wsk,
                input_precision=SKIP_PREC,
            )
            sk = tl.minimum(tl.maximum(sk, 0.0), 1.0)
            tl.store(cb + c * 16, sk, mask=pm[:, None])


@triton.jit
def _skip_weights(q, k, centre, inv_s, off, S: tl.constexpr, AXIS: tl.constexpr):
    if AXIS == 0:
        d = q // S
        kk = k // 5
    else:
        d = q % S
        kk = k % 5
    a = (d.to(tl.float32) - centre) * inv_s - off
    fa = tl.floor(a)
    t = a - fa
    c0 = _cc2(t + 1.0)
    c1 = _cc1(t)
    c2 = _cc1(1.0 - t)
    c3 = _cc2(1.0 - t + 1.0)
    tap = kk[:, None] - (fa.to(tl.int32) + 1)[None, :]
    wv = tl.where(
        tap == 0,
        c0[None, :],
        tl.where(
            tap == 1,
            c1[None, :],
            tl.where(tap == 2, c2[None, :], tl.where(tap == 3, c3[None, :], 0.0)),
        ),
    )
    return wv


@triton.jit
def _skip_patch(
    rgb_ptr,
    r_sc,
    r_sh,
    r_sw,
    i,
    j0,
    h,
    w,
    jy,
    jx,
    centre,
    inv_s,
    kk,
    qq,
    S: tl.constexpr,
    BP: tl.constexpr,
    SKIP_PREC: tl.constexpr,
):
    wsk = _skip_weights(qq, kk, centre, inv_s, -jy, S, 0) * _skip_weights(
        qq, kk, centre, inv_s, jx, S, 1
    )
    pj = j0 + tl.arange(0, BP)
    prow = tl.minimum(tl.maximum(i - 2 + kk // 5, 0), h - 1)
    pcol = tl.minimum(tl.maximum(pj[:, None] - 2 + (kk % 5)[None, :], 0), w - 1)
    pmask = (pj < w)[:, None] & (kk < 25)[None, :]
    pofs = prow[None, :] * r_sh + pcol * r_sw
    sk0 = tl.dot(
        tl.load(rgb_ptr + pofs, mask=pmask, other=0.0), wsk, input_precision=SKIP_PREC
    )
    sk1 = tl.dot(
        tl.load(rgb_ptr + r_sc + pofs, mask=pmask, other=0.0),
        wsk,
        input_precision=SKIP_PREC,
    )
    sk2 = tl.dot(
        tl.load(rgb_ptr + 2 * r_sc + pofs, mask=pmask, other=0.0),
        wsk,
        input_precision=SKIP_PREC,
    )
    sk0 = tl.minimum(tl.maximum(sk0, 0.0), 1.0)
    sk1 = tl.minimum(tl.maximum(sk1, 0.0), 1.0)
    sk2 = tl.minimum(tl.maximum(sk2, 0.0), 1.0)
    s0 = tl.permute(tl.reshape(sk0, (BP, S, S)), (1, 0, 2))
    s1 = tl.permute(tl.reshape(sk1, (BP, S, S)), (1, 0, 2))
    s2 = tl.permute(tl.reshape(sk2, (BP, S, S)), (1, 0, 2))
    return (s0, s1, s2)


@triton.jit
def _k2c_hr_post(
    s4_ptr,
    motd_ptr,
    col_ptr,
    stat_ptr,
    feat_ptr,
    rgb_ptr,
    jit_ptr,
    out_ptr,
    s4n_ptr,
    rn_ptr,
    flag_ptr,
    h,
    w,
    H,
    W,
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
    half,
    width_k,
    inv_cap,
    g_lo,
    g_hi,
    f_box,
    S: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
    SKIP_PREC: tl.constexpr,
    FAST: tl.constexpr,
    ABL: tl.constexpr = 0,
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
    my = tl.load(motd_ptr + i * w + j, mask=pm, other=0.0)
    mx = tl.load(motd_ptr + h * w + i * w + j, mask=pm, other=0.0)
    (iy, ix, inside) = _warp_coords(
        Yf, Xf, my * sf, mx * sf, inv_hm1, inv_wm1, Hm1f, Wm1f
    )
    fy0 = tl.floor(iy)
    fx0 = tl.floor(ix)
    y0 = fy0.to(tl.int32)
    x0 = fx0.to(tl.int32)
    wy1 = iy - fy0
    wx1 = ix - fx0
    wy0 = fy0 + 1.0 - iy
    wx0 = fx0 + 1.0 - ix
    nw = (wx0 * wy0)[:, :, :, None]
    ne = (wx1 * wy0)[:, :, :, None]
    sw = (wx0 * wy1)[:, :, :, None]
    se = (wx1 * wy1)[:, :, :, None]
    my1 = (mask & (y0 + 1 < H))[:, :, :, None]
    mx1 = (mask & (x0 + 1 < W))[:, :, :, None]
    m4 = mask[:, :, :, None]
    c4 = tl.arange(0, 4)[None, None, None, :]
    o00 = (y0 * W + x0)[:, :, :, None] * 4 + c4
    if ABL & 1 == 0:
        v = tl.load(s4_ptr + o00, mask=m4, other=0.0) * nw
        v = tl.fma(tl.load(s4_ptr + o00 + 4, mask=mx1, other=0.0), ne, v)
        v = tl.fma(tl.load(s4_ptr + o00 + 4 * W, mask=my1, other=0.0), sw, v)
        v = tl.fma(tl.load(s4_ptr + o00 + 4 * W + 4, mask=my1 & mx1, other=0.0), se, v)
    else:
        v = nw + ne + sw + se + c4.to(tl.float32)
    (a02, a13) = tl.split(tl.reshape(v, (S, BP, S, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, cw) = tl.split(a13)
    valid = inside.to(tl.float32)
    conf = cw * valid
    jy = tl.load(jit_ptr)
    jx = tl.load(jit_ptr + 1)
    kk = tl.arange(0, 32)
    qq = tl.arange(0, S2)
    if ABL & 2 == 0:
        (s0, s1, s2) = _skip_patch(
            rgb_ptr,
            r_sc,
            r_sh,
            r_sw,
            i,
            j0,
            h,
            w,
            jy,
            jx,
            centre,
            inv_s,
            kk,
            qq,
            S,
            BP,
            SKIP_PREC,
        )
    else:
        s0 = Yf * 0.001
        s1 = Xf * 0.001
        s2 = s0 + s1
    q = dyi * S + dxi
    lr_c = i * c_sh + j * c_sw + q * c_sc
    if ABL & 4 == 0:
        k0 = tl.load(col_ptr + lr_c, mask=mask, other=0.0).to(tl.float32) + s0
        k1 = (
            tl.load(col_ptr + lr_c + S2 * c_sc, mask=mask, other=0.0).to(tl.float32)
            + s1
        )
        k2 = (
            tl.load(col_ptr + lr_c + 2 * S2 * c_sc, mask=mask, other=0.0).to(tl.float32)
            + s2
        )
        lr_t = i * t_sh + j * t_sw + q * t_sc
        a_rel = tl.load(stat_ptr + lr_t, mask=mask, other=0.0)
        a_ret = tl.load(stat_ptr + lr_t + S2 * t_sc, mask=mask, other=0.0)
    else:
        k0 = s0 + 0.1
        k1 = s1 + 0.2
        k2 = s2 + 0.3
        a_rel = s0 - 0.5
        a_ret = s1 + 2.0
    if ABL & 32 == 0:
        rel = _sigmoid(a_rel)
        ret = _sigmoid(a_ret)
    else:
        rel = 1.0 / (1.0 + tl.exp(-a_rel))
        ret = 1.0 / (1.0 + tl.exp(-a_ret))
    py = centre + -jy * sf - dyi.to(tl.float32)
    px = centre + jx * sf - dxi.to(tl.float32)
    py = libdevice.fmod(py + half, sf)
    py = tl.where((py != 0.0) & (py < 0.0), py + sf, py) - half
    px = libdevice.fmod(px + half, sf)
    px = tl.where((px != 0.0) & (px < 0.0), px + sf, px) - half
    radius = tl.sqrt_rn(py * py + px * px) * inv_s
    rk = tl.math.div_rn(radius, width_k)
    if ABL & 32 == 0:
        weight = libdevice.exp(-(rk * rk) * 0.5) * rel
    else:
        weight = tl.exp(-(rk * rk) * 0.5) * rel
    keep = ret * valid
    sn0 = keep * (h0 * conf) + weight * k0
    sn1 = keep * (h1 * conf) + weight * k1
    sn2 = keep * (h2 * conf) + weight * k2
    wn = tl.maximum(keep * conf + weight, 0.0001, propagate_nan=tl.PropagateNan.ALL)
    excess = tl.maximum(wn * inv_cap, 1.0, propagate_nan=tl.PropagateNan.ALL)
    if FAST:
        rex = 1.0 / excess
        sn0 = sn0 * rex
        sn1 = sn1 * rex
        sn2 = sn2 * rex
        wn = wn * rex
        rw_ = 1.0 / wn
        o0 = sn0 * rw_
        o1 = sn1 * rw_
        o2 = sn2 * rw_
    else:
        sn0 = tl.math.div_rn(sn0, excess)
        sn1 = tl.math.div_rn(sn1, excess)
        sn2 = tl.math.div_rn(sn2, excess)
        wn = tl.math.div_rn(wn, excess)
        o0 = tl.math.div_rn(sn0, wn)
        o1 = tl.math.div_rn(sn1, wn)
        o2 = tl.math.div_rn(sn2, wn)
    HW = H * W
    hr = Y * W + X
    if ABL & 8 == 0:
        tl.store(out_ptr + hr, o0, mask=mask)
        tl.store(out_ptr + HW + hr, o1, mask=mask)
        tl.store(out_ptr + 2 * HW + hr, o2, mask=mask)
    ov = tl.reshape(tl.join(tl.join(o0, o2), tl.join(o1, wn)), (S, BP, S, 4))
    tl.store(s4n_ptr + hr[:, :, :, None] * 4 + c4, ov, mask=m4)
    tl.store(rn_ptr + hr, radius + tl.zeros_like(o0), mask=mask)
    jj = j0 + tl.arange(0, BP)
    om = jj < w
    if ABL & 16 == 0:
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
        fc = tl.arange(0, FD)
        fv = tl.load(
            feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
            mask=om[:, None],
            other=0.0,
        ).to(tl.float32)
        fok = tl.abs(fv) <= f_box
        bad_f = tl.max((fok == 0).to(tl.int32), axis=1)
        flag = tl.maximum(bad_blk, bad_f)
    else:
        flag = tl.zeros([BP], tl.int32)
    tl.store(flag_ptr + i * w + jj, flag.to(tl.int8), mask=om)


@triton.jit
def _k2d_hr_post(
    hist_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
    jit_ptr,
    out_ptr,
    s4n_ptr,
    rn_ptr,
    flag_ptr,
    h,
    w,
    H,
    W,
    c_sc,
    c_sh,
    c_sw,
    t_sc,
    t_sh,
    t_sw,
    f_sc,
    f_sh,
    f_sw,
    sf,
    centre,
    inv_s,
    half,
    width_k,
    inv_cap,
    g_lo,
    g_hi,
    f_box,
    S: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
    FAST: tl.constexpr,
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
    hr = Y * W + X
    m4 = mask[:, :, :, None]
    c4 = tl.arange(0, 4)[None, None, None, :]
    v = tl.load(hist_ptr + hr[:, :, :, None] * 4 + c4, mask=m4, other=0.0)
    (a02, a13) = tl.split(tl.reshape(v, (S, BP, S, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, conf) = tl.split(a13)
    q = dyi * S + dxi
    lr_c = i * c_sh + j * c_sw + q * c_sc
    k0 = tl.load(cand_ptr + lr_c, mask=mask, other=0.0)
    k1 = tl.load(cand_ptr + lr_c + S2 * c_sc, mask=mask, other=0.0)
    k2 = tl.load(cand_ptr + lr_c + 2 * S2 * c_sc, mask=mask, other=0.0)
    lr_t = i * t_sh + j * t_sw + q * t_sc
    rel = _sigmoid(tl.load(stat_ptr + lr_t, mask=mask, other=0.0))
    ret = _sigmoid(tl.load(stat_ptr + lr_t + S2 * t_sc, mask=mask, other=0.0))
    jy = tl.load(jit_ptr)
    jx = tl.load(jit_ptr + 1)
    py = centre + -jy * sf - dyi.to(tl.float32)
    px = centre + jx * sf - dxi.to(tl.float32)
    py = libdevice.fmod(py + half, sf)
    py = tl.where((py != 0.0) & (py < 0.0), py + sf, py) - half
    px = libdevice.fmod(px + half, sf)
    px = tl.where((px != 0.0) & (px < 0.0), px + sf, px) - half
    radius = tl.sqrt_rn(py * py + px * px) * inv_s
    rk = tl.math.div_rn(radius, width_k)
    weight = libdevice.exp(-(rk * rk) * 0.5) * rel
    sn0 = ret * (h0 * conf) + weight * k0
    sn1 = ret * (h1 * conf) + weight * k1
    sn2 = ret * (h2 * conf) + weight * k2
    wn = tl.maximum(ret * conf + weight, 0.0001, propagate_nan=tl.PropagateNan.ALL)
    excess = tl.maximum(wn * inv_cap, 1.0, propagate_nan=tl.PropagateNan.ALL)
    if FAST:
        rex = 1.0 / excess
        sn0 = sn0 * rex
        sn1 = sn1 * rex
        sn2 = sn2 * rex
        wn = wn * rex
        rw_ = 1.0 / wn
        o0 = sn0 * rw_
        o1 = sn1 * rw_
        o2 = sn2 * rw_
    else:
        sn0 = tl.math.div_rn(sn0, excess)
        sn1 = tl.math.div_rn(sn1, excess)
        sn2 = tl.math.div_rn(sn2, excess)
        wn = tl.math.div_rn(wn, excess)
        o0 = tl.math.div_rn(sn0, wn)
        o1 = tl.math.div_rn(sn1, wn)
        o2 = tl.math.div_rn(sn2, wn)
    HW = H * W
    tl.store(out_ptr + hr, o0, mask=mask)
    tl.store(out_ptr + HW + hr, o1, mask=mask)
    tl.store(out_ptr + 2 * HW + hr, o2, mask=mask)
    ov = tl.reshape(tl.join(tl.join(o0, o2), tl.join(o1, wn)), (S, BP, S, 4))
    tl.store(s4n_ptr + hr[:, :, :, None] * 4 + c4, ov, mask=m4)
    tl.store(rn_ptr + hr, radius + tl.zeros_like(o0), mask=mask)
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
    if FD > 0:
        fc = tl.arange(0, FD)
        fv = tl.load(
            feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
            mask=om[:, None],
            other=0.0,
        ).to(tl.float32)
        fok = tl.abs(fv) <= f_box
        bad_f = tl.max((fok == 0).to(tl.int32), axis=1)
        flag = tl.maximum(bad_blk, bad_f)
    else:
        flag = bad_blk
    tl.store(flag_ptr + i * w + jj, flag.to(tl.int8), mask=om)


def launch_k2d(
    hist,
    cand,
    stat,
    feat,
    jf,
    out,
    s4n,
    rn,
    flags,
    width_k,
    inv_cap,
    scale,
    bp=16,
    warps=4,
    fast=True,
):
    assert scale == 4, "K2d: x4 only"
    (h, w) = flags.shape
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    _k2d_hr_post[triton.cdiv(w, bp), h](
        hist,
        cand,
        stat,
        feat,
        jf,
        out,
        s4n,
        rn,
        flags,
        h,
        w,
        H,
        W,
        cand.stride(1),
        cand.stride(2),
        cand.stride(3),
        stat.stride(1),
        stat.stride(2),
        stat.stride(3),
        feat.stride(1),
        feat.stride(2),
        feat.stride(3),
        c["sf"],
        c["centre"],
        c["inv_s"],
        _f32(0.5 * scale),
        width_k,
        inv_cap,
        -2.0,
        3.0,
        32.0,
        S=scale,
        BP=bp,
        FD=feat.shape[1],
        FAST=fast,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


@triton.jit
def _k3c_guard(
    flag_ptr,
    s4_ptr,
    r_ptr,
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
        if FD > 0:
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
        hr = Y * W + X
        c4 = tl.arange(0, 4)[None, None, None, :]
        zero = tl.zeros_like(s0)
        ov = tl.reshape(tl.join(tl.join(s0, s2), tl.join(s1, zero)), (S, BP, S, 4))
        tl.store(s4_ptr + hr[:, :, :, None] * 4 + c4, ov, mask=b3[:, :, :, None])
        tl.store(r_ptr + hr, zero + 1.0, mask=b3)


def proj_weights(model):
    s2 = model.scale**2
    dev = model.history_proj.weight.device
    wh = model.history_proj.weight.detach().float().reshape(16, 3, s2)
    wc = model.confidence_proj.weight.detach().float().reshape(4, s2)
    wr = model.reach_proj.weight.detach().float().reshape(4, s2)
    wa = torch.zeros(s2, 4, 32, device=dev)
    wa[:, 0:3, 0:16] = wh.permute(2, 1, 0)
    wa[:, 3, 16:20] = wc.t()
    wrr = torch.zeros(s2, 32, device=dev)
    wrr[:, 20:24] = wr.t()
    ba = torch.zeros(32, device=dev)
    ba[0:16] = model.history_proj.bias.detach().float()
    ba[16:20] = model.confidence_proj.bias.detach().float()
    ba[20:24] = model.reach_proj.bias.detach().float()
    return dict(
        wa=wa.reshape(s2 * 4, 32).contiguous(), wr=wrr.contiguous(), ba=ba.contiguous()
    )


def launch_k1c(
    s4,
    r,
    motd,
    W1,
    stem,
    scale,
    bp=16,
    warps=4,
    prec="tf32",
    abl=0,
    hist=None,
    ofs=7,
    splitdot=False,
    rgb=None,
    jf=None,
    cand=None,
    skip_prec="ieee",
):
    assert scale == 4, "K1c: x4 only"
    (h, w) = motd.shape[-2:]
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    _k1c_hr_pre[triton.cdiv(w, bp), h](
        s4,
        r,
        motd,
        W1["wa"],
        W1["wr"],
        W1["ba"],
        stem,
        hist if hist is not None else s4,
        rgb if cand is not None else s4,
        jf if cand is not None else s4,
        cand if cand is not None else s4,
        h,
        w,
        H,
        W,
        stem.stride(1),
        stem.stride(2),
        stem.stride(3),
        rgb.stride(1) if cand is not None else 0,
        rgb.stride(2) if cand is not None else 0,
        rgb.stride(3) if cand is not None else 0,
        cand.stride(2) if cand is not None else 0,
        cand.stride(3) if cand is not None else 0,
        c["inv_hm1"],
        c["inv_wm1"],
        c["Hm1f"],
        c["Wm1f"],
        c["sf"],
        c["centre"],
        c["inv_s"],
        S=scale,
        BP=bp,
        PREC=prec,
        HIST=hist is not None,
        ABL=abl,
        OFS=ofs,
        SPLITDOT=splitdot,
        SKIP=cand is not None,
        SKIP_PREC=skip_prec,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def launch_k2c(
    s4,
    motd,
    colour,
    stat,
    feat,
    rgb,
    jf,
    out,
    s4n,
    rn,
    flags,
    width_k,
    inv_cap,
    scale,
    bp=16,
    warps=4,
    skip_prec="ieee",
    fast=False,
    abl=0,
):
    assert scale == 4, "K2c: x4 only"
    (h, w) = motd.shape[-2:]
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    _k2c_hr_post[triton.cdiv(w, bp), h](
        s4,
        motd,
        colour,
        stat,
        feat,
        rgb,
        jf,
        out,
        s4n,
        rn,
        flags,
        h,
        w,
        H,
        W,
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
        _f32(0.5 * scale),
        width_k,
        inv_cap,
        -2.0,
        3.0,
        32.0,
        S=scale,
        BP=bp,
        FD=feat.shape[1],
        SKIP_PREC=skip_prec,
        FAST=fast,
        ABL=abl,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def launch_k3c(flags, s4, r, feat, rgb, jf, scale, bp=32, warps=4):
    (h, w) = flags.shape
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    _k3c_guard[triton.cdiv(w, bp), h](
        flags,
        s4,
        r,
        feat,
        rgb,
        jf,
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
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


GROUPED_MAP = (
    [(c, c) for c in range(7)]
    + [(c, c + 9) for c in range(7, 31)]
    + [(c, c + 17) for c in range(31, 63)]
    + [(c, c + 17) for c in range(63, 71)]
)
GROUPED_C = 96


def grouped_map(fd: int = 32):
    return (
        [(c, c) for c in range(7)]
        + [(c, c + 9) for c in range(7, 31)]
        + [(c, c + 17) for c in range(31, 39 + fd)]
    )


def grouped_c(fd: int = 32) -> int:
    return 64 + fd


def stem_grouped_conv(stem_conv):
    b = stem_conv
    fd = b.in_channels - 39
    assert fd >= 0, b.in_channels
    k = b.kernel_size[0]
    conv = torch.nn.Conv2d(grouped_c(fd), b.out_channels, k, padding=k // 2).to(
        device=b.weight.device, dtype=torch.float32
    )
    with torch.no_grad():
        conv.weight.zero_()
        for src, dst in grouped_map(fd):
            conv.weight[:, dst] = b.weight[:, src].float()
        conv.bias.copy_(b.bias.float())
    return conv


@triton.jit
def _k01_prep(
    rgb_ptr,
    dep_ptr,
    mot_ptr,
    jit_ptr,
    pjit_ptr,
    pdep_ptr,
    pfeat_ptr,
    s4_ptr,
    r_ptr,
    wa_ptr,
    wr_ptr,
    ba_ptr,
    stem_ptr,
    hist_ptr,
    motd_ptr,
    rgbp_ptr,
    h,
    w,
    H,
    W,
    r_sc,
    r_sh,
    r_sw,
    d_sh,
    d_sw,
    m_sc,
    m_sh,
    m_sw,
    pd_sh,
    pd_sw,
    pf_sc,
    pf_sh,
    pf_sw,
    s_sc,
    s_sh,
    s_sw,
    inv_lhm1,
    inv_lwm1,
    lhm1f,
    lwm1f,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    centre,
    inv_s,
    BP: tl.constexpr,
    FD: tl.constexpr,
    PREC: tl.constexpr,
    HIST: tl.constexpr,
    DO_HR: tl.constexpr = True,
    MOTD: tl.constexpr = False,
    RGBP: tl.constexpr = False,
    HX: tl.constexpr = False,
):
    S: tl.constexpr = 4
    S2: tl.constexpr = 16
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < w
    best = tl.full([BP], float("-inf"), tl.float32)
    smy = tl.zeros([BP], tl.float32)
    smx = tl.zeros([BP], tl.float32)
    for dy in tl.static_range(3):
        for dx in tl.static_range(3):
            ii = i + dy - 1
            jj = j + dx - 1
            m = pm & (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
            key = -tl.load(dep_ptr + ii * d_sh + jj * d_sw, mask=m, other=0.0)
            better = m & ((key > best) | (key != key))
            best = tl.where(better, key, best)
            vy = tl.load(mot_ptr + ii * m_sh + jj * m_sw, mask=m, other=0.0)
            vx = tl.load(mot_ptr + m_sc + ii * m_sh + jj * m_sw, mask=m, other=0.0)
            smy = tl.where(better, vy, smy)
            smx = tl.where(better, vx, smx)
    d0 = tl.load(jit_ptr) - tl.load(pjit_ptr)
    d1 = tl.load(jit_ptr + 1) - tl.load(pjit_ptr + 1)
    mly = smy + -d0
    mlx = smx + d1
    Yf = (tl.zeros([BP], tl.int32) + i).to(tl.float32)
    Xf = j.to(tl.float32)
    (iy, ix, ins_d) = _warp_coords(Yf, Xf, smy, smx, inv_lhm1, inv_lwm1, lhm1f, lwm1f)
    (y0, x0, y1, x1, nw, ne, sw, se, m00, m01, m10, m11) = _bil_idx(iy, ix, h, w, pm)
    acc = tl.load(pdep_ptr + y0 * pd_sh + x0 * pd_sw, mask=m00, other=0.0) * nw
    acc = tl.fma(
        tl.load(pdep_ptr + y0 * pd_sh + x1 * pd_sw, mask=m01, other=0.0), ne, acc
    )
    acc = tl.fma(
        tl.load(pdep_ptr + y1 * pd_sh + x0 * pd_sw, mask=m10, other=0.0), sw, acc
    )
    acc = tl.fma(
        tl.load(pdep_ptr + y1 * pd_sh + x1 * pd_sw, mask=m11, other=0.0), se, acc
    )
    dc = tl.load(dep_ptr + i * d_sh + j * d_sw, mask=pm, other=1.0)
    disocc = tl.abs(dc - acc) * ins_d.to(tl.float32)
    if FD > 0:
        (iy, ix, ins_f) = _warp_coords(
            Yf, Xf, mly, mlx, inv_lhm1, inv_lwm1, lhm1f, lwm1f
        )
        (y0, x0, y1, x1, nw, ne, sw, se, m00, m01, m10, m11) = _bil_idx(
            iy, ix, h, w, pm
        )
        fc = tl.arange(0, FD)[None, :] * pf_sc
        f = (
            tl.load(
                pfeat_ptr + (y0 * pf_sh + x0 * pf_sw)[:, None] + fc,
                mask=m00[:, None],
                other=0.0,
            ).to(tl.float32)
            * nw[:, None]
        )
        f = tl.fma(
            tl.load(
                pfeat_ptr + (y0 * pf_sh + x1 * pf_sw)[:, None] + fc,
                mask=m01[:, None],
                other=0.0,
            ).to(tl.float32),
            ne[:, None],
            f,
        )
        f = tl.fma(
            tl.load(
                pfeat_ptr + (y1 * pf_sh + x0 * pf_sw)[:, None] + fc,
                mask=m10[:, None],
                other=0.0,
            ).to(tl.float32),
            sw[:, None],
            f,
        )
        f = tl.fma(
            tl.load(
                pfeat_ptr + (y1 * pf_sh + x1 * pf_sw)[:, None] + fc,
                mask=m11[:, None],
                other=0.0,
            ).to(tl.float32),
            se[:, None],
            f,
        )
        f = f * ins_f.to(tl.float32)[:, None]
    k16 = tl.arange(0, 16)
    kk = k16 + (k16 >= 4).to(tl.int32)
    km8 = (k16 < 8)[None, :]
    ei = tl.minimum(tl.maximum(i + kk // 3 - 1, 0), h - 1)
    ej = tl.minimum(tl.maximum(j[:, None] + (kk % 3)[None, :] - 1, 0), w - 1)
    sv = tl.load(
        dep_ptr + ei[None, :] * d_sh + ej * d_sw, mask=pm[:, None] & km8, other=0.0
    )
    sc = tl.maximum(tl.abs(dc), 0.001)
    e = tl.math.div_rn(sv - dc[:, None], sc[:, None])
    e = tl.minimum(tl.maximum(e, -4.0), 4.0)
    e = tl.where(km8, e, 0.0)
    r = tl.load(rgb_ptr + i * r_sh + j * r_sw, mask=pm, other=0.0)
    g = tl.load(rgb_ptr + r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    b = tl.load(rgb_ptr + 2 * r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    c16 = k16[None, :]
    v = tl.where(
        c16 == 0,
        r[:, None],
        tl.where(
            c16 == 1,
            g[:, None],
            tl.where(
                c16 == 2,
                b[:, None],
                tl.where(
                    c16 == 3,
                    dc[:, None],
                    tl.where(
                        c16 == 4,
                        smy[:, None],
                        tl.where(
                            c16 == 5,
                            smx[:, None],
                            tl.where(c16 == 6, disocc[:, None], 0.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    dt = stem_ptr.dtype.element_ty
    base = stem_ptr + i * s_sh + j[:, None] * s_sw
    tl.store(base + c16 * s_sc, v.to(dt), mask=pm[:, None])
    if FD > 0:
        tl.store(
            base + (48 + tl.arange(0, FD))[None, :] * s_sc, f.to(dt), mask=pm[:, None]
        )
    tl.store(base + (48 + FD + c16) * s_sc, e.to(dt), mask=pm[:, None])
    if MOTD:
        tl.store(motd_ptr + i * w + j, smy, mask=pm)
        tl.store(motd_ptr + h * w + i * w + j, smx, mask=pm)
    if RGBP:
        tl.store(rgbp_ptr + i * w + j, r, mask=pm)
        tl.store(rgbp_ptr + h * w + i * w + j, g, mask=pm)
        tl.store(rgbp_ptr + 2 * h * w + i * w + j, b, mask=pm)
    if HX:
        jx = tl.load(jit_ptr + 1)
        u = tl.arange(0, 8)
        dxv = tl.arange(0, 4)
        a = (dxv.to(tl.float32) - centre) * inv_s - jx
        fa = tl.floor(a)
        t = a - fa
        c0 = _cc2(t + 1.0)
        c1 = _cc1(t)
        c2 = _cc1(1.0 - t)
        c3 = _cc2(1.0 - t + 1.0)
        tap = u[None, :] - (fa.to(tl.int32) + 1)[:, None]
        wx = tl.where(
            tap == 0,
            c0[:, None],
            tl.where(
                tap == 1,
                c1[:, None],
                tl.where(tap == 2, c2[:, None], tl.where(tap == 3, c3[:, None], 0.0)),
            ),
        )
        hcols = tl.minimum(tl.maximum(j[:, None] - 2 + u[None, :], 0), w - 1)
        m5 = pm[:, None] & (u < 5)[None, :]
        hxb = rgbp_ptr + (i * w + j)[:, None] * 16 + dxv[None, :]
        for c in tl.static_range(3):
            rrow = tl.load(
                rgb_ptr + c * r_sc + i * r_sh + hcols * r_sw, mask=m5, other=0.0
            )
            hxc = tl.sum(rrow[:, None, :] * wx[None, :, :], axis=2)
            tl.store(hxb + c * 4, hxc, mask=pm[:, None])
    if DO_HR:
        _k01_hr(
            s4_ptr,
            r_ptr,
            wa_ptr,
            wr_ptr,
            ba_ptr,
            base,
            hist_ptr,
            smy,
            smx,
            i,
            j,
            pm,
            H,
            W,
            s_sc,
            inv_hm1,
            inv_wm1,
            Hm1f,
            Wm1f,
            sf,
            BP,
            PREC,
            HIST,
        )


@triton.jit
def _k01_hr(
    s4_ptr,
    r_ptr,
    wa_ptr,
    wr_ptr,
    ba_ptr,
    base,
    hist_ptr,
    smy,
    smx,
    i,
    j,
    pm,
    H,
    W,
    s_sc,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    BP: tl.constexpr,
    PREC: tl.constexpr,
    HIST: tl.constexpr,
):
    S: tl.constexpr = 4
    S2: tl.constexpr = 16
    dt = base.dtype.element_ty
    q = tl.arange(0, S2)
    ch = tl.arange(0, 4)
    Yh = i * S + (q // S)[None, :]
    Xh = j[:, None] * S + (q % S)[None, :]
    mask = pm[:, None] & (q >= 0)[None, :]
    (iy, ix, inside) = _warp_coords(
        Yh.to(tl.float32),
        Xh.to(tl.float32),
        (smy * sf)[:, None],
        (smx * sf)[:, None],
        inv_hm1,
        inv_wm1,
        Hm1f,
        Wm1f,
    )
    fy0 = tl.floor(iy)
    fx0 = tl.floor(ix)
    hy0 = fy0.to(tl.int32)
    hx0 = fx0.to(tl.int32)
    wy1 = iy - fy0
    wx1 = ix - fx0
    wy0 = fy0 + 1.0 - iy
    wx0 = fx0 + 1.0 - ix
    hnw = wx0 * wy0
    hne = wx1 * wy0
    hsw = wx0 * wy1
    hse = wx1 * wy1
    my1 = mask & (hy0 + 1 < H)
    mx1 = mask & (hx0 + 1 < W)
    o00 = hy0 * W + hx0
    c4 = ch[None, None, :]
    hv = (
        tl.load(s4_ptr + o00[:, :, None] * 4 + c4, mask=mask[:, :, None], other=0.0)
        * hnw[:, :, None]
    )
    hv = tl.fma(
        tl.load(
            s4_ptr + (o00 + 1)[:, :, None] * 4 + c4, mask=mx1[:, :, None], other=0.0
        ),
        hne[:, :, None],
        hv,
    )
    hv = tl.fma(
        tl.load(
            s4_ptr + (o00 + W)[:, :, None] * 4 + c4, mask=my1[:, :, None], other=0.0
        ),
        hsw[:, :, None],
        hv,
    )
    hv = tl.fma(
        tl.load(
            s4_ptr + (o00 + W + 1)[:, :, None] * 4 + c4,
            mask=(my1 & mx1)[:, :, None],
            other=0.0,
        ),
        hse[:, :, None],
        hv,
    )
    rw = tl.load(r_ptr + o00, mask=mask, other=0.0) * hnw
    rw = tl.fma(tl.load(r_ptr + o00 + 1, mask=mx1, other=0.0), hne, rw)
    rw = tl.fma(tl.load(r_ptr + o00 + W, mask=my1, other=0.0), hsw, rw)
    rw = tl.fma(tl.load(r_ptr + o00 + W + 1, mask=my1 & mx1, other=0.0), hse, rw)
    ins = inside.to(tl.float32)
    hv = tl.where(c4 == 3, hv * ins[:, :, None], hv)
    rw = tl.where(mask, rw * ins + (1.0 - ins), 0.0)
    if HIST:
        tl.store(
            hist_ptr + (Yh * W + Xh)[:, :, None] * 4 + c4, hv, mask=mask[:, :, None]
        )
    x4 = tl.reshape(hv, (BP, S2 * 4))
    n = tl.arange(0, 32)
    k64 = tl.arange(0, S2 * 4)
    wa = tl.load(wa_ptr + k64[:, None] * 32 + n[None, :])
    wr = tl.load(wr_ptr + q[:, None] * 32 + n[None, :])
    pacc = tl.dot(x4, wa, input_precision=PREC)
    pacc = tl.dot(rw, wr, pacc, input_precision=PREC)
    pacc = pacc + tl.load(ba_ptr + n)[None, :]
    tl.store(base + (16 + n)[None, :] * s_sc, pacc.to(dt), mask=pm[:, None])


def launch_k01(
    rgb,
    depth,
    motion,
    jf,
    pjf,
    prev_depth,
    prev_feat,
    s4,
    r,
    W1,
    stem,
    hist,
    bp=16,
    warps=4,
    prec="tf32",
    motd=None,
    do_hr=True,
    rgbp=None,
    hx=None,
):
    (h, w) = rgb.shape[-2:]
    (H, W) = (4 * h, 4 * w)
    assert stem.shape[1] == grouped_c(prev_feat.shape[1])
    c1 = _consts(h, w, 1)
    c4 = _consts(h, w, 4)
    _k01_prep[triton.cdiv(w, bp), h](
        rgb,
        depth,
        motion,
        jf,
        pjf,
        prev_depth,
        prev_feat,
        s4,
        r,
        W1["wa"],
        W1["wr"],
        W1["ba"],
        stem,
        hist if hist is not None else s4,
        motd if motd is not None else s4,
        rgbp if rgbp is not None else hx if hx is not None else s4,
        h,
        w,
        H,
        W,
        rgb.stride(1),
        rgb.stride(2),
        rgb.stride(3),
        depth.stride(2),
        depth.stride(3),
        motion.stride(1),
        motion.stride(2),
        motion.stride(3),
        prev_depth.stride(2),
        prev_depth.stride(3),
        prev_feat.stride(1),
        prev_feat.stride(2),
        prev_feat.stride(3),
        stem.stride(1),
        stem.stride(2),
        stem.stride(3),
        c1["inv_lhm1"],
        c1["inv_lwm1"],
        c1["lhm1f"],
        c1["lwm1f"],
        c4["inv_hm1"],
        c4["inv_wm1"],
        c4["Hm1f"],
        c4["Wm1f"],
        c4["sf"],
        c4["centre"],
        c4["inv_s"],
        BP=bp,
        FD=prev_feat.shape[1],
        PREC=prec,
        HIST=hist is not None,
        DO_HR=do_hr,
        MOTD=motd is not None,
        RGBP=rgbp is not None,
        HX=hx is not None,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )
