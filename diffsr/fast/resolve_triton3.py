from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from . import conv3 as C3
from . import hr_kernels3 as HK
from . import light3 as L3
from . import resolve_triton as RT
from . import resolve_triton2 as RT2
from .resolve_triton import FP_FUSION, _consts, _f32, _inv, libdevice

DT_CODE = RT2.DT_CODE


@triton.jit
def _rb(x, LOWP: tl.constexpr):
    if LOWP:
        return x.to(tl.bfloat16).to(tl.float32)
    else:
        return x


@triton.jit
def _k_film(
    jit_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    w3_ptr,
    b3_ptr,
    out_ptr,
    NO,
    HID: tl.constexpr,
    NOP: tl.constexpr,
    LOWP: tl.constexpr,
):
    two_pi = 6.2831854820251465
    k4 = tl.arange(0, 4)
    jv = tl.load(jit_ptr + k4 % 2)
    ph = tl.where(k4 < 2, jv, libdevice.sin(jv * two_pi))
    ph = _rb(ph, LOWP)
    hr = tl.arange(0, HID)
    w1 = tl.load(w1_ptr + hr[:, None] * 4 + k4[None, :])
    h1 = _rb(tl.sum(w1 * ph[None, :], axis=1), LOWP) + tl.load(b1_ptr + hr)
    h1 = _rb(tl.maximum(h1, 0.0), LOWP)
    hc = tl.arange(0, HID)
    w2 = tl.load(w2_ptr + hr[:, None] * HID + hc[None, :])
    h2 = _rb(tl.sum(w2 * tl.reshape(h1, (1, HID)), axis=1), LOWP) + tl.load(b2_ptr + hr)
    h2 = _rb(tl.maximum(h2, 0.0), LOWP)
    o = tl.arange(0, NOP)
    om = o < NO
    w3 = tl.load(w3_ptr + o[:, None] * HID + hc[None, :], mask=om[:, None], other=0.0)
    y = tl.load(b3_ptr + o, mask=om, other=0.0) + _rb(
        tl.sum(w3 * tl.reshape(h2, (1, HID)), axis=1), LOWP
    )
    tl.store(out_ptr + o, y, mask=om)


@triton.jit
def _k_heads(
    x_ptr,
    wc_ptr,
    bc_ptr,
    ws_ptr,
    bs_ptr,
    wf_ptr,
    bf_ptr,
    col_ptr,
    stat_ptr,
    feat_ptr,
    h,
    w,
    x_sc,
    x_sh,
    x_sw,
    c_sh,
    c_sw,
    s_sh,
    s_sw,
    f_sh,
    f_sw,
    D: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    NCP: tl.constexpr,
    NS: tl.constexpr,
    NF: tl.constexpr,
    BM: tl.constexpr,
    LOWP: tl.constexpr,
):
    i = tl.program_id(1)
    j = tl.program_id(0) * BM + tl.arange(0, BM)
    pm = j < w
    nc = tl.arange(0, NCP)
    ns = tl.arange(0, NS)
    nf = tl.arange(0, NF)
    acc_c = tl.zeros([BM, NCP], tl.float32)
    acc_s = tl.zeros([BM, NS], tl.float32)
    acc_f = tl.zeros([BM, NF], tl.float32)
    for k0 in tl.static_range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        km = k < D
        x = tl.load(
            x_ptr + i * x_sh + j[:, None] * x_sw + k[None, :] * x_sc,
            mask=pm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        ws = tl.load(
            ws_ptr + k[:, None] * NS + ns[None, :], mask=km[:, None], other=0.0
        )
        acc_s = tl.dot(x, ws, acc_s, input_precision="tf32")
        wc = tl.load(
            wc_ptr + k[:, None] * NCP + nc[None, :], mask=km[:, None], other=0.0
        )
        wf = tl.load(
            wf_ptr + k[:, None] * NF + nf[None, :], mask=km[:, None], other=0.0
        )
        if LOWP:
            xb = x.to(tl.bfloat16)
            acc_c = tl.dot(xb, wc, acc_c)
            acc_f = tl.dot(xb, wf, acc_f)
        else:
            acc_c = tl.dot(x, wc, acc_c, input_precision="tf32")
            acc_f = tl.dot(x, wf, acc_f, input_precision="tf32")
    acc_s = acc_s + tl.load(bs_ptr + ns)[None, :]
    tl.store(
        stat_ptr + i * s_sh + j[:, None] * s_sw + ns[None, :], acc_s, mask=pm[:, None]
    )
    acc_c = _rb(acc_c, LOWP) + tl.load(bc_ptr + nc, mask=nc < NC, other=0.0)[None, :]
    acc_f = _rb(acc_f, LOWP) + tl.load(bf_ptr + nf)[None, :]
    tl.store(
        col_ptr + i * c_sh + j[:, None] * c_sw + nc[None, :],
        acc_c.to(col_ptr.dtype.element_ty),
        mask=pm[:, None] & (nc < NC)[None, :],
    )
    tl.store(
        feat_ptr + i * f_sh + j[:, None] * f_sw + nf[None, :],
        acc_f.to(feat_ptr.dtype.element_ty),
        mask=pm[:, None],
    )


@triton.jit
def _k_heads_skip(
    x_ptr,
    wc_ptr,
    bc_ptr,
    ws_ptr,
    bs_ptr,
    wf_ptr,
    bf_ptr,
    rgb_ptr,
    jit_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
    h,
    w,
    x_sc,
    x_sh,
    x_sw,
    r_sc,
    r_sh,
    r_sw,
    c_sh,
    c_sw,
    s_sh,
    s_sw,
    f_sh,
    f_sw,
    centre,
    inv_s,
    D: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    NF: tl.constexpr,
    BM: tl.constexpr,
    LOWP: tl.constexpr,
    SKIP_PREC: tl.constexpr,
    CAND_IN: tl.constexpr = False,
    F32P: tl.constexpr = "tf32",
    HX: tl.constexpr = False,
):
    i = tl.program_id(1)
    j = tl.program_id(0) * BM + tl.arange(0, BM)
    pm = j < w
    q = tl.arange(0, 16)
    ns = tl.arange(0, NS)
    acc0 = tl.zeros([BM, 16], tl.float32)
    acc1 = tl.zeros([BM, 16], tl.float32)
    acc2 = tl.zeros([BM, 16], tl.float32)
    acc_s = tl.zeros([BM, NS], tl.float32)
    if NF > 0:
        nf = tl.arange(0, NF)
        acc_f = tl.zeros([BM, NF], tl.float32)
    for k0 in tl.static_range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        km = k < D
        x = tl.load(
            x_ptr + i * x_sh + j[:, None] * x_sw + k[None, :] * x_sc,
            mask=pm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        ws = tl.load(
            ws_ptr + k[:, None] * NS + ns[None, :], mask=km[:, None], other=0.0
        )
        acc_s = tl.dot(x, ws, acc_s, input_precision="tf32" if LOWP else F32P)
        wrow = wc_ptr + k[:, None] * 48 + q[None, :]
        w0 = tl.load(wrow, mask=km[:, None], other=0.0)
        w1 = tl.load(wrow + 16, mask=km[:, None], other=0.0)
        w2 = tl.load(wrow + 32, mask=km[:, None], other=0.0)
        if LOWP:
            xb = x.to(tl.bfloat16)
            acc0 = tl.dot(xb, w0, acc0)
            acc1 = tl.dot(xb, w1, acc1)
            acc2 = tl.dot(xb, w2, acc2)
        else:
            acc0 = tl.dot(x, w0, acc0, input_precision=F32P)
            acc1 = tl.dot(x, w1, acc1, input_precision=F32P)
            acc2 = tl.dot(x, w2, acc2, input_precision=F32P)
        if NF > 0:
            wf = tl.load(
                wf_ptr + k[:, None] * NF + nf[None, :], mask=km[:, None], other=0.0
            )
            if LOWP:
                acc_f = tl.dot(x.to(tl.bfloat16), wf, acc_f)
            else:
                acc_f = tl.dot(x, wf, acc_f, input_precision=F32P)
    acc_s = acc_s + tl.load(bs_ptr + ns)[None, :]
    tl.store(
        stat_ptr + i * s_sh + j[:, None] * s_sw + ns[None, :], acc_s, mask=pm[:, None]
    )
    if NF > 0:
        acc_f = _rb(acc_f, LOWP) + tl.load(bf_ptr + nf)[None, :]
        tl.store(
            feat_ptr + i * f_sh + j[:, None] * f_sw + nf[None, :],
            acc_f.to(feat_ptr.dtype.element_ty),
            mask=pm[:, None],
        )
    col0 = _rb(_rb(acc0, LOWP) + tl.load(bc_ptr + q)[None, :], LOWP)
    col1 = _rb(_rb(acc1, LOWP) + tl.load(bc_ptr + 16 + q)[None, :], LOWP)
    col2 = _rb(_rb(acc2, LOWP) + tl.load(bc_ptr + 32 + q)[None, :], LOWP)
    cbase = cand_ptr + i * c_sh + j[:, None] * c_sw + q[None, :]
    if CAND_IN:
        sk0 = tl.load(cbase, mask=pm[:, None], other=0.0)
        sk1 = tl.load(cbase + 16, mask=pm[:, None], other=0.0)
        sk2 = tl.load(cbase + 32, mask=pm[:, None], other=0.0)
    elif HX:
        jy = tl.load(jit_ptr)
        dyq = q // 4
        a = (dyq.to(tl.float32) - centre) * inv_s + jy
        fa = tl.floor(a)
        t = a - fa
        c0 = HK._cc2(t + 1.0)
        c1 = HK._cc1(t)
        c2 = HK._cc1(1.0 - t)
        c3 = HK._cc2(1.0 - t + 1.0)
        fa1 = fa.to(tl.int32) + 1
        sk0 = tl.zeros([BM, 16], tl.float32)
        sk1 = tl.zeros([BM, 16], tl.float32)
        sk2 = tl.zeros([BM, 16], tl.float32)
        for r in tl.static_range(5):
            tp = r - fa1
            wr = tl.where(
                tp == 0,
                c0,
                tl.where(
                    tp == 1, c1, tl.where(tp == 2, c2, tl.where(tp == 3, c3, 0.0))
                ),
            )[None, :]
            rr = tl.minimum(tl.maximum(i - 2 + r, 0), h - 1)
            hb = rgb_ptr + (rr * w + j)[:, None] * 16 + (q % 4)[None, :]
            sk0 = tl.fma(tl.load(hb, mask=pm[:, None], other=0.0), wr, sk0)
            sk1 = tl.fma(tl.load(hb + 4, mask=pm[:, None], other=0.0), wr, sk1)
            sk2 = tl.fma(tl.load(hb + 8, mask=pm[:, None], other=0.0), wr, sk2)
        sk0 = tl.minimum(tl.maximum(sk0, 0.0), 1.0)
        sk1 = tl.minimum(tl.maximum(sk1, 0.0), 1.0)
        sk2 = tl.minimum(tl.maximum(sk2, 0.0), 1.0)
    else:
        jy = tl.load(jit_ptr)
        jx = tl.load(jit_ptr + 1)
        kk = tl.arange(0, 32)
        wsk = HK._skip_weights(q, kk, centre, inv_s, -jy, 4, 0) * HK._skip_weights(
            q, kk, centre, inv_s, jx, 4, 1
        )
        prow = tl.minimum(tl.maximum(i - 2 + kk // 5, 0), h - 1)
        pcol = tl.minimum(tl.maximum(j[:, None] - 2 + (kk % 5)[None, :], 0), w - 1)
        pmask = pm[:, None] & (kk < 25)[None, :]
        pofs = prow[None, :] * r_sh + pcol * r_sw
        sk0 = tl.dot(
            tl.load(rgb_ptr + pofs, mask=pmask, other=0.0),
            wsk,
            input_precision=SKIP_PREC,
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
    tl.store(cbase, col0 + sk0, mask=pm[:, None])
    tl.store(cbase + 16, col1 + sk1, mask=pm[:, None])
    tl.store(cbase + 32, col2 + sk2, mask=pm[:, None])


@triton.jit
def _k2e_heads_post(
    x_ptr,
    wc_ptr,
    bc_ptr,
    ws_ptr,
    bs_ptr,
    wf_ptr,
    bf_ptr,
    rgb_ptr,
    jit_ptr,
    hist_ptr,
    out_ptr,
    s4n_ptr,
    rn_ptr,
    feat_ptr,
    flag_ptr,
    h,
    w,
    H,
    W,
    x_sc,
    x_sh,
    x_sw,
    r_sc,
    r_sh,
    r_sw,
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
    D: tl.constexpr,
    BK: tl.constexpr,
    NF: tl.constexpr,
    BP: tl.constexpr,
    LOWP: tl.constexpr,
    SKIP_PREC: tl.constexpr,
    FAST: tl.constexpr,
):
    S: tl.constexpr = 4
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    p = tl.arange(0, BP)
    j = j0 + p
    pm = j < w
    q = tl.arange(0, 16)
    nf = tl.arange(0, NF)
    acc0 = tl.zeros([BP, 16], tl.float32)
    acc1 = tl.zeros([BP, 16], tl.float32)
    acc2 = tl.zeros([BP, 16], tl.float32)
    accr = tl.zeros([BP, 16], tl.float32)
    acct = tl.zeros([BP, 16], tl.float32)
    accf = tl.zeros([BP, NF], tl.float32)
    for k0 in tl.static_range(0, D, BK):
        k = k0 + tl.arange(0, BK)
        km = k < D
        x = tl.load(
            x_ptr + i * x_sh + j[:, None] * x_sw + k[None, :] * x_sc,
            mask=pm[:, None] & km[None, :],
            other=0.0,
        ).to(tl.float32)
        wsrow = ws_ptr + k[:, None] * 32 + q[None, :]
        accr = tl.dot(
            x, tl.load(wsrow, mask=km[:, None], other=0.0), accr, input_precision="tf32"
        )
        acct = tl.dot(
            x,
            tl.load(wsrow + 16, mask=km[:, None], other=0.0),
            acct,
            input_precision="tf32",
        )
        wrow = wc_ptr + k[:, None] * 48 + q[None, :]
        w0 = tl.load(wrow, mask=km[:, None], other=0.0)
        w1 = tl.load(wrow + 16, mask=km[:, None], other=0.0)
        w2 = tl.load(wrow + 32, mask=km[:, None], other=0.0)
        wf = tl.load(
            wf_ptr + k[:, None] * NF + nf[None, :], mask=km[:, None], other=0.0
        )
        if LOWP:
            xb = x.to(tl.bfloat16)
            acc0 = tl.dot(xb, w0, acc0)
            acc1 = tl.dot(xb, w1, acc1)
            acc2 = tl.dot(xb, w2, acc2)
            accf = tl.dot(xb, wf, accf)
        else:
            acc0 = tl.dot(x, w0, acc0, input_precision="tf32")
            acc1 = tl.dot(x, w1, acc1, input_precision="tf32")
            acc2 = tl.dot(x, w2, acc2, input_precision="tf32")
            accf = tl.dot(x, wf, accf, input_precision="tf32")
    fv = _rb(accf, LOWP) + tl.load(bf_ptr + nf)[None, :]
    fst = fv.to(feat_ptr.dtype.element_ty)
    tl.store(
        feat_ptr + i * f_sh + j[:, None] * f_sw + nf[None, :], fst, mask=pm[:, None]
    )
    bad_f = tl.max(((tl.abs(fst.to(tl.float32)) <= f_box) == 0).to(tl.int32), axis=1)
    rel = _sigmoid3(accr + tl.load(bs_ptr + q)[None, :])
    ret = _sigmoid3(acct + tl.load(bs_ptr + 16 + q)[None, :])
    jy = tl.load(jit_ptr)
    jx = tl.load(jit_ptr + 1)
    kk = tl.arange(0, 32)
    wsk = HK._skip_weights(q, kk, centre, inv_s, -jy, 4, 0) * HK._skip_weights(
        q, kk, centre, inv_s, jx, 4, 1
    )
    prow = tl.minimum(tl.maximum(i - 2 + kk // 5, 0), h - 1)
    pcol = tl.minimum(tl.maximum(j[:, None] - 2 + (kk % 5)[None, :], 0), w - 1)
    pmask = pm[:, None] & (kk < 25)[None, :]
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
    k0v = _rb(_rb(acc0, LOWP) + tl.load(bc_ptr + q)[None, :], LOWP) + tl.minimum(
        tl.maximum(sk0, 0.0), 1.0
    )
    k1v = _rb(_rb(acc1, LOWP) + tl.load(bc_ptr + 16 + q)[None, :], LOWP) + tl.minimum(
        tl.maximum(sk1, 0.0), 1.0
    )
    k2v = _rb(_rb(acc2, LOWP) + tl.load(bc_ptr + 32 + q)[None, :], LOWP) + tl.minimum(
        tl.maximum(sk2, 0.0), 1.0
    )
    k0v = tl.reshape(k0v, (BP, S, S))
    k1v = tl.reshape(k1v, (BP, S, S))
    k2v = tl.reshape(k2v, (BP, S, S))
    rel = tl.reshape(rel, (BP, S, S))
    ret = tl.reshape(ret, (BP, S, S))
    dyi = tl.arange(0, S)[None, :, None]
    dxi = tl.arange(0, S)[None, None, :]
    pp = p[:, None, None]
    mask = (j0 + pp < w) & (dyi >= 0) & (dxi >= 0)
    Y = i * S + dyi
    X = (j0 + pp) * S + dxi
    hr = Y * W + X
    m4 = mask[:, :, :, None]
    c4 = tl.arange(0, 4)[None, None, None, :]
    v = tl.load(hist_ptr + hr[:, :, :, None] * 4 + c4, mask=m4, other=0.0)
    (a02, a13) = tl.split(tl.reshape(v, (BP, S, S, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, conf) = tl.split(a13)
    py = centre + -jy * sf - dyi.to(tl.float32)
    px = centre + jx * sf - dxi.to(tl.float32)
    py = libdevice.fmod(py + half, sf)
    py = tl.where((py != 0.0) & (py < 0.0), py + sf, py) - half
    px = libdevice.fmod(px + half, sf)
    px = tl.where((px != 0.0) & (px < 0.0), px + sf, px) - half
    radius = tl.sqrt_rn(py * py + px * px) * inv_s
    rk = tl.math.div_rn(radius, width_k)
    weight = libdevice.exp(-(rk * rk) * 0.5) * rel
    sn0 = ret * (h0 * conf) + weight * k0v
    sn1 = ret * (h1 * conf) + weight * k1v
    sn2 = ret * (h2 * conf) + weight * k2v
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
    ov = tl.reshape(tl.join(tl.join(o0, o2), tl.join(o1, wn)), (BP, S, S, 4))
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
    bad_blk = tl.max(tl.max(bad, axis=2), axis=1)
    flag = tl.maximum(bad_blk, bad_f)
    tl.store(flag_ptr + i * w + j, flag.to(tl.int8), mask=pm)


@triton.jit
def _sigmoid3(x):
    return tl.math.div_rn(1.0, 1.0 + libdevice.exp(-x))


def launch_k0(
    rgb, depth, motion, jf, pjf, prev_depth, prev_feat, motd, stem, bp=64, warps=2
):
    (h, w) = rgb.shape[-2:]
    c = _consts(h, w, 1)
    RT2._k0_lr_prep[triton.cdiv(w, bp), h](
        rgb,
        depth,
        motion,
        jf,
        pjf,
        prev_depth,
        prev_feat,
        motd,
        stem,
        h,
        w,
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
        c["inv_lhm1"],
        c["inv_lwm1"],
        c["lhm1f"],
        c["lwm1f"],
        BP=bp,
        FD=prev_feat.shape[1],
        PAD=False,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def launch_k1b(prev, motd, P, stem, scale, bp=16, warps=4):
    (h, w) = motd.shape[-2:]
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    RT2._k1b_hr_pre[triton.cdiv(w, bp), h](
        prev,
        motd,
        P["wh"],
        P["bh"],
        P["wc"],
        P["bc"],
        P["wr"],
        P["br"],
        stem,
        h,
        w,
        H,
        W,
        stem.stride(1),
        stem.stride(2),
        stem.stride(3),
        c["inv_hm1"],
        c["inv_wm1"],
        c["Hm1f"],
        c["Wm1f"],
        c["sf"],
        S=scale,
        BP=bp,
        QK=max(16, scale * scale),
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def launch_k2(
    prev, motd, colour, stat, feat, rgb, jf, out, new, flags, P, scale, bp=16, warps=4
):
    (h, w) = motd.shape[-2:]
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    RT._k2_hr_post[triton.cdiv(w, bp), h](
        prev,
        motd,
        colour,
        stat,
        feat,
        rgb,
        jf,
        out,
        new,
        flags,
        h,
        w,
        H,
        W,
        motd.stride(1),
        motd.stride(2),
        motd.stride(3),
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
        P["width_k"],
        P["inv_cap"],
        -2.0,
        3.0,
        32.0,
        S=scale,
        BP=bp,
        FD=feat.shape[1],
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def launch_k3(flags, new, feat, rgb, jf, scale, bp=None, warps=4):
    (h, w) = flags.shape
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    bp = bp or RT._bp(scale)
    RT._k3_guard[triton.cdiv(w, bp), h](
        flags,
        new,
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


def launch_film(jf, F_, out):
    _k_film[1,](
        jf,
        F_["w1"],
        F_["b1"],
        F_["w2"],
        F_["b2"],
        F_["w3"],
        F_["b3"],
        out,
        out.shape[0],
        HID=F_["hid"],
        NOP=triton.next_power_of_2(out.shape[0]),
        LOWP=F_["lowp"],
        num_warps=4 if out.shape[0] <= 128 else 8,
    )


def launch_heads(x, Hd, colour, stat, feat, bm=32, warps=4):
    (h, w) = x.shape[-2:]
    for t in (colour, stat, feat):
        assert t.stride(1) == 1, "head outputs must be channels_last"
    D = x.shape[1]
    _k_heads[triton.cdiv(w, bm), h](
        x,
        Hd["wc"],
        Hd["bc"],
        Hd["ws"],
        Hd["bs"],
        Hd["wf"],
        Hd["bf"],
        colour,
        stat,
        feat,
        h,
        w,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        colour.stride(2),
        colour.stride(3),
        stat.stride(2),
        stat.stride(3),
        feat.stride(2),
        feat.stride(3),
        D=D,
        BK=Hd["bk"],
        NC=Hd["nc"],
        NCP=Hd["ncp"],
        NS=Hd["ns"],
        NF=Hd["nf"],
        BM=bm,
        LOWP=Hd["lowp"],
        num_warps=warps,
    )


def launch_heads_skip(
    x,
    Hd,
    rgb,
    jf,
    cand,
    stat,
    feat,
    bm=32,
    warps=4,
    skip_prec="ieee",
    cand_in=False,
    hx=None,
):
    (h, w) = x.shape[-2:]
    for t in (cand, stat, feat):
        assert t.stride(1) == 1 or t.shape[1] == 0, "head outputs must be channels_last"
    c = _consts(h, w, 4)
    _k_heads_skip[triton.cdiv(w, bm), h](
        x,
        Hd["wc48"],
        Hd["bc"],
        Hd["ws"],
        Hd["bs"],
        Hd["wf"],
        Hd["bf"],
        hx if hx is not None else rgb,
        jf,
        cand,
        stat,
        feat,
        h,
        w,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        rgb.stride(1),
        rgb.stride(2),
        rgb.stride(3),
        cand.stride(2),
        cand.stride(3),
        stat.stride(2),
        stat.stride(3),
        feat.stride(2),
        feat.stride(3),
        c["centre"],
        c["inv_s"],
        D=x.shape[1],
        BK=Hd["bk"],
        NS=Hd["ns"],
        NF=Hd["nf"],
        BM=bm,
        LOWP=Hd["lowp"],
        SKIP_PREC=skip_prec,
        CAND_IN=cand_in,
        F32P=Hd.get("f32p", "tf32"),
        HX=hx is not None,
        num_warps=warps,
    )


def launch_k2e(
    x,
    Hd,
    rgb,
    jf,
    hist,
    out,
    s4n,
    rn,
    feat,
    flags,
    width_k,
    inv_cap,
    bp=32,
    warps=4,
    skip_prec="ieee",
    fast=True,
):
    (h, w) = x.shape[-2:]
    (H, W) = (4 * h, 4 * w)
    assert feat.stride(1) == 1
    c = _consts(h, w, 4)
    _k2e_heads_post[triton.cdiv(w, bp), h](
        x,
        Hd["wc48"],
        Hd["bc"],
        Hd["ws"],
        Hd["bs"],
        Hd["wf"],
        Hd["bf"],
        rgb,
        jf,
        hist,
        out,
        s4n,
        rn,
        feat,
        flags,
        h,
        w,
        H,
        W,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        rgb.stride(1),
        rgb.stride(2),
        rgb.stride(3),
        feat.stride(2),
        feat.stride(3),
        c["sf"],
        c["centre"],
        c["inv_s"],
        _f32(2.0),
        width_k,
        inv_cap,
        -2.0,
        3.0,
        32.0,
        D=x.shape[1],
        BK=Hd["bk"],
        NF=Hd["nf"],
        BP=bp,
        LOWP=Hd["lowp"],
        SKIP_PREC=skip_prec,
        FAST=fast,
        num_warps=warps,
        enable_fp_fusion=FP_FUSION,
    )


def block_convs(blk):
    name = type(blk).__name__
    if name == "Block":
        if (
            blk.kind != "conv3"
            or blk.gate is not None
            or (not isinstance(blk.norm, torch.nn.Identity))
        ):
            raise ValueError("Block must be conv3, no norm, no gate")
        (act, c2) = (blk.mix[0], blk.mix[1])
        c1 = blk.spatial
    elif name == "Conv3Block":
        (c1, act, c2) = (blk.spatial, blk.mix[0], blk.mix[1])
    elif name == "Rep3Block":
        if blk.spatial.fused is None or blk.conv2.fused is None:
            raise ValueError("Rep3Block not folded (call fuse_model)")
        (c1, act, c2) = (blk.spatial.fused, blk.act, blk.conv2.fused)
    else:
        raise ValueError(f"unsupported body block {name}")
    if not isinstance(act, torch.nn.GELU) or act.approximate != "none":
        raise ValueError("block activation must be exact GELU")
    for c in (c1, c2):
        if not (
            isinstance(c, torch.nn.Conv2d)
            and c.kernel_size == (3, 3)
            and (c.padding == (1, 1))
            and (c.stride == (1, 1))
            and (c.groups == 1)
            and (c.bias is not None)
        ):
            raise ValueError("block convs must be 3x3/pad1/stride1 w/ bias")
    return (c1, c2, blk.scale)


DEFAULT_OPTS = dict(
    hr="v4",
    trunk="auto",
    stem_cin=None,
    prep="grouped",
    k01=(16, 4),
    k0=(32, 1),
    k1=(16, 4),
    k1_prec=None,
    heads_f32p="ieee",
    k1_splitdot=False,
    rgb_planar=False,
    skip_mode="hx",
    heads_in_conv=True,
    skip_in_k1=False,
    k2=(16, 2),
    k2_prec="ieee",
    k2_fast=False,
    k2d=(16, 4),
    k2e=(32, 4),
    k3=(256, 4),
    side_stream=True,
    heads_bm=64,
    heads_warps=4,
    conv={},
)


def _cl(t):
    return t.is_contiguous(memory_format=torch.channels_last) and (
        not t.is_contiguous()
    )


class FusedResolve3:
    def __init__(
        self,
        model,
        *,
        amp=torch.bfloat16,
        compile: bool = True,
        mode: str = "default",
        opts: dict | None = None,
    ):
        from .lean_step import check_config

        check_config(model, allow_fd0=True)
        fd = int(model.feature_dim)
        if fd and (fd & fd - 1 or fd < 16):
            raise ValueError(f"feature_dim {fd}: 0 or a power of two >= 16")
        for k, v in (
            ("consistency_gate", "none"),
            ("state_mode", "linear"),
            ("lr_unshuffle", 1),
        ):
            if getattr(model, k, v) != v:
                raise ValueError(f"v3 needs {k}={v!r}")
        if model.scale != 4:
            raise ValueError("v3 fused stage: x4 only")
        if model.stem.in_dim != 39 + fd:
            raise ValueError(
                f"stem input channels {model.stem.in_dim} != 39 + feature_dim"
            )
        if model.stem.kind not in ("conv3", "conv3_film", "jitter1x1"):
            raise ValueError(f"stem kind {model.stem.kind}")
        self.fd = fd
        self.j1 = model.stem.kind == "jitter1x1"
        self.m = model
        self.amp = amp
        self.lowp = amp is not None
        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        o = self.opts
        if fd == 0 and "heads_warps" not in (opts or {}):
            o["heads_warps"] = 2
        if o["trunk"] == "auto":
            sm90 = torch.cuda.get_device_capability()[0] >= 9
            o["trunk"] = (
                "inductor" if model.dim > 96 or (sm90 and model.dim >= 96) else "triton"
            )
        if o["trunk"] in ("triton", "light") and (not self.lowp):
            o["trunk"] = "inductor"
        if self.j1 and o["trunk"] not in ("triton", "light"):
            raise ValueError("stem jitter1x1: Triton / light trunks only")
        if fd != 32 or self.j1:
            if o["prep"] != "grouped" or o["hr"] != "v4":
                raise ValueError(
                    "feature_dim != 32 / jitter1x1 need prep='grouped', hr='v4'"
                )
        if o["prep"] in ("k01", "grouped") and o["hr"] not in ("v3", "v4"):
            o["prep"] = "split"
        if o["prep"] == "k01" and o["hr"] != "v4":
            o["prep"] = "grouped"
        if o["hr"] != "v4" or o["prep"] == "k01":
            o["skip_in_k1"] = False
        self._hx = (
            o["skip_mode"] == "hx"
            and o["hr"] == "v4"
            and (o["prep"] == "grouped")
            and (not o["skip_in_k1"])
            and (not o["rgb_planar"])
        )
        self._lhic = False
        self._hic = bool(
            o["heads_in_conv"]
            and self._hx
            and self.lowp
            and (o["trunk"] == "triton")
            and (len(model.body) > 0)
            and (model.dim <= 64)
            and (fd > 0)
            and (model.dim & model.dim - 1 == 0)
        )
        if o["prep"] in ("k01", "grouped"):
            o["stem_cin"] = HK.grouped_c(fd)
        if o["stem_cin"] is None:
            sm90 = torch.cuda.get_device_capability()[0] >= 9
            o["stem_cin"] = 80 if o["trunk"] == "inductor" and sm90 else 96
        self.cp = int(o["stem_cin"])
        if o["k1_prec"] is None:
            o["k1_prec"] = "bf16" if self.lowp else "ieee"
        assert self.cp >= 40 + fd and self.cp % 8 == 0
        dev = model.stem.base.weight.device
        with torch.no_grad():
            self.P = dict(
                wh=model.history_proj.weight.detach()
                .float()
                .reshape(16, 48)
                .contiguous(),
                bh=model.history_proj.bias.detach().float().contiguous(),
                wc=model.confidence_proj.weight.detach()
                .float()
                .reshape(4, 16)
                .contiguous(),
                bc=model.confidence_proj.bias.detach().float().contiguous(),
                wr=model.reach_proj.weight.detach().float().reshape(4, 16).contiguous(),
                br=model.reach_proj.bias.detach().float().contiguous(),
                width_k=float((F.softplus(model.kernel_width.detach()) + 0.25).item()),
                inv_cap=_inv(1.0 / model.rectifier.alpha_min),
            )
            self.W1 = HK.proj_weights(model)
            self._build_film(dev)
            self._build_heads(dev)
            self._build_trunk(dev)
        self.compiled = compile and o["trunk"] == "inductor"
        self.trunk = (
            torch.compile(self._trunk, mode=mode) if self.compiled else self._trunk
        )
        self._bufs = {}

    def _build_film(self, dev):
        st = self.m.stem
        self.F = None
        self.J = None
        if st.kind == "jitter1x1":
            (l1, l2, l3) = (st.mlp[0], st.mlp[2], st.mlp[4])
            lp = self.lowp

            def wq(t):
                t = t.detach().float()
                return (t.to(torch.bfloat16).float() if lp else t).contiguous()

            def bq(t):
                return t.detach().float().contiguous()

            gidx = torch.tensor(
                [g for (_, g) in HK.grouped_map(self.fd)], dtype=torch.int32, device=dev
            )
            assert gidx.numel() == st.in_dim
            self.J = dict(
                w1=wq(l1.weight),
                b1=bq(l1.bias),
                w2=wq(l2.weight),
                b2=bq(l2.bias),
                w3=wq(l3.weight),
                b3=bq(l3.bias),
                hid=l1.out_features,
                lowp=lp,
                gidx=gidx,
                IN=st.in_dim,
                OUT=st.out_dim,
                wd=torch.zeros(self.cp, st.out_dim, device=dev, dtype=torch.bfloat16),
            )
            assert l3.out_features == st.in_dim * st.out_dim
            return
        if st.kind != "conv3_film":
            return
        (l1, l2, l3) = (st.mlp[0], st.mlp[2], st.mlp[4])
        lp = self.lowp

        def wq(t):
            t = t.detach().float()
            return (t.to(torch.bfloat16).float() if lp else t).contiguous()

        def bq(t):
            return t.detach().float().contiguous()

        self.F = dict(
            w1=wq(l1.weight),
            b1=bq(l1.bias),
            w2=wq(l2.weight),
            b2=bq(l2.bias),
            w3=wq(l3.weight),
            b3=bq(l3.bias),
            hid=l1.out_features,
            lowp=lp,
        )
        assert l1.in_features == 4 and l2.in_features == l1.out_features

    def _build_heads(self, dev):
        m = self.m
        D = m.dim
        (hc, hw, hf) = (m.head_colour, m.head_weight, m.head_feature)
        (nc, ns) = (hc.out_channels, hw.out_channels)
        nf = hf.out_channels if hf is not None else 0
        ncp = triton.next_power_of_2(nc)
        for n in (ns, nf):
            assert n == 0 or (n == triton.next_power_of_2(n) and n >= 16)
        lp = self.lowp
        wdt = torch.bfloat16 if lp else torch.float32

        def wt(conv, npad, dt):
            wv = conv.weight.detach().float().reshape(conv.out_channels, D)
            out = torch.zeros(D, npad, device=dev, dtype=torch.float32)
            out[:, : conv.out_channels] = wv.t()
            return out.to(dt).contiguous()

        def bias(conv):
            return conv.bias.detach().float().contiguous()

        assert nc == 48
        dmy = torch.zeros(16, device=dev)
        self.Hd = dict(
            wc=wt(hc, ncp, wdt),
            wc48=wt(hc, 48, wdt),
            bc=bias(hc),
            ws=wt(hw, ns, torch.float32),
            bs=bias(hw),
            wf=wt(hf, nf, wdt) if nf else dmy.to(wdt),
            bf=bias(hf) if nf else dmy,
            nc=nc,
            ncp=ncp,
            ns=ns,
            nf=nf,
            lowp=lp,
            f32p=self.opts["heads_f32p"],
            bk=32 if D % 32 == 0 else 16,
        )
        if D % self.Hd["bk"]:
            raise ValueError(f"dim {D} not a multiple of 16")

    def _build_trunk(self, dev):
        m = self.m
        b = m.stem.base
        if self.opts["prep"] in ("k01", "grouped"):
            conv = HK.stem_grouped_conv(b)
        else:
            conv = torch.nn.Conv2d(self.cp, b.out_channels, 3, padding=1).to(
                device=b.weight.device, dtype=torch.float32
            )
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, : b.in_channels] = b.weight.float()
                conv.bias.copy_(b.bias.float())
        if self.opts["trunk"] == "light":
            lo = dict(self.opts.get("light") or {})
            D = m.dim
            epi_s = C3.EPI_FILM if self.F is not None else C3.EPI_BIAS
            scfg = self.opts["conv"].get("film")
            if scfg is None:
                if D & D - 1 and D < 64:
                    scfg = {**C3.default_cfg(64, epi_s), "BN": 64}
                else:
                    scfg = C3.default_cfg(D, epi_s)
            if self.j1:
                self.pc_stem = L3.Stem1x1(
                    conv, self.J["wd"], cfg=self.opts["conv"].get("stem1x1")
                )
            else:
                self.pc_stem = C3.make_padconv(conv, epi_s, cin_p=self.cp, cfg=scfg)
            blocks = lo.get("blocks")
            if blocks is None:
                blocks = L3.light_blocks_from_model(m)
            if blocks is None:
                blocks = L3.random_trunk(
                    lo["kind"],
                    D,
                    int(lo["B"]),
                    r=float(lo.get("r", 2.0)),
                    k=int(lo.get("k", 3)),
                    dil=str(lo.get("dil", "1,2")),
                    device=dev,
                    seed=int(lo.get("seed", 0)),
                )
            self.light = L3.LightTrunk(
                blocks,
                act=lo.get("act", "gelu"),
                cfg=lo.get("cfg"),
                shift_sign=int(lo.get("shift_sign", 1)),
            )
            self._lhic = bool(
                self.opts.get("light_heads", False)
                and self._hx
                and self.light.can_fuse_heads
            )
            self.stem_conv = None
            return
        light_body = getattr(m, "body_kind", "plain") in (
            "dwmlp",
            "shiftmlp",
            "pconvglu",
            "poolmlp",
            "linattn",
        )
        if light_body and (self.opts["trunk"] != "inductor" or self.lowp):
            raise ValueError(
                "light bodies: trunk='light' (bf16) or the fp32 inductor trunk"
            )
        self.blocks = [] if light_body else [block_convs(blk) for blk in m.body]
        if self.opts["trunk"] == "triton":
            cc = self.opts["conv"]
            D = m.dim
            epi_s = C3.EPI_FILM if self.F is not None else C3.EPI_BIAS

            def cfg(key, epi):
                return cc.get(key) or C3.default_cfg(D, epi)

            if self.j1:
                self.pc_stem = L3.Stem1x1(conv, self.J["wd"], cfg=cc.get("stem1x1"))
            else:
                self.pc_stem = C3.make_padconv(
                    conv, epi_s, cin_p=self.cp, cfg=cfg("film", epi_s)
                )
            if self._hic:
                (c1, c2, sc) = self.blocks[-1]
                self.pc_heads = C3.PadConvHeads(
                    c2,
                    self.Hd,
                    cfg=cc.get("headsconv") or C3.default_cfg(D, C3.EPI_RES),
                )
            self.pc_blocks = [
                (
                    C3.make_padconv(c1, C3.EPI_GELU, cfg=cfg("gelu", C3.EPI_GELU)),
                    C3.make_padconv(c2, C3.EPI_RES, cfg=cfg("res", C3.EPI_RES)),
                    s.detach().float().reshape(-1).contiguous(),
                )
                for (c1, c2, s) in self.blocks
            ]
            self.stem_conv = None
            return
        if _cl(b.weight):
            conv = conv.to(memory_format=torch.channels_last)
        if self.lowp:
            conv.weight.data = conv.weight.data.to(torch.bfloat16)
            for c1, c2, _ in self.blocks:
                for c in (c1, c2):
                    c.weight.data = c.weight.data.to(torch.bfloat16)
        self.stem_conv = conv

    def _trunk(self, stem_in, gb):
        out = self.stem_conv(stem_in)
        if gb is not None:
            (g, be) = gb.chunk(2)
            out = (out * (g[None, :, None, None] + 1.0) + be[None, :, None, None]).to(
                out.dtype
            )
        return self.m.body(out)

    def _ctx(self):
        return (
            torch.autocast("cuda", self.amp, cache_enabled=False)
            if self.lowp
            else contextlib.nullcontext()
        )

    def buffers(self, rgb):
        (h, w) = rgb.shape[-2:]
        k = (h, w)
        if k in self._bufs:
            return self._bufs[k]
        dev = rgb.device
        m = self.m
        dt = self.amp if self.lowp else torch.float32
        cl = torch.channels_last
        b = dict(
            motd=torch.empty((1, 2, h, w), device=dev, dtype=torch.float32),
            colour=torch.empty(
                (1, m.head_colour.out_channels, h, w),
                device=dev,
                dtype=dt,
                memory_format=cl,
            ),
            stat=torch.empty(
                (1, m.head_weight.out_channels, h, w),
                device=dev,
                dtype=torch.float32,
                memory_format=cl,
            ),
            flags=torch.zeros((h, w), device=dev, dtype=torch.int8),
            rgbp=torch.empty((1, 3, h, w), device=dev, dtype=torch.float32),
            hx=torch.empty((h * w * 16,), device=dev, dtype=torch.float32),
            cand=torch.empty(
                (1, 48, h, w), device=dev, dtype=torch.float32, memory_format=cl
            )
            if self.opts["hr"] == "v4"
            else None,
            hist=torch.empty(
                (h * m.scale * w * m.scale, 4), device=dev, dtype=torch.float32
            )
            if self.opts["hr"] in ("v4", "v5")
            else None,
            gb=torch.zeros((2 * m.dim,), device=dev, dtype=torch.float32)
            if self.F is not None
            else None,
        )
        if self.opts["trunk"] == "light":
            D = m.dim
            b["pad_stem"] = C3.PadBuf(self.cp, h, w, torch.bfloat16, dev)
            b["stem"] = b["pad_stem"].view
            b["pad_s"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
            b["pad_a"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
            b["pad_b"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
            b["pad_h"] = (
                C3.PadBuf(D, h, w, torch.bfloat16, dev)
                if self.light.needs_scratch
                else None
            )
        elif self.opts["trunk"] == "triton":
            D = m.dim
            b["pad_stem"] = C3.PadBuf(self.cp, h, w, torch.bfloat16, dev)
            b["stem"] = b["pad_stem"].view
            b["pad_s"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
            b["pad_h"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
            b["pad_xf"] = C3.PadBuf(D, h, w, torch.float32, dev)
            b["pad_xb"] = C3.PadBuf(D, h, w, torch.bfloat16, dev)
        else:
            cf = cl if _cl(self.stem_conv.weight) else torch.contiguous_format
            b["stem"] = torch.empty(
                (1, self.cp, h, w), device=dev, dtype=dt, memory_format=cf
            ).zero_()
        self._bufs[k] = b
        return b

    def new_state_buffers(self, rgb):
        (h, w) = rgb.shape[-2:]
        s = self.m.scale
        dev = rgb.device
        dt = self.amp if self.lowp else torch.float32
        d = dict(
            out=torch.empty((1, 3, h * s, w * s), device=dev, dtype=torch.float32),
            feat=torch.empty(
                (1, self.Hd["nf"], h, w),
                device=dev,
                dtype=dt,
                memory_format=torch.channels_last,
            ),
        )
        if self.opts["hr"] in ("v3", "v4", "v5"):
            (d["s4"], d["r"]) = HK.alloc4(h * s, w * s, dev)
        else:
            d["new"] = torch.empty(
                (1, 5, h * s, w * s), device=dev, dtype=torch.float32
            )
        return d

    def state_bufs(self, state):
        if self.opts["hr"] in ("v3", "v4", "v5"):
            (s4, r) = HK.pack4(state)
            return dict(s4=s4, r=r)
        return dict(new=RT.packed_from_state(state))

    def state_tuple(self, nb, depth, jitter):
        feat = nb["feat"]
        jf = jitter if jitter.dtype == torch.float32 else jitter.float()
        if self.opts["hr"] in ("v3", "v4", "v5"):
            (colour, conf, reach) = HK.views4(nb["s4"], nb["r"])
        else:
            new = nb["new"]
            (colour, conf, reach) = (new[:, 0:3], new[:, 3:4], new[:, 4:5])
        empty = colour[:, :0]
        return (colour, conf, depth, feat, reach, None, None, colour, empty, jf, empty)

    def run_into(
        self, rgb, depth, motion, jitter, prev, prev_depth, prev_feat, prev_jit, nb
    ):
        m = self.m
        s = m.scale
        o = self.opts
        b = self.buffers(rgb)
        jf = jitter if jitter.dtype == torch.float32 else jitter.float()
        pjf = prev_jit if prev_jit.dtype == torch.float32 else prev_jit.float()
        side = None
        if o["side_stream"] and (self.F is not None or self.J is not None):
            if getattr(self, "_side", None) is None:
                self._side = torch.cuda.Stream()
            side = self._side
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                if self.F is not None:
                    launch_film(jf, self.F, b["gb"])
                if self.J is not None:
                    L3.launch_j1x1(jf, self.J)
        if o["prep"] == "k01":
            HK.launch_k01(
                rgb,
                depth,
                motion,
                jf,
                pjf,
                prev_depth,
                prev_feat,
                prev["s4"],
                prev["r"],
                self.W1,
                b["stem"],
                b["hist"],
                *o["k01"],
                prec=o["k1_prec"],
            )
        elif o["prep"] == "grouped":
            HK.launch_k01(
                rgb,
                depth,
                motion,
                jf,
                pjf,
                prev_depth,
                prev_feat,
                prev["s4"],
                prev["r"],
                self.W1,
                b["stem"],
                None,
                *o["k0"],
                motd=b["motd"],
                do_hr=False,
                rgbp=b["rgbp"] if o["rgb_planar"] else None,
                hx=b["hx"] if self._hx else None,
            )
        else:
            launch_k0(
                rgb,
                depth,
                motion,
                jf,
                pjf,
                prev_depth,
                prev_feat,
                b["motd"],
                b["stem"],
                *o["k0"],
            )
        if o["prep"] == "k01":
            pass
        elif o["hr"] in ("v3", "v4", "v5"):
            HK.launch_k1c(
                prev["s4"],
                prev["r"],
                b["motd"],
                self.W1,
                b["stem"],
                s,
                *o["k1"],
                prec=o["k1_prec"],
                hist=b["hist"],
                ofs=16 if o["prep"] == "grouped" else 7,
                splitdot=o["k1_splitdot"],
                rgb=rgb if o["skip_in_k1"] else None,
                jf=jf,
                cand=b["cand"] if o["skip_in_k1"] else None,
                skip_prec=o["k2_prec"],
            )
        else:
            launch_k1b(prev["new"], b["motd"], self.P, b["stem"], s, *o["k1"])
        gb = b["gb"] if self.F is not None else None
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        else:
            if self.F is not None:
                launch_film(jf, self.F, b["gb"])
            if self.J is not None:
                L3.launch_j1x1(jf, self.J)
        lhic = o["trunk"] == "light" and self._lhic
        if o["trunk"] == "light":
            self.pc_stem(b["pad_stem"], b["pad_s"], gb=gb)
            heads = (
                dict(
                    Hd=self.Hd,
                    hx=b["hx"],
                    jf=jf,
                    cand=b["cand"],
                    stat=b["stat"],
                    feat=nb["feat"],
                )
                if lhic
                else None
            )
            x = self.light(b["pad_s"], b["pad_a"], b["pad_b"], b["pad_h"], heads=heads)
            x = x.view if x is not None else None
        elif o["trunk"] == "triton":
            x = self._trunk_triton(b, gb, heads=(jf, nb["feat"]) if self._hic else None)
        else:
            with self._ctx():
                x = self.trunk(b["stem"], gb)
        feat = nb["feat"]
        if self._hic or lhic:
            HK.launch_k2d(
                b["hist"],
                b["cand"],
                b["stat"],
                feat,
                jf,
                nb["out"],
                nb["s4"],
                nb["r"],
                b["flags"],
                self.P["width_k"],
                self.P["inv_cap"],
                s,
                *o["k2d"],
                fast=o["k2_fast"],
            )
            HK.launch_k3c(b["flags"], nb["s4"], nb["r"], feat, rgb, jf, s, *o["k3"])
            return nb
        if o["hr"] == "v5":
            launch_k2e(
                x,
                self.Hd,
                rgb,
                jf,
                b["hist"],
                nb["out"],
                nb["s4"],
                nb["r"],
                feat,
                b["flags"],
                self.P["width_k"],
                self.P["inv_cap"],
                *o["k2e"],
                skip_prec=o["k2_prec"],
                fast=o["k2_fast"],
            )
            HK.launch_k3c(b["flags"], nb["s4"], nb["r"], feat, rgb, jf, s, *o["k3"])
            return nb
        if o["hr"] == "v4":
            srgb = b["rgbp"] if o["rgb_planar"] and o["prep"] == "grouped" else rgb
            launch_heads_skip(
                x,
                self.Hd,
                srgb,
                jf,
                b["cand"],
                b["stat"],
                feat,
                bm=o["heads_bm"],
                warps=o["heads_warps"],
                skip_prec=o["k2_prec"],
                cand_in=o["skip_in_k1"],
                hx=b["hx"] if self._hx else None,
            )
            HK.launch_k2d(
                b["hist"],
                b["cand"],
                b["stat"],
                feat,
                jf,
                nb["out"],
                nb["s4"],
                nb["r"],
                b["flags"],
                self.P["width_k"],
                self.P["inv_cap"],
                s,
                *o["k2d"],
                fast=o["k2_fast"],
            )
            HK.launch_k3c(b["flags"], nb["s4"], nb["r"], feat, rgb, jf, s, *o["k3"])
            return nb
        launch_heads(
            x,
            self.Hd,
            b["colour"],
            b["stat"],
            feat,
            bm=o["heads_bm"],
            warps=o["heads_warps"],
        )
        if o["hr"] == "v3":
            HK.launch_k2c(
                prev["s4"],
                b["motd"],
                b["colour"],
                b["stat"],
                feat,
                rgb,
                jf,
                nb["out"],
                nb["s4"],
                nb["r"],
                b["flags"],
                self.P["width_k"],
                self.P["inv_cap"],
                s,
                *o["k2"],
                skip_prec=o["k2_prec"],
                fast=o["k2_fast"],
            )
            HK.launch_k3c(b["flags"], nb["s4"], nb["r"], feat, rgb, jf, s, *o["k3"])
        else:
            launch_k2(
                prev["new"],
                b["motd"],
                b["colour"],
                b["stat"],
                feat,
                rgb,
                jf,
                nb["out"],
                nb["new"],
                b["flags"],
                self.P,
                s,
                *o["k2"],
            )
            launch_k3(b["flags"], nb["new"], feat, rgb, jf, s, *o["k3"])
        return nb

    def _trunk_triton(self, b, gb, heads=None):
        (ps, pxs, ph, pxf, pxb) = (
            b["pad_stem"],
            b["pad_s"],
            b["pad_h"],
            b["pad_xf"],
            b["pad_xb"],
        )
        self.pc_stem(ps, pxs, gb=gb)
        (xin, res) = (pxs, pxs)
        nblk = len(self.pc_blocks)
        for k, (c1, c2, sc) in enumerate(self.pc_blocks):
            c1(xin, ph)
            if heads is not None and k == nblk - 1:
                (jf, feat) = heads
                self.pc_heads(ph, res, sc, b["hx"], jf, b["cand"], b["stat"], feat)
                return None
            c2(ph, pxf, res=res, scale=sc, y2=pxb if k < nblk - 1 else None)
            (xin, res) = (pxb, pxf)
        return pxf.view if nblk else pxs.view

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
        prev = self.state_bufs(state)
        nb = self.new_state_buffers(rgb)
        self.run_into(
            rgb, depth, motion, jitter, prev, state[2], state[3], state[9], nb
        )
        return (nb["out"], self.state_tuple(nb, depth, jitter), {})


def capture(fn, warm: int = 3):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ret = fn()
    return (g, ret)


class FixedGraph:
    def __init__(self, runner: FusedResolve3, rgb, depth, motion, jitter, state):
        self.r = runner
        self.prev = runner.state_bufs(state)
        self.args = (
            rgb,
            depth,
            motion,
            jitter,
            self.prev,
            state[2],
            state[3],
            state[9],
        )
        self.nb = runner.new_state_buffers(rgb)
        (self.graph, _) = capture(lambda: runner.run_into(*self.args, self.nb))

    def __call__(self):
        self.graph.replay()
        return self.nb["out"]


class GraphedStep:
    def __init__(self, runner: FusedResolve3):
        self.r = runner
        self.slots = None
        self.graphs = [None, None]

    def _alloc(self, rgb, depth, motion):

        def one():
            nb = self.r.new_state_buffers(rgb)
            nb.update(
                rgb=torch.empty_like(rgb),
                depth=torch.empty_like(depth),
                motion=torch.empty_like(motion),
                jit=torch.zeros((1, 2), device=rgb.device, dtype=torch.float32),
            )
            return nb

        self.slots = [one(), one()]

    def _slot_of(self, state):
        if self.slots is None:
            return None
        for k, sl in enumerate(self.slots):
            key = sl["s4"] if "s4" in sl else sl["new"]
            if (
                state[0].data_ptr() == key.data_ptr()
                and state[3].data_ptr() == sl["feat"].data_ptr()
                and (state[2].data_ptr() == sl["depth"].data_ptr())
                and (state[9].data_ptr() == sl["jit"].data_ptr())
            ):
                return k
        return None

    def step(self, rgb, depth, motion, jitter, state, init_hr=None, init_mask=None):
        r = self.r
        if state is None:
            return type(r.m).step(
                r.m,
                rgb,
                depth,
                motion,
                jitter,
                None,
                init_hr=init_hr,
                init_mask=init_mask,
            )
        if self.slots is None:
            self._alloc(rgb, depth, motion)
        src = self._slot_of(state)
        if src is None:
            src = 1
            sl = self.slots[src]
            for k_, v_ in r.state_bufs(state).items():
                sl[k_].copy_(v_)
            sl["feat"].copy_(state[3])
            sl["depth"].copy_(state[2])
            sl["jit"].copy_(state[9])
        dst = 1 - src
        (d, sv) = (self.slots[dst], self.slots[src])
        d["rgb"].copy_(rgb)
        d["depth"].copy_(depth)
        d["motion"].copy_(motion)
        d["jit"].copy_(jitter)
        if self.graphs[dst] is None:
            prev = {k_: sv[k_] for k_ in ("s4", "r", "new") if k_ in sv}
            args = (
                d["rgb"],
                d["depth"],
                d["motion"],
                d["jit"],
                prev,
                sv["depth"],
                sv["feat"],
                sv["jit"],
                d,
            )
            (self.graphs[dst], _) = capture(lambda: r.run_into(*args), warm=2)
        self.graphs[dst].replay()
        return (d["out"], r.state_tuple(d, d["depth"], d["jit"]), {})
