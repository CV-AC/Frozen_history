from __future__ import annotations
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from .resolve_triton import (
    FP_FUSION,
    _bicubic_skip,
    _bilinear_plane,
    _bilinear_setup,
    _consts,
    _f32,
    _inv,
    _sigmoid,
    _warp_coords,
    libdevice,
    packed_from_state,
)


@triton.jit
def _kt1_hr_pre(
    prev_ptr,
    mot_ptr,
    wh_ptr,
    bh_ptr,
    wc_ptr,
    bc_ptr,
    wr_ptr,
    br_ptr,
    out_ptr,
    th,
    tw,
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
    T: tl.constexpr,
    BP: tl.constexpr,
    NH: tl.constexpr,
    NC: tl.constexpr,
    NCP: tl.constexpr,
):
    QK: tl.constexpr = T * T
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < tw
    q = tl.arange(0, QK)
    Y = i * T + (q // T)[None, :]
    X = j[:, None] * T + (q % T)[None, :]
    mask = pm[:, None] & (q >= 0)[None, :]
    my = tl.load(mot_ptr + Y // S * m_sh + X // S * m_sw, mask=mask, other=0.0)
    mx = tl.load(mot_ptr + m_sc + Y // S * m_sh + X // S * m_sw, mask=mask, other=0.0)
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
    qq = q[:, None]
    oh = tl.arange(0, NH)[None, :]
    w0 = tl.load(wh_ptr + oh * 3 * QK + qq)
    w1 = tl.load(wh_ptr + oh * 3 * QK + QK + qq)
    w2 = tl.load(wh_ptr + oh * 3 * QK + 2 * QK + qq)
    acc = tl.dot(c0, w0, input_precision="ieee")
    acc = tl.dot(c1, w1, acc, input_precision="ieee")
    acc = tl.dot(c2, w2, acc, input_precision="ieee")
    acc = acc + tl.load(bh_ptr + tl.arange(0, NH))[None, :]
    oc = tl.arange(0, NCP)[None, :]
    mc = oc < NC
    wcv = tl.load(wc_ptr + oc * QK + qq, mask=mc & (qq >= 0), other=0.0)
    accc = (
        tl.dot(cw, wcv, input_precision="ieee")
        + tl.load(bc_ptr + tl.arange(0, NCP), mask=tl.arange(0, NCP) < NC, other=0.0)[
            None, :
        ]
    )
    wrv = tl.load(wr_ptr + oc * QK + qq, mask=mc & (qq >= 0), other=0.0)
    accr = (
        tl.dot(rw, wrv, input_precision="ieee")
        + tl.load(br_ptr + tl.arange(0, NCP), mask=tl.arange(0, NCP) < NC, other=0.0)[
            None, :
        ]
    )
    thw = th * tw
    base = out_ptr + i * tw + j[:, None]
    tl.store(base + oh * thw, acc, mask=pm[:, None])
    tl.store(base + (NH + oc) * thw, accc, mask=pm[:, None] & mc)
    tl.store(base + (NH + NC + oc) * thw, accr, mask=pm[:, None] & mc)


@triton.jit
def _kt2_hr_post(
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
    th,
    tw,
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
    T: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
):
    T2: tl.constexpr = T * T
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    dyi = tl.arange(0, T)[:, None, None]
    pi = tl.arange(0, BP)[None, :, None]
    dxi = tl.arange(0, T)[None, None, :]
    j = j0 + pi
    pm = j < tw
    mask = pm & (dyi >= 0) & (dxi >= 0)
    Y = i * T + dyi
    X = j * T + dxi
    Yf = Y.to(tl.float32)
    Xf = X.to(tl.float32)
    my = tl.load(mot_ptr + Y // S * m_sh + X // S * m_sw, mask=mask, other=0.0)
    mx = tl.load(mot_ptr + m_sc + Y // S * m_sh + X // S * m_sw, mask=mask, other=0.0)
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
    q = dyi * T + dxi
    lr_c = i * c_sh + j * c_sw + q * c_sc
    k0 = tl.load(col_ptr + lr_c, mask=mask, other=0.0).to(tl.float32) + s0
    k1 = tl.load(col_ptr + lr_c + T2 * c_sc, mask=mask, other=0.0).to(tl.float32) + s1
    k2 = (
        tl.load(col_ptr + lr_c + 2 * T2 * c_sc, mask=mask, other=0.0).to(tl.float32)
        + s2
    )
    lr_t = i * t_sh + j * t_sw + q * t_sc
    rel = _sigmoid(tl.load(stat_ptr + lr_t, mask=mask, other=0.0))
    ret = _sigmoid(tl.load(stat_ptr + lr_t + T2 * t_sc, mask=mask, other=0.0))
    py = centre + -jy * sf - (dyi % S).to(tl.float32)
    px = centre + jx * sf - (dxi % S).to(tl.float32)
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
    tl.store(new_ptr + 4 * HW + hr, radius + tl.zeros_like(o0), mask=mask)
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
    om = jj < tw
    fc = tl.arange(0, FD)
    fv = tl.load(
        feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
        mask=om[:, None],
        other=0.0,
    ).to(tl.float32)
    bad_f = tl.max(((tl.abs(fv) <= f_box) == 0).to(tl.int32), axis=1)
    tl.store(flag_ptr + i * tw + jj, tl.maximum(bad_blk, bad_f).to(tl.int8), mask=om)


@triton.jit
def _kt3_guard(
    flag_ptr,
    new_ptr,
    feat_ptr,
    rgb_ptr,
    jit_ptr,
    th,
    tw,
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
    T: tl.constexpr,
    BP: tl.constexpr,
    FD: tl.constexpr,
):
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    jj = j0 + tl.arange(0, BP)
    om = jj < tw
    bad = tl.zeros([BP], dtype=tl.int32)
    for di in tl.static_range(3):
        for dj in tl.static_range(3):
            ii = i + di - 1
            jn = jj + dj - 1
            m = om & (ii >= 0) & (ii < th) & (jn >= 0) & (jn < tw)
            bad = tl.maximum(
                bad, tl.load(flag_ptr + ii * tw + jn, mask=m, other=0).to(tl.int32)
            )
    if tl.max(bad, axis=0) > 0:
        badm = (bad > 0) & om
        fc = tl.arange(0, FD)
        tl.store(
            feat_ptr + i * f_sh + jj[:, None] * f_sw + fc[None, :] * f_sc,
            tl.zeros([BP, FD], dtype=tl.float32).to(feat_ptr.dtype.element_ty),
            mask=badm[:, None],
        )
        dyi = tl.arange(0, T)[:, None, None]
        pi = tl.arange(0, BP)[None, :, None]
        dxi = tl.arange(0, T)[None, None, :]
        j = j0 + pi
        b3 = (bad > 0)[None, :, None] & (j < tw) & (dyi >= 0) & (dxi >= 0)
        Y = i * T + dyi
        X = j * T + dxi
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


def _bpt(T):
    return {2: 64, 4: 32, 8: 8}.get(T, 16)


@triton_op("rrta_fast::hr_pre_t", mutates_args={})
def hr_pre_t(
    prev: torch.Tensor,
    motion: torch.Tensor,
    wh: torch.Tensor,
    bh: torch.Tensor,
    wc: torch.Tensor,
    bc: torch.Tensor,
    wr: torch.Tensor,
    br: torch.Tensor,
    scale: int,
    k: int,
) -> torch.Tensor:
    (hl, wl) = motion.shape[-2:]
    (th, tw) = (hl // k, wl // k)
    (H, W) = (hl * scale, wl * scale)
    T = scale * k
    (nh, nc) = (16 * k * k, 4 * k * k)
    out = torch.empty((1, nh + 2 * nc, th, tw), device=prev.device, dtype=torch.float32)
    c = _consts(hl, wl, scale)
    bp = 16 if T * T >= 64 else 32
    wrap_triton(_kt1_hr_pre)[triton.cdiv(tw, bp), th](
        prev,
        motion,
        wh,
        bh,
        wc,
        bc,
        wr,
        br,
        out,
        th,
        tw,
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
        T=T,
        BP=bp,
        NH=nh,
        NC=nc,
        NCP=max(16, nc),
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )
    return out


@triton_op("rrta_fast::hr_post_t", mutates_args={})
def hr_post_t(
    prev: torch.Tensor,
    motion: torch.Tensor,
    colour: torch.Tensor,
    stat: torch.Tensor,
    feat: torch.Tensor,
    rgb: torch.Tensor,
    jitter: torch.Tensor,
    scale: int,
    k: int,
    width_k: float,
    inv_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (hl, wl) = motion.shape[-2:]
    (th, tw) = (hl // k, wl // k)
    (H, W) = (hl * scale, wl * scale)
    T = scale * k
    out = torch.empty((1, 3, H, W), device=prev.device, dtype=torch.float32)
    new = torch.empty((1, 5, H, W), device=prev.device, dtype=torch.float32)
    flags = torch.empty((th, tw), device=prev.device, dtype=torch.int8)
    c = _consts(hl, wl, scale)
    bp = _bpt(T)
    wrap_triton(_kt2_hr_post)[triton.cdiv(tw, bp), th](
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
        th,
        tw,
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
        T=T,
        BP=bp,
        FD=feat.shape[1],
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )
    return (out, new, flags)


@triton_op("rrta_fast::guard_t", mutates_args={"new", "feat"})
def guard_t(
    flags: torch.Tensor,
    new: torch.Tensor,
    feat: torch.Tensor,
    rgb: torch.Tensor,
    jitter: torch.Tensor,
    scale: int,
    k: int,
) -> None:
    (th, tw) = flags.shape
    (hl, wl) = (th * k, tw * k)
    (H, W) = (hl * scale, wl * scale)
    T = scale * k
    c = _consts(hl, wl, scale)
    bp = _bpt(T)
    wrap_triton(_kt3_guard)[triton.cdiv(tw, bp), th](
        flags,
        new,
        feat,
        rgb,
        jitter,
        th,
        tw,
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
        T=T,
        BP=bp,
        FD=feat.shape[1],
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )


def check_config_k(m) -> None:
    from .lean_step import check_config

    check_config(m)
    for k, v in (("consistency_gate", "none"), ("state_mode", "linear")):
        if getattr(m, k, v) != v:
            raise ValueError(f"fused K path needs {k}={v!r}")


class FusedResolveK:
    def __init__(self, model, compile: bool = True, mode: str = "default"):
        check_config_k(model)
        self.m = model
        self.k = int(getattr(model, "lr_unshuffle", 1))
        (s, k) = (model.scale, self.k)
        T2 = (s * k) ** 2
        with torch.no_grad():
            self.wh = (
                model.history_proj.weight.detach()
                .float()
                .reshape(16 * k * k, 3 * T2)
                .contiguous()
            )
            self.bh = model.history_proj.bias.detach().float().contiguous()
            self.wc = (
                model.confidence_proj.weight.detach()
                .float()
                .reshape(4 * k * k, T2)
                .contiguous()
            )
            self.bc = model.confidence_proj.bias.detach().float().contiguous()
            self.wr = (
                model.reach_proj.weight.detach()
                .float()
                .reshape(4 * k * k, T2)
                .contiguous()
            )
            self.br = model.reach_proj.bias.detach().float().contiguous()
            self.width_k = float(
                (F.softplus(model.kernel_width.detach()) + 0.25).item()
            )
        self.inv_cap = _inv(1.0 / model.rectifier.alpha_min)
        self.inner = torch.compile(self._inner, mode=mode) if compile else self._inner

    def _lo(self, t):
        return F.pixel_unshuffle(t, self.k) if self.k > 1 else t

    def _inner(self, rgb, depth, motion, jitter, prev, prev_depth, prev_feat, prev_jit):
        import diffsr.rrta as R

        m = self.m
        (s, k) = (m.scale, self.k)
        motion = R.dilate_motion(motion, depth)
        motion_lr = m._lr_motion(motion, jitter, prev_jit)
        proj = hr_pre_t(
            prev, motion, self.wh, self.bh, self.wc, self.bc, self.wr, self.br, s, k
        )
        (warped_depth, depth_inside) = R.warp(prev_depth, motion)
        fmot = m._trunk_motion(motion_lr, depth) if k > 1 else motion_lr
        (warped_feat, feat_inside) = R.warp(prev_feat, fmot)
        feature_hist = warped_feat * feat_inside
        disocclusion = (depth - warped_depth).abs() * depth_inside
        stem_in = torch.cat(
            [
                self._lo(rgb),
                self._lo(depth),
                self._lo(motion),
                self._lo(disocclusion),
                proj,
                feature_hist,
                self._lo(R.depth_edges(depth)),
            ],
            dim=1,
        )
        x = m.body(m.stem(stem_in, jitter))
        colour = m.head_colour(x)
        stat = m._head(m.head_weight, x)
        feat = m.head_feature(x)
        jf = jitter.float()
        (out, new, flags) = hr_post_t(
            prev, motion, colour, stat, feat, rgb, jf, s, k, self.width_k, self.inv_cap
        )
        guard_t(flags, new, feat, rgb, jf, s, k)
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
