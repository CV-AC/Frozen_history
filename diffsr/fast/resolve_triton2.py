from __future__ import annotations
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from .resolve_triton import (
    FP_FUSION,
    FusedResolve,
    _bp,
    _consts,
    _f32,
    _warp_coords,
    guard,
    hr_post,
    libdevice,
)

STEM_C = 71
STEM_CP = 72
DT_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
CODE_DT = {v: k for (k, v) in DT_CODE.items()}


@triton.jit
def _bil_idx(iy, ix, H, W, mask):
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
    my0 = (y0 >= 0) & (y0 < H)
    my1 = (y1 >= 0) & (y1 < H)
    mx0 = (x0 >= 0) & (x0 < W)
    mx1 = (x1 >= 0) & (x1 < W)
    return (
        y0,
        x0,
        y1,
        x1,
        nw,
        ne,
        sw,
        se,
        mask & my0 & mx0,
        mask & my0 & mx1,
        mask & my1 & mx0,
        mask & my1 & mx1,
    )


@triton.jit
def _k0_lr_prep(
    rgb_ptr,
    dep_ptr,
    mot_ptr,
    jit_ptr,
    pjit_ptr,
    pdep_ptr,
    pfeat_ptr,
    motd_ptr,
    stem_ptr,
    h,
    w,
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
    BP: tl.constexpr,
    FD: tl.constexpr,
    PAD: tl.constexpr,
):
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
    (iy, ix, ins_f) = _warp_coords(Yf, Xf, mly, mlx, inv_lhm1, inv_lwm1, lhm1f, lwm1f)
    (y0, x0, y1, x1, nw, ne, sw, se, m00, m01, m10, m11) = _bil_idx(iy, ix, h, w, pm)
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
    k = tl.arange(0, 8)
    kk = k + (k >= 4).to(tl.int32)
    ei = tl.minimum(tl.maximum(i + kk // 3 - 1, 0), h - 1)
    ej = tl.minimum(tl.maximum(j[:, None] + (kk % 3)[None, :] - 1, 0), w - 1)
    sv = tl.load(dep_ptr + ei[None, :] * d_sh + ej * d_sw, mask=pm[:, None], other=0.0)
    sc = tl.maximum(tl.abs(dc), 0.001)
    e = tl.math.div_rn(sv - dc[:, None], sc[:, None])
    e = tl.minimum(tl.maximum(e, -4.0), 4.0)
    r = tl.load(rgb_ptr + i * r_sh + j * r_sw, mask=pm, other=0.0)
    g = tl.load(rgb_ptr + r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    b = tl.load(rgb_ptr + 2 * r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    c8 = k[None, :]
    v = tl.where(
        c8 == 0,
        r[:, None],
        tl.where(
            c8 == 1,
            g[:, None],
            tl.where(
                c8 == 2,
                b[:, None],
                tl.where(
                    c8 == 3,
                    dc[:, None],
                    tl.where(
                        c8 == 4,
                        smy[:, None],
                        tl.where(c8 == 5, smx[:, None], disocc[:, None]),
                    ),
                ),
            ),
        ),
    )
    dt = stem_ptr.dtype.element_ty
    base = stem_ptr + i * s_sh + j[:, None] * s_sw
    tl.store(base + c8 * s_sc, v.to(dt), mask=pm[:, None] & (c8 < 7))
    tl.store(base + (31 + tl.arange(0, FD))[None, :] * s_sc, f.to(dt), mask=pm[:, None])
    tl.store(base + (63 + c8) * s_sc, e.to(dt), mask=pm[:, None])
    if PAD:
        tl.store(
            stem_ptr + i * s_sh + j * s_sw + 71 * s_sc,
            tl.zeros([BP], tl.float32).to(dt),
            mask=pm,
        )
    tl.store(motd_ptr + i * w + j, smy, mask=pm)
    tl.store(motd_ptr + h * w + i * w + j, smx, mask=pm)


@triton.jit
def _k1b_hr_pre(
    prev_ptr,
    motd_ptr,
    wh_ptr,
    bh_ptr,
    wc_ptr,
    bc_ptr,
    wr_ptr,
    br_ptr,
    stem_ptr,
    h,
    w,
    H,
    W,
    s_sc,
    s_sh,
    s_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    S: tl.constexpr,
    BP: tl.constexpr,
    QK: tl.constexpr,
):
    S2: tl.constexpr = S * S
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < w
    q = tl.arange(0, QK)
    qm = q < S2
    Y = i * S + (q // S)[None, :]
    X = j[:, None] * S + (q % S)[None, :]
    mask = pm[:, None] & qm[None, :]
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
    (y0, x0, y1, x1, nw, ne, sw, se, m00, m01, m10, m11) = _bil_idx(iy, ix, H, W, mask)
    o00 = y0 * W + x0
    o01 = y0 * W + x1
    o10 = y1 * W + x0
    o11 = y1 * W + x1
    HW = H * W
    ins = inside.to(tl.float32)
    c0 = tl.load(prev_ptr + o00, mask=m00, other=0.0) * nw
    c0 = tl.fma(tl.load(prev_ptr + o01, mask=m01, other=0.0), ne, c0)
    c0 = tl.fma(tl.load(prev_ptr + o10, mask=m10, other=0.0), sw, c0)
    c0 = tl.fma(tl.load(prev_ptr + o11, mask=m11, other=0.0), se, c0)
    p1 = prev_ptr + HW
    c1 = tl.load(p1 + o00, mask=m00, other=0.0) * nw
    c1 = tl.fma(tl.load(p1 + o01, mask=m01, other=0.0), ne, c1)
    c1 = tl.fma(tl.load(p1 + o10, mask=m10, other=0.0), sw, c1)
    c1 = tl.fma(tl.load(p1 + o11, mask=m11, other=0.0), se, c1)
    p2 = prev_ptr + 2 * HW
    c2 = tl.load(p2 + o00, mask=m00, other=0.0) * nw
    c2 = tl.fma(tl.load(p2 + o01, mask=m01, other=0.0), ne, c2)
    c2 = tl.fma(tl.load(p2 + o10, mask=m10, other=0.0), sw, c2)
    c2 = tl.fma(tl.load(p2 + o11, mask=m11, other=0.0), se, c2)
    p3 = prev_ptr + 3 * HW
    cw = tl.load(p3 + o00, mask=m00, other=0.0) * nw
    cw = tl.fma(tl.load(p3 + o01, mask=m01, other=0.0), ne, cw)
    cw = tl.fma(tl.load(p3 + o10, mask=m10, other=0.0), sw, cw)
    cw = tl.fma(tl.load(p3 + o11, mask=m11, other=0.0), se, cw)
    cw = cw * ins
    p4 = prev_ptr + 4 * HW
    rw = tl.load(p4 + o00, mask=m00, other=0.0) * nw
    rw = tl.fma(tl.load(p4 + o01, mask=m01, other=0.0), ne, rw)
    rw = tl.fma(tl.load(p4 + o10, mask=m10, other=0.0), sw, rw)
    rw = tl.fma(tl.load(p4 + o11, mask=m11, other=0.0), se, rw)
    rw = tl.where(mask, rw * ins + (1.0 - ins), 0.0)
    o = tl.arange(0, 16)
    qq = q[:, None]
    oo = o[None, :]
    w0 = tl.load(wh_ptr + oo * 3 * S2 + qq, mask=qm[:, None], other=0.0)
    w1 = tl.load(wh_ptr + oo * 3 * S2 + S2 + qq, mask=qm[:, None], other=0.0)
    w2 = tl.load(wh_ptr + oo * 3 * S2 + 2 * S2 + qq, mask=qm[:, None], other=0.0)
    acc = tl.dot(c0, w0, input_precision="ieee")
    acc = tl.dot(c1, w1, acc, input_precision="ieee")
    acc = tl.dot(c2, w2, acc, input_precision="ieee")
    acc = acc + tl.load(bh_ptr + o)[None, :]
    m4 = qm[:, None] & (oo < 4)
    wcv = tl.load(wc_ptr + oo * S2 + qq, mask=m4, other=0.0)
    accc = (
        tl.dot(cw, wcv, input_precision="ieee")
        + tl.load(bc_ptr + o, mask=o < 4, other=0.0)[None, :]
    )
    wrv = tl.load(wr_ptr + oo * S2 + qq, mask=m4, other=0.0)
    accr = (
        tl.dot(rw, wrv, input_precision="ieee")
        + tl.load(br_ptr + o, mask=o < 4, other=0.0)[None, :]
    )
    dt = stem_ptr.dtype.element_ty
    base = stem_ptr + i * s_sh + j[:, None] * s_sw
    tl.store(base + (7 + oo) * s_sc, acc.to(dt), mask=pm[:, None])
    tl.store(base + (23 + oo) * s_sc, accc.to(dt), mask=pm[:, None] & (oo < 4))
    tl.store(base + (27 + oo) * s_sc, accr.to(dt), mask=pm[:, None] & (oo < 4))


def _k0_bp(w):
    return 64


@triton_op("rrta_fast::lr_prep", mutates_args={})
def lr_prep(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    motion: torch.Tensor,
    jitter: torch.Tensor,
    prev_jitter: torch.Tensor,
    prev_depth: torch.Tensor,
    prev_feat: torch.Tensor,
    dt_code: int,
    channels_last: bool,
    pad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    (h, w) = rgb.shape[-2:]
    motd = torch.empty((1, 2, h, w), device=rgb.device, dtype=torch.float32)
    stem = torch.empty(
        (1, STEM_CP if pad else STEM_C, h, w),
        device=rgb.device,
        dtype=CODE_DT[dt_code],
        memory_format=torch.channels_last if channels_last else torch.contiguous_format,
    )
    c = _consts(h, w, 1)
    bp = _k0_bp(w)
    wrap_triton(_k0_lr_prep)[triton.cdiv(w, bp), h](
        rgb,
        depth,
        motion,
        jitter,
        prev_jitter,
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
        PAD=pad,
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )
    return (motd, stem)


@triton_op("rrta_fast::hr_pre_into", mutates_args={"stem"})
def hr_pre_into(
    prev: torch.Tensor,
    motd: torch.Tensor,
    wh: torch.Tensor,
    bh: torch.Tensor,
    wc: torch.Tensor,
    bc: torch.Tensor,
    wr: torch.Tensor,
    br: torch.Tensor,
    stem: torch.Tensor,
    scale: int,
) -> None:
    (h, w) = motd.shape[-2:]
    (H, W) = (h * scale, w * scale)
    c = _consts(h, w, scale)
    bp = 32
    qk = max(16, scale * scale)
    wrap_triton(_k1b_hr_pre)[triton.cdiv(w, bp), h](
        prev,
        motd,
        wh,
        bh,
        wc,
        bc,
        wr,
        br,
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
        QK=qk,
        num_warps=4,
        enable_fp_fusion=FP_FUSION,
    )


class PaddedStem(torch.nn.Module):
    def __init__(self, stem, cin: int = STEM_CP):
        super().__init__()
        if stem.kind not in ("conv3", "conv3_film"):
            raise ValueError(f"PaddedStem: stem kind {stem.kind!r}")
        b = stem.base
        conv = torch.nn.Conv2d(cin, b.out_channels, 3, padding=1).to(
            device=b.weight.device, dtype=b.weight.dtype
        )
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, : b.in_channels] = b.weight
            conv.bias.copy_(b.bias)
        if b.weight.is_contiguous(memory_format=torch.channels_last) and (
            not b.weight.is_contiguous()
        ):
            conv = conv.to(memory_format=torch.channels_last)
        self.base = conv
        self.kind = stem.kind
        self.mlp = stem.mlp

    def forward(self, x, jitter):
        out = self.base(x)
        if self.kind == "conv3_film":
            ph = torch.cat((jitter, torch.sin(2 * torch.pi * jitter)), dim=1)
            (gamma, beta) = self.mlp(ph).chunk(2, dim=1)
            out = out * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return out


class FusedResolve2(FusedResolve):
    def _inner2(
        self,
        rgb,
        depth,
        motion,
        jitter,
        prev,
        prev_depth,
        prev_feat,
        prev_jit,
        dt_code: int,
        cl: bool,
    ):
        m = self.m
        s = m.scale
        jf = jitter.float()
        pad = self.stem_pad is not None
        (motd, stem_in) = lr_prep(
            rgb,
            depth,
            motion,
            jf,
            prev_jit.float(),
            prev_depth,
            prev_feat,
            dt_code,
            cl,
            pad,
        )
        hr_pre_into(
            prev, motd, self.wh, self.bh, self.wc, self.bc, self.wr, self.br, stem_in, s
        )
        stem = self.stem_pad if pad else m.stem
        x = m.body(stem(stem_in, jitter))
        colour = m.head_colour(x)
        stat = m._head(m.head_weight, x)
        feat = m.head_feature(x)
        (out, new, flags) = hr_post(
            prev, motd, colour, stat, feat, rgb, jf, s, self.width_k, self.inv_cap
        )
        guard(flags, new, feat, rgb, jf, s)
        return (out, new, feat)

    def __init__(
        self,
        model,
        compile: bool = True,
        mode: str = "default",
        pad_stem: bool = True,
        **kw,
    ):
        super().__init__(model, compile=False, **kw)
        if model.stem.in_dim != STEM_C:
            raise ValueError(
                f"stem expects {model.stem.in_dim} channels, K0 writes {STEM_C}"
            )
        self.compiled = compile
        self._pad_req = bool(pad_stem)
        self.stem_pad = None
        self.inner2 = (
            torch.compile(self._inner2, mode=mode) if compile else self._inner2
        )

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
        from .resolve_triton import packed_from_state

        if self._pad_req and self.stem_pad is None:
            self.stem_pad = PaddedStem(self.m.stem).eval()
        prev = packed_from_state(state)
        dt = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else torch.float32
        )
        wgt = self.m.stem.base.weight
        cl = bool(
            wgt.is_contiguous(memory_format=torch.channels_last)
            and (not wgt.is_contiguous())
        )
        (out, new, feat) = self.inner2(
            rgb,
            depth,
            motion,
            jitter,
            prev,
            state[2],
            state[3],
            state[9],
            DT_CODE[dt],
            cl,
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


def install2(model, **kw) -> FusedResolve2:
    runner = FusedResolve2(model, **kw)
    model.step = runner.step
    return runner
