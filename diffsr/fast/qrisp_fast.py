from __future__ import annotations
import torch
import triton
import triton.language as tl
from . import conv3 as C3
from .conv3 import PadBuf, PadConv, pack_weight, _epilogue
from .resolve_triton import FP_FUSION, _consts, _warp_coords


@triton.jit
def _conv3x3_pad_relu(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    COUT_P: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = tl.load(x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :])
        b = tl.load(w_ptr + (tap * CIN + k0 + kk)[:, None] * COUT_P + n[None, :])
        acc = tl.dot(a, b, acc)
    p = p0 + m
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    nm = n < COUT
    v = (
        acc.to(tl.bfloat16).to(tl.float32)
        + tl.load(b_ptr + n, mask=nm, other=0.0)[None, :]
    )
    v = tl.maximum(v, 0.0)
    tl.store(
        y_ptr + p[:, None] * COUT + n[None, :],
        v.to(tl.bfloat16),
        mask=pm[:, None] & nm[None, :],
    )


class ReluConv(PadConv):
    def __call__(self, x: PadBuf, y: PadBuf):
        assert x.C == self.cin_p and y.C == self.cout
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm), self.cout_p // self.bn)
        _conv3x3_pad_relu[grid](
            x.buf,
            self.w,
            self.b,
            y.buf,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            CIN=self.cin_p,
            COUT=self.cout,
            COUT_P=self.cout_p,
            BM=self.bm,
            BN=self.bn,
            BK=self.bk,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


@triton.jit
def _kq_prep(
    rgb_ptr,
    dep_ptr,
    mot_ptr,
    jit_ptr,
    pjit_ptr,
    s4_ptr,
    stem_ptr,
    hist_ptr,
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
    s_sc,
    s_sh,
    s_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    BP: tl.constexpr,
):
    S: tl.constexpr = 4
    S2: tl.constexpr = 16
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < w
    bi = i // 2 * 2
    bj = j // 2 * 2
    best = tl.full([BP], float("-inf"), tl.float32)
    py = tl.zeros([BP], tl.float32)
    px = tl.zeros([BP], tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            ii = bi + dy
            jj = bj + dx
            m = pm & (ii < h) & (jj < w)
            key = -tl.load(dep_ptr + ii * d_sh + jj * d_sw, mask=m, other=0.0)
            better = m & ((key > best) | (key != key))
            best = tl.where(better, key, best)
            py = tl.where(
                better, tl.load(mot_ptr + ii * m_sh + jj * m_sw, mask=m, other=0.0), py
            )
            px = tl.where(
                better,
                tl.load(mot_ptr + m_sc + ii * m_sh + jj * m_sw, mask=m, other=0.0),
                px,
            )
    mhy = (py + (tl.load(pjit_ptr) - tl.load(jit_ptr))) * sf
    mhx = (px + (tl.load(pjit_ptr + 1) - tl.load(jit_ptr + 1))) * sf
    q = tl.arange(0, S2)
    ch = tl.arange(0, 4)
    Yh = i * S + (q // S)[None, :]
    Xh = j[:, None] * S + (q % S)[None, :]
    mask = pm[:, None] & (q >= 0)[None, :]
    (iy, ix, inside) = _warp_coords(
        Yh.to(tl.float32),
        Xh.to(tl.float32),
        mhy[:, None],
        mhx[:, None],
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
    my1 = mask & (y0 + 1 < H)
    mx1 = mask & (x0 + 1 < W)
    o00 = y0 * W + x0
    c4 = ch[None, None, :]
    v = (
        tl.load(s4_ptr + o00[:, :, None] * 4 + c4, mask=mask[:, :, None], other=0.0)
        * (wx0 * wy0)[:, :, None]
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + 1)[:, :, None] * 4 + c4, mask=mx1[:, :, None], other=0.0
        ),
        (wx1 * wy0)[:, :, None],
        v,
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + W)[:, :, None] * 4 + c4, mask=my1[:, :, None], other=0.0
        ),
        (wx0 * wy1)[:, :, None],
        v,
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + W + 1)[:, :, None] * 4 + c4,
            mask=(my1 & mx1)[:, :, None],
            other=0.0,
        ),
        (wx1 * wy1)[:, :, None],
        v,
    )
    tl.store(hist_ptr + (Yh * W + Xh)[:, :, None] * 4 + c4, v, mask=mask[:, :, None])
    k16 = tl.arange(0, 16)[None, :]
    r = tl.load(rgb_ptr + i * r_sh + j * r_sw, mask=pm, other=0.0)
    g = tl.load(rgb_ptr + r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    b = tl.load(rgb_ptr + 2 * r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    d = tl.load(dep_ptr + i * d_sh + j * d_sw, mask=pm, other=0.0)
    a16 = tl.where(
        k16 == 0,
        r[:, None],
        tl.where(
            k16 == 1,
            g[:, None],
            tl.where(k16 == 2, b[:, None], tl.where(k16 == 3, d[:, None], 0.0)),
        ),
    )
    dt = stem_ptr.dtype.element_ty
    base = stem_ptr + i * s_sh + j[:, None] * s_sw
    tl.store(base + k16 * s_sc, a16.to(dt), mask=pm[:, None])
    v64 = tl.reshape(tl.permute(v, (0, 2, 1)), (BP, 64))
    n64 = tl.arange(0, 64)[None, :]
    tl.store(base + (16 + n64) * s_sc, v64.to(dt), mask=pm[:, None])


@triton.jit
def _kq_post(
    o_ptr, hist_ptr, out_ptr, s4n_ptr, h, w, H, W, o_sc, o_sh, o_sw, BP: tl.constexpr
):
    S: tl.constexpr = 4
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    dyi = tl.arange(0, S)[:, None, None]
    pi = tl.arange(0, BP)[None, :, None]
    dxi = tl.arange(0, S)[None, None, :]
    j = j0 + pi
    mask = (j < w) & (dyi >= 0) & (dxi >= 0)
    Y = i * S + dyi
    X = j * S + dxi
    hr = Y * W + X
    q = dyi * S + dxi
    ob = o_ptr + i * o_sh + j * o_sw + q * o_sc
    al = tl.load(ob, mask=mask, other=0.0).to(tl.float32)
    k0 = tl.load(ob + 16 * o_sc, mask=mask, other=0.0).to(tl.float32)
    k1 = tl.load(ob + 32 * o_sc, mask=mask, other=0.0).to(tl.float32)
    k2 = tl.load(ob + 48 * o_sc, mask=mask, other=0.0).to(tl.float32)
    fe = tl.load(ob + 64 * o_sc, mask=mask, other=0.0).to(tl.float32)
    m4 = mask[:, :, :, None]
    c4 = tl.arange(0, 4)[None, None, None, :]
    v = tl.load(hist_ptr + hr[:, :, :, None] * 4 + c4, mask=m4, other=0.0)
    (a02, a13) = tl.split(tl.reshape(v, (S, BP, S, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, _) = tl.split(a13)
    a = 1.0 / (1.0 + tl.exp(-al))
    y0 = a * k0 + (1.0 - a) * h0
    y1 = a * k1 + (1.0 - a) * h1
    y2 = a * k2 + (1.0 - a) * h2
    HW = H * W
    tl.store(out_ptr + hr, y0, mask=mask)
    tl.store(out_ptr + HW + hr, y1, mask=mask)
    tl.store(out_ptr + 2 * HW + hr, y2, mask=mask)
    ov = tl.reshape(tl.join(tl.join(y0, y2), tl.join(y1, fe)), (S, BP, S, 4))
    tl.store(s4n_ptr + hr[:, :, :, None] * 4 + c4, ov, mask=m4)


class QrispFast:
    def __init__(self, net, cfg: dict | None = None):
        cfg = dict(cfg or {})
        assert net.scale == 4 and net.hist == 4
        f = net.f
        self.net = net
        first = net.first
        self.cin = int(cfg.get("cin_p", 96))
        conv = torch.nn.Conv2d(self.cin, f, 3, padding=1).to(first.weight.device)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0:4] = first.weight[:, 0:4].float()
            conv.weight[:, 16:80] = first.weight[:, 4:68].float()
            conv.bias.copy_(first.bias.float())
        c64 = C3.default_cfg(64, C3.EPI_GELU)
        cst = C3.default_cfg(64, C3.EPI_FILM)
        if self.cin == 80:
            cst = {**cst, "BK": 16}
        self.c_first = ReluConv(
            conv, C3.EPI_BIAS, cin_p=self.cin, cfg=cfg.get("first", cst)
        )
        self.c_mid = [
            ReluConv(c, C3.EPI_BIAS, cfg=cfg.get("mid", c64)) for c in net.mid
        ]
        self.post = cfg.get("post", "fused")
        if self.post == "fused":
            assert net.last.out_channels == 80
            lc = dict(cfg.get("last", c64))
            lc["BN"] = 80
            self.c_last = QpostConv(net.last, C3.EPI_BIAS, cfg=lc)
        else:
            self.c_last = C3.make_padconv(
                net.last, C3.EPI_BIAS, cfg=cfg.get("last", c64)
            )
        self.bp0 = cfg.get("prep_bp", 16)
        self.bp1 = cfg.get("post_bp", 16)
        self._bufs = {}

    def buffers(self, rgb):
        (h, w) = rgb.shape[-2:]
        if (h, w) in self._bufs:
            return self._bufs[h, w]
        dev = rgb.device
        f = self.net.f
        b = dict(
            pin=PadBuf(self.cin, h, w, torch.bfloat16, dev),
            pa=PadBuf(f, h, w, torch.bfloat16, dev),
            pb=PadBuf(f, h, w, torch.bfloat16, dev),
            po=PadBuf(80, h, w, torch.bfloat16, dev),
            hist=torch.empty((16 * h * w, 4), device=dev, dtype=torch.float32),
        )
        self._bufs[h, w] = b
        return b

    def run_into(self, rgb, depth, motion, jitter, s4, prev_j, out, s4n):
        b = self.buffers(rgb)
        (h, w) = rgb.shape[-2:]
        (H, W) = (4 * h, 4 * w)
        c = _consts(h, w, 4)
        stem = b["pin"].view
        _kq_prep[triton.cdiv(w, self.bp0), h](
            rgb,
            depth,
            motion,
            jitter,
            prev_j,
            s4,
            stem,
            b["hist"],
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
            stem.stride(1),
            stem.stride(2),
            stem.stride(3),
            c["inv_hm1"],
            c["inv_wm1"],
            c["Hm1f"],
            c["Wm1f"],
            c["sf"],
            BP=self.bp0,
            num_warps=4,
            enable_fp_fusion=FP_FUSION,
        )
        self.c_first(b["pin"], b["pa"])
        (x, y) = (b["pa"], b["pb"])
        for conv in self.c_mid:
            conv(x, y)
            (x, y) = (y, x)
        if self.post == "fused":
            self.c_last(x, b["hist"], out, s4n)
            return out
        self.c_last(x, b["po"])
        o = b["po"].view
        _kq_post[triton.cdiv(w, self.bp1), h](
            o,
            b["hist"],
            out,
            s4n,
            h,
            w,
            H,
            W,
            o.stride(1),
            o.stride(2),
            o.stride(3),
            BP=self.bp1,
            num_warps=4,
            enable_fp_fusion=FP_FUSION,
        )
        return out


class QrispFixedGraph:
    def __init__(self, fast: QrispFast, rgb, depth, motion, jitter, state):
        from .resolve_triton3 import capture

        (hist, prev_j) = state
        (H, W) = hist.shape[-2:]
        self.s4 = hist.float().permute(0, 2, 3, 1).contiguous()
        self.pj = prev_j.float().contiguous()
        self.jf = jitter.float().contiguous()
        self.out = torch.empty((1, 3, H, W), device=rgb.device)
        self.s4n = torch.empty_like(self.s4)
        args = (rgb, depth, motion, self.jf, self.s4, self.pj, self.out, self.s4n)
        (self.graph, _) = capture(lambda: fast.run_into(*args))

    def __call__(self):
        self.graph.replay()
        return self.out


@triton.jit
def _conv3x3_pad_qpost(
    x_ptr,
    w_ptr,
    b_ptr,
    hist_ptr,
    out_ptr,
    s4n_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    Wh,
    Hh,
    CIN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    COUT: tl.constexpr = 80
    pid = tl.program_id(0)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    q = tl.arange(0, 16)
    kk = tl.arange(0, BK)
    aa = tl.zeros([BM, 16], dtype=tl.float32)
    a0 = tl.zeros([BM, 16], dtype=tl.float32)
    a1 = tl.zeros([BM, 16], dtype=tl.float32)
    a2 = tl.zeros([BM, 16], dtype=tl.float32)
    af = tl.zeros([BM, 16], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = tl.load(x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :])
        wrow = w_ptr + (tap * CIN + k0 + kk)[:, None] * COUT + q[None, :]
        aa = tl.dot(a, tl.load(wrow), aa)
        a0 = tl.dot(a, tl.load(wrow + 16), a0)
        a1 = tl.dot(a, tl.load(wrow + 32), a1)
        a2 = tl.dot(a, tl.load(wrow + 48), a2)
        af = tl.dot(a, tl.load(wrow + 64), af)
    p = p0 + m
    rowp = (p - SL) // WP
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    i = rowp - 1
    j = col - 1
    msk = pm[:, None] & (q >= 0)[None, :]
    al = (
        (aa.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    k0v = (
        (a0.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + 16 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    k1v = (
        (a1.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + 32 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    k2v = (
        (a2.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + 48 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    fe = (
        (af.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + 64 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    Y = i[:, None] * 4 + (q // 4)[None, :]
    X = j[:, None] * 4 + (q % 4)[None, :]
    hr = Y * Wh + X
    c4 = tl.arange(0, 4)[None, None, :]
    v = tl.load(hist_ptr + hr[:, :, None] * 4 + c4, mask=msk[:, :, None], other=0.0)
    (a02, a13) = tl.split(tl.reshape(v, (BM, 16, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, _) = tl.split(a13)
    s = 1.0 / (1.0 + tl.exp(-al))
    y0 = s * k0v + (1.0 - s) * h0
    y1 = s * k1v + (1.0 - s) * h1
    y2 = s * k2v + (1.0 - s) * h2
    HW = Hh * Wh
    tl.store(out_ptr + hr, y0, mask=msk)
    tl.store(out_ptr + HW + hr, y1, mask=msk)
    tl.store(out_ptr + 2 * HW + hr, y2, mask=msk)
    ov = tl.reshape(tl.join(tl.join(y0, y2), tl.join(y1, fe)), (BM, 16, 4))
    tl.store(s4n_ptr + hr[:, :, None] * 4 + c4, ov, mask=msk[:, :, None])


class QpostConv(PadConv):
    def __call__(self, x: PadBuf, hist, out, s4n):
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm),)
        _conv3x3_pad_qpost[grid](
            x.buf,
            self.w,
            self.b,
            hist,
            out,
            s4n,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            4 * x.W,
            4 * x.H,
            CIN=self.cin_p,
            BM=self.bm,
            BK=self.bk,
            STAGES=self.stages,
            num_warps=self.warps,
        )


@triton.jit
def _kq_prep2(
    rgb_ptr,
    dep_ptr,
    mot_ptr,
    jit_ptr,
    pjit_ptr,
    s4_ptr,
    stem_ptr,
    hist_ptr,
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
    s_sc,
    s_sh,
    s_sw,
    inv_hm1,
    inv_wm1,
    Hm1f,
    Wm1f,
    sf,
    BP: tl.constexpr,
    DEJ: tl.constexpr,
    JITC: tl.constexpr,
):
    S: tl.constexpr = 4
    S2: tl.constexpr = 16
    i = tl.program_id(1)
    j = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = j < w
    bi = i // 2 * 2
    bj = j // 2 * 2
    best = tl.full([BP], float("-inf"), tl.float32)
    py = tl.zeros([BP], tl.float32)
    px = tl.zeros([BP], tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            ii = bi + dy
            jj = bj + dx
            m = pm & (ii < h) & (jj < w)
            key = -tl.load(dep_ptr + ii * d_sh + jj * d_sw, mask=m, other=0.0)
            better = m & ((key > best) | (key != key))
            best = tl.where(better, key, best)
            py = tl.where(
                better, tl.load(mot_ptr + ii * m_sh + jj * m_sw, mask=m, other=0.0), py
            )
            px = tl.where(
                better,
                tl.load(mot_ptr + m_sc + ii * m_sh + jj * m_sw, mask=m, other=0.0),
                px,
            )
    jy = tl.load(jit_ptr)
    jx = tl.load(jit_ptr + 1)
    if DEJ == 1:
        py = py + (jy - tl.load(pjit_ptr))
        px = px - (jx - tl.load(pjit_ptr + 1))
    elif DEJ == 2:
        py = py + (tl.load(pjit_ptr) - jy)
        px = px + (tl.load(pjit_ptr + 1) - jx)
    mhy = py * sf
    mhx = px * sf
    q = tl.arange(0, S2)
    ch = tl.arange(0, 4)
    Yh = i * S + (q // S)[None, :]
    Xh = j[:, None] * S + (q % S)[None, :]
    mask = pm[:, None] & (q >= 0)[None, :]
    (iy, ix, inside) = _warp_coords(
        Yh.to(tl.float32),
        Xh.to(tl.float32),
        mhy[:, None],
        mhx[:, None],
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
    my1 = mask & (y0 + 1 < H)
    mx1 = mask & (x0 + 1 < W)
    o00 = y0 * W + x0
    c4 = ch[None, None, :]
    v = (
        tl.load(s4_ptr + o00[:, :, None] * 4 + c4, mask=mask[:, :, None], other=0.0)
        * (wx0 * wy0)[:, :, None]
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + 1)[:, :, None] * 4 + c4, mask=mx1[:, :, None], other=0.0
        ),
        (wx1 * wy0)[:, :, None],
        v,
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + W)[:, :, None] * 4 + c4, mask=my1[:, :, None], other=0.0
        ),
        (wx0 * wy1)[:, :, None],
        v,
    )
    v = tl.fma(
        tl.load(
            s4_ptr + (o00 + W + 1)[:, :, None] * 4 + c4,
            mask=(my1 & mx1)[:, :, None],
            other=0.0,
        ),
        (wx1 * wy1)[:, :, None],
        v,
    )
    tl.store(hist_ptr + (Yh * W + Xh)[:, :, None] * 4 + c4, v, mask=mask[:, :, None])
    k16 = tl.arange(0, 16)[None, :]
    r = tl.load(rgb_ptr + i * r_sh + j * r_sw, mask=pm, other=0.0)
    g = tl.load(rgb_ptr + r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    b = tl.load(rgb_ptr + 2 * r_sc + i * r_sh + j * r_sw, mask=pm, other=0.0)
    d = tl.load(dep_ptr + i * d_sh + j * d_sw, mask=pm, other=0.0)
    a16 = tl.where(
        k16 == 0,
        r[:, None],
        tl.where(
            k16 == 1,
            g[:, None],
            tl.where(k16 == 2, b[:, None], tl.where(k16 == 3, d[:, None], 0.0)),
        ),
    )
    if JITC:
        z0 = 2.0 * -jy
        z1 = 2.0 * jx
        a16 = tl.where(k16 == 4, z0, tl.where(k16 == 5, z1, a16))
    dt = stem_ptr.dtype.element_ty
    base = stem_ptr + i * s_sh + j[:, None] * s_sw
    tl.store(base + k16 * s_sc, a16.to(dt), mask=pm[:, None])
    v64 = tl.reshape(tl.permute(v, (0, 2, 1)), (BP, 64))
    n64 = tl.arange(0, 64)[None, :]
    tl.store(base + (16 + n64) * s_sc, v64.to(dt), mask=pm[:, None])


@triton.jit
def _conv3x3_pad_qpost2(
    x_ptr,
    w_ptr,
    b_ptr,
    hist_ptr,
    out_ptr,
    s4n_ptr,
    flag_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    Wh,
    Hh,
    CIN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    STAGES: tl.constexpr,
    CRELU: tl.constexpr,
    GUARD: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    COUT: tl.constexpr = 80
    pid = tl.program_id(0)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    q = tl.arange(0, 16)
    kk = tl.arange(0, BK)
    aa = tl.zeros([BM, 16], dtype=tl.float32)
    a0 = tl.zeros([BM, 16], dtype=tl.float32)
    a1 = tl.zeros([BM, 16], dtype=tl.float32)
    a2 = tl.zeros([BM, 16], dtype=tl.float32)
    af = tl.zeros([BM, 16], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = tl.load(x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :])
        wrow = w_ptr + (tap * CIN + k0 + kk)[:, None] * COUT + q[None, :]
        aa = tl.dot(a, tl.load(wrow), aa)
        a0 = tl.dot(a, tl.load(wrow + 16), a0)
        a1 = tl.dot(a, tl.load(wrow + 32), a1)
        a2 = tl.dot(a, tl.load(wrow + 48), a2)
        af = tl.dot(a, tl.load(wrow + 64), af)
    p = p0 + m
    rowp = (p - SL) // WP
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    i = rowp - 1
    j = col - 1
    msk = pm[:, None] & (q >= 0)[None, :]
    al = (aa + tl.load(b_ptr + q)[None, :]).to(tl.bfloat16).to(tl.float32)
    k0v = (a0 + tl.load(b_ptr + 16 + q)[None, :]).to(tl.bfloat16).to(tl.float32)
    k1v = (a1 + tl.load(b_ptr + 32 + q)[None, :]).to(tl.bfloat16).to(tl.float32)
    k2v = (a2 + tl.load(b_ptr + 48 + q)[None, :]).to(tl.bfloat16).to(tl.float32)
    fe = (af + tl.load(b_ptr + 64 + q)[None, :]).to(tl.bfloat16).to(tl.float32)
    if CRELU:
        k0v = tl.maximum(k0v, 0.0)
        k1v = tl.maximum(k1v, 0.0)
        k2v = tl.maximum(k2v, 0.0)
    Y = i[:, None] * 4 + (q // 4)[None, :]
    X = j[:, None] * 4 + (q % 4)[None, :]
    hr = Y * Wh + X
    c4 = tl.arange(0, 4)[None, None, :]
    v = tl.load(hist_ptr + hr[:, :, None] * 4 + c4, mask=msk[:, :, None], other=0.0)
    (a02, a13) = tl.split(tl.reshape(v, (BM, 16, 2, 2)))
    (h0, h2) = tl.split(a02)
    (h1, _) = tl.split(a13)
    s = tl.math.div_rn(1.0, 1.0 + tl.exp(-al))
    y0 = s * k0v + (1.0 - s) * h0
    y1 = s * k1v + (1.0 - s) * h1
    y2 = s * k2v + (1.0 - s) * h2
    HW = Hh * Wh
    tl.store(out_ptr + hr, y0, mask=msk)
    tl.store(out_ptr + HW + hr, y1, mask=msk)
    tl.store(out_ptr + 2 * HW + hr, y2, mask=msk)
    ov = tl.reshape(tl.join(tl.join(y0, y2), tl.join(y1, fe)), (BM, 16, 4))
    tl.store(s4n_ptr + hr[:, :, None] * 4 + c4, ov, mask=msk[:, :, None])
    if GUARD:
        ok = (
            (y0 >= -2.0)
            & (y0 <= 3.0)
            & (y1 >= -2.0)
            & (y1 <= 3.0)
            & (y2 >= -2.0)
            & (y2 <= 3.0)
            & (tl.abs(fe) <= 1000.0)
        )
        bad = tl.max((msk & (ok == 0)).to(tl.int32), axis=1)
        tl.store(flag_ptr + i * W + j, bad.to(tl.int8), mask=pm)


@triton.jit
def _kq_guard(flag_ptr, s4n_ptr, h, w, W, BP: tl.constexpr):
    i = tl.program_id(1)
    j0 = tl.program_id(0) * BP
    jj = j0 + tl.arange(0, BP)
    om = jj < w
    bad = tl.zeros([BP], dtype=tl.int32)
    for di in tl.static_range(3):
        for dj in tl.static_range(3):
            ii = i + di - 1
            jn = jj + dj - 1
            mm = om & (ii >= 0) & (ii < h) & (jn >= 0) & (jn < w)
            bad = tl.maximum(
                bad, tl.load(flag_ptr + ii * w + jn, mask=mm, other=0).to(tl.int32)
            )
    if tl.max(bad, axis=0) > 0:
        q = tl.arange(0, 16)
        Y = i * 4 + (q // 4)[None, :]
        X = jj[:, None] * 4 + (q % 4)[None, :]
        hr = Y * (4 * w) + X
        c4 = tl.arange(0, 4)[None, None, :]
        bm = ((bad > 0) & om)[:, None] & (q >= 0)[None, :]
        tl.store(
            s4n_ptr + hr[:, :, None] * 4 + c4,
            tl.zeros([BP, 16, 4], tl.float32),
            mask=bm[:, :, None],
        )


class _Conv:
    @staticmethod
    def make(w, b):
        c = torch.nn.Conv2d(w.shape[1], w.shape[0], 3, padding=1).to(w.device)
        with torch.no_grad():
            c.weight.copy_(w.float())
            c.bias.copy_(b.float())
        return c


def relu_conv(conv, cin_p=None, cfg=None, kind="auto"):
    cout = conv.weight.shape[0]
    if kind == "kn":
        return C3.PadConvKN(conv, C3.EPI_RELU, cin_p=cin_p, cfg=cfg)
    if cout & cout - 1 == 0 and kind in ("auto", "ptr"):
        return ReluConv(conv, C3.EPI_BIAS, cin_p=cin_p, cfg=cfg)
    return C3.make_padconv(conv, C3.EPI_RELU, cin_p=cin_p, cfg=cfg)


class QrispFast2:
    def __init__(self, spec: dict, cfg: dict | None = None):
        cfg = dict(cfg or {})
        (wf, bf) = spec["first"]
        f = wf.shape[0]
        self.f = f
        self.jitc = bool(spec.get("jitc", True))
        self.dej = int(spec.get("dejitter", 0))
        self.crelu = bool(spec.get("crelu", True))
        self.guard = bool(spec.get("guard", True))
        ncin = wf.shape[1]
        nfix = 6 if self.jitc else 4
        assert ncin == nfix + 64, (ncin, nfix)
        self.cin = int(cfg.get("cin_p", 96))
        w = torch.zeros(f, self.cin, 3, 3, device=wf.device)
        w[:, 0:nfix] = wf[:, 0:nfix].float()
        w[:, 16:80] = wf[:, nfix:].float()
        pow2 = f & f - 1 == 0
        kind = cfg.get("conv_kind", "auto")
        c64 = C3.default_cfg(64, C3.EPI_GELU)
        cst = C3.default_cfg(64, C3.EPI_FILM)
        if not pow2:
            c64 = {k: v for (k, v) in c64.items() if k != "BK"}
            cst = {k: v for (k, v) in cst.items() if k != "BK"}
        elif c64.get("BK", 64) > f:
            c64["BK"] = f
        self.c_first = relu_conv(
            _Conv.make(w, bf), cin_p=self.cin, cfg=cfg.get("first", cst), kind=kind
        )
        self.c_mid = [
            relu_conv(_Conv.make(mw, mb), cfg=cfg.get("mid", c64), kind=kind)
            for (mw, mb) in spec["mid"]
        ]
        (wl, bl) = spec["last"]
        assert wl.shape[0] == 80
        lc = dict(cfg.get("last", c64))
        lc["BN"] = 80
        if f % lc.get("BK", 64):
            lc["BK"] = 16 if f % 16 == 0 else f
        self.c_last = PadConv(_Conv.make(wl, bl), C3.EPI_BIAS, cfg=lc)
        self.c_last.b = bl.float().to(torch.bfloat16).float().contiguous()
        self.bp0 = cfg.get("prep_bp", 16)
        self._bufs = {}

    def buffers(self, rgb):
        (h, w) = rgb.shape[-2:]
        if (h, w) in self._bufs:
            return self._bufs[h, w]
        dev = rgb.device
        b = dict(
            pin=PadBuf(self.cin, h, w, torch.bfloat16, dev),
            pa=PadBuf(self.f, h, w, torch.bfloat16, dev),
            pb=PadBuf(self.f, h, w, torch.bfloat16, dev),
            hist=torch.empty((16 * h * w, 4), device=dev, dtype=torch.float32),
            flags=torch.zeros((h, w), device=dev, dtype=torch.int8),
        )
        self._bufs[h, w] = b
        return b

    def run_into(self, rgb, depth, motion, jitter, s4, prev_j, out, s4n):
        b = self.buffers(rgb)
        (h, w) = rgb.shape[-2:]
        (H, W) = (4 * h, 4 * w)
        c = _consts(h, w, 4)
        stem = b["pin"].view
        _kq_prep2[triton.cdiv(w, self.bp0), h](
            rgb,
            depth,
            motion,
            jitter,
            prev_j,
            s4,
            stem,
            b["hist"],
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
            stem.stride(1),
            stem.stride(2),
            stem.stride(3),
            c["inv_hm1"],
            c["inv_wm1"],
            c["Hm1f"],
            c["Wm1f"],
            c["sf"],
            BP=self.bp0,
            DEJ=self.dej,
            JITC=self.jitc,
            num_warps=4,
            enable_fp_fusion=FP_FUSION,
        )
        self.c_first(b["pin"], b["pa"])
        (x, y) = (b["pa"], b["pb"])
        for conv in self.c_mid:
            conv(x, y)
            (x, y) = (y, x)
        cl = self.c_last
        grid = (triton.cdiv(x.p_hi - x.p_lo, cl.bm),)
        _conv3x3_pad_qpost2[grid](
            x.buf,
            cl.w,
            cl.b,
            b["hist"],
            out,
            s4n,
            b["flags"],
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            W,
            H,
            CIN=cl.cin_p,
            BM=cl.bm,
            BK=cl.bk,
            STAGES=cl.stages,
            CRELU=self.crelu,
            GUARD=self.guard,
            num_warps=cl.warps,
        )
        if self.guard:
            _kq_guard[triton.cdiv(w, 32), h](
                b["flags"], s4n, h, w, W, BP=32, num_warps=4
            )
        return out


def spec_from_qrispnet(net, jitter):
    with torch.no_grad():
        (wf, wl) = net.kernels(jitter[:1])
    return dict(
        first=(wf[0], net.first_bias.detach()),
        mid=[(c.weight.detach(), c.bias.detach()) for c in net.mid],
        last=(wl[0], net.last_bias.detach()),
        jitc=True,
        dejitter=1 if net.dejitter else 0,
        crelu=True,
        guard=bool(net.state_guard),
    )


def spec_random(f, m, device, seed=0):
    g = torch.Generator().manual_seed(seed)

    def conv(cin, cout):
        k = 1.0 / (cin * 9) ** 0.5
        w = (torch.rand(cout, cin, 3, 3, generator=g) * 2 - 1) * k
        b = (torch.rand(cout, generator=g) * 2 - 1) * k
        return (w.to(device), b.to(device))

    return dict(
        first=conv(70, f),
        mid=[conv(f, f) for _ in range(m)],
        last=conv(f, 80),
        jitc=True,
        dejitter=0,
        crelu=True,
        guard=True,
    )


class QrispFixedGraph2:
    def __init__(self, fast: QrispFast2, rgb, depth, motion, jitter, state):
        from .resolve_triton3 import capture

        (hist, prev_j) = state
        (H, W) = hist.shape[-2:]
        self.s4 = hist.float().permute(0, 2, 3, 1).contiguous()
        self.pj = prev_j.float().contiguous()
        self.jf = jitter.float().contiguous()
        self.out = torch.empty((1, 3, H, W), device=rgb.device)
        self.s4n = torch.empty_like(self.s4)
        args = (rgb, depth, motion, self.jf, self.s4, self.pj, self.out, self.s4n)
        (self.graph, _) = capture(lambda: fast.run_into(*args))

    def __call__(self):
        self.graph.replay()
        return self.out
