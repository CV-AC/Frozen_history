from __future__ import annotations
import torch
import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice
(EPI_BIAS, EPI_GELU, EPI_RES, EPI_FILM, EPI_RELU) = (0, 1, 2, 3, 4)


@triton.jit
def _conv3x3_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    H,
    W,
    x_ps,
    y_ps,
    r_ps,
    CIN_P: tl.constexpr,
    COUT: tl.constexpr,
    COUT_P: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    EPI: tl.constexpr,
    STAGES: tl.constexpr,
):
    BM: tl.constexpr = BH * BW
    NK: tl.constexpr = CIN_P // BK
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    ntw = tl.cdiv(W, BW)
    th = pid // ntw
    tw = pid % ntw
    m = tl.arange(0, BM)
    r = th * BH + m // BW
    c = tw * BW + m % BW
    n = pid_n * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        rr = r + (tap // 3 - 1)
        cc = c + (tap % 3 - 1)
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        a = tl.load(
            x_ptr + (rr * W + cc)[:, None] * x_ps + (k0 + kk)[None, :],
            mask=ok[:, None],
            other=0.0,
        )
        a = a.to(tl.bfloat16)
        b = tl.load(w_ptr + (tap * CIN_P + k0 + kk)[:, None] * COUT_P + n[None, :])
        acc = tl.dot(a, b, acc)
    pm = (r < H) & (c < W)
    nm = n < COUT
    om = pm[:, None] & nm[None, :]
    pix = r * W + c
    v = (
        acc.to(tl.bfloat16).to(tl.float32)
        + tl.load(b_ptr + n, mask=nm, other=0.0)[None, :]
    )
    yp = y_ptr + pix[:, None] * y_ps + n[None, :]
    if EPI == 1:
        v = v * 0.5 * (libdevice.erf(v * 0.7071067811865476) + 1.0)
        tl.store(yp, v.to(tl.bfloat16), mask=om)
    elif EPI == 2:
        res = tl.load(r_ptr + pix[:, None] * r_ps + n[None, :], mask=om, other=0.0).to(
            tl.float32
        )
        sc = tl.load(s_ptr + n, mask=nm, other=0.0)
        v = res + sc[None, :] * v
        tl.store(yp, v, mask=om)
    elif EPI == 3:
        g = tl.load(g_ptr + n, mask=nm, other=0.0)
        be = tl.load(g_ptr + COUT + n, mask=nm, other=0.0)
        v = v * (g + 1.0)[None, :] + be[None, :]
        tl.store(yp, v.to(tl.bfloat16), mask=om)
    else:
        tl.store(yp, v.to(y_ptr.dtype.element_ty), mask=om)


def pack_weight(conv: torch.nn.Conv2d, cin_p: int, cout_p: int):
    w = conv.weight.detach().float()
    (cout, cin) = w.shape[:2]
    out = torch.zeros(9, cin_p, cout_p, device=w.device)
    out[:, :cin, :cout] = w.permute(2, 3, 1, 0).reshape(9, cin, cout)
    return out.reshape(9 * cin_p, cout_p).to(torch.bfloat16).contiguous()


class ConvSpec:
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        (cout, cin) = conv.weight.shape[:2]
        (self.cin, self.cout) = (cin, cout)
        self.cin_p = cin_p or cin
        self.epi = epi
        cfg = dict(cfg or {})
        self.bn = cfg.get("BN", min(64, triton.next_power_of_2(cout)))
        self.cout_p = triton.cdiv(cout, self.bn) * self.bn
        self.bk = cfg.get(
            "BK", 64 if self.cin_p % 64 == 0 else 32 if self.cin_p % 32 == 0 else 16
        )
        assert self.cin_p % self.bk == 0
        self.bh = cfg.get("BH", 8)
        self.bw = cfg.get("BW", 16)
        self.warps = cfg.get("warps", 4)
        self.stages = cfg.get("stages", 3)
        self.w = pack_weight(conv, self.cin_p, self.cout_p)
        self.b = conv.bias.detach().float().contiguous()

    def __call__(self, x, y, res=None, scale=None, gb=None):
        (H, W) = x.shape[-2:]
        grid = (
            triton.cdiv(H, self.bh) * triton.cdiv(W, self.bw),
            self.cout_p // self.bn,
        )
        dummy = self.b
        _conv3x3_kernel[grid](
            x,
            self.w,
            self.b,
            res if res is not None else dummy,
            scale if scale is not None else dummy,
            gb if gb is not None else dummy,
            y,
            H,
            W,
            x.stride(3),
            y.stride(3),
            res.stride(3) if res is not None else 0,
            CIN_P=self.cin_p,
            COUT=self.cout,
            COUT_P=self.cout_p,
            BK=self.bk,
            BN=self.bn,
            BH=self.bh,
            BW=self.bw,
            EPI=self.epi,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


class PadBuf:
    def __init__(self, C, H, W, dtype, device, slack=None):
        (self.C, self.H, self.W) = (C, H, W)
        self.WP = W + 2
        self.SL = slack or W + 3 + 512
        self.NP = self.SL + (H + 2) * self.WP + self.SL
        self.buf = torch.zeros(self.NP, C, device=device, dtype=dtype)
        self.view = torch.as_strided(
            self.buf,
            (1, C, H, W),
            (self.NP * C, 1, self.WP * C, C),
            (self.SL + self.WP + 1) * C,
        )
        self.p_lo = self.SL + self.WP
        self.p_hi = self.SL + (H + 1) * self.WP


@triton.jit
def _conv3x3_pad(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
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
    EPI: tl.constexpr,
    Y2: tl.constexpr,
    STAGES: tl.constexpr,
    CACHE: tl.constexpr = "",
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
        a = tl.load(
            x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :],
            cache_modifier=CACHE,
        )
        b = tl.load(w_ptr + (tap * CIN + k0 + kk)[:, None] * COUT_P + n[None, :])
        acc = tl.dot(a, b, acc)
    p = p0 + m
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    nm = n < COUT
    om = pm[:, None] & nm[None, :]
    v = (
        acc.to(tl.bfloat16).to(tl.float32)
        + tl.load(b_ptr + n, mask=nm, other=0.0)[None, :]
    )
    o = p[:, None] * COUT + n[None, :]
    if EPI == 1:
        v = v * 0.5 * (libdevice.erf(v * 0.7071067811865476) + 1.0)
        tl.store(y_ptr + o, v.to(tl.bfloat16), mask=om)
    elif EPI == 2:
        res = tl.load(r_ptr + o, mask=om, other=0.0).to(tl.float32)
        sc = tl.load(s_ptr + n, mask=nm, other=0.0)
        v = res + sc[None, :] * v
        tl.store(y_ptr + o, v, mask=om)
        if Y2:
            tl.store(y2_ptr + o, v.to(tl.bfloat16), mask=om)
    elif EPI == 3:
        g = tl.load(g_ptr + n, mask=nm, other=0.0)
        be = tl.load(g_ptr + COUT + n, mask=nm, other=0.0)
        v = v * (g + 1.0)[None, :] + be[None, :]
        tl.store(y_ptr + o, v.to(tl.bfloat16), mask=om)
    else:
        tl.store(y_ptr + o, v.to(y_ptr.dtype.element_ty), mask=om)


class PadConv:
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        (cout, cin) = conv.weight.shape[:2]
        (self.cin, self.cout) = (cin, cout)
        self.cin_p = cin_p or cin
        self.epi = epi
        cfg = dict(cfg or {})
        self.bn = cfg.get("BN", min(64, triton.next_power_of_2(cout)))
        self.cout_p = triton.cdiv(cout, self.bn) * self.bn
        self.bk = cfg.get(
            "BK", 64 if self.cin_p % 64 == 0 else 32 if self.cin_p % 32 == 0 else 16
        )
        assert self.cin_p % self.bk == 0
        self.bm = cfg.get("BM", 128)
        self.warps = cfg.get("warps", 4)
        self.stages = cfg.get("stages", 3)
        self.cache = cfg.get("cache", "")
        self.w = pack_weight(conv, self.cin_p, self.cout_p)
        self.b = conv.bias.detach().float().contiguous()

    def __call__(
        self,
        x: PadBuf,
        y: PadBuf,
        res: PadBuf | None = None,
        scale=None,
        gb=None,
        y2: PadBuf | None = None,
    ):
        assert x.C == self.cin_p and y.C == self.cout
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm), self.cout_p // self.bn)
        d = self.b
        _conv3x3_pad[grid](
            x.buf,
            self.w,
            self.b,
            res.buf if res is not None else d,
            scale if scale is not None else d,
            gb if gb is not None else d,
            y.buf,
            y2.buf if y2 is not None else d,
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
            EPI=self.epi,
            Y2=y2 is not None,
            STAGES=self.stages,
            CACHE=self.cache,
            num_warps=self.warps,
        )
        return y


@triton.jit
def _epilogue(
    acc,
    n,
    nm,
    p,
    pm,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    COUT: tl.constexpr,
    EPI: tl.constexpr,
    Y2: tl.constexpr,
):
    om = pm[:, None] & nm[None, :]
    v = (
        acc.to(tl.bfloat16).to(tl.float32)
        + tl.load(b_ptr + n, mask=nm, other=0.0)[None, :]
    )
    o = p[:, None] * COUT + n[None, :]
    if EPI == 1:
        v = v * 0.5 * (libdevice.erf(v * 0.7071067811865476) + 1.0)
        tl.store(y_ptr + o, v.to(tl.bfloat16), mask=om)
    elif EPI == 2:
        res = tl.load(r_ptr + o, mask=om, other=0.0).to(tl.float32)
        sc = tl.load(s_ptr + n, mask=nm, other=0.0)
        v = res + sc[None, :] * v
        tl.store(y_ptr + o, v, mask=om)
        if Y2:
            tl.store(y2_ptr + o, v.to(tl.bfloat16), mask=om)
    elif EPI == 3:
        g = tl.load(g_ptr + n, mask=nm, other=0.0)
        be = tl.load(g_ptr + COUT + n, mask=nm, other=0.0)
        v = v * (g + 1.0)[None, :] + be[None, :]
        tl.store(y_ptr + o, v.to(tl.bfloat16), mask=om)
    elif EPI == 4:
        tl.store(y_ptr + o, tl.maximum(v, 0.0).to(tl.bfloat16), mask=om)
    else:
        tl.store(y_ptr + o, v.to(y_ptr.dtype.element_ty), mask=om)


@triton.jit
def _conv3x3_pad_split(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BM: tl.constexpr,
    BN1: tl.constexpr,
    BN2: tl.constexpr,
    BK: tl.constexpr,
    EPI: tl.constexpr,
    Y2: tl.constexpr,
    STAGES: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    pid = tl.program_id(0)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    n1 = tl.arange(0, BN1)
    n2 = BN1 + tl.arange(0, BN2)
    kk = tl.arange(0, BK)
    acc1 = tl.zeros([BM, BN1], dtype=tl.float32)
    acc2 = tl.zeros([BM, BN2], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = tl.load(x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :])
        wrow = (tap * CIN + k0 + kk)[:, None] * COUT
        b1 = tl.load(w_ptr + wrow + n1[None, :])
        b2 = tl.load(w_ptr + wrow + n2[None, :])
        acc1 = tl.dot(a, b1, acc1)
        acc2 = tl.dot(a, b2, acc2)
    p = p0 + m
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    _epilogue(
        acc1,
        n1,
        n1 < COUT,
        p,
        pm,
        b_ptr,
        r_ptr,
        s_ptr,
        g_ptr,
        y_ptr,
        y2_ptr,
        COUT,
        EPI,
        Y2,
    )
    _epilogue(
        acc2,
        n2,
        n2 < COUT,
        p,
        pm,
        b_ptr,
        r_ptr,
        s_ptr,
        g_ptr,
        y_ptr,
        y2_ptr,
        COUT,
        EPI,
        Y2,
    )


class PadConvSplit(PadConv):
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        cfg = dict(cfg or {})
        cout = conv.weight.shape[0]
        bn1 = cfg.pop("BN1", 1 << cout.bit_length() - 1)
        bn2 = cout - bn1
        assert bn2 >= 16 and bn2 & bn2 - 1 == 0, (cout, bn1, bn2)
        cfg["BN"] = cout
        super().__init__(conv, epi, cin_p=cin_p, cfg=cfg)
        (self.bn1, self.bn2) = (bn1, bn2)

    def __call__(self, x, y, res=None, scale=None, gb=None, y2=None):
        assert x.C == self.cin_p and y.C == self.cout
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm),)
        d = self.b
        _conv3x3_pad_split[grid](
            x.buf,
            self.w,
            self.b,
            res.buf if res is not None else d,
            scale if scale is not None else d,
            gb if gb is not None else d,
            y.buf,
            y2.buf if y2 is not None else d,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            CIN=self.cin_p,
            COUT=self.cout,
            BM=self.bm,
            BN1=self.bn1,
            BN2=self.bn2,
            BK=self.bk,
            EPI=self.epi,
            Y2=y2 is not None,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


def make_padconv(conv, epi, cin_p=None, cfg=None):
    cout = conv.weight.shape[0]
    if cout & cout - 1 == 0 or (cfg and "BN" in cfg):
        return PadConv(conv, epi, cin_p=cin_p, cfg=cfg)
    return PadConvSplit(conv, epi, cin_p=cin_p, cfg=cfg)


@triton.jit
def _conv3x3_tma(
    x_desc,
    w_desc,
    w2_desc,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BN2: tl.constexpr,
    BK: tl.constexpr,
    EPI: tl.constexpr,
    Y2: tl.constexpr,
    STAGES: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    p0 = P_LO + pid * BM
    n0 = pid_n * BN
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    if BN2 > 0:
        acc2 = tl.zeros([BM, BN2], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = x_desc.load([p0 + off, k0])
        b = w_desc.load([tap * CIN + k0, n0])
        acc = tl.dot(a, b, acc)
        if BN2 > 0:
            b2 = w2_desc.load([tap * CIN + k0, 0])
            acc2 = tl.dot(a, b2, acc2)
    p = p0 + tl.arange(0, BM)
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    n = n0 + tl.arange(0, BN)
    _epilogue(
        acc,
        n,
        n < COUT,
        p,
        pm,
        b_ptr,
        r_ptr,
        s_ptr,
        g_ptr,
        y_ptr,
        y2_ptr,
        COUT,
        EPI,
        Y2,
    )
    if BN2 > 0:
        n2 = BN + tl.arange(0, BN2)
        _epilogue(
            acc2,
            n2,
            n2 < COUT,
            p,
            pm,
            b_ptr,
            r_ptr,
            s_ptr,
            g_ptr,
            y_ptr,
            y2_ptr,
            COUT,
            EPI,
            Y2,
        )


class TmaConv:
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        from triton.tools.tensor_descriptor import TensorDescriptor

        self.TD = TensorDescriptor
        (cout, cin) = conv.weight.shape[:2]
        (self.cin, self.cout) = (cin, cout)
        self.cin_p = cin_p or cin
        self.epi = epi
        cfg = dict(cfg or {})
        pow2 = cout & cout - 1 == 0
        if pow2:
            self.bn = cfg.get("BN", min(64, cout))
            self.bn2 = 0
            self.cout_p = triton.cdiv(cout, self.bn) * self.bn
        else:
            self.bn = cfg.get("BN1", 1 << cout.bit_length() - 1)
            self.bn2 = cout - self.bn
            assert self.bn2 >= 16 and self.bn2 & self.bn2 - 1 == 0
            self.cout_p = cout
        self.bk = cfg.get(
            "BK", 64 if self.cin_p % 64 == 0 else 32 if self.cin_p % 32 == 0 else 16
        )
        self.bm = cfg.get("BM", 128)
        self.warps = cfg.get("warps", 4)
        self.stages = cfg.get("stages", 4)
        self.w = pack_weight(conv, self.cin_p, self.cout_p)
        self.b = conv.bias.detach().float().contiguous()
        self.wd = TensorDescriptor(
            self.w, [9 * self.cin_p, self.cout_p], [self.cout_p, 1], [self.bk, self.bn]
        )
        if self.bn2:
            self.w2 = self.w[:, self.bn :].contiguous()
            self.wd2 = TensorDescriptor(
                self.w2, [9 * self.cin_p, self.bn2], [self.bn2, 1], [self.bk, self.bn2]
            )
        else:
            self.wd2 = self.wd
        self._xd = {}

    def _xdesc(self, x: PadBuf):
        key = x.buf.data_ptr()
        if key not in self._xd:
            self._xd[key] = self.TD(x.buf, [x.NP, x.C], [x.C, 1], [self.bm, self.bk])
        return self._xd[key]

    def __call__(self, x: PadBuf, y: PadBuf, res=None, scale=None, gb=None, y2=None):
        assert x.C == self.cin_p and y.C == self.cout
        grid = (
            triton.cdiv(x.p_hi - x.p_lo, self.bm),
            1 if self.bn2 else self.cout_p // self.bn,
        )
        d = self.b
        _conv3x3_tma[grid](
            self._xdesc(x),
            self.wd,
            self.wd2,
            self.b,
            res.buf if res is not None else d,
            scale if scale is not None else d,
            gb if gb is not None else d,
            y.buf,
            y2.buf if y2 is not None else d,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            CIN=self.cin_p,
            COUT=self.cout,
            BM=self.bm,
            BN=self.bn,
            BN2=self.bn2,
            BK=self.bk,
            EPI=self.epi,
            Y2=y2 is not None,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


def make_conv(conv, epi, cin_p=None, cfg=None, kind="auto"):
    if kind == "auto":
        kind = "ptr"
    if kind == "persist":
        return PersistConv(conv, epi, cin_p=cin_p, cfg=cfg)
    if kind == "kn":
        return PadConvKN(conv, epi, cin_p=cin_p, cfg=cfg)
    if kind == "tma":
        return TmaConv(conv, epi, cin_p=cin_p, cfg=cfg)
    return make_padconv(conv, epi, cin_p=cin_p, cfg=cfg)


@triton.jit
def _tap(x_ptr, p, kk, off, CIN: tl.constexpr, CACHE: tl.constexpr):
    return tl.load(x_ptr + (p + off)[:, None] * CIN + kk[None, :], cache_modifier=CACHE)


@triton.jit
def _conv3x3_persist(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    NTILES,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BM: tl.constexpr,
    EPI: tl.constexpr,
    Y2: tl.constexpr,
    CACHE: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    kk = tl.arange(0, CIN)
    n = tl.arange(0, COUT)
    wr = kk[:, None] * COUT + n[None, :]
    w0 = tl.load(w_ptr + 0 * CIN * COUT + wr)
    w1 = tl.load(w_ptr + 1 * CIN * COUT + wr)
    w2 = tl.load(w_ptr + 2 * CIN * COUT + wr)
    w3 = tl.load(w_ptr + 3 * CIN * COUT + wr)
    w4 = tl.load(w_ptr + 4 * CIN * COUT + wr)
    w5 = tl.load(w_ptr + 5 * CIN * COUT + wr)
    w6 = tl.load(w_ptr + 6 * CIN * COUT + wr)
    w7 = tl.load(w_ptr + 7 * CIN * COUT + wr)
    w8 = tl.load(w_ptr + 8 * CIN * COUT + wr)
    m = tl.arange(0, BM)
    nm = n < COUT
    for tile in tl.range(pid, NTILES, nprog, num_stages=STAGES):
        p = P_LO + tile * BM + m
        acc = tl.zeros([BM, COUT], dtype=tl.float32)
        acc = tl.dot(_tap(x_ptr, p, kk, -WP - 1, CIN, CACHE), w0, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, -WP, CIN, CACHE), w1, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, -WP + 1, CIN, CACHE), w2, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, -1, CIN, CACHE), w3, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, 0, CIN, CACHE), w4, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, 1, CIN, CACHE), w5, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, WP - 1, CIN, CACHE), w6, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, WP, CIN, CACHE), w7, acc)
        acc = tl.dot(_tap(x_ptr, p, kk, WP + 1, CIN, CACHE), w8, acc)
        col = (p - SL) % WP
        pm = (p < P_HI) & (col >= 1) & (col <= W)
        _epilogue(
            acc, n, nm, p, pm, b_ptr, r_ptr, s_ptr, g_ptr, y_ptr, y2_ptr, COUT, EPI, Y2
        )


class PersistConv(PadConv):
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        cfg = dict(cfg or {})
        cout = conv.weight.shape[0]
        cfg["BN"] = cout
        super().__init__(conv, epi, cin_p=cin_p, cfg=cfg)
        assert self.cin_p & self.cin_p - 1 == 0 and self.cin_p <= 64
        assert cout & cout - 1 == 0 and cout <= 64
        self.cache = cfg.get("cache", "")
        self.nprog = cfg.get("nprog", None)

    def __call__(self, x, y, res=None, scale=None, gb=None, y2=None):
        assert x.C == self.cin_p and y.C == self.cout
        ntiles = triton.cdiv(x.p_hi - x.p_lo, self.bm)
        nsm = torch.cuda.get_device_properties(x.buf.device).multi_processor_count
        nprog = min(ntiles, self.nprog or nsm)
        d = self.b
        _conv3x3_persist[nprog,](
            x.buf,
            self.w,
            self.b,
            res.buf if res is not None else d,
            scale if scale is not None else d,
            gb if gb is not None else d,
            y.buf,
            y2.buf if y2 is not None else d,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            ntiles,
            CIN=self.cin_p,
            COUT=self.cout,
            BM=self.bm,
            EPI=self.epi,
            Y2=y2 is not None,
            CACHE=self.cache,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


_CFG = {
    (9, 64, EPI_GELU): dict(BM=128, BK=64, warps=4, stages=3),
    (9, 64, EPI_RES): dict(BM=128, BK=32, warps=4, stages=4),
    (9, 64, EPI_FILM): dict(BM=256, BK=32, warps=4, stages=4),
    (9, 96, EPI_GELU): dict(BM=128, BK=32, warps=8, stages=4),
    (9, 96, EPI_RES): dict(BM=128, BK=32, warps=8, stages=4),
    (9, 96, EPI_FILM): dict(BM=128, BK=32, warps=8, stages=4),
    (8, 64, EPI_GELU): dict(BM=128, BK=32, warps=4, stages=4),
    (8, 64, EPI_RES): dict(BM=128, BK=32, warps=4, stages=4),
    (8, 64, EPI_FILM): dict(BM=128, BK=32, warps=4, stages=4),
    (8, 96, EPI_GELU): dict(BM=128, BK=32, warps=4, stages=3),
    (8, 96, EPI_RES): dict(BM=128, BK=32, warps=4, stages=4),
    (8, 96, EPI_FILM): dict(BM=128, BK=32, warps=4, stages=3),
}


def default_cfg(cout, epi):
    major = torch.cuda.get_device_capability()[0]
    e = EPI_FILM if epi in (EPI_FILM, EPI_BIAS) else epi
    return dict(_CFG.get((min(major, 9), cout, e), {}))


@triton.jit
def _conv3x3_pad_kn(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    g_ptr,
    y_ptr,
    y2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    KA: tl.constexpr,
    KB: tl.constexpr,
    NA: tl.constexpr,
    NB: tl.constexpr,
    BM: tl.constexpr,
    EPI: tl.constexpr,
    Y2: tl.constexpr,
    STAGES: tl.constexpr,
):
    CIN: tl.constexpr = KA + KB
    COUT: tl.constexpr = NA + NB
    pid = tl.program_id(0)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    ka = tl.arange(0, KA)
    kb = KA + tl.arange(0, KB)
    na = tl.arange(0, NA)
    nb = NA + tl.arange(0, NB)
    acc_a = tl.zeros([BM, NA], dtype=tl.float32)
    acc_b = tl.zeros([BM, NB], dtype=tl.float32)
    for tap in tl.range(0, 9, num_stages=STAGES):
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        rows = (p0 + off + m)[:, None] * CIN
        a1 = tl.load(x_ptr + rows + ka[None, :])
        a2 = tl.load(x_ptr + rows + kb[None, :])
        wa = (tap * CIN + ka)[:, None] * COUT
        wb = (tap * CIN + kb)[:, None] * COUT
        acc_a = tl.dot(a1, tl.load(w_ptr + wa + na[None, :]), acc_a)
        acc_b = tl.dot(a1, tl.load(w_ptr + wa + nb[None, :]), acc_b)
        acc_a = tl.dot(a2, tl.load(w_ptr + wb + na[None, :]), acc_a)
        acc_b = tl.dot(a2, tl.load(w_ptr + wb + nb[None, :]), acc_b)
    p = p0 + m
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    _epilogue(
        acc_a,
        na,
        na < COUT,
        p,
        pm,
        b_ptr,
        r_ptr,
        s_ptr,
        g_ptr,
        y_ptr,
        y2_ptr,
        COUT,
        EPI,
        Y2,
    )
    _epilogue(
        acc_b,
        nb,
        nb < COUT,
        p,
        pm,
        b_ptr,
        r_ptr,
        s_ptr,
        g_ptr,
        y_ptr,
        y2_ptr,
        COUT,
        EPI,
        Y2,
    )


class PadConvKN(PadConv):
    def __init__(self, conv, epi, cin_p=None, cfg=None):
        cfg = dict(cfg or {})
        cout = conv.weight.shape[0]
        cfg["BN"] = cout
        cfg.setdefault("BK", 16)
        super().__init__(conv, epi, cin_p=cin_p, cfg=cfg)

        def halves(c):
            a = 1 << c.bit_length() - 1
            return (a, c - a) if c != a else (c // 2, c // 2)

        (self.ka, self.kb) = halves(self.cin_p)
        (self.na, self.nb) = halves(cout)
        for v in (self.kb, self.nb):
            assert v >= 16 and v & v - 1 == 0, (self.cin_p, cout)

    def __call__(self, x, y, res=None, scale=None, gb=None, y2=None):
        assert x.C == self.cin_p and y.C == self.cout
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm),)
        d = self.b
        _conv3x3_pad_kn[grid](
            x.buf,
            self.w,
            self.b,
            res.buf if res is not None else d,
            scale if scale is not None else d,
            gb if gb is not None else d,
            y.buf,
            y2.buf if y2 is not None else d,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            KA=self.ka,
            KB=self.kb,
            NA=self.na,
            NB=self.nb,
            BM=self.bm,
            EPI=self.epi,
            Y2=y2 is not None,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y


@triton.jit
def _cc1h(x):
    return tl.fma(tl.fma(x, 1.25, -2.25) * x, x, 1.0)


@triton.jit
def _cc2h(x):
    return tl.fma(tl.fma(tl.fma(x, -0.75, 3.75), x, -6.0), x, 3.0)


@triton.jit
def _conv3x3_pad_heads(
    x_ptr,
    w_ptr,
    b_ptr,
    r_ptr,
    s_ptr,
    wc_ptr,
    bc_ptr,
    ws_ptr,
    bs_ptr,
    wf_ptr,
    bf_ptr,
    hx_ptr,
    jit_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    h,
    w,
    c_sh,
    c_sw,
    s_sh,
    s_sw,
    f_sh,
    f_sw,
    centre,
    inv_s,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    NS: tl.constexpr,
    NF: tl.constexpr,
    STAGES: tl.constexpr,
):
    NK: tl.constexpr = CIN // BK
    pid = tl.program_id(0)
    p0 = P_LO + pid * BM
    m = tl.arange(0, BM)
    n = tl.arange(0, COUT)
    kk = tl.arange(0, BK)
    acc = tl.zeros([BM, COUT], dtype=tl.float32)
    for t in tl.range(0, 9 * NK, num_stages=STAGES):
        tap = t // NK
        k0 = t % NK * BK
        off = (tap // 3 - 1) * WP + (tap % 3 - 1)
        a = tl.load(x_ptr + (p0 + off + m)[:, None] * CIN + (k0 + kk)[None, :])
        b = tl.load(w_ptr + (tap * CIN + k0 + kk)[:, None] * COUT + n[None, :])
        acc = tl.dot(a, b, acc)
    p = p0 + m
    rowp = (p - SL) // WP
    col = (p - SL) % WP
    pm = (p < P_HI) & (col >= 1) & (col <= W)
    i = rowp - 1
    j = col - 1
    o = p[:, None] * COUT + n[None, :]
    v = acc.to(tl.bfloat16).to(tl.float32) + tl.load(b_ptr + n)[None, :]
    res = tl.load(r_ptr + o, mask=pm[:, None], other=0.0).to(tl.float32)
    x = res + tl.load(s_ptr + n)[None, :] * v
    xb = x.to(tl.bfloat16)
    q = tl.arange(0, 16)
    ns = tl.arange(0, NS)
    nf = tl.arange(0, NF)
    kd = tl.arange(0, COUT)
    stat = (
        tl.dot(
            x, tl.load(ws_ptr + kd[:, None] * NS + ns[None, :]), input_precision="tf32"
        )
        + tl.load(bs_ptr + ns)[None, :]
    )
    fv = tl.dot(xb, tl.load(wf_ptr + kd[:, None] * NF + nf[None, :]))
    fv = fv.to(tl.bfloat16).to(tl.float32) + tl.load(bf_ptr + nf)[None, :]
    wrow = wc_ptr + kd[:, None] * 48 + q[None, :]
    c0 = tl.dot(xb, tl.load(wrow))
    c1 = tl.dot(xb, tl.load(wrow + 16))
    c2 = tl.dot(xb, tl.load(wrow + 32))
    c0 = (
        (c0.to(tl.bfloat16).to(tl.float32) + tl.load(bc_ptr + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    c1 = (
        (c1.to(tl.bfloat16).to(tl.float32) + tl.load(bc_ptr + 16 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    c2 = (
        (c2.to(tl.bfloat16).to(tl.float32) + tl.load(bc_ptr + 32 + q)[None, :])
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    jy = tl.load(jit_ptr)
    dyq = q // 4
    a_ = (dyq.to(tl.float32) - centre) * inv_s + jy
    fa = tl.floor(a_)
    tt = a_ - fa
    k0c = _cc2h(tt + 1.0)
    k1c = _cc1h(tt)
    k2c = _cc1h(1.0 - tt)
    k3c = _cc2h(1.0 - tt + 1.0)
    fa1 = fa.to(tl.int32) + 1
    sk0 = tl.zeros([BM, 16], tl.float32)
    sk1 = tl.zeros([BM, 16], tl.float32)
    sk2 = tl.zeros([BM, 16], tl.float32)
    jc = tl.minimum(tl.maximum(j, 0), w - 1)
    for r in tl.static_range(5):
        tp = r - fa1
        wr = tl.where(
            tp == 0,
            k0c,
            tl.where(tp == 1, k1c, tl.where(tp == 2, k2c, tl.where(tp == 3, k3c, 0.0))),
        )[None, :]
        rr = tl.minimum(tl.maximum(i - 2 + r, 0), h - 1)
        hb = hx_ptr + (rr * w + jc)[:, None] * 16 + (q % 4)[None, :]
        sk0 = tl.fma(tl.load(hb, mask=pm[:, None], other=0.0), wr, sk0)
        sk1 = tl.fma(tl.load(hb + 4, mask=pm[:, None], other=0.0), wr, sk1)
        sk2 = tl.fma(tl.load(hb + 8, mask=pm[:, None], other=0.0), wr, sk2)
    sk0 = tl.minimum(tl.maximum(sk0, 0.0), 1.0)
    sk1 = tl.minimum(tl.maximum(sk1, 0.0), 1.0)
    sk2 = tl.minimum(tl.maximum(sk2, 0.0), 1.0)
    iw = tl.minimum(tl.maximum(i, 0), h - 1)
    tl.store(
        stat_ptr + (iw * s_sh + jc * s_sw)[:, None] + ns[None, :],
        stat,
        mask=pm[:, None],
    )
    tl.store(
        feat_ptr + (iw * f_sh + jc * f_sw)[:, None] + nf[None, :],
        fv.to(feat_ptr.dtype.element_ty),
        mask=pm[:, None],
    )
    cb = cand_ptr + (iw * c_sh + jc * c_sw)[:, None] + q[None, :]
    tl.store(cb, c0 + sk0, mask=pm[:, None])
    tl.store(cb + 16, c1 + sk1, mask=pm[:, None])
    tl.store(cb + 32, c2 + sk2, mask=pm[:, None])


class PadConvHeads(PadConv):
    def __init__(self, conv, Hd, cfg=None):
        cfg = dict(cfg or {})
        cout = conv.weight.shape[0]
        cfg["BN"] = cout
        super().__init__(conv, EPI_RES, cfg=cfg)
        assert cout & cout - 1 == 0 and cout <= 64, cout
        assert Hd["lowp"] and Hd["ns"] == 32 and (Hd["nc"] == 48)
        self.Hd = Hd

    def __call__(self, x, res, scale, hx, jf, cand, stat, feat):
        Hd = self.Hd
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm),)
        (h, w) = (x.H, x.W)
        _conv3x3_pad_heads[grid](
            x.buf,
            self.w,
            self.b,
            res.buf,
            scale,
            Hd["wc48"],
            Hd["bc"],
            Hd["ws"],
            Hd["bs"],
            Hd["wf"],
            Hd["bf"],
            hx,
            jf,
            cand,
            stat,
            feat,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            h,
            w,
            cand.stride(2),
            cand.stride(3),
            stat.stride(2),
            stat.stride(3),
            feat.stride(2),
            feat.stride(3),
            1.5,
            0.25,
            CIN=self.cin_p,
            COUT=self.cout,
            BM=self.bm,
            BK=self.bk,
            NS=Hd["ns"],
            NF=Hd["nf"],
            STAGES=self.stages,
            num_warps=self.warps,
        )
