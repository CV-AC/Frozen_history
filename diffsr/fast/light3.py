from __future__ import annotations
import math
import torch
import triton
import triton.language as tl
from . import conv3 as C3
from .conv3 import PadBuf

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice
(MIX_DW, MIX_SHIFT, MIX_POOL) = (0, 1, 2)
(ACT_RELU, ACT_GELU_TANH, ACT_GELU_ERF, ACT_GELU_TANH_FAST) = (0, 1, 2, 3)
ACTS = {
    "relu": ACT_RELU,
    "gelu": ACT_GELU_TANH_FAST,
    "gelu_fast": ACT_GELU_TANH_FAST,
    "gelu_tanh": ACT_GELU_TANH,
    "gelu_erf": ACT_GELU_ERF,
}
_CFG3 = {
    (9, "shiftmlp", 48): dict(bm=128, warps=4, persist=False),
    (9, "shiftmlp", 64): dict(bm=64, warps=4, persist=False),
    (9, "shiftmlp", 96): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
    (9, "dwmlp", 48): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=3,
        stages=2,
        split=dict(bm=32, cc=16, warps=1),
    ),
    (9, "dwmlp", 64): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=2,
        stages=3,
        split=dict(bm=32, cc=64, warps=2),
    ),
    (9, "dwmlp", 96): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=2,
        stages=1,
        split=dict(bm=32, cc=32, warps=8),
    ),
    (9, "poolmlp", 48): dict(bm=128, warps=4, persist=False),
    (9, "poolmlp", 64): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
    (9, "poolmlp", 96): dict(bm=64, warps=4, persist=True, mult=2, stages=2),
    (8, "shiftmlp", 48): dict(bm=64, warps=4, persist=True, mult=2, stages=2, svec=-1),
    (8, "shiftmlp", 64): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
    (8, "shiftmlp", 96): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
    (8, "dwmlp", 48): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=3,
        stages=2,
        split=dict(bm=256, cc=16, warps=4),
    ),
    (8, "dwmlp", 64): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=2,
        stages=3,
        split=dict(bm=64, cc=64, warps=4),
    ),
    (8, "dwmlp", 96): dict(
        bm=64,
        warps=4,
        persist=True,
        mult=2,
        stages=1,
        split=dict(bm=32, cc=32, warps=2),
    ),
    (8, "poolmlp", 48): dict(bm=128, warps=4, persist=False),
    (8, "poolmlp", 64): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
    (8, "poolmlp", 96): dict(bm=64, warps=4, persist=True, mult=2, stages=1),
}
_CFG3_DEFAULT = dict(bm=64, warps=4, persist=True, mult=2, stages=1)
_NSM = {}


def cfg3(kind: str, C: int, device=None) -> dict:
    dev = torch.device(device or "cuda")
    major = torch.cuda.get_device_capability(dev)[0]
    return dict(_CFG3.get((major, kind, int(C)), _CFG3_DEFAULT))


def num_sms(device) -> int:
    k = str(device)
    if k not in _NSM:
        _NSM[k] = torch.cuda.get_device_properties(device).multi_processor_count
    return _NSM[k]


def split2(n: int) -> tuple[int, int]:
    a = 1 << int(n).bit_length() - 1
    b = n - a
    if b and (b & b - 1 or b < 16):
        raise ValueError(f"width {n} is not a + b with powers of two >= 16")
    return (a, b)


@triton.jit
def _rbf(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _act(t, ACT: tl.constexpr):
    if ACT == 0:
        return tl.maximum(t, 0.0)
    elif ACT == 3:
        return (
            0.5
            * t
            * (1.0 + _tanh_approx(0.7978845608028654 * (t + 0.044715 * t * t * t)))
        )
    elif ACT == 1:
        return (
            0.5
            * t
            * (1.0 + libdevice.tanh(0.7978845608028654 * (t + 0.044715 * t * t * t)))
        )
    else:
        return t * 0.5 * (libdevice.erf(t * 0.7071067811865476) + 1.0)


@triton.jit
def _mlp(
    ha,
    hb,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    CA: tl.constexpr,
    CB: tl.constexpr,
    HA: tl.constexpr,
    HB: tl.constexpr,
    ACT: tl.constexpr,
):
    C: tl.constexpr = CA + CB
    RC: tl.constexpr = HA + HB
    ka = tl.arange(0, CA)
    na = tl.arange(0, HA)
    t_a = tl.dot(ha, tl.load(w1_ptr + ka[:, None] * RC + na[None, :]))
    if CB > 0:
        kb = CA + tl.arange(0, CB)
        t_a = tl.dot(hb, tl.load(w1_ptr + kb[:, None] * RC + na[None, :]), t_a)
    u_a = _act(_rbf(t_a) + tl.load(b1_ptr + na)[None, :], ACT).to(tl.bfloat16)
    ca = tl.arange(0, CA)
    o_a = tl.dot(u_a, tl.load(w2_ptr + na[:, None] * C + ca[None, :]))
    if CB > 0:
        cb = CA + tl.arange(0, CB)
        o_b = tl.dot(u_a, tl.load(w2_ptr + na[:, None] * C + cb[None, :]))
    else:
        o_b = o_a
    if HB > 0:
        nb = HA + tl.arange(0, HB)
        t_b = tl.dot(ha, tl.load(w1_ptr + ka[:, None] * RC + nb[None, :]))
        if CB > 0:
            kb2 = CA + tl.arange(0, CB)
            t_b = tl.dot(hb, tl.load(w1_ptr + kb2[:, None] * RC + nb[None, :]), t_b)
        u_b = _act(_rbf(t_b) + tl.load(b1_ptr + nb)[None, :], ACT).to(tl.bfloat16)
        o_a = tl.dot(u_b, tl.load(w2_ptr + nb[:, None] * C + ca[None, :]), o_a)
        if CB > 0:
            cb2 = CA + tl.arange(0, CB)
            o_b = tl.dot(u_b, tl.load(w2_ptr + nb[:, None] * C + cb2[None, :]), o_b)
    o_a = _rbf(o_a) + tl.load(b2_ptr + ca)[None, :]
    if CB > 0:
        cb3 = CA + tl.arange(0, CB)
        o_b = _rbf(o_b) + tl.load(b2_ptr + cb3)[None, :]
    return (o_a, o_b)


@triton.jit
def _dw_chunk(
    x_ptr,
    p,
    row,
    col,
    pm,
    H,
    W,
    WP,
    c,
    wdw_ptr,
    bdw_ptr,
    C: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
):
    acc = tl.zeros([p.shape[0], c.shape[0]], dtype=tl.float32)
    for ty in tl.static_range(KS):
        for tx in tl.static_range(KS):
            dy = (ty - KS // 2) * DIL
            dx = (tx - KS // 2) * DIL
            ok = (
                pm & (row + dy >= 0) & (row + dy < H) & (col + dx >= 0) & (col + dx < W)
            )
            v = tl.load(
                x_ptr + (p + dy * WP + dx)[:, None] * C + c[None, :],
                mask=ok[:, None],
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(wdw_ptr + (ty * KS + tx) * C + c)
            acc += v * wv[None, :]
    return (_rbf(acc) + tl.load(bdw_ptr + c)[None, :]).to(tl.bfloat16)


@triton.jit
def _shift_chunk(
    x_ptr,
    p,
    row,
    col,
    pm,
    H,
    W,
    WP,
    c,
    C: tl.constexpr,
    DIL: tl.constexpr,
    SGN: tl.constexpr,
):
    G: tl.constexpr = C // 8
    g = c // G
    dyr = tl.where(g == 0, -DIL, tl.where(g == 1, DIL, 0)) * SGN
    dxr = tl.where(g == 2, -DIL, tl.where(g == 3, DIL, 0)) * SGN
    rr = row[:, None] + dyr[None, :]
    cc = col[:, None] + dxr[None, :]
    ok = pm[:, None] & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return tl.load(
        x_ptr + (p[:, None] + (dyr * WP + dxr)[None, :]) * C + c[None, :],
        mask=ok,
        other=0.0,
    )


@triton.jit
def _pool_chunk(x_ptr, p, row, col, pm, H, W, WP, c, xc, g1, C: tl.constexpr):
    acc = tl.zeros([p.shape[0], c.shape[0]], dtype=tl.float32)
    for ty in tl.static_range(3):
        for tx in tl.static_range(3):
            ok = (
                pm
                & (row + ty - 1 >= 0)
                & (row + ty - 1 < H)
                & (col + tx - 1 >= 0)
                & (col + tx - 1 < W)
            )
            acc += tl.load(
                x_ptr + (p + (ty - 1) * WP + (tx - 1))[:, None] * C + c[None, :],
                mask=ok[:, None],
                other=0.0,
            ).to(tl.float32)
    nr = 3 - (row == 0).to(tl.int32) - (row == H - 1).to(tl.int32)
    ncl = 3 - (col == 0).to(tl.int32) - (col == W - 1).to(tl.int32)
    cnt = (nr * ncl).to(tl.float32)
    pool = _rbf(acc / cnt[:, None])
    return xc + g1[None, :] * (pool - xc)


@triton.jit
def _light_block(
    x_ptr,
    y_ptr,
    wdw_ptr,
    bdw_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    g_ptr,
    g2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    H,
    CA: tl.constexpr,
    CB: tl.constexpr,
    HA: tl.constexpr,
    HB: tl.constexpr,
    BM: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
    SGN: tl.constexpr,
):
    C: tl.constexpr = CA + CB
    pid = tl.program_id(0)
    p = P_LO + pid * BM + tl.arange(0, BM)
    rowp = (p - SL) // WP
    colp = (p - SL) % WP
    row = rowp - 1
    col = colp - 1
    pm = (p < P_HI) & (col >= 0) & (col < W)
    ca = tl.arange(0, CA)
    xa = tl.load(x_ptr + p[:, None] * C + ca[None, :], mask=pm[:, None], other=0.0).to(
        tl.float32
    )
    if CB > 0:
        cb = CA + tl.arange(0, CB)
        xb = tl.load(
            x_ptr + p[:, None] * C + cb[None, :], mask=pm[:, None], other=0.0
        ).to(tl.float32)
    else:
        cb = ca
        xb = xa
    if MIX == 0:
        ha = _dw_chunk(
            x_ptr, p, row, col, pm, H, W, WP, ca, wdw_ptr, bdw_ptr, C, KS, DIL
        )
        hb = ha
        if CB > 0:
            hb = _dw_chunk(
                x_ptr, p, row, col, pm, H, W, WP, cb, wdw_ptr, bdw_ptr, C, KS, DIL
            )
        ra = xa
        rb = xb
    elif MIX == 1:
        ha = _shift_chunk(x_ptr, p, row, col, pm, H, W, WP, ca, C, DIL, SGN)
        hb = ha
        if CB > 0:
            hb = _shift_chunk(x_ptr, p, row, col, pm, H, W, WP, cb, C, DIL, SGN)
        ra = xa
        rb = xb
    else:
        ra = _pool_chunk(
            x_ptr, p, row, col, pm, H, W, WP, ca, xa, tl.load(g_ptr + ca), C
        )
        ha = ra.to(tl.bfloat16)
        rb = ra
        hb = ha
        if CB > 0:
            rb = _pool_chunk(
                x_ptr, p, row, col, pm, H, W, WP, cb, xb, tl.load(g_ptr + cb), C
            )
            hb = rb.to(tl.bfloat16)
    (oa, ob) = _mlp(ha, hb, w1_ptr, b1_ptr, w2_ptr, b2_ptr, CA, CB, HA, HB, ACT)
    if MIX == 2:
        ga = tl.load(g2_ptr + ca)
    else:
        ga = tl.load(g_ptr + ca)
    ya = ra + ga[None, :] * oa
    tl.store(y_ptr + p[:, None] * C + ca[None, :], ya.to(tl.bfloat16), mask=pm[:, None])
    if CB > 0:
        if MIX == 2:
            gb = tl.load(g2_ptr + cb)
        else:
            gb = tl.load(g_ptr + cb)
        yb = rb + gb[None, :] * ob
        tl.store(
            y_ptr + p[:, None] * C + cb[None, :], yb.to(tl.bfloat16), mask=pm[:, None]
        )


@triton.jit
def _light_pconvglu(
    x_ptr,
    y_ptr,
    wp_ptr,
    bp_ptr,
    w1p_ptr,
    w1x_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    g_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    H,
    CA: tl.constexpr,
    CB: tl.constexpr,
    C4: tl.constexpr,
    C4P: tl.constexpr,
    HA: tl.constexpr,
    HB: tl.constexpr,
    BM: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
):
    C: tl.constexpr = CA + CB
    RC: tl.constexpr = HA + HB
    R2: tl.constexpr = 2 * RC
    pid = tl.program_id(0)
    p = P_LO + pid * BM + tl.arange(0, BM)
    rowp = (p - SL) // WP
    colp = (p - SL) % WP
    row = rowp - 1
    col = colp - 1
    pm = (p < P_HI) & (col >= 0) & (col < W)
    ca = tl.arange(0, CA)
    xa = tl.load(x_ptr + p[:, None] * C + ca[None, :], mask=pm[:, None], other=0.0)
    if CB > 0:
        cb = CA + tl.arange(0, CB)
        xb = tl.load(x_ptr + p[:, None] * C + cb[None, :], mask=pm[:, None], other=0.0)
    else:
        cb = ca
        xb = xa
    c4 = tl.arange(0, C4P)
    acc_p = tl.zeros([BM, C4P], dtype=tl.float32)
    for tap in tl.static_range(9):
        dy = (tap // 3 - 1) * DIL
        dx = (tap % 3 - 1) * DIL
        ok = pm & (row + dy >= 0) & (row + dy < H) & (col + dx >= 0) & (col + dx < W)
        a = tl.load(
            x_ptr + (p + dy * WP + dx)[:, None] * C + c4[None, :],
            mask=ok[:, None] & (c4 < C4)[None, :],
            other=0.0,
        )
        acc_p = tl.dot(
            a,
            tl.load(wp_ptr + tap * C4P * C4P + c4[:, None] * C4P + c4[None, :]),
            acc_p,
        )
    hp = (_rbf(acc_p) + tl.load(bp_ptr + c4)[None, :]).to(tl.bfloat16)
    na = tl.arange(0, HA)
    ta = tl.dot(hp, tl.load(w1p_ptr + c4[:, None] * R2 + na[None, :]))
    ta = tl.dot(xa, tl.load(w1x_ptr + ca[:, None] * R2 + na[None, :]), ta)
    tg = tl.dot(hp, tl.load(w1p_ptr + c4[:, None] * R2 + RC + na[None, :]))
    tg = tl.dot(xa, tl.load(w1x_ptr + ca[:, None] * R2 + RC + na[None, :]), tg)
    if CB > 0:
        ta = tl.dot(xb, tl.load(w1x_ptr + cb[:, None] * R2 + na[None, :]), ta)
        tg = tl.dot(xb, tl.load(w1x_ptr + cb[:, None] * R2 + RC + na[None, :]), tg)
    ua = (
        (_rbf(ta) + tl.load(b1_ptr + na)[None, :])
        * _act(_rbf(tg) + tl.load(b1_ptr + RC + na)[None, :], ACT)
    ).to(tl.bfloat16)
    oa = tl.dot(ua, tl.load(w2_ptr + na[:, None] * C + ca[None, :]))
    if CB > 0:
        ob = tl.dot(ua, tl.load(w2_ptr + na[:, None] * C + cb[None, :]))
    else:
        ob = oa
    if HB > 0:
        nb = HA + tl.arange(0, HB)
        tb = tl.dot(hp, tl.load(w1p_ptr + c4[:, None] * R2 + nb[None, :]))
        tb = tl.dot(xa, tl.load(w1x_ptr + ca[:, None] * R2 + nb[None, :]), tb)
        tgb = tl.dot(hp, tl.load(w1p_ptr + c4[:, None] * R2 + RC + nb[None, :]))
        tgb = tl.dot(xa, tl.load(w1x_ptr + ca[:, None] * R2 + RC + nb[None, :]), tgb)
        if CB > 0:
            tb = tl.dot(xb, tl.load(w1x_ptr + cb[:, None] * R2 + nb[None, :]), tb)
            tgb = tl.dot(
                xb, tl.load(w1x_ptr + cb[:, None] * R2 + RC + nb[None, :]), tgb
            )
        ub = (
            (_rbf(tb) + tl.load(b1_ptr + nb)[None, :])
            * _act(_rbf(tgb) + tl.load(b1_ptr + RC + nb)[None, :], ACT)
        ).to(tl.bfloat16)
        oa = tl.dot(ub, tl.load(w2_ptr + nb[:, None] * C + ca[None, :]), oa)
        if CB > 0:
            ob = tl.dot(ub, tl.load(w2_ptr + nb[:, None] * C + cb[None, :]), ob)
    ya = xa.to(tl.float32) + tl.load(g_ptr + ca)[None, :] * (
        _rbf(oa) + tl.load(b2_ptr + ca)[None, :]
    )
    tl.store(y_ptr + p[:, None] * C + ca[None, :], ya.to(tl.bfloat16), mask=pm[:, None])
    if CB > 0:
        yb = xb.to(tl.float32) + tl.load(g_ptr + cb)[None, :] * (
            _rbf(ob) + tl.load(b2_ptr + cb)[None, :]
        )
        tl.store(
            y_ptr + p[:, None] * C + cb[None, :], yb.to(tl.bfloat16), mask=pm[:, None]
        )


def _bf(t):
    return t.detach().float().to(torch.bfloat16).float()


def random_block(
    kind: str,
    C: int,
    r: float = 2.0,
    k: int = 3,
    dil: int = 1,
    device="cuda",
    gen: torch.Generator | None = None,
    pconv_ratio: float = 0.25,
) -> dict:
    g = gen or torch.Generator().manual_seed(0)
    RC = int(round(C * r))

    def u(shape, fan_in):
        b = 1.0 / math.sqrt(fan_in)
        return ((torch.rand(shape, generator=g) * 2 - 1) * b).to(device)

    blk = dict(
        kind=kind, C=C, RC=RC, k=k, dil=dil, gamma=torch.full((C,), 0.1, device=device)
    )
    if kind in ("dwmlp", "shiftmlp", "poolmlp"):
        blk.update(w1=u((RC, C), C), b1=u((RC,), C), w2=u((C, RC), RC), b2=u((C,), RC))
    if kind == "dwmlp":
        blk.update(dw_w=u((C, k, k), k * k), dw_b=u((C,), k * k))
    if kind == "poolmlp":
        blk["gamma2"] = torch.full((C,), 0.1, device=device)
    if kind == "pconvglu":
        C4 = int(round(C * pconv_ratio))
        blk.update(
            C4=C4,
            p_w=u((C4, C4, 3, 3), C4 * 9),
            p_b=u((C4,), C4 * 9),
            w1=u((2 * RC, C), C),
            b1=u((2 * RC,), C),
            w2=u((C, RC), RC),
            b2=u((C,), RC),
        )
    return blk


class LightBlock:
    def __init__(
        self, blk: dict, act: str = "gelu", cfg: dict | None = None, shift_sign: int = 1
    ):
        cfg = dict(cfg or {})
        self.kind = blk["kind"]
        (C, RC) = (blk["C"], blk["RC"])
        (self.C, self.RC) = (C, RC)
        (self.CA, self.CB) = tuple(cfg["csplit"]) if "csplit" in cfg else split2(C)
        assert self.CA + self.CB == C
        (self.HA, self.HB) = split2(RC)
        self.act = ACTS[act]
        self.dil = int(blk.get("dil", 1))
        self.ks = int(blk.get("k", 3))
        self.sgn = int(shift_sign)
        self.bm = cfg.get("BM", 64)
        self.warps = cfg.get("warps", 4)
        dev = blk["gamma"].device
        self.l3 = cfg3(self.kind, C, dev)
        self.l3.update(
            {
                k: cfg[k]
                for k in ("bm", "warps", "persist", "mult", "stages", "svec", "split")
                if k in cfg
            }
        )
        if self.kind != "dwmlp" or not self.l3.get("split"):
            self.l3.pop("split", None)
        self.v1 = bool(cfg.get("v1", False))
        self.g = blk["gamma"].detach().float().contiguous()
        self.g2 = (
            blk["gamma2"].detach().float().contiguous() if "gamma2" in blk else self.g
        )
        if self.kind == "pconvglu":
            C4 = blk["C4"]
            C4P = max(16, triton.next_power_of_2(C4))
            (self.C4, self.C4P) = (C4, C4P)
            wp = torch.zeros(9, C4P, C4P, device=dev)
            wp[:, :C4, :C4] = _bf(blk["p_w"]).permute(2, 3, 1, 0).reshape(9, C4, C4)
            self.wp = wp.to(torch.bfloat16).contiguous()
            bp = torch.zeros(C4P, device=dev)
            bp[:C4] = blk["p_b"].float()
            self.bp = bp
            w1 = _bf(blk["w1"])
            w1p = torch.zeros(C4P, 2 * RC, device=dev)
            w1p[:C4] = w1[:, :C4].t()
            self.w1p = w1p.to(torch.bfloat16).contiguous()
            w1x = w1.t().clone()
            w1x[:C4] = 0
            self.w1x = w1x.to(torch.bfloat16).contiguous()
            self.b1 = blk["b1"].detach().float().contiguous()
            self.w2 = _bf(blk["w2"]).t().contiguous().to(torch.bfloat16)
            self.b2 = blk["b2"].detach().float().contiguous()
            return
        self.w1 = _bf(blk["w1"]).t().contiguous().to(torch.bfloat16)
        self.b1 = blk["b1"].detach().float().contiguous()
        self.w2 = _bf(blk["w2"]).t().contiguous().to(torch.bfloat16)
        self.b2 = blk["b2"].detach().float().contiguous()
        if self.kind == "dwmlp":
            k = self.ks
            self.wdw = _bf(blk["dw_w"]).reshape(C, k * k).t().contiguous()
            self.bdw = blk["dw_b"].detach().float().contiguous()
        else:
            self.wdw = self.b1
            self.bdw = self.b1
        self.mix = {"dwmlp": MIX_DW, "shiftmlp": MIX_SHIFT, "poolmlp": MIX_POOL}[
            self.kind
        ]

    @property
    def needs_scratch(self) -> bool:
        return (
            not self.v1
            and self.kind != "pconvglu"
            and (self.l3.get("split") is not None)
        )

    @property
    def can_fuse_heads(self) -> bool:
        return not self.v1 and self.kind != "pconvglu"

    def __call__(
        self, x: PadBuf, y: PadBuf, h: PadBuf | None = None, heads: dict | None = None
    ):
        assert x.C == self.C and y.C == self.C
        grid = (triton.cdiv(x.p_hi - x.p_lo, self.bm),)
        if self.kind == "pconvglu":
            _light_pconvglu[grid](
                x.buf,
                y.buf,
                self.wp,
                self.bp,
                self.w1p,
                self.w1x,
                self.b1,
                self.w2,
                self.b2,
                self.g,
                x.p_lo,
                x.p_hi,
                x.SL,
                x.WP,
                x.W,
                x.H,
                CA=self.CA,
                CB=self.CB,
                C4=self.C4,
                C4P=self.C4P,
                HA=self.HA,
                HB=self.HB,
                BM=self.bm,
                DIL=self.dil,
                ACT=self.act,
                num_warps=self.warps,
            )
            return y
        if not self.v1:
            return launch_block3(self, x, y, h=h, heads=heads, **self.l3)
        assert heads is None
        _light_block[grid](
            x.buf,
            y.buf,
            self.wdw,
            self.bdw,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.g,
            self.g2,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            x.H,
            CA=self.CA,
            CB=self.CB,
            HA=self.HA,
            HB=self.HB,
            BM=self.bm,
            MIX=self.mix,
            KS=self.ks,
            DIL=self.dil,
            ACT=self.act,
            SGN=self.sgn,
            num_warps=self.warps,
        )
        return y


@triton.jit
def _la_reduce(
    k_ptr,
    v_ptr,
    kv_ptr,
    ks_ptr,
    NPIX,
    W,
    k_sc,
    k_sh,
    k_sw,
    v_sc,
    v_sh,
    v_sw,
    NT,
    C: tl.constexpr,
    CP: tl.constexpr,
    DH: tl.constexpr,
    BM: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    c = tl.arange(0, CP)
    cm = c < C
    acc = tl.zeros([CP, CP], tl.float32)
    ks = tl.zeros([CP], tl.float32)
    for tile in range(pid, NT, nprog):
        n = tile * BM + tl.arange(0, BM)
        nm = n < NPIX
        i = n // W
        j = n % W
        m = nm[:, None] & cm[None, :]
        kt = tl.load(
            k_ptr + (i * k_sh + j * k_sw)[:, None] + c[None, :] * k_sc,
            mask=m,
            other=0.0,
        )
        vt = tl.load(
            v_ptr + (i * v_sh + j * v_sw)[:, None] + c[None, :] * v_sc,
            mask=m,
            other=0.0,
        )
        acc = tl.dot(tl.trans(kt), vt, acc)
        ks += tl.sum(kt.to(tl.float32), axis=0)
    hd = (c[:, None] // DH == c[None, :] // DH) & cm[:, None] & cm[None, :]
    tl.atomic_add(kv_ptr + c[:, None] * CP + c[None, :], acc, mask=hd)
    tl.atomic_add(ks_ptr + c, ks, mask=cm)


def la_reduce(k, v, heads: int, bm: int = 128, mult: int = 2):
    (_, C, H, W) = k.shape
    dh = C // heads
    CP = max(16, triton.next_power_of_2(C))
    kv = torch.zeros(CP, CP, device=k.device, dtype=torch.float32)
    ks = torch.zeros(CP, device=k.device, dtype=torch.float32)
    npix = H * W
    nt = triton.cdiv(npix, bm)
    grid = (min(nt, num_sms(k.device) * mult),)
    _la_reduce[grid](
        k,
        v,
        kv,
        ks,
        npix,
        W,
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        nt,
        C=C,
        CP=CP,
        DH=dh,
        BM=bm,
        num_warps=4,
    )
    kvh = torch.stack(
        [kv[h * dh : (h + 1) * dh, h * dh : (h + 1) * dh] for h in range(heads)]
    )
    return (kvh, ks[:C].reshape(heads, dh))


class TorchBlock:
    needs_scratch = False
    can_fuse_heads = False

    def __init__(self, blk: dict, compile: bool = True):
        self.mod = blk["module"]
        self.kind = blk["kind"]
        self.C = blk["C"]
        self.heads = int(blk["heads"])
        self.eps = float(getattr(self.mod, "eps", 1e-06))
        c = (lambda f: torch.compile(f)) if compile else lambda f: f
        self.fa = c(self._qkv)
        self.fb = c(self._rest)

    def _qkv(self, x):
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            (q, k, v) = self.mod.qkv(x).chunk(3, dim=1)
            return (torch.relu(q), torch.relu(k), v)

    def _rest(self, x, q, kv, ks):
        (b, C, H, W) = q.shape
        hh = self.heads
        dh = C // hh
        with torch.autocast("cuda", enabled=False):
            qf = q.float().reshape(b, hh, dh, H * W)
            num = torch.einsum("bhdn,hde->bhen", qf, kv)
            den = torch.einsum("bhdn,hd->bhn", qf, ks)
            o = (num / (den.unsqueeze(2) + self.eps)).reshape(b, C, H, W).to(q.dtype)
        m = self.mod
        with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
            z = x + m.scale1 * m.proj(o)
            return z + m.scale2 * m.pw2(m.act(m.pw1(z)))

    def __call__(self, x: PadBuf, y: PadBuf, h=None, heads=None):
        assert heads is None
        xv = x.view
        (q, k, v) = self.fa(xv)
        (kv, ks) = la_reduce(k, v, self.heads)
        y.view.copy_(self.fb(xv, q, kv, ks))
        return y


class LightTrunk:
    def __init__(
        self,
        blocks: list[dict],
        act: str = "gelu",
        cfg: dict | None = None,
        shift_sign: int = 1,
    ):
        self.blocks = [
            TorchBlock(b)
            if b["kind"] == "linattn"
            else LightBlock(b, act=b.get("act", act), cfg=cfg, shift_sign=shift_sign)
            for b in blocks
        ]
        self.needs_scratch = any((b.needs_scratch for b in self.blocks))
        self.can_fuse_heads = bool(self.blocks) and self.blocks[-1].can_fuse_heads
        self._h = {}

    def __call__(
        self,
        x: PadBuf,
        a: PadBuf,
        b: PadBuf,
        h: PadBuf | None = None,
        heads: dict | None = None,
    ) -> PadBuf:
        if h is None and self.needs_scratch:
            k = (x.C, x.H, x.W, str(x.buf.device))
            if k not in self._h:
                self._h[k] = PadBuf(x.C, x.H, x.W, x.buf.dtype, x.buf.device)
            h = self._h[k]
        (cur, nxt) = (x, a)
        nb = len(self.blocks)
        for k, blk in enumerate(self.blocks):
            if heads is not None and k == nb - 1:
                blk(cur, nxt, h, heads=heads)
                return None
            blk(cur, nxt, h)
            cur = nxt
            nxt = b if nxt is a else a
        return cur


def random_trunk(
    kind: str,
    C: int,
    B: int,
    r: float = 2.0,
    k: int = 3,
    dil: str = "1,2",
    device="cuda",
    seed: int = 0,
    attn_idx: int | None = None,
) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    dils = [int(v) for v in str(dil).split(",")]
    return [
        random_block(kind, C, r=r, k=k, dil=dils[i % len(dils)], device=device, gen=g)
        for i in range(B)
    ]


def light_blocks_from_model(model):
    from .light_extract import body_dicts

    return body_dicts(model)


@triton.jit
def _block_tile(
    x_ptr,
    y_ptr,
    wdw_ptr,
    bdw_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    g_ptr,
    g2_ptr,
    p,
    SL,
    WP,
    W,
    H,
    P_HI,
    CA: tl.constexpr,
    CB: tl.constexpr,
    RC: tl.constexpr,
    HC: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
    SGN: tl.constexpr,
):
    C: tl.constexpr = CA + CB
    rowp = (p - SL) // WP
    colp = (p - SL) % WP
    row = rowp - 1
    col = colp - 1
    pm = (p < P_HI) & (col >= 0) & (col < W)
    ca = tl.arange(0, CA)
    xa = tl.load(x_ptr + p[:, None] * C + ca[None, :], mask=pm[:, None], other=0.0)
    if CB > 0:
        cb = CA + tl.arange(0, CB)
        xb = tl.load(x_ptr + p[:, None] * C + cb[None, :], mask=pm[:, None], other=0.0)
    if MIX == 0:
        ha = _dw_chunk(
            x_ptr, p, row, col, pm, H, W, WP, ca, wdw_ptr, bdw_ptr, C, KS, DIL
        )
        if CB > 0:
            hb = _dw_chunk(
                x_ptr, p, row, col, pm, H, W, WP, cb, wdw_ptr, bdw_ptr, C, KS, DIL
            )
    elif MIX == 1:
        ha = _shift_chunk(x_ptr, p, row, col, pm, H, W, WP, ca, C, DIL, SGN)
        if CB > 0:
            hb = _shift_chunk(x_ptr, p, row, col, pm, H, W, WP, cb, C, DIL, SGN)
    else:
        za = _pool_chunk(
            x_ptr,
            p,
            row,
            col,
            pm,
            H,
            W,
            WP,
            ca,
            xa.to(tl.float32),
            tl.load(g_ptr + ca),
            C,
        )
        ha = za.to(tl.bfloat16)
        if CB > 0:
            zb = _pool_chunk(
                x_ptr,
                p,
                row,
                col,
                pm,
                H,
                W,
                WP,
                cb,
                xb.to(tl.float32),
                tl.load(g_ptr + cb),
                C,
            )
            hb = zb.to(tl.bfloat16)
    oa = tl.zeros([p.shape[0], CA], dtype=tl.float32)
    if CB > 0:
        ob = tl.zeros([p.shape[0], CB], dtype=tl.float32)
    for h0 in tl.static_range(0, RC, HC):
        nh = h0 + tl.arange(0, HC)
        t = tl.dot(ha, tl.load(w1_ptr + ca[:, None] * RC + nh[None, :]))
        if CB > 0:
            t = tl.dot(hb, tl.load(w1_ptr + cb[:, None] * RC + nh[None, :]), t)
        u = _act(_rbf(t) + tl.load(b1_ptr + nh)[None, :], ACT).to(tl.bfloat16)
        oa = tl.dot(u, tl.load(w2_ptr + nh[:, None] * C + ca[None, :]), oa)
        if CB > 0:
            ob = tl.dot(u, tl.load(w2_ptr + nh[:, None] * C + cb[None, :]), ob)
    if MIX == 2:
        ga = tl.load(g2_ptr + ca)
        ra = za
    else:
        ga = tl.load(g_ptr + ca)
        ra = xa.to(tl.float32)
    ya = ra + ga[None, :] * (_rbf(oa) + tl.load(b2_ptr + ca)[None, :])
    tl.store(y_ptr + p[:, None] * C + ca[None, :], ya.to(tl.bfloat16), mask=pm[:, None])
    if CB > 0:
        if MIX == 2:
            gb = tl.load(g2_ptr + cb)
            rb = zb
        else:
            gb = tl.load(g_ptr + cb)
            rb = xb.to(tl.float32)
        yb = rb + gb[None, :] * (_rbf(ob) + tl.load(b2_ptr + cb)[None, :])
        tl.store(
            y_ptr + p[:, None] * C + cb[None, :], yb.to(tl.bfloat16), mask=pm[:, None]
        )


@triton.jit
def _light_block2(
    x_ptr,
    y_ptr,
    wdw_ptr,
    bdw_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    g_ptr,
    g2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    H,
    NTILES,
    CA: tl.constexpr,
    CB: tl.constexpr,
    RC: tl.constexpr,
    HC: tl.constexpr,
    BM: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
    SGN: tl.constexpr,
    PERSIST: tl.constexpr,
    STAGES: tl.constexpr,
    SVEC: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    if PERSIST:
        nprog = tl.num_programs(0)
        for tile in tl.range(pid, NTILES, nprog, num_stages=STAGES):
            p = P_LO + tile * BM + tl.arange(0, BM)
            _block_tile(
                x_ptr,
                y_ptr,
                wdw_ptr,
                bdw_ptr,
                w1_ptr,
                b1_ptr,
                w2_ptr,
                b2_ptr,
                g_ptr,
                g2_ptr,
                p,
                SL,
                WP,
                W,
                H,
                P_HI,
                CA,
                CB,
                RC,
                HC,
                MIX,
                KS,
                DIL,
                ACT,
                SGN,
            )
    else:
        p = P_LO + pid * BM + tl.arange(0, BM)
        _block_tile(
            x_ptr,
            y_ptr,
            wdw_ptr,
            bdw_ptr,
            w1_ptr,
            b1_ptr,
            w2_ptr,
            b2_ptr,
            g_ptr,
            g2_ptr,
            p,
            SL,
            WP,
            W,
            H,
            P_HI,
            CA,
            CB,
            RC,
            HC,
            MIX,
            KS,
            DIL,
            ACT,
            SGN,
        )


def launch_block2(
    lb, x: PadBuf, y: PadBuf, hc=64, bm=128, warps=4, persist=False, mult=2, stages=2
):
    ntiles = triton.cdiv(x.p_hi - x.p_lo, bm)
    if persist:
        grid = (min(ntiles, num_sms(x.buf.device) * mult),)
    else:
        grid = (ntiles,)
    hc = min(hc, lb.RC)
    while lb.RC % hc:
        hc //= 2
    _light_block2[grid](
        x.buf,
        y.buf,
        lb.wdw,
        lb.bdw,
        lb.w1,
        lb.b1,
        lb.w2,
        lb.b2,
        lb.g,
        lb.g2,
        x.p_lo,
        x.p_hi,
        x.SL,
        x.WP,
        x.W,
        x.H,
        ntiles,
        CA=lb.CA,
        CB=lb.CB,
        RC=lb.RC,
        HC=hc,
        BM=bm,
        MIX=lb.mix,
        KS=lb.ks,
        DIL=lb.dil,
        ACT=lb.act,
        SGN=lb.sgn,
        PERSIST=persist,
        STAGES=stages,
        num_warps=warps,
    )
    return y


@triton.jit
def _mix3(
    xbase,
    p,
    row,
    col,
    pm,
    H,
    W,
    WP,
    c,
    wdw_ptr,
    bdw_ptr,
    xc,
    g1,
    C: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    SGN: tl.constexpr,
    x_ptr=None,
    SVEC: tl.constexpr = 0,
):
    if MIX == 0:
        acc = tl.zeros([p.shape[0], c.shape[0]], dtype=tl.float32)
        for ty in tl.static_range(KS):
            for tx in tl.static_range(KS):
                dy = (ty - KS // 2) * DIL
                dx = (tx - KS // 2) * DIL
                ok = (
                    pm
                    & (row + dy >= 0)
                    & (row + dy < H)
                    & (col + dx >= 0)
                    & (col + dx < W)
                )
                v = tl.load(xbase + (dy * WP + dx) * C, mask=ok[:, None], other=0.0).to(
                    tl.float32
                )
                acc += v * tl.load(wdw_ptr + (ty * KS + tx) * C + c)[None, :]
        h = (_rbf(acc) + tl.load(bdw_ptr + c)[None, :]).to(tl.bfloat16)
        r = xc.to(tl.float32)
    elif MIX == 1:
        G: tl.constexpr = C // 8
        gi = c // G
        dyr = tl.where(gi == 0, -DIL, tl.where(gi == 1, DIL, 0)) * SGN
        dxr = tl.where(gi == 2, -DIL, tl.where(gi == 3, DIL, 0)) * SGN
        rr = row[:, None] + dyr[None, :]
        cc = col[:, None] + dxr[None, :]
        ok = pm[:, None] & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        if SVEC > 0:
            o2 = p[:, None] * C + c[None, :] + ((dyr * WP + dxr) * C)[None, :]
            o2 = tl.max_contiguous(tl.multiple_of(o2, [1, SVEC]), [1, SVEC])
            h = tl.load(x_ptr + o2, mask=ok, other=0.0)
        else:
            h = tl.load(xbase + ((dyr * WP + dxr) * C)[None, :], mask=ok, other=0.0)
        r = xc.to(tl.float32)
    elif MIX == 3:
        h = tl.load(wdw_ptr + p[:, None] * C + c[None, :], mask=pm[:, None], other=0.0)
        r = xc.to(tl.float32)
    else:
        acc = tl.zeros([p.shape[0], c.shape[0]], dtype=tl.float32)
        for ty in tl.static_range(3):
            for tx in tl.static_range(3):
                ok = (
                    pm
                    & (row + ty - 1 >= 0)
                    & (row + ty - 1 < H)
                    & (col + tx - 1 >= 0)
                    & (col + tx - 1 < W)
                )
                acc += tl.load(
                    xbase + ((ty - 1) * WP + (tx - 1)) * C, mask=ok[:, None], other=0.0
                ).to(tl.float32)
        nr = 3 - (row == 0).to(tl.int32) - (row == H - 1).to(tl.int32)
        ncl = 3 - (col == 0).to(tl.int32) - (col == W - 1).to(tl.int32)
        pool = _rbf(acc / (nr * ncl).to(tl.float32)[:, None])
        xf = xc.to(tl.float32)
        r = xf + g1[None, :] * (pool - xf)
        h = r.to(tl.bfloat16)
    return (h, r)


@triton.jit
def _heads_epi(
    ya,
    yb,
    ca,
    cb,
    row,
    col,
    pm,
    hd_wc,
    hd_bc,
    hd_ws,
    hd_bs,
    hd_wf,
    hd_bf,
    hx_ptr,
    jit_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
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
    CA: tl.constexpr,
    CB: tl.constexpr,
    NS: tl.constexpr,
    NF: tl.constexpr,
):
    BM: tl.constexpr = ya.shape[0]
    q = tl.arange(0, 16)
    ns = tl.arange(0, NS)
    stat = tl.dot(
        ya, tl.load(hd_ws + ca[:, None] * NS + ns[None, :]), input_precision="tf32"
    )
    xa = ya.to(tl.bfloat16)
    wra = hd_wc + ca[:, None] * 48 + q[None, :]
    c0 = tl.dot(xa, tl.load(wra))
    c1 = tl.dot(xa, tl.load(wra + 16))
    c2 = tl.dot(xa, tl.load(wra + 32))
    if CB > 0:
        stat = tl.dot(
            yb,
            tl.load(hd_ws + cb[:, None] * NS + ns[None, :]),
            stat,
            input_precision="tf32",
        )
        xb = yb.to(tl.bfloat16)
        wrb = hd_wc + cb[:, None] * 48 + q[None, :]
        c0 = tl.dot(xb, tl.load(wrb), c0)
        c1 = tl.dot(xb, tl.load(wrb + 16), c1)
        c2 = tl.dot(xb, tl.load(wrb + 32), c2)
    stat = stat + tl.load(hd_bs + ns)[None, :]
    c0 = _rbf(_rbf(c0) + tl.load(hd_bc + q)[None, :])
    c1 = _rbf(_rbf(c1) + tl.load(hd_bc + 16 + q)[None, :])
    c2 = _rbf(_rbf(c2) + tl.load(hd_bc + 32 + q)[None, :])
    i = row
    j = col
    jy = tl.load(jit_ptr)
    dyq = q // 4
    a_ = (dyq.to(tl.float32) - centre) * inv_s + jy
    fa = tl.floor(a_)
    tt = a_ - fa
    k0c = C3._cc2h(tt + 1.0)
    k1c = C3._cc1h(tt)
    k2c = C3._cc1h(1.0 - tt)
    k3c = C3._cc2h(1.0 - tt + 1.0)
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
    if NF > 0:
        nf = tl.arange(0, NF)
        fv = tl.dot(xa, tl.load(hd_wf + ca[:, None] * NF + nf[None, :]))
        if CB > 0:
            fv = tl.dot(xb, tl.load(hd_wf + cb[:, None] * NF + nf[None, :]), fv)
        fv = _rbf(fv) + tl.load(hd_bf + nf)[None, :]
        tl.store(
            feat_ptr + (iw * f_sh + jc * f_sw)[:, None] + nf[None, :],
            fv.to(feat_ptr.dtype.element_ty),
            mask=pm[:, None],
        )
    cbp = cand_ptr + (iw * c_sh + jc * c_sw)[:, None] + q[None, :]
    tl.store(cbp, c0 + sk0, mask=pm[:, None])
    tl.store(cbp + 16, c1 + sk1, mask=pm[:, None])
    tl.store(cbp + 32, c2 + sk2, mask=pm[:, None])


@triton.jit
def _tile3(
    x_ptr,
    y_ptr,
    wdw_ptr,
    bdw_ptr,
    g_ptr,
    g2_ptr,
    p,
    SL,
    WP,
    W,
    H,
    P_HI,
    w1aa,
    w1ba,
    w1ab,
    w1bb,
    w2aa,
    w2ab,
    w2ba,
    w2bb,
    b1a,
    b1b,
    b2a,
    b2b,
    hd_wc,
    hd_bc,
    hd_ws,
    hd_bs,
    hd_wf,
    hd_bf,
    hx_ptr,
    jit_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
    c_sh,
    c_sw,
    s_sh,
    s_sw,
    f_sh,
    f_sw,
    centre,
    inv_s,
    CA: tl.constexpr,
    CB: tl.constexpr,
    HA: tl.constexpr,
    HB: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
    SGN: tl.constexpr,
    SVEC: tl.constexpr,
    HEADS: tl.constexpr,
    NS: tl.constexpr,
    NF: tl.constexpr,
):
    C: tl.constexpr = CA + CB
    rowp = (p - SL) // WP
    col = (p - SL) % WP - 1
    row = rowp - 1
    pm = (p < P_HI) & (col >= 0) & (col < W)
    ca = tl.arange(0, CA)
    cb = ca
    if CB > 0:
        cb = CA + tl.arange(0, CB)
    xba = x_ptr + p[:, None] * C + ca[None, :]
    xa = tl.load(xba, mask=pm[:, None], other=0.0)
    (ha, ra) = _mix3(
        xba,
        p,
        row,
        col,
        pm,
        H,
        W,
        WP,
        ca,
        wdw_ptr,
        bdw_ptr,
        xa,
        tl.load(g_ptr + ca),
        C,
        MIX,
        KS,
        DIL,
        SGN,
        x_ptr,
        SVEC,
    )
    hb = ha
    rb = ra
    if CB > 0:
        xbb = x_ptr + p[:, None] * C + cb[None, :]
        xb = tl.load(xbb, mask=pm[:, None], other=0.0)
        (hb, rb) = _mix3(
            xbb,
            p,
            row,
            col,
            pm,
            H,
            W,
            WP,
            cb,
            wdw_ptr,
            bdw_ptr,
            xb,
            tl.load(g_ptr + cb),
            C,
            MIX,
            KS,
            DIL,
            SGN,
            x_ptr,
            SVEC,
        )
    t = tl.dot(ha, w1aa)
    if CB > 0:
        t = tl.dot(hb, w1ba, t)
    u = _act(_rbf(t) + b1a[None, :], ACT).to(tl.bfloat16)
    oa = tl.dot(u, w2aa)
    ob = oa
    if CB > 0:
        ob = tl.dot(u, w2ab)
    if HB > 0:
        t2 = tl.dot(ha, w1ab)
        if CB > 0:
            t2 = tl.dot(hb, w1bb, t2)
        u2 = _act(_rbf(t2) + b1b[None, :], ACT).to(tl.bfloat16)
        oa = tl.dot(u2, w2ba, oa)
        if CB > 0:
            ob = tl.dot(u2, w2bb, ob)
    if MIX == 2:
        ga = tl.load(g2_ptr + ca)
    else:
        ga = tl.load(g_ptr + ca)
    ya = ra + ga[None, :] * (_rbf(oa) + b2a[None, :])
    yb = ya
    if CB > 0:
        if MIX == 2:
            gb = tl.load(g2_ptr + cb)
        else:
            gb = tl.load(g_ptr + cb)
        yb = rb + gb[None, :] * (_rbf(ob) + b2b[None, :])
    if HEADS:
        _heads_epi(
            ya,
            yb,
            ca,
            cb,
            row,
            col,
            pm,
            hd_wc,
            hd_bc,
            hd_ws,
            hd_bs,
            hd_wf,
            hd_bf,
            hx_ptr,
            jit_ptr,
            cand_ptr,
            stat_ptr,
            feat_ptr,
            H,
            W,
            c_sh,
            c_sw,
            s_sh,
            s_sw,
            f_sh,
            f_sw,
            centre,
            inv_s,
            CA,
            CB,
            NS,
            NF,
        )
    else:
        tl.store(
            y_ptr + p[:, None] * C + ca[None, :], ya.to(tl.bfloat16), mask=pm[:, None]
        )
        if CB > 0:
            tl.store(
                y_ptr + p[:, None] * C + cb[None, :],
                yb.to(tl.bfloat16),
                mask=pm[:, None],
            )


@triton.jit
def _light_block3(
    x_ptr,
    y_ptr,
    wdw_ptr,
    bdw_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    g_ptr,
    g2_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    H,
    NT,
    hd_wc,
    hd_bc,
    hd_ws,
    hd_bs,
    hd_wf,
    hd_bf,
    hx_ptr,
    jit_ptr,
    cand_ptr,
    stat_ptr,
    feat_ptr,
    c_sh,
    c_sw,
    s_sh,
    s_sw,
    f_sh,
    f_sw,
    centre,
    inv_s,
    CA: tl.constexpr,
    CB: tl.constexpr,
    HA: tl.constexpr,
    HB: tl.constexpr,
    BM: tl.constexpr,
    MIX: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
    ACT: tl.constexpr,
    SGN: tl.constexpr,
    PERSIST: tl.constexpr,
    STAGES: tl.constexpr,
    SVEC: tl.constexpr = 0,
    HEADS: tl.constexpr = False,
    NS: tl.constexpr = 32,
    NF: tl.constexpr = 32,
):
    C: tl.constexpr = CA + CB
    RC: tl.constexpr = HA + HB
    ka = tl.arange(0, CA)
    na = tl.arange(0, HA)
    w1aa = tl.load(w1_ptr + ka[:, None] * RC + na[None, :])
    w2aa = tl.load(w2_ptr + na[:, None] * C + ka[None, :])
    b1a = tl.load(b1_ptr + na)
    b2a = tl.load(b2_ptr + ka)
    w1ba = w1aa
    w1ab = w1aa
    w1bb = w1aa
    w2ab = w2aa
    w2ba = w2aa
    w2bb = w2aa
    b1b = b1a
    b2b = b2a
    if CB > 0:
        kb = CA + tl.arange(0, CB)
        w1ba = tl.load(w1_ptr + kb[:, None] * RC + na[None, :])
        w2ab = tl.load(w2_ptr + na[:, None] * C + kb[None, :])
        b2b = tl.load(b2_ptr + kb)
    if HB > 0:
        nb = HA + tl.arange(0, HB)
        w1ab = tl.load(w1_ptr + ka[:, None] * RC + nb[None, :])
        w2ba = tl.load(w2_ptr + nb[:, None] * C + ka[None, :])
        b1b = tl.load(b1_ptr + nb)
        if CB > 0:
            kb2 = CA + tl.arange(0, CB)
            w1bb = tl.load(w1_ptr + kb2[:, None] * RC + nb[None, :])
            w2bb = tl.load(w2_ptr + nb[:, None] * C + kb2[None, :])
    pid = tl.program_id(0)
    if PERSIST:
        nprog = tl.num_programs(0)
        for tile in tl.range(pid, NT, nprog, num_stages=STAGES):
            p = P_LO + tile * BM + tl.arange(0, BM)
            _tile3(
                x_ptr,
                y_ptr,
                wdw_ptr,
                bdw_ptr,
                g_ptr,
                g2_ptr,
                p,
                SL,
                WP,
                W,
                H,
                P_HI,
                w1aa,
                w1ba,
                w1ab,
                w1bb,
                w2aa,
                w2ab,
                w2ba,
                w2bb,
                b1a,
                b1b,
                b2a,
                b2b,
                hd_wc,
                hd_bc,
                hd_ws,
                hd_bs,
                hd_wf,
                hd_bf,
                hx_ptr,
                jit_ptr,
                cand_ptr,
                stat_ptr,
                feat_ptr,
                c_sh,
                c_sw,
                s_sh,
                s_sw,
                f_sh,
                f_sw,
                centre,
                inv_s,
                CA,
                CB,
                HA,
                HB,
                MIX,
                KS,
                DIL,
                ACT,
                SGN,
                SVEC,
                HEADS,
                NS,
                NF,
            )
    else:
        p = P_LO + pid * BM + tl.arange(0, BM)
        _tile3(
            x_ptr,
            y_ptr,
            wdw_ptr,
            bdw_ptr,
            g_ptr,
            g2_ptr,
            p,
            SL,
            WP,
            W,
            H,
            P_HI,
            w1aa,
            w1ba,
            w1ab,
            w1bb,
            w2aa,
            w2ab,
            w2ba,
            w2bb,
            b1a,
            b1b,
            b2a,
            b2b,
            hd_wc,
            hd_bc,
            hd_ws,
            hd_bs,
            hd_wf,
            hd_bf,
            hx_ptr,
            jit_ptr,
            cand_ptr,
            stat_ptr,
            feat_ptr,
            c_sh,
            c_sw,
            s_sh,
            s_sw,
            f_sh,
            f_sw,
            centre,
            inv_s,
            CA,
            CB,
            HA,
            HB,
            MIX,
            KS,
            DIL,
            ACT,
            SGN,
            SVEC,
            HEADS,
            NS,
            NF,
        )


def launch_block3(
    lb,
    x: PadBuf,
    y: PadBuf,
    bm=64,
    warps=4,
    persist=True,
    mult=2,
    stages=1,
    act=None,
    svec=0,
    split=None,
    h=None,
    heads=None,
):
    (mix, wdw) = (lb.mix, lb.wdw)
    if svec == -1:
        svec = svec_for(lb.C)
    if split is not None:
        assert lb.mix == MIX_DW and h is not None
        launch_dw_stencil(lb, x, h, **split)
        (mix, wdw) = (3, h.buf)
    ntiles = triton.cdiv(x.p_hi - x.p_lo, bm)
    if persist:
        grid = (min(ntiles, num_sms(x.buf.device) * mult),)
    else:
        grid = (ntiles,)
    if heads is not None:
        Hd = heads["Hd"]
        (cand, stat, feat) = (heads["cand"], heads["stat"], heads["feat"])
        nf = Hd["nf"]
        hargs = (
            Hd["wc48"],
            Hd["bc"],
            Hd["ws"],
            Hd["bs"],
            Hd["wf"],
            Hd["bf"],
            heads["hx"],
            heads["jf"],
            cand,
            stat,
            feat if nf else stat,
            cand.stride(2),
            cand.stride(3),
            stat.stride(2),
            stat.stride(3),
            feat.stride(2) if nf else 0,
            feat.stride(3) if nf else 0,
            1.5,
            0.25,
        )
        hkw = dict(HEADS=True, NS=Hd["ns"], NF=nf)
    else:
        d = lb.b1
        hargs = (d,) * 11 + (0,) * 6 + (0.0, 0.0)
        hkw = dict(HEADS=False)
    _light_block3[grid](
        x.buf,
        y.buf,
        wdw,
        lb.bdw,
        lb.w1,
        lb.b1,
        lb.w2,
        lb.b2,
        lb.g,
        lb.g2,
        x.p_lo,
        x.p_hi,
        x.SL,
        x.WP,
        x.W,
        x.H,
        ntiles,
        *hargs,
        CA=lb.CA,
        CB=lb.CB,
        HA=lb.HA,
        HB=lb.HB,
        BM=bm,
        MIX=mix,
        KS=lb.ks,
        DIL=lb.dil,
        ACT=lb.act if act is None else ACTS[act],
        SGN=lb.sgn,
        PERSIST=persist,
        STAGES=stages,
        SVEC=svec if mix == MIX_SHIFT else 0,
        num_warps=warps,
        **hkw,
    )
    return y


@triton.jit
def _dw_stencil(
    x_ptr,
    h_ptr,
    wdw_ptr,
    bdw_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    H,
    C: tl.constexpr,
    CC: tl.constexpr,
    BM: tl.constexpr,
    KS: tl.constexpr,
    DIL: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.program_id(1) * CC + tl.arange(0, CC)
    p = P_LO + pid * BM + tl.arange(0, BM)
    rowp = (p - SL) // WP
    col = (p - SL) % WP - 1
    row = rowp - 1
    pm = (p < P_HI) & (col >= 0) & (col < W)
    xb = x_ptr + p[:, None] * C + c[None, :]
    acc = tl.zeros([BM, CC], dtype=tl.float32)
    for ty in tl.static_range(KS):
        for tx in tl.static_range(KS):
            dy = (ty - KS // 2) * DIL
            dx = (tx - KS // 2) * DIL
            ok = (
                pm & (row + dy >= 0) & (row + dy < H) & (col + dx >= 0) & (col + dx < W)
            )
            v = tl.load(xb + (dy * WP + dx) * C, mask=ok[:, None], other=0.0).to(
                tl.float32
            )
            acc += v * tl.load(wdw_ptr + (ty * KS + tx) * C + c)[None, :]
    h = (_rbf(acc) + tl.load(bdw_ptr + c)[None, :]).to(tl.bfloat16)
    tl.store(h_ptr + p[:, None] * C + c[None, :], h, mask=pm[:, None])


def launch_dw_stencil(lb, x: PadBuf, h: PadBuf, bm=128, cc=32, warps=4):
    cc = min(cc, lb.C & -lb.C)
    grid = (triton.cdiv(x.p_hi - x.p_lo, bm), lb.C // cc)
    _dw_stencil[grid](
        x.buf,
        h.buf,
        lb.wdw,
        lb.bdw,
        x.p_lo,
        x.p_hi,
        x.SL,
        x.WP,
        x.W,
        x.H,
        C=lb.C,
        CC=cc,
        BM=bm,
        KS=lb.ks,
        DIL=lb.dil,
        num_warps=warps,
    )
    return h


def svec_for(C: int) -> int:
    g = C // 8
    return min(8, g & -g)


@triton.jit
def _rbl(x, LOWP: tl.constexpr):
    if LOWP:
        return x.to(tl.bfloat16).to(tl.float32)
    return x


@triton.jit
def _k_j1x1(
    jit_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    w3_ptr,
    b3_ptr,
    gidx_ptr,
    wd_ptr,
    NO,
    IN,
    OUT,
    HID: tl.constexpr,
    BLK: tl.constexpr,
    LOWP: tl.constexpr,
):
    two_pi = 6.2831854820251465
    k4 = tl.arange(0, 4)
    jv = tl.load(jit_ptr + k4 % 2)
    ph = _rbl(tl.where(k4 < 2, jv, libdevice.sin(jv * two_pi)), LOWP)
    hr = tl.arange(0, HID)
    w1 = tl.load(w1_ptr + hr[:, None] * 4 + k4[None, :])
    h1 = _rbl(tl.sum(w1 * ph[None, :], axis=1), LOWP) + tl.load(b1_ptr + hr)
    h1 = _rbl(tl.maximum(h1, 0.0), LOWP)
    hc = tl.arange(0, HID)
    w2 = tl.load(w2_ptr + hr[:, None] * HID + hc[None, :])
    h2 = _rbl(tl.sum(w2 * tl.reshape(h1, (1, HID)), axis=1), LOWP) + tl.load(
        b2_ptr + hr
    )
    h2 = _rbl(tl.maximum(h2, 0.0), LOWP)
    o = tl.program_id(0) * BLK + tl.arange(0, BLK)
    om = o < NO
    w3 = tl.load(w3_ptr + o[:, None] * HID + hc[None, :], mask=om[:, None], other=0.0)
    y = tl.load(b3_ptr + o, mask=om, other=0.0) + _rbl(
        tl.sum(w3 * tl.reshape(h2, (1, HID)), axis=1), LOWP
    )
    oc = o // IN
    ic = o % IN
    g = tl.load(gidx_ptr + ic, mask=om, other=0)
    tl.store(wd_ptr + g * OUT + oc, y.to(wd_ptr.dtype.element_ty), mask=om)


def launch_j1x1(jf, J, blk=256):
    no = J["IN"] * J["OUT"]
    _k_j1x1[triton.cdiv(no, blk),](
        jf,
        J["w1"],
        J["b1"],
        J["w2"],
        J["b2"],
        J["w3"],
        J["b3"],
        J["gidx"],
        J["wd"],
        no,
        J["IN"],
        J["OUT"],
        HID=J["hid"],
        BLK=blk,
        LOWP=J["lowp"],
        num_warps=4,
    )


@triton.jit
def _stem1x1(
    x_ptr,
    y_ptr,
    wb_ptr,
    wd_ptr,
    b_ptr,
    P_LO,
    P_HI,
    SL,
    WP,
    W,
    NT,
    KA: tl.constexpr,
    KB: tl.constexpr,
    CA: tl.constexpr,
    CB: tl.constexpr,
    BM: tl.constexpr,
    PERSIST: tl.constexpr,
    STAGES: tl.constexpr,
):
    CIN: tl.constexpr = KA + KB
    C: tl.constexpr = CA + CB
    ka = tl.arange(0, KA)
    ca = tl.arange(0, CA)
    wb_aa = tl.load(wb_ptr + ka[:, None] * C + ca[None, :])
    wd_aa = tl.load(wd_ptr + ka[:, None] * C + ca[None, :])
    b_a = tl.load(b_ptr + ca)
    wb_ba = wb_aa
    wd_ba = wd_aa
    wb_ab = wb_aa
    wd_ab = wd_aa
    wb_bb = wb_aa
    wd_bb = wd_aa
    b_b = b_a
    kb = ka
    cb = ca
    if KB > 0:
        kb = KA + tl.arange(0, KB)
        wb_ba = tl.load(wb_ptr + kb[:, None] * C + ca[None, :])
        wd_ba = tl.load(wd_ptr + kb[:, None] * C + ca[None, :])
    if CB > 0:
        cb = CA + tl.arange(0, CB)
        wb_ab = tl.load(wb_ptr + ka[:, None] * C + cb[None, :])
        wd_ab = tl.load(wd_ptr + ka[:, None] * C + cb[None, :])
        b_b = tl.load(b_ptr + cb)
        if KB > 0:
            wb_bb = tl.load(wb_ptr + kb[:, None] * C + cb[None, :])
            wd_bb = tl.load(wd_ptr + kb[:, None] * C + cb[None, :])
    pid = tl.program_id(0)
    if PERSIST:
        lo = pid
        step = tl.num_programs(0)
        hi = NT
    else:
        lo = pid
        step = 1
        hi = pid + 1
    for tile in tl.range(lo, hi, step, num_stages=STAGES):
        p = P_LO + tile * BM + tl.arange(0, BM)
        col = (p - SL) % WP - 1
        pm = (p < P_HI) & (col >= 0) & (col < W)
        xa = tl.load(
            x_ptr + p[:, None] * CIN + ka[None, :], mask=pm[:, None], other=0.0
        )
        ta = tl.dot(xa, wb_aa)
        da = tl.dot(xa, wd_aa)
        if KB > 0:
            xb = tl.load(
                x_ptr + p[:, None] * CIN + kb[None, :], mask=pm[:, None], other=0.0
            )
            ta = tl.dot(xb, wb_ba, ta)
            da = tl.dot(xb, wd_ba, da)
        va = _rbf(ta) + b_a[None, :] + _rbf(da)
        tl.store(
            y_ptr + p[:, None] * C + ca[None, :], va.to(tl.bfloat16), mask=pm[:, None]
        )
        if CB > 0:
            tb = tl.dot(xa, wb_ab)
            db = tl.dot(xa, wd_ab)
            if KB > 0:
                tb = tl.dot(xb, wb_bb, tb)
                db = tl.dot(xb, wd_bb, db)
            vb = _rbf(tb) + b_b[None, :] + _rbf(db)
            tl.store(
                y_ptr + p[:, None] * C + cb[None, :],
                vb.to(tl.bfloat16),
                mask=pm[:, None],
            )


class Stem1x1:
    def __init__(self, conv, wd, cfg=None):
        cfg = dict(cfg or {})
        (cout, cin) = conv.weight.shape[:2]
        assert conv.kernel_size == (1, 1) and wd.shape == (cin, cout)
        (self.cin, self.cout) = (cin, cout)
        (self.KA, self.KB) = split2(cin)
        (self.CA, self.CB) = split2(cout)
        self.wb = (
            conv.weight.detach()
            .float()
            .reshape(cout, cin)
            .t()
            .contiguous()
            .to(torch.bfloat16)
        )
        self.b = conv.bias.detach().float().contiguous()
        self.wd = wd
        self.bm = cfg.get("bm", 64)
        self.warps = cfg.get("warps", 4)
        self.persist = cfg.get("persist", True)
        self.mult = cfg.get("mult", 3)
        self.stages = cfg.get("stages", 3)

    def __call__(self, x: PadBuf, y: PadBuf, gb=None):
        assert x.C == self.cin and y.C == self.cout
        nt = triton.cdiv(x.p_hi - x.p_lo, self.bm)
        grid = (min(nt, num_sms(x.buf.device) * self.mult),) if self.persist else (nt,)
        _stem1x1[grid](
            x.buf,
            y.buf,
            self.wb,
            self.wd,
            self.b,
            x.p_lo,
            x.p_hi,
            x.SL,
            x.WP,
            x.W,
            nt,
            KA=self.KA,
            KB=self.KB,
            CA=self.CA,
            CB=self.CB,
            BM=self.bm,
            PERSIST=self.persist,
            STAGES=self.stages,
            num_warps=self.warps,
        )
        return y
