from __future__ import annotations
import torch


def _act_name(mod) -> str:
    if isinstance(mod, torch.nn.ReLU):
        return "relu"
    if isinstance(mod, torch.nn.GELU):
        if mod.approximate == "tanh":
            return "gelu"
        return "gelu_erf"
    raise ValueError(f"unsupported light activation {mod!r}")


def _pw(conv, cin, cout):
    if not (
        isinstance(conv, torch.nn.Conv2d)
        and conv.kernel_size == (1, 1)
        and (conv.stride == (1, 1))
        and (conv.groups == 1)
        and (conv.bias is not None)
        and (conv.in_channels == cin)
        and (conv.out_channels == cout)
    ):
        raise ValueError(f"expected a 1x1 conv {cin}->{cout} with bias, got {conv!r}")
    w = conv.weight.detach().float().reshape(cout, cin)
    return (w, conv.bias.detach().float())


def _vec(p, C):
    v = p.detach().float().reshape(-1)
    if v.numel() != C:
        raise ValueError(f"layer scale of {v.numel()} channels, expected {C}")
    return v


def block_dict(blk) -> dict:
    name = type(blk).__name__
    if name == "DWMLPBlock":
        dw = blk.dwconv
        C = dw.in_channels
        k = dw.kernel_size[0]
        d = dw.dilation[0]
        if not (
            dw.groups == C
            and dw.out_channels == C
            and (dw.kernel_size == (k, k))
            and (dw.dilation == (d, d))
            and (dw.padding == (d * (k - 1) // 2,) * 2)
            and (dw.stride == (1, 1))
            and (dw.bias is not None)
            and (dw.padding_mode == "zeros")
        ):
            raise ValueError(f"unsupported depthwise conv {dw!r}")
        RC = blk.pw1.out_channels
        (w1, b1) = _pw(blk.pw1, C, RC)
        (w2, b2) = _pw(blk.pw2, RC, C)
        return dict(
            kind="dwmlp",
            C=C,
            RC=RC,
            k=k,
            dil=d,
            gamma=_vec(blk.scale, C),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            dw_w=dw.weight.detach().float().reshape(C, k, k),
            dw_b=dw.bias.detach().float(),
            act=_act_name(blk.act),
        )
    if name == "ShiftMLPBlock":
        C = blk.pw1.in_channels
        if C % 8:
            raise ValueError(
                f"shiftmlp needs C % 8 == 0 (got {C}): the kernel's groups are c // (C / 8)"
            )
        RC = blk.pw1.out_channels
        (w1, b1) = _pw(blk.pw1, C, RC)
        (w2, b2) = _pw(blk.pw2, RC, C)
        return dict(
            kind="shiftmlp",
            C=C,
            RC=RC,
            k=1,
            dil=int(blk.shift),
            gamma=_vec(blk.scale, C),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            act=_act_name(blk.act),
        )
    if name == "PoolMLPBlock":
        pool = blk.pool
        if not (
            isinstance(pool, torch.nn.AvgPool2d)
            and pool.kernel_size in (3, (3, 3))
            and (pool.stride in (1, (1, 1)))
            and (pool.padding in (1, (1, 1)))
            and (not pool.count_include_pad)
            and (not pool.ceil_mode)
            and (pool.divisor_override is None)
        ):
            raise ValueError(f"unsupported pool {pool!r}")
        C = blk.pw1.in_channels
        RC = blk.pw1.out_channels
        (w1, b1) = _pw(blk.pw1, C, RC)
        (w2, b2) = _pw(blk.pw2, RC, C)
        return dict(
            kind="poolmlp",
            C=C,
            RC=RC,
            k=3,
            dil=1,
            gamma=_vec(blk.scale1, C),
            gamma2=_vec(blk.scale2, C),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            act=_act_name(blk.act),
        )
    if name == "PConvGLUBlock":
        pc = blk.pconv
        cp = int(blk.cp)
        d = pc.dilation[0]
        if not (
            pc.in_channels == cp
            and pc.out_channels == cp
            and (pc.kernel_size == (3, 3))
            and (pc.dilation == (d, d))
            and (pc.padding == (d, d))
            and (pc.stride == (1, 1))
            and (pc.groups == 1)
            and (pc.bias is not None)
            and (pc.padding_mode == "zeros")
        ):
            raise ValueError(f"unsupported partial conv {pc!r}")
        C = blk.pw1.in_channels
        RC = blk.pw2.in_channels
        (w1, b1) = _pw(blk.pw1, C, 2 * RC)
        (w2, b2) = _pw(blk.pw2, RC, C)
        return dict(
            kind="pconvglu",
            C=C,
            RC=RC,
            k=3,
            dil=d,
            C4=cp,
            p_w=pc.weight.detach().float(),
            p_b=pc.bias.detach().float(),
            gamma=_vec(blk.scale, C),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            act=_act_name(blk.act),
        )
    if name == "LinAttnBlock":
        C = blk.qkv.in_channels
        (wq, bq) = _pw(blk.qkv, C, 3 * C)
        (wo, bo) = _pw(blk.proj, C, C)
        RC = blk.pw1.out_channels
        (w1, b1) = _pw(blk.pw1, C, RC)
        (w2, b2) = _pw(blk.pw2, RC, C)
        return dict(
            kind="linattn",
            C=C,
            RC=RC,
            heads=int(blk.heads),
            attn_dim=int(blk.attn_dim),
            wqkv=wq,
            bqkv=bq,
            wo=wo,
            bo=bo,
            gamma=_vec(blk.scale1, C),
            gamma2=_vec(blk.scale2, C),
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            act=_act_name(blk.act),
            module=blk,
        )
    raise ValueError(f"not a light block: {name}")


def body_dicts(model) -> list[dict] | None:
    kind = getattr(model, "body_kind", "plain")
    if kind not in ("dwmlp", "shiftmlp", "pconvglu", "poolmlp", "linattn"):
        return None
    return [block_dict(b) for b in model.body]
