from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "make_body",
    "fuse_model",
    "attach_rep_scope",
    "BODY_KINDS",
    "INNER_KINDS",
    "RepConv3",
    "Conv3Block",
    "Rep3Block",
    "PConvBlock",
    "MBV2Block",
    "StarBlock",
    "UNetBody",
    "unet_stage_blocks",
    "REP_PRESETS",
    "LIGHT_KINDS",
    "LIGHT_EPS",
    "light_body",
    "DWMLPBlock",
    "ShiftMLPBlock",
    "PConvGLUBlock",
    "PoolMLPBlock",
    "LinAttnBlock",
    "spatial_shift",
    "shift_bounds",
    "linear_attention",
    "parse_dilations",
    "parse_attn_idx",
]


def _act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"act must be gelu|relu|silu, got {name!r}")


def _round8(x: float) -> int:
    return max(8, int(round(x / 8.0)) * 8)


_SOBEL_X = ((1.0, 0.0, -1.0), (2.0, 0.0, -2.0), (1.0, 0.0, -1.0))
_SOBEL_Y = ((1.0, 2.0, 1.0), (0.0, 0.0, 0.0), (-1.0, -2.0, -1.0))
_LAPLACE = ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
REP_PRESETS = {
    "repvgg": ("c3", "c1", "id"),
    "ecb": ("c3", "c1c3", "sobx", "soby", "lap", "id"),
    "acb": ("c3", "c13", "c31", "c1", "id"),
    "full": ("c3", "c1", "id", "c13", "c31", "c1c3", "sobx", "soby", "lap"),
}
_BRANCHES = ("c3", "c1", "id", "c13", "c31", "c1c3", "sobx", "soby", "lap")


def _parse_branches(spec: str) -> tuple[str, ...]:
    if spec in REP_PRESETS:
        return REP_PRESETS[spec]
    parts = tuple((p for p in spec.replace(",", "+").split("+") if p))
    bad = [p for p in parts if p not in _BRANCHES]
    if bad or not parts:
        raise ValueError(
            f"rep branches must be a preset {sorted(REP_PRESETS)} or a '+'-list of {_BRANCHES}; got {spec!r}"
        )
    return parts


class RepConv3(nn.Module):
    def __init__(
        self, cin: int, cout: int, branches: str = "ecb", c1c3_mult: float = 2.0
    ) -> None:
        super().__init__()
        (self.cin, self.cout) = (cin, cout)
        self.branches = _parse_branches(branches)
        if "id" in self.branches and cin != cout:
            self.branches = tuple((b for b in self.branches if b != "id"))
        b = set(self.branches)
        self.c3 = nn.Conv2d(cin, cout, 3, padding=1) if "c3" in b else None
        self.c1 = nn.Conv2d(cin, cout, 1) if "c1" in b else None
        self.c13 = nn.Conv2d(cin, cout, (1, 3), padding=(0, 1)) if "c13" in b else None
        self.c31 = nn.Conv2d(cin, cout, (3, 1), padding=(1, 0)) if "c31" in b else None
        self.id_scale = nn.Parameter(torch.ones(cout)) if "id" in b else None
        if "c1c3" in b:
            mid = _round8(cin * c1c3_mult)
            self.c1c3_a = nn.Conv2d(cin, mid, 1)
            self.c1c3_b = nn.Conv2d(mid, cout, 3)
        else:
            self.c1c3_a = self.c1c3_b = None
        self.edge = nn.ModuleDict()
        self.edge_scale = nn.ParameterDict()
        self.edge_bias = nn.ParameterDict()
        for name, mask in (("sobx", _SOBEL_X), ("soby", _SOBEL_Y), ("lap", _LAPLACE)):
            if name in b:
                self.edge[name] = nn.Conv2d(cin, cout, 1, bias=False)
                self.edge_scale[name] = nn.Parameter(torch.randn(cout) * 0.001)
                self.edge_bias[name] = nn.Parameter(torch.randn(cout) * 0.001)
                self.register_buffer(
                    f"mask_{name}",
                    torch.tensor(mask).view(1, 1, 3, 3),
                    persistent=False,
                )
        self.fused: nn.Conv2d | None = None

    force_branches = False
    _scope = False
    _kcache = None

    def kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        ref = next((p for p in self.parameters()))
        w = torch.zeros(self.cout, self.cin, 3, 3, device=ref.device, dtype=ref.dtype)
        b = torch.zeros(self.cout, device=ref.device, dtype=ref.dtype)
        if self.c3 is not None:
            w = w + self.c3.weight
            b = b + self.c3.bias
        if self.c1 is not None:
            w = w + F.pad(self.c1.weight, (1, 1, 1, 1))
            b = b + self.c1.bias
        if self.c13 is not None:
            w = w + F.pad(self.c13.weight, (0, 0, 1, 1))
            b = b + self.c13.bias
        if self.c31 is not None:
            w = w + F.pad(self.c31.weight, (1, 1, 0, 0))
            b = b + self.c31.bias
        if self.id_scale is not None:
            eye = torch.zeros_like(w)
            idx = torch.arange(self.cout, device=w.device)
            eye[idx, idx, 1, 1] = 1.0
            w = w + eye * self.id_scale.view(-1, 1, 1, 1)
        if self.c1c3_a is not None:
            wa = self.c1c3_a.weight.flatten(1)
            wb = self.c1c3_b.weight
            w = w + torch.einsum("omhw,mi->oihw", wb, wa)
            b = b + self.c1c3_b.bias + torch.einsum("omhw,m->o", wb, self.c1c3_a.bias)
        for name, conv in self.edge.items():
            mask = getattr(self, f"mask_{name}").to(w.dtype)
            sc = self.edge_scale[name].view(-1, 1, 1, 1)
            w1 = conv.weight
            w = w + sc * w1 * mask
            b = b + self.edge_bias[name]
        return (w, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.fused(x)
        if self.force_branches:
            return self.forward_branches(x)
        (w, b) = self.cached_kernel()
        return F.conv2d(x, w, b, padding=1)

    def cached_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        grad = torch.is_grad_enabled()
        if grad and (not self._scope):
            return self.kernel()
        key = (
            grad,
            torch.is_inference_mode_enabled(),
            tuple(((p._version, p.data_ptr(), p.dtype) for p in self.parameters())),
        )
        if self._kcache is not None and self._kcache[0] == key:
            return self._kcache[1]
        wb = self.kernel()
        self._kcache = (key, wb)
        return wb

    @staticmethod
    def _pad_with_bias(y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        y = F.pad(y, (1, 1, 1, 1))
        bv = bias.view(1, -1, 1, 1).to(y.dtype)
        y[:, :, 0:1, :] = bv
        y[:, :, -1:, :] = bv
        y[:, :, :, 0:1] = bv
        y[:, :, :, -1:] = bv
        return y

    def forward_branches(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        if self.c3 is not None:
            out = out + self.c3(x)
        if self.c1 is not None:
            out = out + self.c1(x)
        if self.c13 is not None:
            out = out + self.c13(x)
        if self.c31 is not None:
            out = out + self.c31(x)
        if self.id_scale is not None:
            out = out + x * self.id_scale.view(1, -1, 1, 1)
        if self.c1c3_a is not None:
            y = self._pad_with_bias(self.c1c3_a(x), self.c1c3_a.bias)
            out = out + self.c1c3_b(y)
        for name, conv in self.edge.items():
            y = F.pad(conv(x), (1, 1, 1, 1))
            mask = getattr(self, f"mask_{name}").to(y.dtype)
            kern = self.edge_scale[name].view(-1, 1, 1, 1) * mask
            out = out + F.conv2d(y, kern, self.edge_bias[name], groups=self.cout)
        return out

    @torch.no_grad()
    def fuse(self) -> None:
        self._kcache = None
        if self.fused is not None:
            return
        (w, b) = self.kernel()
        conv = nn.Conv2d(self.cin, self.cout, 3, padding=1).to(w.device, w.dtype)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        for name in ("c3", "c1", "c13", "c31", "c1c3_a", "c1c3_b"):
            setattr(self, name, None)
        self.id_scale = None
        self.edge = nn.ModuleDict()
        self.edge_scale = nn.ParameterDict()
        self.edge_bias = nn.ParameterDict()
        self.fused = conv

    def extra_repr(self) -> str:
        return f"{self.cin}->{self.cout}, branches={'+'.join(self.branches)}, fused={self.fused is not None}"


class _Residual(nn.Module):
    def __init__(self, dim: int, layer_scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))


class Conv3Block(_Residual):
    def __init__(
        self, dim: int, layer_scale: float = 0.1, act: str = "gelu", **_
    ) -> None:
        super().__init__(dim, layer_scale)
        self.spatial = nn.Conv2d(dim, dim, 3, padding=1)
        self.mix = nn.Sequential(_act(act), nn.Conv2d(dim, dim, 3, padding=1))

    def forward(self, x):
        return x + self.scale * self.mix(self.spatial(x))


class Rep3Block(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        rep_branches: str = "ecb",
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        self.spatial = RepConv3(dim, dim, rep_branches)
        self.act = _act(act)
        self.conv2 = RepConv3(dim, dim, rep_branches)

    def forward(self, x):
        return x + self.scale * self.conv2(self.act(self.spatial(x)))


class PConvBlock(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        pconv_ratio: int = 4,
        expand: float = 2.0,
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        self.cp = max(1, dim // int(pconv_ratio))
        hidden = _round8(dim * expand)
        self.pconv = nn.Conv2d(self.cp, self.cp, 3, padding=1, bias=False)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = _act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        y = torch.cat((self.pconv(x[:, : self.cp]), x[:, self.cp :]), dim=1)
        return x + self.scale * self.pw2(self.act(self.pw1(y)))


class MBV2Block(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        hidden = _round8(dim * expand)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act1 = _act(act)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act2 = _act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return x + self.scale * self.pw2(self.act2(self.dwconv(self.act1(self.pw1(x)))))


class StarBlock(_Residual):
    def __init__(
        self, dim: int, layer_scale: float = 0.1, act: str = "gelu", **_
    ) -> None:
        super().__init__(dim, layer_scale)
        self.spatial = nn.Conv2d(dim, 2 * dim, 3, padding=1)
        self.act = _act(act)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        (a, b) = self.spatial(x).chunk(2, dim=1)
        return x + self.scale * self.proj(self.act(a) * b)


INNER = {
    "conv3": Conv3Block,
    "rep3": Rep3Block,
    "pconv": PConvBlock,
    "mbv2": MBV2Block,
    "star": StarBlock,
}
INNER_KINDS = tuple(INNER)


def _stack(inner: str, dim: int, n: int, **kw) -> nn.Sequential:
    return nn.Sequential(*[INNER[inner](dim, **kw) for _ in range(n)])


def unet_stage_blocks(blocks: int, levels: int, spec: str = "") -> list[int]:
    n = 2 * levels + 1
    if spec:
        out = [int(v) for v in spec.replace("-", ",").split(",") if v != ""]
        if len(out) != n or min(out) < 0:
            raise ValueError(
                f"unet_blocks {spec!r} needs {n} counts >= 0 for levels={levels}"
            )
        return out
    mid = max(1, int(blocks) - 2 * levels)
    return [1] * levels + [mid] + [1] * levels


class UNetBody(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: int = 8,
        levels: int = 1,
        width_mult: float = 2.0,
        inner: str = "conv3",
        unet_blocks: str = "",
        down: str = "conv2",
        up: str = "shuffle",
        **kw,
    ) -> None:
        super().__init__()
        if levels not in (1, 2, 3):
            raise ValueError("unet levels must be 1..3")
        if inner not in INNER:
            raise ValueError(f"unet inner must be one of {INNER_KINDS}")
        (self.levels, self.inner) = (int(levels), inner)
        (self.down_kind, self.up_kind) = (down, up)
        self.widths = [dim] + [
            _round8(dim * width_mult ** (i + 1)) for i in range(levels)
        ]
        self.counts = unet_stage_blocks(blocks, levels, unet_blocks)
        L = levels
        self.enc = nn.ModuleList(
            [_stack(inner, self.widths[i], self.counts[i], **kw) for i in range(L)]
        )
        self.mid = _stack(inner, self.widths[L], self.counts[L], **kw)
        self.dec = nn.ModuleList(
            [
                _stack(inner, self.widths[i], self.counts[2 * L - i], **kw)
                for i in range(L)
            ]
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(L):
            (a, b) = (self.widths[i], self.widths[i + 1])
            if down == "conv2":
                self.downs.append(nn.Conv2d(a, b, 2, stride=2))
            elif down == "conv3":
                self.downs.append(nn.Conv2d(a, b, 3, stride=2, padding=1))
            else:
                raise ValueError("unet down must be conv2|conv3")
            if up == "shuffle":
                self.ups.append(
                    nn.Sequential(nn.Conv2d(b, 4 * a, 1), nn.PixelShuffle(2))
                )
            elif up == "tconv":
                self.ups.append(nn.ConvTranspose2d(b, a, 2, stride=2))
            else:
                raise ValueError("unet up must be shuffle|tconv")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (skips, sizes) = ([], [])
        for i in range(self.levels):
            x = self.enc[i](x)
            skips.append(x)
            (h, w) = x.shape[-2:]
            sizes.append((h, w))
            if h % 2 or w % 2:
                x = F.pad(x, (0, w % 2, 0, h % 2))
            x = self.downs[i](x)
        x = self.mid(x)
        for i in reversed(range(self.levels)):
            (h, w) = sizes[i]
            x = self.ups[i](x)[..., :h, :w] + skips[i]
            x = self.dec[i](x)
        return x

    def extra_repr(self) -> str:
        return f"levels={self.levels}, widths={self.widths}, blocks={self.counts}, inner={self.inner}, down={self.down_kind}, up={self.up_kind}"


LIGHT_KINDS = ("dwmlp", "shiftmlp", "pconvglu", "poolmlp", "linattn")
LIGHT_EPS = 1e-06


def _light_act(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "relu":
        return nn.ReLU()
    raise ValueError(
        f"light bodies take act gelu (tanh approximation) | relu, got {name!r}"
    )


def _int_list(spec, what: str) -> tuple[int, ...]:
    if isinstance(spec, int):
        return (int(spec),)
    if isinstance(spec, str):
        try:
            return tuple((int(p) for p in spec.replace(" ", "").split(",") if p != ""))
        except ValueError:
            raise ValueError(
                f"{what} must be comma-separated ints like '1,2', got {spec!r}"
            ) from None
    return tuple((int(v) for v in spec))


def parse_dilations(spec) -> tuple[int, ...]:
    out = _int_list(spec, "body_dilations")
    if not out or min(out) < 1:
        raise ValueError(
            f"body_dilations needs at least one dilation, all >= 1; got {spec!r}"
        )
    return out


def parse_attn_idx(spec, blocks: int) -> tuple[int, ...]:
    out = _int_list(spec, "body_attn_idx") if spec not in (None, "") else ()
    if not out:
        out = (int(blocks) // 2,)
    if len(set(out)) != len(out) or min(out) < 0 or max(out) >= int(blocks):
        raise ValueError(
            f"body_attn_idx {spec!r} must be distinct indices in [0, {int(blocks)})"
        )
    return tuple(sorted(out))


def _hidden(dim: int, expand: float) -> int:
    h = int(round(float(expand) * dim))
    if h < 1:
        raise ValueError(f"body_expand {expand} gives an empty MLP")
    return h


def _partial(dim: int, ratio) -> int:
    r = float(ratio)
    if r >= 1.0:
        cp = dim // int(r)
    elif r > 0.0:
        cp = int(round(dim * r))
    else:
        cp = 0
    if not 1 <= cp <= dim:
        raise ValueError(f"body_pconv_ratio {ratio} gives {cp} of {dim} channels")
    return cp


def shift_bounds(dim: int) -> tuple[int, int, int, int, int]:
    return tuple((k * int(dim) // 8 for k in range(5)))


def spatial_shift(x: torch.Tensor, d: int) -> torch.Tensor:
    d = int(d)
    (b0, b1, b2, b3, b4) = shift_bounds(x.shape[1])
    (h, w) = x.shape[-2:]
    xp = F.pad(x[:, :b4], (d, d, d, d))
    return torch.cat(
        (
            xp[:, b0:b1, 0:h, d : d + w],
            xp[:, b1:b2, 2 * d : 2 * d + h, d : d + w],
            xp[:, b2:b3, d : d + h, 0:w],
            xp[:, b3:b4, d : d + h, 2 * d : 2 * d + w],
            x[:, b4:],
        ),
        dim=1,
    )


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    eps: float = LIGHT_EPS,
) -> torch.Tensor:
    (b, c, h, w) = q.shape
    dh = c // heads
    dt = q.dtype
    ct = torch.promote_types(dt, torch.float32)
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf = q.to(ct).reshape(b, heads, dh, h * w)
        kf = k.to(ct).reshape(b, heads, dh, h * w)
        vf = v.to(ct).reshape(b, heads, dh, h * w)
        kv = torch.einsum("bhdn,bhen->bhde", kf, vf)
        ksum = kf.sum(dim=-1)
        num = torch.einsum("bhdn,bhde->bhen", qf, kv)
        den = torch.einsum("bhdn,bhd->bhn", qf, ksum)
        out = num / (den.unsqueeze(2) + eps)
    return out.reshape(b, c, h, w).to(dt)


class DWMLPBlock(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        dw_kernel: int = 3,
        dilation: int = 1,
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        (k, d) = (int(dw_kernel), int(dilation))
        if k < 1 or k % 2 == 0:
            raise ValueError(f"body_dw_kernel must be odd >= 1, got {k}")
        hidden = _hidden(dim, expand)
        self.dwconv = nn.Conv2d(
            dim, dim, k, padding=d * (k - 1) // 2, dilation=d, groups=dim
        )
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = _light_act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return x + self.scale * self.pw2(self.act(self.pw1(self.dwconv(x))))


class ShiftMLPBlock(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        dilation: int = 1,
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        self.shift = int(dilation)
        hidden = _hidden(dim, expand)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = _light_act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        return x + self.scale * self.pw2(
            self.act(self.pw1(spatial_shift(x, self.shift)))
        )

    def extra_repr(self) -> str:
        return f"shift={self.shift}, bounds={shift_bounds(self.pw1.in_channels)}"


class PConvGLUBlock(_Residual):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        pconv_ratio=4,
        dilation: int = 1,
        **_,
    ) -> None:
        super().__init__(dim, layer_scale)
        self.cp = _partial(dim, pconv_ratio)
        d = int(dilation)
        hidden = _hidden(dim, expand)
        self.pconv = nn.Conv2d(self.cp, self.cp, 3, padding=d, dilation=d)
        self.pw1 = nn.Conv2d(dim, 2 * hidden, 1)
        self.act = _light_act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        h = torch.cat((self.pconv(x[:, : self.cp]), x[:, self.cp :]), dim=1)
        (a, g) = self.pw1(h).chunk(2, dim=1)
        return x + self.scale * self.pw2(a * self.act(g))


class PoolMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        **_,
    ) -> None:
        super().__init__()
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
        self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        hidden = _hidden(dim, expand)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = _light_act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        z = x + self.scale1 * (self.pool(x) - x)
        return z + self.scale2 * self.pw2(self.act(self.pw1(z)))


class LinAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale: float = 0.1,
        act: str = "gelu",
        expand: float = 2.0,
        attn_dim: int = 16,
        **_,
    ) -> None:
        super().__init__()
        a = int(attn_dim)
        if a < 1 or dim % a:
            raise ValueError(f"body_attn_dim {a} must divide dim {dim}")
        (self.heads, self.attn_dim) = (dim // a, a)
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        hidden = _hidden(dim, expand)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = _light_act(act)
        self.pw2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        (q, k, v) = self.qkv(x).chunk(3, dim=1)
        o = linear_attention(F.relu(q), F.relu(k), v, self.heads)
        z = x + self.scale1 * self.proj(o)
        return z + self.scale2 * self.pw2(self.act(self.pw1(z)))

    def extra_repr(self) -> str:
        return f"heads={self.heads}, attn_dim={self.attn_dim}"


LIGHT = {
    "dwmlp": DWMLPBlock,
    "shiftmlp": ShiftMLPBlock,
    "pconvglu": PConvGLUBlock,
    "poolmlp": PoolMLPBlock,
}


def light_body(
    kind: str,
    dim: int,
    blocks: int,
    *,
    layer_scale: float = 0.1,
    act: str = "gelu",
    expand: float = 2.0,
    pconv_ratio=4,
    dw_kernel: int = 3,
    dilations="1,2",
    attn_idx="",
    attn_dim: int = 16,
) -> nn.Sequential:
    if kind not in LIGHT_KINDS:
        raise ValueError(f"light kind must be one of {LIGHT_KINDS}")
    dil = parse_dilations(dilations)
    attn = (
        set(parse_attn_idx(attn_idx, blocks))
        if kind == "linattn" and int(blocks) > 0
        else set()
    )
    base = DWMLPBlock if kind == "linattn" else LIGHT[kind]
    out = []
    for i in range(int(blocks)):
        if i in attn:
            out.append(
                LinAttnBlock(
                    dim,
                    layer_scale=layer_scale,
                    act=act,
                    expand=expand,
                    attn_dim=attn_dim,
                )
            )
        else:
            out.append(
                base(
                    dim,
                    layer_scale=layer_scale,
                    act=act,
                    expand=expand,
                    pconv_ratio=pconv_ratio,
                    dw_kernel=dw_kernel,
                    dilation=dil[i % len(dil)],
                )
            )
    return nn.Sequential(*out)


BODY_KINDS = ("conv3", "rep3", "pconv", "mbv2", "star", "unet") + LIGHT_KINDS


def make_body(
    kind: str,
    dim: int,
    blocks: int,
    *,
    layer_scale: float = 0.1,
    act: str = "gelu",
    expand: float = 2.0,
    pconv_ratio: int = 4,
    rep_branches: str = "ecb",
    inner: str = "conv3",
    unet_levels: int = 1,
    width_mult: float = 2.0,
    unet_blocks: str = "",
    down: str = "conv2",
    up: str = "shuffle",
    dw_kernel: int = 3,
    dilations="1,2",
    attn_idx="",
    attn_dim: int = 16,
) -> nn.Module:
    if kind != "linattn" and (attn_idx not in (None, "") or int(attn_dim) != 16):
        raise ValueError(
            "body_attn_idx / body_attn_dim are read only by body='linattn'"
        )
    if kind in LIGHT_KINDS:
        return light_body(
            kind,
            dim,
            blocks,
            layer_scale=layer_scale,
            act=act,
            expand=expand,
            pconv_ratio=pconv_ratio,
            dw_kernel=dw_kernel,
            dilations=dilations,
            attn_idx=attn_idx,
            attn_dim=attn_dim,
        )
    kw = dict(
        layer_scale=layer_scale,
        act=act,
        expand=expand,
        pconv_ratio=pconv_ratio,
        rep_branches=rep_branches,
    )
    if kind == "unet":
        return UNetBody(
            dim,
            blocks,
            levels=unet_levels,
            width_mult=width_mult,
            inner=inner,
            unet_blocks=unet_blocks,
            down=down,
            up=up,
            **kw,
        )
    if kind not in INNER:
        raise ValueError(f"body must be one of {BODY_KINDS}, got {kind!r}")
    return _stack(kind, dim, blocks, **kw)


def _rep_scope_enter(module: nn.Module, *_):
    for m in module.modules():
        if isinstance(m, RepConv3):
            (m._scope, m._kcache) = (True, None)


def _rep_scope_exit(module: nn.Module, *_):
    for m in module.modules():
        if isinstance(m, RepConv3):
            (m._scope, m._kcache) = (False, None)


def attach_rep_scope(model: nn.Module) -> bool:
    if not any((isinstance(m, RepConv3) for m in model.modules())):
        return False
    model.register_forward_pre_hook(_rep_scope_enter)
    model.register_forward_hook(_rep_scope_exit)
    return True


@torch.no_grad()
def fuse_model(model: nn.Module) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, RepConv3) and m.fused is None:
            m.fuse()
            n += 1
    return n


def _selftest(sizes=((66, 66), (270, 480)), dims=(32,), device="cpu") -> None:
    torch.manual_seed(0)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for dim in dims:
        for spec in ("repvgg", "ecb", "acb", "full"):
            rc = RepConv3(dim, dim, spec).double().to(device)
            for p in rc.parameters():
                p.data.normal_(0, 0.2)
            x = torch.randn(2, dim, 13, 17, dtype=torch.float64, device=device)
            a = rc.forward_branches(x)
            b = rc(x)
            rc.fuse()
            c = rc(x)
            e1 = (a - b).abs().max().item()
            e2 = (a - c).abs().max().item()
            print(
                f"  rep {spec:<7} branch-vs-kernel {e1:.2e}  branch-vs-fused {e2:.2e}"
            )
            assert e1 < 1e-09 and e2 < 1e-09, spec

        class _Loop(nn.Module):
            def __init__(self, body):
                super().__init__()
                self.body = body

            def forward(self, x, steps=5):
                for _ in range(steps):
                    x = self.body(x)
                return x

        torch.manual_seed(1)
        loop = _Loop(make_body("rep3", dim, 2)).double().to(device)
        x0 = torch.randn(1, dim, 12, 14, dtype=torch.float64, device=device)
        grads = []
        for scoped in (False, True):
            if scoped:
                assert attach_rep_scope(loop)
            loop.zero_grad(set_to_none=True)
            loop(x0).square().mean().backward()
            grads.append([p.grad.clone() for p in loop.parameters()])
            assert all(
                (m._kcache is None for m in loop.modules() if isinstance(m, RepConv3))
            )
        gerr = max(((a - b).abs().max().item() for (a, b) in zip(*grads)))
        print(f"  rep3 scoped-cache vs per-call kernel: grad max|diff| {gerr:.2e}")
        assert gerr < 1e-10
        configs = [
            ("conv3", {}),
            ("rep3", {}),
            ("rep3", dict(rep_branches="repvgg")),
            ("pconv", {}),
            ("pconv", dict(pconv_ratio=2)),
            ("mbv2", {}),
            ("star", {}),
            ("unet", {}),
            ("unet", dict(inner="rep3")),
            ("unet", dict(inner="pconv")),
            ("unet", dict(unet_levels=2)),
            ("unet", dict(up="tconv", down="conv3")),
            ("unet", dict(unet_blocks="0,4,1")),
        ]
        for kind, kw in configs:
            body = make_body(kind, dim, 4, **kw).to(device)
            for h, w in sizes:
                x = torch.randn(1, dim, h, w, device=device, requires_grad=True)
                y = body(x)
                assert y.shape == x.shape, (kind, kw, y.shape)
                y.square().mean().backward()
                g = sum(
                    (
                        p.grad.abs().sum()
                        for p in body.parameters()
                        if p.grad is not None
                    )
                )
                ok = (
                    bool(torch.isfinite(y).all())
                    and bool(torch.isfinite(g))
                    and bool(torch.isfinite(x.grad).all())
                )
                n_nograd = sum(
                    (
                        1
                        for p in body.parameters()
                        if p.grad is None or not bool((p.grad != 0).any())
                    )
                )
                print(
                    f"  {kind:<6} {str(kw):<40} {h}x{w} finite={ok} params={sum((p.numel() for p in body.parameters()))} no-grad-params={n_nograd}"
                )
                assert ok and n_nograd == 0, (kind, kw)
                body.zero_grad(set_to_none=True)
            if kind == "rep3" or kw.get("inner") == "rep3":
                body.eval()
                for m in body.modules():
                    if isinstance(m, RepConv3):
                        m.force_branches = True
                        for n_, p in m.named_parameters():
                            if "edge" in n_:
                                p.data.normal_(0, 0.05)
                x = torch.randn(1, dim, *sizes[-1], device=device)
                with torch.no_grad():
                    ref = body(x)
                    fuse_model(body)
                    out = body(x)
                err = (ref - out).abs().max().item()
                rel = err / ref.abs().max().item()
                print(
                    f"  {kind} {kw} branches-vs-fused fp32 max|diff| {err:.2e} (rel {rel:.1e})"
                )
                assert err < 0.0001
    print("fast_blocks selftest OK")


if __name__ == "__main__":
    _selftest()
