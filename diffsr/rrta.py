from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .samples import parse_depth_spec as _parse_depth_spec
from .samples import splat_gather as _splat_gather
from .samples import registered_sample_positions as _registered_positions

__all__ = [
    "RRTANet",
    "Stem",
    "create_rrta",
    "warp",
    "dilate_motion",
    "sample_phase",
    "registered_upsample",
    "jitter_image_offset",
    "depth_edges",
    "Block",
    "RRTA_SIZES",
]


def warp(x: torch.Tensor, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    (batch, _, height, width) = x.shape
    gdtype = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    rows = torch.arange(height, device=x.device, dtype=gdtype)
    cols = torch.arange(width, device=x.device, dtype=gdtype)
    (grid_y, grid_x) = torch.meshgrid(rows, cols, indexing="ij")
    src_y = grid_y[None] + motion[:, 0].to(gdtype)
    src_x = grid_x[None] + motion[:, 1].to(gdtype)
    norm_y = 2.0 * src_y / max(height - 1, 1) - 1.0
    norm_x = 2.0 * src_x / max(width - 1, 1) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    if x.dtype == gdtype:
        out = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
    else:
        out = F.grid_sample(
            x.to(gdtype),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        if not torch.is_autocast_enabled(x.device.type):
            out = out.to(x.dtype)
    inside = (norm_x.abs() <= 1.0) & (norm_y.abs() <= 1.0)
    return (out, inside[:, None].to(x.dtype))


def dilate_motion(
    motion: torch.Tensor, depth: torch.Tensor, near_is_small: bool = True
) -> torch.Tensor:
    key = -depth if near_is_small else depth
    (_, index) = F.max_pool2d(key, 3, stride=1, padding=1, return_indices=True)
    flat = motion.flatten(2)
    picked = flat.gather(2, index.flatten(2).expand(-1, motion.shape[1], -1))
    return picked.view_as(motion)


def _hi(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


def jitter_image_offset(jitter: torch.Tensor) -> torch.Tensor:
    return torch.stack((-jitter[:, 0], jitter[:, 1]), dim=1)


def registered_upsample(
    rgb: torch.Tensor, jitter: torch.Tensor, scale: int
) -> torch.Tensor:
    (batch, _, height, width) = rgb.shape
    (H, W) = (height * scale, width * scale)
    centre = 0.5 * (scale - 1)
    gdtype = torch.float64 if rgb.dtype == torch.float64 else torch.float32
    off = jitter_image_offset(jitter.to(gdtype))
    rows = torch.arange(H, device=rgb.device, dtype=gdtype)
    cols = torch.arange(W, device=rgb.device, dtype=gdtype)
    u = (rows[None, :] - centre) / scale - off[:, 0:1]
    v = (cols[None, :] - centre) / scale - off[:, 1:2]
    norm_y = 2.0 * u / max(height - 1, 1) - 1.0
    norm_x = 2.0 * v / max(width - 1, 1) - 1.0
    grid = torch.stack(
        (norm_x[:, None, :].expand(-1, H, -1), norm_y[:, :, None].expand(-1, -1, W)),
        dim=-1,
    )
    out = F.grid_sample(
        rgb.to(gdtype), grid, mode="bicubic", padding_mode="border", align_corners=True
    )
    return out.clamp(0, 1).to(rgb.dtype)


def sample_phase(
    scale: int, jitter: torch.Tensor, height: int, width: int, geometry: str = "legacy"
) -> torch.Tensor:
    if geometry not in ("legacy", "image"):
        raise ValueError(f"geometry must be 'legacy' or 'image', not {geometry!r}")
    (device, dtype) = (jitter.device, jitter.dtype)
    rows = torch.arange(height, device=device, dtype=dtype) % scale
    cols = torch.arange(width, device=device, dtype=dtype) % scale
    centre = 0.5 * scale - 0.5
    if geometry == "image":
        jitter = jitter_image_offset(jitter)
    dy = centre + jitter[:, 0, None, None] * scale - rows[None, :, None]
    dx = centre + jitter[:, 1, None, None] * scale - cols[None, None, :]
    if geometry == "image":
        half = 0.5 * scale
        dy = torch.remainder(dy + half, scale) - half
        dx = torch.remainder(dx + half, scale) - half
    dy = dy.expand(-1, height, width)
    dx = dx.expand(-1, height, width)
    return torch.stack((dy, dx), dim=1)


def depth_edges(depth: torch.Tensor, eps: float = 0.001) -> torch.Tensor:
    padded = F.pad(depth, (1, 1, 1, 1), mode="replicate")
    shifts = [
        padded[:, :, dy : dy + depth.shape[-2], dx : dx + depth.shape[-1]]
        for dy in (0, 1, 2)
        for dx in (0, 1, 2)
        if (dy, dx) != (1, 1)
    ]
    scale = depth.abs().clamp_min(eps)
    return torch.cat([((s - depth) / scale).clamp(-4.0, 4.0) for s in shifts], dim=1)


def _rgb_to_ycocg(x: torch.Tensor) -> torch.Tensor:
    (r, g, b) = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
    return torch.cat(
        (0.25 * r + 0.5 * g + 0.25 * b, 0.5 * (r - b), -0.25 * r + 0.5 * g - 0.25 * b),
        dim=1,
    )


def _ycocg_to_rgb(x: torch.Tensor) -> torch.Tensor:
    (y, co, cg) = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
    return torch.cat((y + co - cg, y + cg, y - co - cg), dim=1)


class RankRectifier(nn.Module):
    MODES = ("learned", "classic", "moments", "none")

    def __init__(
        self,
        gamma_max: float = 2.0,
        alpha_min: float = 1.0 / 16.0,
        phase_weight_init: float = -2.0,
        lam: float = 4.0,
        ycocg: bool = True,
        mode: str = "learned",
        phase_blend: bool = True,
        scale: int = 4,
        phase_memory: bool = False,
        band_split: bool = False,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.gamma_max = float(gamma_max)
        self.alpha_min = float(alpha_min)
        self.lam = float(lam)
        self.ycocg = bool(ycocg)
        self.mode = mode
        self.phase_blend = bool(phase_blend)
        self.scale = int(scale)
        self.phase_weight = nn.Parameter(torch.tensor(float(phase_weight_init)))
        self.phase_memory = bool(phase_memory)
        self.advantage_weight = nn.Parameter(torch.tensor(0.0))
        self.band_split = bool(band_split)
        self.band_offset = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def order_statistics(y: torch.Tensor) -> tuple[torch.Tensor, ...]:
        y_max = F.max_pool2d(y, 3, stride=1, padding=1)
        y_min = -F.max_pool2d(-y, 3, stride=1, padding=1)
        y_mu = F.avg_pool2d(y, 3, stride=1, padding=1, count_include_pad=False)
        return (y_min, y_mu, y_max)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        weights: torch.Tensor,
        gain: torch.Tensor,
        logit: torch.Tensor,
        valid: torch.Tensor,
        phase: torch.Tensor | None = None,
        history_reach: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = _rgb_to_ycocg(current) if self.ycocg else current
        h = _rgb_to_ycocg(history) if self.ycocg else history
        if self.mode == "none":
            rectified = h
            disagreement = torch.zeros_like(h[:, :1])
        else:
            (y_min, y_mu, y_max) = self.order_statistics(y)
            spread = (y_max - y_min).clamp_min(1e-05)
            if self.mode == "classic":
                (centre, half) = (0.5 * (y_min + y_max), 0.5 * spread)
            elif self.mode == "moments":
                second = F.avg_pool2d(
                    y * y, 3, stride=1, padding=1, count_include_pad=False
                )
                variance = (second - y_mu * y_mu).clamp_min(1e-10)
                sigma = variance.sqrt()
                centre = y_mu
                half = self.gamma_max * torch.sigmoid(gain) * 2.0 * sigma
            else:
                w = weights.softmax(dim=1)
                centre = w[:, 0:1] * y_min + w[:, 1:2] * y_mu + w[:, 2:3] * y_max
                half = self.gamma_max * torch.sigmoid(gain) * spread
            (low, high) = (centre - half, centre + half)
            rectified = torch.minimum(torch.maximum(h, low), high)
            excess = (h - rectified) / spread
            disagreement = excess.abs().mean(dim=1, keepdim=True)
        drive = logit - self.lam * disagreement
        reach = None
        if phase is not None:
            reach = phase.pow(2).sum(dim=1, keepdim=True).sqrt() / self.scale
            if self.phase_blend:
                drive = drive - F.softplus(self.phase_weight) * reach
            if self.phase_memory and history_reach is not None:
                advantage = history_reach - reach
                drive = drive + F.softplus(self.advantage_weight) * advantage
        alpha = torch.sigmoid(drive)
        alpha = self.alpha_min + (1.0 - self.alpha_min) * alpha
        alpha = alpha + (1.0 - alpha) * (1.0 - valid)
        if self.band_split:
            y_lo = F.avg_pool2d(y, 3, stride=1, padding=1, count_include_pad=False)
            h_lo = F.avg_pool2d(
                rectified, 3, stride=1, padding=1, count_include_pad=False
            )
            shift = self.band_offset
            alpha_lo = torch.sigmoid(
                torch.logit(alpha.clamp(0.0001, 1 - 0.0001)) + shift
            )
            alpha_hi = torch.sigmoid(
                torch.logit(alpha.clamp(0.0001, 1 - 0.0001)) - shift
            )
            blended = (
                alpha_lo * y_lo
                + (1.0 - alpha_lo) * h_lo
                + alpha_hi * (y - y_lo)
                + (1.0 - alpha_hi) * (rectified - h_lo)
            )
        else:
            blended = alpha * y + (1.0 - alpha) * rectified
        out = _ycocg_to_rgb(blended) if self.ycocg else blended
        new_reach = None
        if reach is not None:
            base = reach if history_reach is None else history_reach
            new_reach = alpha * reach + (1.0 - alpha) * base
        return (out, alpha, disagreement, new_reach)


class Stem(nn.Module):
    KINDS = ("jitter1x1", "conv3", "conv3_film")

    def __init__(
        self, in_dim: int, out_dim: int, kind: str = "conv3_film", hidden: int = 64
    ) -> None:
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"stem must be one of {self.KINDS}")
        (self.kind, self.in_dim, self.out_dim) = (kind, in_dim, out_dim)
        if kind == "jitter1x1":
            self.base = nn.Conv2d(in_dim, out_dim, 1)
            self.mlp = nn.Sequential(
                nn.Linear(4, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, in_dim * out_dim),
            )
        else:
            self.base = nn.Conv2d(in_dim, out_dim, 3, padding=1)
            self.mlp = (
                nn.Sequential(
                    nn.Linear(4, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, 2 * out_dim),
                )
                if kind == "conv3_film"
                else None
            )
        if self.mlp is not None:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def phase(jitter: torch.Tensor) -> torch.Tensor:
        return torch.cat((jitter, torch.sin(2 * torch.pi * jitter)), dim=1)

    def forward(self, x: torch.Tensor, jitter: torch.Tensor) -> torch.Tensor:
        if self.kind == "jitter1x1":
            delta = self.mlp(self.phase(jitter)).view(-1, self.out_dim, self.in_dim)
            return self.base(x) + torch.einsum("boi,bihw->bohw", delta, x)
        out = self.base(x)
        if self.kind == "conv3_film":
            gamma_beta = self.mlp(self.phase(jitter))
            (gamma, beta) = gamma_beta.chunk(2, dim=1)
            out = out * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return out


class Block(nn.Module):
    KINDS = ("dw7", "conv3")

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        layer_scale: float = 0.1,
        norm: str = "none",
        kind: str = "dw7",
        squeeze: int = 0,
    ) -> None:
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"block must be one of {self.KINDS}")
        self.kind = kind
        self.norm = nn.GroupNorm(1, dim) if norm == "group" else nn.Identity()
        if kind == "dw7":
            self.spatial = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            hidden = int(dim * mlp_ratio)
            self.mix = nn.Sequential(
                nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1)
            )
        else:
            self.spatial = nn.Conv2d(dim, dim, 3, padding=1)
            self.mix = nn.Sequential(nn.GELU(), nn.Conv2d(dim, dim, 3, padding=1))
        self.gate = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, max(4, dim // squeeze), 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(4, dim // squeeze), dim, 1),
                nn.Sigmoid(),
            )
            if squeeze
            else None
        )
        self.scale = nn.Parameter(layer_scale * torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mix(self.norm(self.spatial(x)))
        if self.gate is not None:
            h = h * self.gate(h)
        return x + self.scale * h


RRTA_SIZES: dict[str, dict] = {
    "t": dict(dim=32, blocks=4),
    "s": dict(dim=48, blocks=6),
    "m": dict(dim=64, blocks=8),
}


class RRTANet(nn.Module):
    _SMR_SIGMA_FLOOR = 0.2

    def __init__(
        self,
        scale: int = 4,
        dim: int = 32,
        blocks: int = 4,
        mlp_ratio: float = 2.0,
        layer_scale: float = 0.1,
        gamma_max: float = 2.0,
        alpha_min: float = 1.0 / 16.0,
        phase_weight_init: float = -2.0,
        lam: float = 4.0,
        ycocg: bool = True,
        motion_dilation: bool = True,
        rectify: str = "learned",
        recurrent: bool = True,
        phase_blend: bool = True,
        band_lowres: bool = True,
        feature_dim: int = 8,
        norm: str = "none",
        depth_edge: bool = True,
        stem: str = "conv3_film",
        jitter_head: bool = True,
        phase_memory: bool = True,
        block: str = "conv3",
        band_split: bool = True,
        hr_refine: int = 16,
        hr_layers: int = 3,
        frames_in: int = 1,
        squeeze: int = 0,
        dual_rate: float = 0.0,
        canonical_phase: bool = False,
        accumulate: str = "blend",
        reject: str = "none",
        kernel: str = "gauss",
        accum_cap: float = 0.0,
        static_gate: float = 0.0,
        global_motion: bool = False,
        dejitter: bool = False,
        first_order: bool = False,
        accum_alpha_min: float = 0.0,
        static_cap: float = 64.0,
        phase_geometry: str = "legacy",
        registered_skip: bool = False,
        lr_rejitter: bool = False,
        depth_jit: bool = True,
        fp32_accum: bool = False,
        skip_dead: bool = False,
        readout: str = "accum",
        vcap_max: float = 0.0,
        vcap_min: float = 4.0,
        vcap_v0: float = 2.0,
        hist_clip: str = "none",
        smr_sigma: float = 0.6,
        smr_lambda_init: float = 1.0,
        smr_lambda_eps: float = 0.01,
        smr_input_eps: float = 0.05,
        smr_depth: str = "far0.1",
        lr_unshuffle: int = 1,
        feature_bound: float = 0.0,
        consistency_gate: str = "none",
        state_mode: str = "linear",
        gate_shorten: str = "off",
        gate_shorten_init: float = 0.01,
        body: str = "plain",
        body_inner: str = "conv3",
        body_width_mult: float = 2.0,
        body_unet_levels: int = 1,
        body_unet_blocks: str = "",
        body_pconv_ratio: int = 4,
        body_expand: float = 2.0,
        body_rep_branches: str = "ecb",
        body_act: str = "gelu",
        body_dw_kernel: int = 3,
        body_dilations: str = "1,2",
        body_attn_idx: str = "",
        body_attn_dim: int = 16,
    ) -> None:
        super().__init__()
        lr_unshuffle = int(lr_unshuffle)
        if lr_unshuffle < 1:
            raise ValueError("lr_unshuffle must be >= 1 (1 = off)")
        if lr_unshuffle > 1 and canonical_phase:
            raise ValueError(
                "canonical_phase rolls the head's s x s sample slot; it is not defined for lr_unshuffle > 1"
            )
        self.lr_unshuffle = lr_unshuffle
        if consistency_gate not in ("none", "sample"):
            raise ValueError("consistency_gate must be 'none' or 'sample'")
        if state_mode not in ("linear", "gru"):
            raise ValueError("state_mode must be 'linear' or 'gru'")
        if consistency_gate != "none" and accumulate == "samples":
            raise NotImplementedError(
                "consistency_gate='sample' is not implemented for the SMR path (accumulate='samples'); use accumulate='normalized'"
            )
        if consistency_gate != "none" and accumulate != "normalized":
            raise ValueError(
                "consistency_gate='sample' needs accumulate='normalized' (its window shortening acts on the normalized accumulator's retention)"
            )
        if state_mode == "gru" and int(feature_dim) <= 0:
            raise ValueError("state_mode='gru' needs feature_dim > 0")
        if gate_shorten not in ("off", "on"):
            raise ValueError("gate_shorten must be 'off' or 'on'")
        if gate_shorten == "on" and consistency_gate == "none":
            raise ValueError(
                "gate_shorten='on' needs consistency_gate='sample' (it shortens the window by r)"
            )
        if not gate_shorten_init > 0.0:
            raise ValueError("gate_shorten_init is softplus(g) at init; must be > 0")
        self.consistency_gate = consistency_gate
        self.state_mode = state_mode
        self.gate_shorten = gate_shorten
        self.gate_shorten_init = float(gate_shorten_init)
        if not feature_bound >= 0.0:
            raise ValueError("feature_bound must be >= 0 (0 = off)")
        self.feature_bound = float(feature_bound)
        (self.scale, self.dim) = (int(scale), int(dim))
        self.motion_dilation = bool(motion_dilation)
        self.recurrent = bool(recurrent)
        self.band_lowres = bool(band_lowres)
        self.feature_dim = int(feature_dim)
        self.depth_edge = bool(depth_edge)
        self.phase_memory = bool(phase_memory)
        self.frames_in = max(1, int(frames_in))
        self.dual_rate = float(dual_rate)
        self.canonical_phase = bool(canonical_phase)
        if accumulate not in ("blend", "normalized", "samples", "direct"):
            raise ValueError(
                "accumulate must be 'blend', 'normalized', 'samples' or 'direct'"
            )
        if accumulate == "direct":
            bad = [
                name
                for (name, on) in (
                    ("reject", reject != "none"),
                    ("first_order", first_order),
                    ("hist_clip", hist_clip != "none"),
                    ("static_gate", static_gate > 0.0),
                    ("accum_alpha_min", accum_alpha_min > 0.0),
                    ("accum_cap", accum_cap > 0.0),
                    ("dual_rate", dual_rate > 0.0),
                    ("vcap_max", vcap_max > 0.0),
                    ("readout", readout != "accum"),
                    ("consistency_gate", consistency_gate != "none"),
                )
                if on
            ]
            if bad:
                raise ValueError(f"accumulate='direct' does not combine with {bad}")
        self.accumulate = accumulate
        self.accum_cap = float(accum_cap)
        self.static_gate = float(static_gate)
        self.global_motion = bool(global_motion)
        self.dejitter = bool(dejitter)
        self.first_order = bool(first_order)
        self.accum_alpha_min = float(accum_alpha_min)
        if first_order:
            self.mls_gain = nn.Parameter(torch.tensor(1.0))
        self.static_cap = float(static_cap)
        if phase_geometry not in ("legacy", "image"):
            raise ValueError("phase_geometry must be 'legacy' or 'image'")
        self.phase_geometry = phase_geometry
        self.registered_skip = bool(registered_skip)
        self.lr_rejitter = bool(lr_rejitter)
        self.depth_jit = bool(depth_jit)
        self.fp32_accum = bool(fp32_accum)
        self.skip_dead = bool(skip_dead)
        if readout not in ("accum", "residual", "fresh"):
            raise ValueError("readout must be 'accum', 'residual' or 'fresh'")
        if readout != "accum" and accumulate not in ("normalized", "samples"):
            raise ValueError(
                "readout != 'accum' needs accumulate='normalized' (or 'samples' for readout='residual')"
            )
        if readout == "fresh" and accumulate == "samples":
            raise ValueError(
                "readout='fresh' is the samples path's own MAP read-out; use readout 'accum' or 'residual'"
            )
        if readout != "accum" and first_order:
            raise ValueError(
                "readout != 'accum' is not combined with first_order (both are read-outs)"
            )
        self.readout = readout
        self.vcap_max = float(vcap_max)
        self.vcap_min = float(vcap_min)
        self.vcap_v0 = float(vcap_v0)
        if hist_clip not in ("none", "var"):
            raise ValueError("hist_clip must be 'none' or 'var'")
        if hist_clip != "none" and accumulate != "normalized":
            raise ValueError("hist_clip needs accumulate='normalized'")
        self.hist_clip = hist_clip
        if self.vcap_max > 0.0 and accumulate not in ("normalized", "samples"):
            raise ValueError("vcap_max needs accumulate='normalized' or 'samples'")
        self.smr_sigma = float(smr_sigma)
        self.smr_lambda_init = float(smr_lambda_init)
        self.smr_lambda_eps = float(smr_lambda_eps)
        self.smr_input_eps = float(smr_input_eps)
        self.smr_depth = str(smr_depth)
        self.smr_depth_spec = None
        if accumulate == "samples":
            bad = [
                name
                for (name, on) in (
                    ("reject", reject != "none"),
                    ("first_order", first_order),
                    ("hist_clip", hist_clip != "none"),
                    ("static_gate", static_gate > 0.0),
                    ("accum_alpha_min", accum_alpha_min > 0.0),
                    ("dual_rate", dual_rate > 0.0),
                )
                if on
            ]
            if bad:
                raise ValueError(f"accumulate='samples' does not combine with {bad}")
            if not self.smr_sigma > self._SMR_SIGMA_FLOOR:
                raise ValueError(f"smr_sigma must exceed {self._SMR_SIGMA_FLOOR} HR px")
            if not (
                self.smr_lambda_init > self.smr_lambda_eps
                and self.smr_lambda_eps > 0.0
                and (self.smr_input_eps > 0.0)
            ):
                raise ValueError(
                    "need smr_lambda_init > smr_lambda_eps > 0 and smr_input_eps > 0"
                )
            self.smr_depth_spec = _parse_depth_spec(self.smr_depth)
        self.register_buffer("wbar_ema", torch.tensor(0.5), persistent=False)
        if kernel not in ("none", "gauss"):
            raise ValueError("kernel must be 'none' or 'gauss'")
        self.kernel = kernel
        if reject not in ("none", "mismatch", "variance", "ztest"):
            raise ValueError("reject must be 'none', 'mismatch', 'variance' or 'ztest'")
        self.reject = reject
        self.needs_m2 = reject in ("variance", "ztest")
        if reject != "none" and accumulate == "normalized":
            self.reject_gain = nn.Parameter(torch.tensor(0.0))
            self.reject_gamma = (
                nn.Parameter(torch.tensor(0.55)) if reject == "ztest" else None
            )
        else:
            self.reject_gain = None
            self.reject_gamma = None
        s2 = (self.scale * self.lr_unshuffle) ** 2
        k2 = self.lr_unshuffle**2
        if self.accumulate == "direct":
            self.history_proj = None
            hist_ch = 3 * s2
        else:
            self.history_proj = nn.Conv2d(3 * s2, 16 * k2, 1)
            hist_ch = 16 * k2
        self.confidence_proj = nn.Conv2d(s2, 4 * k2, 1)
        self.stem = Stem(
            (3 + 1 + 2 + 1 + 4) * k2
            + hist_ch
            + self.feature_dim
            + (2 if self.global_motion else 0)
            + (8 * k2 if self.depth_edge else 0)
            + (4 * k2 if self.phase_memory else 0)
            + (3 * k2 + 2) * (self.frames_in - 1)
            + (3 * k2 if self.consistency_gate != "none" else 0),
            dim,
            kind=stem,
        )
        self.block_kind = str(block)
        self.body_kind = str(body)
        if self.body_kind == "plain":
            self.body = nn.Sequential(
                *[
                    Block(
                        dim, mlp_ratio, layer_scale, norm, kind=block, squeeze=squeeze
                    )
                    for _ in range(blocks)
                ]
            )
        else:
            from .fast_blocks import attach_rep_scope, make_body

            if norm != "none" or squeeze:
                raise ValueError("body != 'plain' has no norm/squeeze option")
            self.body = make_body(
                self.body_kind,
                dim,
                blocks,
                layer_scale=layer_scale,
                act=body_act,
                expand=body_expand,
                pconv_ratio=body_pconv_ratio,
                rep_branches=body_rep_branches,
                inner=body_inner,
                unet_levels=body_unet_levels,
                width_mult=body_width_mult,
                unet_blocks=body_unet_blocks,
                dw_kernel=body_dw_kernel,
                dilations=body_dilations,
                attn_idx=body_attn_idx,
                attn_dim=body_attn_dim,
            )
            attach_rep_scope(self)
        self.jitter_head = bool(jitter_head)
        self.hr_refine = int(hr_refine)
        self.hr_layers = int(hr_layers)
        self.head_colour = nn.Conv2d(dim, 3 * s2, 1)
        self.head_film = (
            nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 2 * 3 * s2)
            )
            if self.jitter_head
            else None
        )
        if self.head_film is not None:
            nn.init.zeros_(self.head_film[-1].weight)
            nn.init.zeros_(self.head_film[-1].bias)
        band_out = 7 if self.band_lowres else 7 * s2
        self.head_band = nn.Conv2d(dim, band_out, 1)
        self.head_feature = (
            nn.Conv2d(dim, self.feature_dim * (2 if self.state_mode == "gru" else 1), 1)
            if self.feature_dim
            else None
        )
        nn.init.zeros_(self.head_colour.weight)
        nn.init.zeros_(self.head_colour.bias)
        nn.init.zeros_(self.head_band.weight)
        with torch.no_grad():
            bias = torch.zeros(7, 1 if self.band_lowres else s2)
            bias[0] = 2.0
            bias[2] = 2.0
            ratio = 1.0 / (2.0 * max(gamma_max, 1e-06))
            bias[3:6] = math.log(ratio / (1.0 - ratio)) if ratio < 1.0 else 0.0
            bias[6] = 0.0
            self.head_band.bias.copy_(bias.reshape(-1))
        if self.accumulate in ("normalized", "samples"):
            self.head_weight = nn.Conv2d(dim, 2 * s2, 1)
            nn.init.zeros_(self.head_weight.weight)
            nn.init.zeros_(self.head_weight.bias)
            with torch.no_grad():
                self.head_weight.bias[s2:] = math.log((1.0 - alpha_min) / alpha_min)
            if self.accumulate == "normalized":
                self.kernel_width = nn.Parameter(torch.tensor(0.0))
        else:
            self.head_weight = None
        if self.dual_rate > 0.0:
            self.head_fast = nn.Conv2d(dim, 1 * s2, 1)
            nn.init.zeros_(self.head_fast.weight)
            nn.init.zeros_(self.head_fast.bias)
            self.trust = nn.Parameter(torch.tensor([2.0, -2.0]))
        else:
            self.head_fast = None
        if self.hr_refine:
            width = self.hr_refine
            layers: list[nn.Module] = [nn.Conv2d(8, width, 3, padding=1)]
            for _ in range(max(0, self.hr_layers - 2)):
                layers += [nn.GELU(), nn.Conv2d(width, width, 3, padding=1)]
            layers += [nn.GELU(), nn.Conv2d(width, 3, 3, padding=1)]
            self.refine = nn.Sequential(*layers)
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)
        else:
            self.refine = None
        self.rectifier = RankRectifier(
            gamma_max,
            alpha_min,
            phase_weight_init,
            lam,
            ycocg,
            mode=rectify,
            phase_blend=phase_blend,
            scale=scale,
            phase_memory=phase_memory,
            band_split=band_split,
        )
        self.reach_proj = nn.Conv2d(s2, 4 * k2, 1) if self.phase_memory else None
        if self.readout == "residual":
            self.head_delta = nn.Conv2d(dim, 3 * s2, 1)
            nn.init.zeros_(self.head_delta.weight)
            nn.init.zeros_(self.head_delta.bias)
        else:
            self.head_delta = None
        if self.readout == "fresh":
            self.head_mu = nn.Conv2d(dim, s2, 1)
            nn.init.zeros_(self.head_mu.weight)
            with torch.no_grad():
                self.head_mu.bias.fill_(math.log(math.expm1(0.05)))
        else:
            self.head_mu = None
        if self.hist_clip == "var":
            self.head_clip = nn.Conv2d(dim, s2, 1)
            nn.init.zeros_(self.head_clip.weight)
            with torch.no_grad():
                self.head_clip.bias.fill_(math.log(math.expm1(0.75)))
        else:
            self.head_clip = None
        if self.accumulate == "samples":
            self.head_lambda = nn.Conv2d(dim, s2, 1)
            nn.init.zeros_(self.head_lambda.weight)
            with torch.no_grad():
                self.head_lambda.bias.fill_(
                    math.log(math.expm1(self.smr_lambda_init - self.smr_lambda_eps))
                )
            self.smr_sigma_param = nn.Parameter(
                torch.tensor(
                    math.log(math.expm1(self.smr_sigma - self._SMR_SIGMA_FLOOR))
                )
            )
        else:
            self.head_lambda = None
            self.smr_sigma_param = None
        if self.consistency_gate != "none":
            self.gate_gain = (
                nn.Parameter(torch.tensor(math.log(math.expm1(self.gate_shorten_init))))
                if self.gate_shorten == "on"
                else None
            )
            self.gru_gate_r = (
                nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
                if self.state_mode == "gru"
                else None
            )
        else:
            self.gate_gain = None
            self.gru_gate_r = None

    GATE_EPS = 0.02
    GATE_W_EPS = 0.001

    def _consistency(
        self,
        rgb: torch.Tensor,
        jitter: torch.Tensor,
        colour_hist: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        hr_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (height, width) = rgb.shape[-2:]
        (big_h, big_w) = hr_size
        with (
            torch.no_grad(),
            torch.autocast(device_type=rgb.device.type, enabled=False),
        ):
            x = _hi(rgb)
            dt = x.dtype
            q = _registered_positions(
                self.scale, jitter.to(dt), height, width, convention="image"
            )
            grid = torch.stack(
                (
                    2.0 * q[:, 1] / max(big_w - 1, 1) - 1.0,
                    2.0 * q[:, 0] / max(big_h - 1, 1) - 1.0,
                ),
                dim=-1,
            )
            hist = torch.cat(
                (colour_hist.to(dt), confidence.to(dt), valid.to(dt)), dim=1
            )
            at_q = F.grid_sample(
                hist, grid, mode="bilinear", padding_mode="border", align_corners=True
            )
            h_q = at_q[:, :3]
            w_q = at_q[:, 3:4].clamp_min(0.0)
            ok = (w_q > self.GATE_W_EPS) & (at_q[:, 4:5] > 0.5)
            luma = 0.25 * x[:, 0:1] + 0.5 * x[:, 1:2] + 0.25 * x[:, 2:3]
            mu = F.avg_pool2d(luma, 3, stride=1, padding=1, count_include_pad=False)
            m2 = F.avg_pool2d(
                luma * luma, 3, stride=1, padding=1, count_include_pad=False
            )
            sigma = (m2 - mu * mu).clamp_min(0.0).sqrt()
            d = (x - h_q).abs().mean(dim=1, keepdim=True)
            d = torch.where(ok, d, torch.zeros_like(d))
            r = d / (sigma + self.GATE_EPS)
        return (r, w_q, ok, d)

    track_wbar = False

    def _trunk_motion(self, motion: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        k = self.lr_unshuffle
        (_, index) = F.max_pool2d(-depth, k, stride=k, return_indices=True)
        flat = motion.flatten(2)
        picked = flat.gather(2, index.flatten(2).expand(-1, motion.shape[1], -1))
        return picked.view(motion.shape[0], motion.shape[1], *index.shape[-2:]) / k

    def _lo(self, t: torch.Tensor) -> torch.Tensor:
        if self.lr_unshuffle > 1:
            return F.pixel_unshuffle(t, self.lr_unshuffle)
        return t

    def _lr_motion(
        self,
        motion: torch.Tensor,
        jitter: torch.Tensor,
        previous_jitter: torch.Tensor | None,
    ) -> torch.Tensor:
        if previous_jitter is None:
            return motion
        d = jitter.to(motion.dtype) - previous_jitter.to(motion.dtype)
        return motion + torch.stack((-d[:, 0], d[:, 1]), dim=1)[:, :, None, None]

    def _skip(
        self, rgb: torch.Tensor, jitter: torch.Tensor, hr_size: tuple[int, int]
    ) -> torch.Tensor:
        return registered_upsample(rgb, jitter, self.scale)

    def _vcap_frames(self, motion_hr: torch.Tensor) -> torch.Tensor:
        speed = motion_hr.pow(2).sum(dim=1, keepdim=True).sqrt()
        return torch.clamp_min(
            self.vcap_max / (1.0 + speed / self.vcap_v0), self.vcap_min
        )

    def smr_sigma_value(self) -> torch.Tensor:
        return self._SMR_SIGMA_FLOOR + F.softplus(self.smr_sigma_param)

    def step(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        jitter: torch.Tensor,
        state: tuple | None,
        init_hr: torch.Tensor | None = None,
        init_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple, dict]:
        (batch, _, height, width) = rgb.shape
        scale = self.scale
        hr_size = (height * scale, width * scale)
        kun = self.lr_unshuffle
        tscale = scale * kun
        if kun > 1 and (height % kun or width % kun):
            raise ValueError(
                f"lr_unshuffle={kun} needs an LR size divisible by {kun}, got {height}x{width}"
            )
        (theight, twidth) = (height // kun, width // kun)
        smr = self.accumulate == "samples"
        if smr and init_hr is not None:
            raise NotImplementedError(
                "init_hr (GT warm start) would put a non-sample image into the SMR sample state; not supported"
            )
        if not self.recurrent:
            state = None
        if self.motion_dilation:
            motion = dilate_motion(motion, depth)
        diag: dict = {}
        if state is None:
            if self.registered_skip:
                colour_hist = self._skip(rgb, jitter, hr_size)
            else:
                colour_hist = F.interpolate(
                    rgb, size=hr_size, mode="bicubic", align_corners=False
                ).clamp(0, 1)
            confidence = torch.zeros(
                batch, 1, *hr_size, device=rgb.device, dtype=rgb.dtype
            )
            valid = torch.zeros_like(confidence)
            reach_hist = torch.full_like(confidence, 1.0)
            if init_hr is not None:
                gate = (
                    torch.ones(batch, device=rgb.device, dtype=rgb.dtype)
                    if init_mask is None
                    else init_mask.to(device=rgb.device, dtype=rgb.dtype)
                ).view(-1, 1, 1, 1)
                warm = init_hr.to(colour_hist.dtype)
                colour_hist = gate * warm + (1.0 - gate) * colour_hist
                if self.accumulate == "normalized":
                    w0 = 8.0 * self.wbar_ema.to(rgb.dtype).clamp_min(0.001)
                    if self.vcap_max <= 0.0:
                        legacy_cap = (
                            self.accum_cap
                            if self.accum_cap > 0.0
                            else 1.0 / self.rectifier.alpha_min
                        )
                        w0 = w0.clamp_max(legacy_cap)
                else:
                    w0 = torch.tensor(
                        min(8.0, 1.0 / self.rectifier.alpha_min),
                        device=rgb.device,
                        dtype=rgb.dtype,
                    )
                confidence = gate * w0 * torch.ones_like(confidence)
                valid = gate * torch.ones_like(valid)
                radius0 = (
                    sample_phase(
                        scale,
                        jitter.to(rgb.dtype),
                        *hr_size,
                        geometry=self.phase_geometry,
                    )
                    .pow(2)
                    .sum(1, keepdim=True)
                    .sqrt()
                    / scale
                )
                reach_hist = gate * radius0 + (1.0 - gate) * reach_hist
            fast_hist = colour_hist if self.dual_rate > 0.0 else None
            m2_hist = colour_hist * colour_hist if self.needs_m2 else colour_hist[:, :0]
            m1_hist = (
                torch.zeros(batch, 2, *hr_size, device=rgb.device, dtype=rgb.dtype)
                if self.first_order
                else colour_hist[:, :0]
            )
            disocclusion = torch.zeros_like(depth)
            past_rgb = (
                rgb.repeat(1, self.frames_in - 1, 1, 1) if self.frames_in > 1 else None
            )
            past_jitter = (
                torch.zeros(
                    batch, 2 * (self.frames_in - 1), device=rgb.device, dtype=rgb.dtype
                )
                if self.frames_in > 1
                else None
            )
            feature_hist = torch.zeros(
                batch,
                self.feature_dim,
                theight,
                twidth,
                device=rgb.device,
                dtype=rgb.dtype,
            )
            motion_hr = None
        else:
            (
                previous_colour,
                previous_confidence,
                previous_depth,
                previous_feature,
                previous_reach,
                previous_rgb,
                previous_jitter,
                previous_fast,
                previous_m2,
                previous_jitter_full,
                previous_m1,
            ) = state
            if self.dejitter and previous_jitter_full is not None:
                delta = jitter.to(motion.dtype) - previous_jitter_full
                motion = (
                    motion
                    + torch.stack((delta[:, 0], -delta[:, 1]), dim=1)[:, :, None, None]
                )
            motion_lr = (
                self._lr_motion(motion, jitter, previous_jitter_full)
                if self.lr_rejitter
                else motion
            )
            motion_hr = F.interpolate(motion, size=hr_size, mode="nearest") * scale
            (warped, inside) = warp(
                torch.cat(
                    (
                        previous_colour,
                        previous_confidence,
                        previous_reach,
                        previous_m2,
                        previous_m1,
                    ),
                    dim=1,
                ),
                motion_hr,
            )
            (colour_hist, confidence) = (warped[:, :3], warped[:, 3:4] * inside)
            _o = 5
            m2_hist = warped[:, _o : _o + 3] if self.needs_m2 else warped[:, :0]
            _o += 3 if self.needs_m2 else 0
            m1_hist = warped[:, _o : _o + 2] if self.first_order else warped[:, :0]
            fast_hist = (
                warp(previous_fast, motion_hr)[0] if self.dual_rate > 0.0 else None
            )
            reach_hist = warped[:, 4:5] * inside + (1.0 - inside)
            valid = inside
            if kun > 1:
                (warped_depth, depth_inside) = warp(
                    previous_depth,
                    motion if self.lr_rejitter and (not self.depth_jit) else motion_lr,
                )
                if self.feature_dim:
                    (warped_feat, feat_inside) = warp(
                        previous_feature, self._trunk_motion(motion_lr, depth)
                    )
                    feature_hist = warped_feat * feat_inside
                else:
                    feature_hist = torch.zeros(
                        batch, 0, theight, twidth, device=rgb.device, dtype=rgb.dtype
                    )
            elif self.lr_rejitter and (not self.depth_jit):
                (warped_depth, depth_inside) = warp(previous_depth, motion)
                if self.feature_dim:
                    (warped_feat, feat_inside) = warp(previous_feature, motion_lr)
                    feature_hist = warped_feat * feat_inside
                else:
                    feature_hist = torch.zeros(
                        batch, 0, height, width, device=rgb.device, dtype=rgb.dtype
                    )
            else:
                lr_state = previous_depth
                if self.feature_dim:
                    lr_state = torch.cat((lr_state, previous_feature), dim=1)
                (warped_lr, depth_inside) = warp(lr_state, motion_lr)
                warped_depth = warped_lr[:, :1]
                feature_hist = (
                    warped_lr[:, 1:] * depth_inside
                    if self.feature_dim
                    else torch.zeros(
                        batch, 0, height, width, device=rgb.device, dtype=rgb.dtype
                    )
                )
            disocclusion = (depth - warped_depth).abs() * depth_inside
            if self.frames_in > 1:
                (warped_rgb, rgb_inside) = warp(previous_rgb, motion_lr)
                past_rgb = warped_rgb * rgb_inside
                past_jitter = previous_jitter
            else:
                past_rgb = past_jitter = None
        confidence_in = confidence
        if smr:
            s_hat = _hi(colour_hist) * _hi(valid)
            w_hat = _hi(confidence)
            if self.registered_skip:
                prior_in = self._skip(rgb, jitter, hr_size)
            else:
                prior_in = F.interpolate(
                    rgb, size=hr_size, mode="bicubic", align_corners=False
                ).clamp(0, 1)
            eps_in = self.smr_input_eps
            colour_hist = (s_hat + eps_in * _hi(prior_in)) / (w_hat + eps_in)
            confidence_in = torch.log1p(w_hat)
            dtest_hr = None
            if state is not None and self.smr_depth_spec is not None:
                (mode, rel) = self.smr_depth_spec
                err = (_hi(depth) - _hi(warped_depth)) / _hi(depth).abs().clamp_min(
                    1e-06
                )
                reject = err > rel if mode == "far" else err.abs() > rel
                dtest = (~reject).to(s_hat.dtype) * _hi(depth_inside)
                dtest_hr = F.interpolate(dtest, size=hr_size, mode="nearest")
        gate_r = gate_wq = gate_d = None
        if self.consistency_gate != "none":
            (gate_r, gate_wq, gate_ok, gate_d) = self._consistency(
                rgb, jitter, colour_hist, confidence, valid, hr_size
            )
            okf = gate_ok.to(gate_r.dtype)
            diag["gate_r_mean"] = gate_r.mean().detach()
            diag["gate_r_mean_hist"] = (
                gate_r.sum() / okf.sum().clamp_min(1.0)
            ).detach()
            diag["gate_r_gt2"] = (gate_r > 2.0).to(gate_r.dtype).mean().detach()
            diag["gate_hist_frac"] = okf.mean().detach()
            diag["gate_d_mean_hist"] = (
                gate_d.sum() / okf.sum().clamp_min(1.0)
            ).detach()
            if self.gate_gain is not None:
                diag["gate_g"] = F.softplus(self.gate_gain.detach().float())
        history_lr = (
            F.pixel_unshuffle(colour_hist, tscale)
            if self.history_proj is None
            else self.history_proj(F.pixel_unshuffle(colour_hist, tscale))
        )
        confidence_lr = self.confidence_proj(F.pixel_unshuffle(confidence_in, tscale))
        parts = [
            self._lo(rgb),
            self._lo(depth),
            self._lo(motion),
            self._lo(disocclusion),
            history_lr,
            confidence_lr,
        ]
        if self.global_motion:
            spread = motion.flatten(2).std(dim=2).amax(dim=1)
            magnitude = motion.pow(2).sum(1).sqrt().flatten(1).mean(dim=1)
            stat_g = torch.log1p(torch.stack((spread, magnitude), dim=1))
            parts.append(stat_g[:, :, None, None].expand(-1, -1, theight, twidth))
        if self.reach_proj is not None:
            parts.append(self.reach_proj(F.pixel_unshuffle(reach_hist, tscale)))
        if self.feature_dim:
            parts.append(feature_hist)
        if self.depth_edge:
            parts.append(self._lo(depth_edges(depth)))
        if self.frames_in > 1:
            relative = (
                past_jitter.view(batch, -1, 2)
                - jitter.to(past_jitter.dtype)[:, None, :]
            ).reshape(batch, -1)
            parts.append(self._lo(past_rgb))
            parts.append(relative[:, :, None, None].expand(-1, -1, theight, twidth))
        if gate_r is not None:
            parts.append(
                self._lo(
                    torch.cat(
                        (torch.log1p(gate_r), torch.log1p(gate_wq), gate_d), dim=1
                    ).to(rgb.dtype)
                )
            )
        x = self.body(self.stem(torch.cat(parts, dim=1), jitter))
        colour = (
            self._head(self.head_colour, x)
            if self.accumulate == "direct"
            else self.head_colour(x)
        )
        if self.canonical_phase:
            if self.phase_geometry == "image":
                o = jitter_image_offset(_hi(jitter))
                off = ((o + 0.5) * scale).floor().long() % scale
            else:
                off = ((jitter.to(colour.dtype) + 0.5) * scale).round().long() % scale
            groups = colour.view(batch, 3, scale, scale, height, width)
            rolled = torch.stack(
                [
                    torch.roll(
                        groups[b], shifts=(int(off[b, 0]), int(off[b, 1])), dims=(1, 2)
                    )
                    for b in range(batch)
                ]
            )
            colour = rolled.view(batch, 3 * scale * scale, height, width)
        if self.head_film is not None:
            gb = self.head_film(Stem.phase(jitter.to(x.dtype)))
            (gamma, beta) = gb.chunk(2, dim=1)
            colour = colour * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        candidate = F.pixel_shuffle(colour, tscale)
        if self.registered_skip:
            skip = self._skip(rgb, jitter, hr_size)
        else:
            skip = F.interpolate(rgb, size=hr_size, mode="bicubic", align_corners=False)
        candidate = candidate + skip
        dead = self.skip_dead and (
            self.head_weight is not None or self.accumulate == "direct"
        )
        if not dead:
            raw_band = self.head_band(x)
            band = (
                F.interpolate(
                    raw_band, size=hr_size, mode="bilinear", align_corners=False
                )
                if self.band_lowres
                else F.pixel_shuffle(raw_band, tscale)
            )
        if self.refine is not None:
            candidate = candidate + self.refine(
                torch.cat((candidate, colour_hist, confidence, reach_hist), dim=1)
            )
        phase = sample_phase(
            scale, jitter.to(rgb.dtype), *hr_size, geometry=self.phase_geometry
        )
        if not dead:
            (out, alpha, disagreement, new_reach) = self.rectifier(
                candidate,
                colour_hist,
                band[:, 0:3],
                band[:, 3:6],
                band[:, 6:7],
                valid,
                phase,
                reach_hist if self.phase_memory else None,
            )
            if new_reach is None:
                new_reach = torch.ones_like(confidence)
        if smr:
            candidate = _hi(candidate)
            valid = _hi(valid)
            stat = _hi(F.pixel_shuffle(self._head(self.head_weight, x), tscale))
            rho = torch.sigmoid(stat[:, 0:1])
            retention = torch.sigmoid(stat[:, 1:2])
            lam = (
                F.softplus(
                    _hi(F.pixel_shuffle(self._head(self.head_lambda, x), tscale))
                )
                + self.smr_lambda_eps
            )
            rgb_hi = _hi(rgb)
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                sigma = self.smr_sigma_value().to(rgb_hi.dtype)
                (s_f, w_f) = _splat_gather(
                    rgb_hi,
                    jitter.to(rgb_hi.dtype),
                    scale,
                    sigma,
                    taps=3,
                    convention="image",
                    normalize="analytic",
                )
            w_frame = rho * w_f
            keep = retention * valid
            if dtest_hr is not None:
                keep = keep * dtest_hr
            s_new = keep * s_hat + rho * s_f
            w_new = keep * w_hat + w_frame
            if self.vcap_max > 0.0:
                if motion_hr is None:
                    motion_hr = (
                        F.interpolate(motion, size=hr_size, mode="nearest") * scale
                    )
                frames = self._vcap_frames(_hi(motion_hr))
                cap = frames * w_frame.detach().mean(
                    dim=(1, 2, 3), keepdim=True
                ).clamp_min(1e-06)
                diag["vcap_frames"] = frames.mean().detach()
            else:
                cap = (
                    self.accum_cap
                    if self.accum_cap > 0.0
                    else 1.0 / self.rectifier.alpha_min
                )
            excess = (w_new / cap).clamp_min(1.0)
            s_new = s_new / excess
            w_new = w_new / excess
            out = (s_new + lam * candidate) / (w_new + lam)
            gate = lam / (w_new + lam)
            if self.readout == "residual":
                delta = F.pixel_shuffle(self._head(self.head_delta, x), tscale)
                out = out + delta
                diag["delta_abs"] = delta.abs().mean().detach()
            state_colour = s_new
            new_confidence = w_new
            alpha = w_frame / excess / w_new.clamp_min(1e-12)
            disagreement = torch.zeros_like(w_new)
            new_reach = _hi(phase).pow(2).sum(1, keepdim=True).sqrt() / scale
            smr_aux = {
                "gate": gate,
                "lam": lam,
                "W": w_new,
                "rho": rho,
                "retention": retention,
                "keep": keep,
            }
            diag["smr_sigma"] = sigma.detach()
            diag["gate_mean"] = gate.mean().detach()
            diag["lam_mean"] = lam.mean().detach()
            diag["rho_mean"] = rho.mean().detach()
            diag["ret_mean"] = retention.mean().detach()
            diag["ret_sat"] = (retention >= 0.999).to(w_new.dtype).mean().detach()
            diag["w_mean"] = w_new.mean().detach()
            diag["w_over_cap_max"] = (w_new / cap).max().detach()
            diag["alpha_mean"] = alpha.mean().detach()
            diag["vcap_bind"] = (excess > 1.0).to(w_new.dtype).mean().detach()
            if dtest_hr is not None:
                diag["dtest_rej"] = (
                    (1.0 - dtest_hr) * valid
                ).sum().detach() / valid.sum().clamp_min(1.0).detach()
        elif self.head_weight is not None:
            if self.fp32_accum:
                candidate = _hi(candidate)
                colour_hist = _hi(colour_hist)
                confidence = _hi(confidence)
                valid = _hi(valid)
                phase = _hi(phase)
                m2_hist = _hi(m2_hist)
                m1_hist = _hi(m1_hist)
            stat = F.pixel_shuffle(self._head(self.head_weight, x), tscale)
            reliability = torch.sigmoid(stat[:, 0:1])
            retention_logit = stat[:, 1:2]
            if self.reject == "ztest" and self.reject_gain is not None:
                mu = colour_hist
                sigma = (m2_hist - mu * mu).clamp_min(0.0).mean(dim=1, keepdim=True)
                sigma = (sigma + 1e-08).sqrt()
                gamma = F.softplus(self.reject_gamma) + 0.1
                z = (candidate - mu).abs().mean(dim=1, keepdim=True) / (
                    gamma * sigma + 0.001
                )
                retention_logit = retention_logit + F.softplus(self.reject_gain) * (
                    1.0 - 0.5 * z * z
                )
            elif self.reject == "variance" and self.reject_gain is not None:
                var = (m2_hist - colour_hist * colour_hist).clamp_min(0.0)
                var = (var.mean(dim=1, keepdim=True) + 1e-08).sqrt()
                var = var / (colour_hist.abs().mean(dim=1, keepdim=True) + 0.05)
                retention_logit = retention_logit - F.softplus(self.reject_gain) * var
            elif self.reject_gain is not None:
                mismatch = (candidate - colour_hist).abs().mean(dim=1, keepdim=True)
                mismatch = mismatch / (candidate.abs().mean(dim=1, keepdim=True) + 0.05)
                retention_logit = (
                    retention_logit - F.softplus(self.reject_gain) * mismatch
                )
            retention = torch.sigmoid(retention_logit)
            radius = phase.pow(2).sum(1, keepdim=True).sqrt() / scale
            if self.kernel == "none":
                weight = reliability
            else:
                width = F.softplus(self.kernel_width) + 0.25
                weight = torch.exp(-(radius / width).pow(2) * 0.5) * reliability
            if self.training or self.track_wbar:
                with torch.no_grad():
                    self.wbar_ema.mul_(0.98).add_(0.02 * weight.detach().float().mean())
            hist_colour = colour_hist
            if self.hist_clip == "var":
                ycc = self.rectifier.ycocg
                y_c = _rgb_to_ycocg(candidate) if ycc else candidate
                h_c = _rgb_to_ycocg(colour_hist) if ycc else colour_hist
                mu_c = F.avg_pool2d(
                    y_c, 3, stride=1, padding=1, count_include_pad=False
                )
                sq_c = F.avg_pool2d(
                    y_c * y_c, 3, stride=1, padding=1, count_include_pad=False
                )
                sigma_c = ((sq_c - mu_c * mu_c).clamp_min(0.0) + 1e-06).sqrt()
                gamma_c = 0.5 + F.softplus(
                    F.pixel_shuffle(self._head(self.head_clip, x), tscale)
                )
                clipped = torch.minimum(
                    torch.maximum(h_c, mu_c - gamma_c * sigma_c),
                    mu_c + gamma_c * sigma_c,
                )
                shift = clipped - h_c
                hist_colour = colour_hist + (_ycocg_to_rgb(shift) if ycc else shift)
                diag["clip_frac"] = (
                    (shift != 0).any(1, keepdim=True).to(valid.dtype) * valid
                ).sum().detach() / valid.sum().clamp_min(1.0).detach()
                diag["clip_gamma"] = gamma_c.mean().detach()
            s_hist = hist_colour * confidence
            w_hist = confidence
            cap = (
                self.accum_cap
                if self.accum_cap > 0.0
                else 1.0 / self.rectifier.alpha_min
            )
            if self.vcap_max > 0.0:
                if motion_hr is None:
                    motion_hr = (
                        F.interpolate(motion, size=hr_size, mode="nearest") * scale
                    )
                frames = self._vcap_frames(motion_hr.to(weight.dtype))
                cap = frames * weight.detach().mean(dim=(1, 2, 3), keepdim=True)
                diag["vcap_frames"] = frames.mean().detach()
            if self.static_gate > 0.0:
                spread = motion.flatten(2).std(dim=2).amax(dim=1)
                gate = (spread < self.static_gate).to(retention.dtype)
                gate = gate.view(-1, 1, 1, 1)
                retention = retention + gate * (1.0 - retention)
                cap = gate * self.static_cap + (1.0 - gate) * cap
            keep = retention * valid
            if self.gate_gain is not None:
                r_up = F.interpolate(gate_r, scale_factor=scale, mode="nearest").to(
                    keep.dtype
                )
                shrink = torch.exp(-F.softplus(self.gate_gain) * r_up)
                keep = keep * shrink
                diag["gate_shrink"] = (
                    (shrink * valid).sum() / valid.sum().clamp_min(1.0)
                ).detach()
            s_new = keep * s_hist + weight * candidate
            w_new = (keep * w_hist + weight).clamp_min(0.0001)
            if self.accum_alpha_min > 0.0:
                cap = torch.minimum(
                    torch.as_tensor(
                        cap, device=w_new.device, dtype=w_new.dtype
                    ).expand_as(w_new)
                    if not torch.is_tensor(cap)
                    else cap,
                    weight / self.accum_alpha_min,
                )
            excess = (w_new / cap).clamp_min(1.0)
            s_new = s_new / excess
            w_new = w_new / excess
            out = s_new / w_new
            alpha = weight / w_new
            new_confidence = w_new
            if self.readout != "accum":
                state_colour = out
                if self.readout == "residual":
                    delta = F.pixel_shuffle(self._head(self.head_delta, x), tscale)
                    out = out + delta
                    diag["delta_abs"] = delta.abs().mean().detach()
                else:
                    mu = F.softplus(
                        F.pixel_shuffle(self._head(self.head_mu, x), tscale)
                    )
                    out = (s_new + mu * candidate) / (w_new + mu)
                    diag["mu"] = mu.mean().detach()
            if diag or self.fp32_accum or self.vcap_max > 0.0:
                diag["w_mean"] = w_new.mean().detach()
                diag["alpha_mean"] = alpha.mean().detach()
                diag["ret_sat"] = (retention >= 0.999).to(w_new.dtype).mean().detach()
                if self.vcap_max > 0.0:
                    diag["vcap_bind"] = (excess > 1.0).to(w_new.dtype).mean().detach()
            m2_new = (
                (keep * m2_hist * w_hist + weight * candidate * candidate) / w_new
                if self.needs_m2
                else m2_hist
            )
            if self.first_order:
                m1_new = (keep * m1_hist * w_hist + weight * phase) / w_new
                pad = F.pad(out, (1, 1, 1, 1), mode="replicate")
                gy = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]) * 0.5
                gx = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]) * 0.5
                state_colour = out
                out = out - self.mls_gain * (m1_new[:, 0:1] * gy + m1_new[:, 1:2] * gx)
            else:
                m1_new = m1_hist
            disagreement = torch.zeros_like(w_new)
            new_reach = radius
        elif self.accumulate == "direct":
            out = candidate
            alpha = torch.full_like(confidence, float("nan"))
            new_confidence = torch.ones_like(confidence)
            disagreement = torch.zeros_like(confidence)
            new_reach = phase.pow(2).sum(1, keepdim=True).sqrt() / scale
            with torch.no_grad():
                dev = (out.detach().float() - colour_hist.detach().float()).abs().mean(
                    1, keepdim=True
                ) * valid.float()
                diag["hist_dev"] = dev.sum() / valid.float().sum().clamp_min(1.0)
                diag["hist_frac"] = valid.float().mean()
        if not smr and (
            not (
                self.accumulate == "normalized"
                and (self.first_order or self.readout != "accum")
            )
        ):
            state_colour = out
        new_fast = None
        if self.head_fast is not None:
            fast_logit = F.pixel_shuffle(self.head_fast(x), tscale)
            a_fast = self.dual_rate + (1.0 - self.dual_rate) * torch.sigmoid(fast_logit)
            new_fast = a_fast * candidate + (1.0 - a_fast) * fast_hist
            gap = (new_fast - out).abs().mean(dim=1, keepdim=True)
            keep_slow = torch.sigmoid(self.trust[0] + self.trust[1] * gap / 0.05)
            out = keep_slow * out + (1.0 - keep_slow) * new_fast
            aux_gap = gap
        if self.head_weight is None and self.accumulate != "direct":
            new_confidence = (
                1.0 + (1.0 - alpha.mean(1, keepdim=True)) * confidence
            ).clamp(0.0, 1.0 / self.rectifier.alpha_min)
        new_m2 = (
            (m2_new if self.accumulate == "normalized" else out * out)
            if self.needs_m2
            else out[:, :0]
        )
        if not (self.first_order and self.accumulate == "normalized"):
            m1_new = m1_hist
        if self.head_feature is not None and self.state_mode == "gru":
            ab = _hi(self._head(self.head_feature, x))
            gate_a = ab[:, : self.feature_dim]
            if gate_r is not None:
                r_state = F.max_pool2d(gate_r, kun) if kun > 1 else gate_r
                gate_a = gate_a + F.softplus(self.gru_gate_r) * r_state.to(gate_a.dtype)
            z = torch.sigmoid(gate_a)
            u = torch.tanh(ab[:, self.feature_dim :])
            new_feature = torch.lerp(_hi(feature_hist), u, z).clamp(-1.0, 1.0)
            diag["gru_z_mean"] = z.mean().detach()
            diag["gru_h_absmax"] = new_feature.abs().max().detach()
        else:
            new_feature = (
                self.head_feature(x)
                if self.head_feature is not None
                else torch.zeros(
                    batch, 0, theight, twidth, device=rgb.device, dtype=rgb.dtype
                )
            )
        if self.feature_bound > 0.0:
            new_feature = self.feature_bound * torch.tanh(
                new_feature / self.feature_bound
            )
        aux = {
            "alpha": alpha,
            "disagreement": disagreement,
            "candidate": candidate,
            "phase": phase,
        }
        if smr:
            aux.update(smr_aux)
        if gate_r is not None:
            aux["gate_r"] = gate_r
        if diag:
            aux["diag"] = diag
        if self.frames_in > 1:
            next_rgb = torch.cat((rgb, past_rgb[:, :-3]), dim=1)
            next_jitter = torch.cat((jitter.to(rgb.dtype), past_jitter[:, :-2]), dim=1)
        else:
            next_rgb = next_jitter = None
        if new_fast is None:
            new_fast = state_colour
        carried_jitter = jitter.float() if self.lr_rejitter else jitter.to(out.dtype)
        if self.state_guard and (not smr):
            (
                state_colour,
                new_confidence,
                new_feature,
                new_reach,
                new_fast,
                new_m2,
                m1_new,
                aux["guard_px"],
            ) = self._guard_state(
                state_colour,
                new_confidence,
                new_feature,
                new_reach,
                new_fast,
                new_m2,
                m1_new,
                skip,
                tscale,
            )
        elif self.state_guard and smr:
            (state_colour, new_confidence, new_feature, new_reach, aux["guard_px"]) = (
                self._guard_state_smr(
                    state_colour, new_confidence, new_feature, new_reach, tscale
                )
            )
            new_fast = state_colour
            diag["guard_px"] = aux["guard_px"]
        return (
            out,
            (
                state_colour,
                new_confidence,
                depth,
                new_feature,
                new_reach,
                next_rgb,
                next_jitter,
                new_fast,
                new_m2,
                carried_jitter,
                m1_new,
            ),
            aux,
        )

    state_guard = True
    STATE_GUARD_COLOUR = (-2.0, 3.0)
    STATE_GUARD_FEATURE = 32.0
    STATE_GUARD_DILATE = 1

    def _guard_state(self, colour, conf, feat, reach, fast, m2, m1, skip, scale: int):
        (lo, hi) = self.STATE_GUARD_COLOUR
        ok = ((colour >= lo) & (colour <= hi)).all(1, keepdim=True) & torch.isfinite(
            conf
        )
        for extra in (m2, m1):
            if extra.shape[1]:
                ok = ok & torch.isfinite(extra).all(1, keepdim=True)
        bad_lr = F.max_pool2d((~ok).to(colour.dtype), scale) > 0
        if feat.shape[1]:
            bad_lr = bad_lr | ~(feat.abs() <= self.STATE_GUARD_FEATURE).all(
                1, keepdim=True
            )
        d = int(self.STATE_GUARD_DILATE)
        if d > 0:
            bad_lr = (
                F.max_pool2d(bad_lr.to(colour.dtype), 2 * d + 1, stride=1, padding=d)
                > 0
            )
        if feat.shape[1]:
            feat = torch.where(bad_lr, 0.0, feat)
        bad = (
            F.interpolate(bad_lr.to(colour.dtype), scale_factor=scale, mode="nearest")
            > 0
        )
        reset = skip.clamp(0, 1).to(colour.dtype)
        same_fast = fast is colour
        colour = torch.where(bad, reset, colour)
        conf = torch.where(bad, 0.0, conf)
        reach = torch.where(bad, 1.0, reach)
        fast = colour if same_fast else torch.where(bad, reset.to(fast.dtype), fast)
        if m2.shape[1]:
            m2 = torch.where(bad, (reset * reset).to(m2.dtype), m2)
        if m1.shape[1]:
            m1 = torch.where(bad, 0.0, m1)
        return (colour, conf, feat, reach, fast, m2, m1, bad_lr.sum().detach())

    def _guard_state_smr(self, s, w, feat, reach, scale: int):
        ok = torch.isfinite(s).all(1, keepdim=True) & torch.isfinite(w)
        bad_lr = F.max_pool2d((~ok).to(w.dtype), scale) > 0
        if feat.shape[1]:
            bad_lr = bad_lr | ~(feat.abs() <= self.STATE_GUARD_FEATURE).all(
                1, keepdim=True
            )
        d = int(self.STATE_GUARD_DILATE)
        if d > 0:
            bad_lr = (
                F.max_pool2d(bad_lr.to(w.dtype), 2 * d + 1, stride=1, padding=d) > 0
            )
        if feat.shape[1]:
            feat = torch.where(bad_lr, 0.0, feat)
        bad = F.interpolate(bad_lr.to(w.dtype), scale_factor=scale, mode="nearest") > 0
        s = torch.where(bad, 0.0, s)
        w = torch.where(bad, 0.0, w)
        reach = torch.where(bad, 1.0, reach)
        return (s, w, feat, reach, bad_lr.sum().detach())

    def _head(self, conv: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.fp32_accum:
            return conv(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return conv(_hi(x))

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        motion: torch.Tensor,
        jitter: torch.Tensor,
        state: tuple | None = None,
        return_aux: bool = False,
        init_hr: torch.Tensor | None = None,
        init_mask: torch.Tensor | None = None,
    ):
        (outputs, auxes) = ([], [])
        for t in range(rgb.shape[1]):
            if t == 0 and state is None and (init_hr is not None):
                (out, state, aux) = self.step(
                    rgb[:, t],
                    depth[:, t],
                    motion[:, t],
                    jitter[:, t],
                    None,
                    init_hr=init_hr,
                    init_mask=init_mask,
                )
            else:
                (out, state, aux) = self.step(
                    rgb[:, t], depth[:, t], motion[:, t], jitter[:, t], state
                )
            outputs.append(out)
            if return_aux:
                auxes.append(aux)
        stacked = torch.stack(outputs, dim=1)
        return (stacked, state, auxes) if return_aux else (stacked, state)

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, dim={self.dim}, rectify={self.rectifier.mode}, recurrent={self.recurrent}, depth_edge={self.depth_edge}, stem={self.stem.kind}, jitter_head={self.jitter_head}, block={self.block_kind}, band_split={self.rectifier.band_split}, hr_refine={self.hr_refine}x{self.hr_layers}, frames_in={self.frames_in}, squeeze={getattr(self.body[0] if isinstance(self.body, nn.Sequential) and len(self.body) else None, 'gate', None) is not None}, dual_rate={self.dual_rate}, canonical={self.canonical_phase}, gamma_max={self.rectifier.gamma_max}, alpha_min={self.rectifier.alpha_min:.4f}"
            + (f", body={self.body_kind}" if self.body_kind != "plain" else "")
        )


def create_rrta(size: str = "t", scale: int = 4, **overrides) -> RRTANet:
    if size not in RRTA_SIZES:
        raise KeyError(f"unknown size {size!r}; available: {sorted(RRTA_SIZES)}")
    preset = dict(RRTA_SIZES[size])
    preset.update(overrides)
    return RRTANet(scale=scale, **preset)


def _selftest() -> None:
    torch.manual_seed(0)
    rect = RankRectifier(
        gamma_max=2.0, alpha_min=1.0 / 16.0, lam=4.0, ycocg=False
    ).double()
    (b, h, w) = (2, 9, 11)
    current = torch.rand(b, 3, h, w, dtype=torch.float64)
    history = torch.rand(b, 3, h, w, dtype=torch.float64) * 2 - 0.5
    valid = torch.ones(b, 1, h, w, dtype=torch.float64)
    big = 30.0
    weights = torch.zeros(b, 3, h, w, dtype=torch.float64)
    weights[:, 0] = big
    weights[:, 2] = big
    weights[:, 1] = -big
    gain = torch.zeros(b, 3, h, w, dtype=torch.float64)
    gain = torch.full((b, 3, h, w), -1.0986122886681098, dtype=torch.float64)
    (y_min, _, y_max) = RankRectifier.order_statistics(current)
    (out, alpha, _, _) = rect(
        current,
        history,
        weights,
        gain,
        torch.full((b, 1, h, w), -30.0, dtype=torch.float64),
        valid,
    )
    reference = torch.minimum(torch.maximum(history, y_min), y_max)
    recovered = (out - rect.alpha_min * current) / (1 - rect.alpha_min)
    error = (recovered - reference).abs().max().item()
    print(f"classic TAA clamp is a special case      max err {error:.2e}")
    assert error < 1e-09, error
    weights = torch.randn(b, 3, h, w, dtype=torch.float64)
    gain = torch.randn(b, 3, h, w, dtype=torch.float64)
    logit = torch.randn(b, 1, h, w, dtype=torch.float64) * 5
    (_, alpha, _, _) = rect(current, history, weights, gain, logit, valid)
    assert alpha.min() >= rect.alpha_min - 1e-12, alpha.min().item()
    assert alpha.max() <= 1.0 + 1e-12, alpha.max().item()
    print(
        f"alpha in [{rect.alpha_min:.4f}, 1]                     observed [{alpha.min():.4f}, {alpha.max():.4f}]"
    )
    (_, alpha0, _, _) = rect(
        current, history, weights, gain, logit, torch.zeros_like(valid)
    )
    assert (alpha0 - 1.0).abs().max() < 1e-12
    (out0, _, _, _) = rect(
        current, history, weights, gain, logit, torch.zeros_like(valid)
    )
    print(
        f"no history -> pure current sample        max err {(out0 - current).abs().max():.2e}"
    )
    assert (out0 - current).abs().max() < 1e-12
    y = _rgb_to_ycocg(current) if rect.ycocg else current
    (y_min, y_mu, y_max) = RankRectifier.order_statistics(y)
    spread = (y_max - y_min).clamp_min(1e-05)
    half = rect.gamma_max * torch.sigmoid(gain) * spread
    assert (2 * half <= 2 * rect.gamma_max * spread + 1e-12).all()
    print(f"band width <= 2*gamma_max*range          gamma_max={rect.gamma_max}")
    centre_w = torch.zeros(b, 3, h, w, dtype=torch.float64)
    centre_w[:, 1] = big
    (_, y_mu_c, _) = RankRectifier.order_statistics(current)
    (_, _, disagreement, _) = rect(current, y_mu_c, centre_w, gain, logit, valid)
    print(
        f"history inside the band is untouched     max excess {disagreement.abs().max():.2e}"
    )
    assert disagreement.abs().max() < 1e-12
    flat = torch.full((b, 3, h, w), 0.37, dtype=torch.float64)
    (out_flat, _, _, _) = rect(flat, flat, weights, gain, logit, valid)
    print(
        f"constant field is a fixed point          max err {(out_flat - 0.37).abs().max():.2e}"
    )
    assert (out_flat - 0.37).abs().max() < 1e-12
    net = create_rrta("t", 4)
    rgb = torch.rand(1, 3, 3, 16, 20, requires_grad=True)
    (out, _) = net(
        rgb,
        torch.rand(1, 3, 1, 16, 20),
        torch.randn(1, 3, 2, 16, 20),
        torch.rand(1, 3, 2) - 0.5,
    )
    out.mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert all((torch.isfinite(g).all() for g in grads))
    assert torch.isfinite(rgb.grad).all()
    print(f"finite gradients through {len(grads)} tensors over 3 steps")


if __name__ == "__main__":
    _selftest()
