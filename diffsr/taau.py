from __future__ import annotations
import math
from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F
from . import samples as smp

__all__ = [
    "BH_K",
    "VEL_ALPHA",
    "VEL_PX",
    "TAAUConfig",
    "TAAU",
    "rgb_to_ycocg",
    "ycocg_to_rgb",
    "reconstruct",
    "frame_context",
    "taau_update",
    "PIPELINE",
    "code_md5",
]
BH_K = 2.29
(VEL_ALPHA, VEL_PX) = (0.2, 40.0)
PIPELINE = {
    "qrisp_colour": "MipBiasJit",
    "motion_jit": False,
    "motion_lerp": 0.0,
    "depth_jit": False,
}


def rgb_to_ycocg(x: torch.Tensor) -> torch.Tensor:
    (r, g, b) = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
    return torch.cat(
        (0.25 * r + 0.5 * g + 0.25 * b, 0.5 * (r - b), -0.25 * r + 0.5 * g - 0.25 * b),
        dim=1,
    )


def ycocg_to_rgb(x: torch.Tensor) -> torch.Tensor:
    (y, co, cg) = (x[:, 0:1], x[:, 1:2], x[:, 2:3])
    return torch.cat((y + co - cg, y + cg, y - co - cg), dim=1)


@dataclass(frozen=True)
class TAAUConfig:
    alpha_min: float = 0.1
    gamma: float = 1.25
    luma_weight: bool = False
    blend: str = "decay"
    rho: float = 1.0
    dilate: bool = True
    velocity: bool = False
    depth_rel: float = 0.1
    clip_mode: str = "clip"
    taps: int = 5

    def __post_init__(self):
        if self.blend not in ("decay", "floor"):
            raise ValueError(f"blend must be 'decay' or 'floor', not {self.blend!r}")
        if self.clip_mode not in ("clip", "clamp"):
            raise ValueError("clip_mode must be 'clip' or 'clamp'")
        if not 0.0 < self.alpha_min < VEL_ALPHA:
            raise ValueError(f"alpha_min must be in (0, {VEL_ALPHA})")

    @property
    def sigma(self) -> float:
        return self.rho / math.sqrt(2.0 * BH_K)

    def group(self) -> tuple:
        return (
            float(self.rho),
            bool(self.dilate),
            int(self.taps),
            float(self.depth_rel),
            self.clip_mode,
        )

    def name(self) -> str:
        return f"{self.blend}_r{self.rho:g}_{('dil' if self.dilate else 'nodil')}_{('vel' if self.velocity else 'novel')}_a{self.alpha_min:.4g}_g{self.gamma:g}_{('luma' if self.luma_weight else 'noluma')}"

    def to_dict(self) -> dict:
        return asdict(self)


def reconstruct(
    lr: torch.Tensor, jitter: torch.Tensor, scale: int, rho: float, taps: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = rho / math.sqrt(2.0 * BH_K)
    (S, W) = smp.splat_gather(
        lr,
        jitter,
        scale,
        sigma=sigma,
        taps=taps,
        convention="image",
        normalize="analytic",
        border="zero",
        kernel="gauss",
    )
    c = S / W.clamp_min(1e-30)
    return (c, W)


def _dilated(
    motion: torch.Tensor, depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m = smp.dilate_motion(motion, depth, near_is_small=True)
    z = -F.max_pool2d(-depth, 3, stride=1, padding=1)
    return (m, z)


def frame_context(
    lr: torch.Tensor,
    depth: torch.Tensor,
    motion: torch.Tensor,
    jitter: torch.Tensor,
    prev_depth: torch.Tensor | None,
    scale: int,
    rhos=(1.0,),
    dilate_opts=(True,),
    depth_rel: float = 0.1,
    taps: int = 5,
) -> dict:
    (lr, depth, motion) = (lr.float(), depth.float(), motion.float())
    jitter = jitter.float()
    ctx = {
        "splat": {
            float(r): reconstruct(lr, jitter, scale, float(r), taps) for r in rhos
        },
        "moments": smp.sample_moments(
            rgb_to_ycocg(lr), jitter, scale, k=3, convention="image"
        ),
        "motion": {},
        "depth_rel": float(depth_rel),
        "taps": int(taps),
    }
    for dil in dilate_opts:
        (m, z) = _dilated(motion, depth) if dil else (motion, depth)
        m_hr = smp.motion_to_hr(m, scale)
        speed = m_hr.pow(2).sum(1, keepdim=True).sqrt()
        if prev_depth is None:
            valid = None
        else:
            (v_lr, _) = smp.depth_test(
                z, prev_depth.float(), m, rel=depth_rel, mode="far"
            )
            valid = F.interpolate(v_lr, scale_factor=scale, mode="nearest")
        ctx["motion"][bool(dil)] = (m_hr, valid, speed)
    return ctx


def _param(cfgs, key, device) -> torch.Tensor:
    return torch.tensor(
        [float(getattr(c, key)) for c in cfgs], dtype=torch.float32, device=device
    ).view(-1, 1, 1, 1)


def taau_update(
    cfgs: list[TAAUConfig],
    ctx: dict,
    state: dict | None,
    force_alpha: float | None = None,
) -> tuple[torch.Tensor, dict, dict]:
    if isinstance(cfgs, TAAUConfig):
        cfgs = [cfgs]
    g = {c.group() for c in cfgs}
    if len(g) != 1:
        raise ValueError(
            f"configs do not share (rho, dilate, taps, depth_rel, clip_mode): {g}"
        )
    (rho, dilate, taps, depth_rel, clip_mode) = g.pop()
    if taps != ctx["taps"] or depth_rel != ctx["depth_rel"]:
        raise ValueError("frame context built with other taps/depth_rel")
    n = len(cfgs)
    (c, w) = ctx["splat"][rho]
    (height, width) = c.shape[-2:]
    if state is None:
        out = c.clamp(0.0, 1.0).expand(n, -1, -1, -1).contiguous()
        return (
            out,
            {"color": out, "A": w.expand(n, -1, -1, -1).contiguous()},
            {"alpha_mean": [1.0] * n, "valid_frac": 0.0},
        )
    dev = c.device
    (m_hr, valid, speed) = ctx["motion"][dilate]
    (hist, inside) = smp.warp_catmull_rom(
        state["color"].reshape(1, n * 3, height, width),
        m_hr,
        premultiplied=False,
        clamp_to_neighbourhood=False,
    )
    hist = hist.view(n, 3, height, width)
    v = inside if valid is None else inside * valid
    ok = v > 0.5
    gamma = _param(cfgs, "gamma", dev)
    hy = smp.variance_clip(rgb_to_ycocg(hist), ctx["moments"], gamma, mode=clip_mode)
    hc = ycocg_to_rgb(hy)
    amin = _param(cfgs, "alpha_min", dev)
    vel = _param(cfgs, "velocity", dev) > 0.5
    floor = torch.where(
        vel, amin + (VEL_ALPHA - amin) * (speed / VEL_PX).clamp(0.0, 1.0), amin
    )
    (a_w, _) = smp.warp_bilinear(state["A"].reshape(1, n, height, width), m_hr)
    a_h = (1.0 - floor) * a_w.view(n, 1, height, width).clamp_min(0.0) * v
    alpha_decay = w / (a_h + w).clamp_min(1e-30)
    alpha_floor = torch.maximum(floor, w / (w + (1.0 / amin - 1.0)))
    decay = torch.tensor([c_.blend == "decay" for c_ in cfgs], device=dev).view(
        -1, 1, 1, 1
    )
    alpha = torch.where(decay, alpha_decay, alpha_floor)
    a_new = torch.where(decay, a_h + w, w.expand(n, -1, -1, -1))
    luma = torch.tensor([c_.luma_weight for c_ in cfgs], device=dev).view(-1, 1, 1, 1)
    if bool(luma.any()):
        yc = rgb_to_ycocg(c)[:, :1]
        wc = alpha / (1.0 + yc)
        wh = (1.0 - alpha) / (1.0 + hy[:, :1])
        alpha = torch.where(luma, wc / (wc + wh).clamp_min(1e-30), alpha)
    alpha = torch.where(ok, alpha, torch.ones_like(alpha))
    if force_alpha is not None:
        alpha = torch.full_like(alpha, float(force_alpha))
    out = torch.lerp(hc, c.expand_as(hc), alpha.expand_as(hc)).clamp(0.0, 1.0)
    aux = {
        "alpha_mean": alpha.flatten(1).mean(1).tolist(),
        "valid_frac": float(v.mean()),
    }
    return (out, {"color": out, "A": a_new}, aux)


def code_md5() -> dict:
    import hashlib
    from pathlib import Path

    here = Path(__file__).resolve()
    return {
        f"diffsr/{p.name}": hashlib.md5(p.read_bytes()).hexdigest()
        for p in (here, here.parent / "samples.py")
    }


class TAAU:
    def __init__(self, scale: int, cfg: TAAUConfig, force_alpha: float | None = None):
        self.scale = int(scale)
        self.cfg = cfg
        self.force_alpha = force_alpha
        self.dejitter = False
        self.static_gate = 0.0
        self.static_cap = 64.0
        self.stats = {
            "steps": 0,
            "nonfinite_steps": 0,
            "min": float("inf"),
            "max": float("-inf"),
        }

    def step(self, lr, depth, motion, jitter, state):
        if self.dejitter or self.static_gate:
            raise RuntimeError(
                "dejitter/static_gate overrides do not apply to the TAAU baseline"
            )
        if lr.shape[0] != 1:
            raise ValueError("TAAU.step expects batch 1 (one frame)")
        prev_depth = None if state is None else state["depth"]
        ctx = frame_context(
            lr,
            depth,
            motion,
            jitter,
            prev_depth,
            self.scale,
            rhos=(self.cfg.rho,),
            dilate_opts=(self.cfg.dilate,),
            depth_rel=self.cfg.depth_rel,
            taps=self.cfg.taps,
        )
        taau_state = (
            None if state is None else {"color": state["color"], "A": state["A"]}
        )
        (out, st, aux) = taau_update([self.cfg], ctx, taau_state, self.force_alpha)
        st["depth"] = depth.float()
        s = self.stats
        s["steps"] += 1
        if not bool(torch.isfinite(out).all()):
            s["nonfinite_steps"] += 1
        s["min"] = min(s["min"], float(out.min()))
        s["max"] = max(s["max"], float(out.max()))
        return (out, st, aux)

    def __repr__(self) -> str:
        return f"TAAU(x{self.scale}, {self.cfg})"


def _selftest() -> None:
    torch.manual_seed(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{('PASS' if cond else 'FAIL')}] {name} {detail}")

    x = torch.rand(2, 3, 5, 7, dtype=torch.float64)
    check(
        "YCoCg round trip", torch.allclose(ycocg_to_rgb(rgb_to_ycocg(x)), x, atol=1e-12)
    )
    cfg = TAAUConfig(rho=1.0)
    check("sigma = rho/sqrt(4.58)", abs(cfg.sigma - 1 / math.sqrt(4.58)) < 1e-12)
    (s, rho) = (4, 2.0)
    lr = torch.zeros(1, 3, 6, 6, dtype=torch.float64)
    lr[:, :, 3, 3] = 1.0
    jit = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    (S, W) = smp.splat_gather(lr, jit, s, sigma=rho / math.sqrt(2 * BH_K), taps=5)
    pos = smp.registered_sample_positions(s, jit, 6, 6)[0, :, 3, 3]
    (yy, xx) = (13, 15)
    d2 = (pos[0] - yy) ** 2 + (pos[1] - xx) ** 2
    sig2 = (rho / math.sqrt(2 * BH_K)) ** 2
    expect = math.exp(-BH_K * float(d2) / rho**2) * s * s / (2 * math.pi * sig2)
    check(
        "kernel is exp(-2.29 (d/rho)^2)",
        abs(float(S[0, 0, yy, xx]) - expect) < 1e-12,
        f"{float(S[0, 0, yy, xx]):.6g} vs {expect:.6g}",
    )
    lr = torch.full((1, 3, 8, 8), 0.37, dtype=torch.float64)
    (c_, w_) = reconstruct(lr, jit, s, 1.0)
    check("constant image", torch.allclose(c_, torch.full_like(c_, 0.37)))
    (h, w) = (12, 16)
    model = TAAU(4, TAAUConfig(alpha_min=0.1, gamma=1.0), force_alpha=1.0)
    st = None
    err = 0.0
    for t in range(3):
        lr = torch.rand(1, 3, h, w)
        dep = torch.rand(1, 1, h, w) * 0.1 + 0.01
        mot = torch.randn(1, 2, h, w) * 0.5
        jt = torch.rand(1, 2) - 0.5
        (out, st, _) = model.step(lr, dep, mot, jt, st)
        (c_ref, _) = reconstruct(lr, jt, 4, 1.0)
        err = max(err, float((out - c_ref.clamp(0, 1)).abs().max()))
    check("force_alpha=1 == spatial reconstruction", err == 0.0, f"max {err:.2e}")
    cfgs = [
        TAAUConfig(alpha_min=a, gamma=gm, luma_weight=lw, blend=b, velocity=vv)
        for (a, gm, lw, b, vv) in (
            (1 / 16, 1.0, False, "decay", False),
            (0.1, 1.5, True, "floor", True),
            (0.125, 1.25, True, "decay", True),
            (1 / 16, 1.25, False, "floor", False),
        )
    ]
    (sb, ss) = (None, [None] * len(cfgs))
    prev = None
    worst = 0.0
    for t in range(6):
        lr = torch.rand(1, 3, h, w)
        dep = torch.rand(1, 1, h, w) * 0.1 + 0.01
        mot = torch.randn(1, 2, h, w) * 0.7
        jt = torch.rand(1, 2) - 0.5
        ctx = frame_context(lr, dep, mot, jt, prev, 4)
        (ob, sb, _) = taau_update(cfgs, ctx, sb)
        for k, cfg_ in enumerate(cfgs):
            (o1, ss[k], _) = taau_update([cfg_], ctx, ss[k])
            worst = max(worst, float((o1[0] - ob[k]).abs().max()))
        prev = dep
    check("batched == per-config", worst < 1e-05, f"max {worst:.2e}")
    for blend in ("decay", "floor"):
        model = TAAU(4, TAAUConfig(alpha_min=1 / 16, blend=blend, gamma=1.5))
        hr = torch.rand(1, 3, 4 * h, 4 * w)
        st = None
        fin = True
        for t in range(20):
            jt = torch.rand(1, 2) - 0.5
            pos = smp.registered_sample_positions(4, jt, h, w)[0]
            iy = pos[0].round().long().clamp(0, 4 * h - 1)
            ix = pos[1].round().long().clamp(0, 4 * w - 1)
            lr = hr[0][:, iy, ix][None]
            (out, st, aux) = model.step(
                lr, torch.ones(1, 1, h, w) * 0.05, torch.zeros(1, 2, h, w), jt, st
            )
            fin &= (
                bool(torch.isfinite(out).all())
                and float(out.min()) >= 0
                and (float(out.max()) <= 1)
            )
        check(
            f"{blend}: finite, in [0,1], history valid",
            fin and aux["valid_frac"] > 0.9,
            f"alpha_mean {aux['alpha_mean'][0]:.3f}",
        )
    print("ALL PASS" if ok else "SOME TESTS FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
