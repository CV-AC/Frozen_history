from __future__ import annotations
import dataclasses
import torch
import torch.nn.functional as F
from . import samples as smp
from . import taau

__all__ = ["minmax_box", "frame_context", "TAAClamp", "clamp_config"]


def clamp_config(**kw) -> taau.TAAUConfig:
    kw = dict(kw)
    kw["gamma"] = 1.0
    kw["clip_mode"] = "clamp"
    return taau.TAAUConfig(**kw)


def minmax_box(
    lr_ycocg: torch.Tensor, jitter, scale: int, k: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    r = k // 2
    xp = F.pad(lr_ycocg, (r, r, r, r), mode="replicate")
    mx = F.max_pool2d(xp, k, stride=1)
    mn = -F.max_pool2d(-xp, k, stride=1)
    c = lr_ycocg.shape[1]
    both = smp.gather_nearest(torch.cat((mn, mx), 1), jitter, scale, "image")
    (mn_h, mx_h) = (both[:, :c], both[:, c:])
    return (0.5 * (mn_h + mx_h), 0.5 * (mx_h - mn_h).clamp_min(0.0))


def frame_context(
    lr,
    depth,
    motion,
    jitter,
    prev_depth,
    scale,
    rhos=(1.0,),
    dilate_opts=(True,),
    depth_rel=0.1,
    taps=5,
) -> dict:
    ctx = taau.frame_context(
        lr,
        depth,
        motion,
        jitter,
        prev_depth,
        scale,
        rhos=rhos,
        dilate_opts=dilate_opts,
        depth_rel=depth_rel,
        taps=taps,
    )
    ctx["moments"] = minmax_box(taau.rgb_to_ycocg(lr.float()), jitter.float(), scale)
    ctx["rectification"] = "minmax_clamp"
    return ctx


class TAAClamp(taau.TAAU):
    def __init__(
        self, scale: int, cfg: taau.TAAUConfig, force_alpha: float | None = None
    ):
        if cfg.gamma != 1.0 or cfg.clip_mode != "clamp":
            cfg = dataclasses.replace(cfg, gamma=1.0, clip_mode="clamp")
        super().__init__(scale, cfg, force_alpha)

    def step(self, lr, depth, motion, jitter, state):
        if lr.shape[0] != 1:
            raise ValueError("TAAClamp.step expects batch 1 (one frame)")
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
        st_in = None if state is None else {"color": state["color"], "A": state["A"]}
        (out, st, aux) = taau.taau_update([self.cfg], ctx, st_in, self.force_alpha)
        st["depth"] = depth.float()
        s = self.stats
        s["steps"] += 1
        if not bool(torch.isfinite(out).all()):
            s["nonfinite_steps"] += 1
        s["min"] = min(s["min"], float(out.min()))
        s["max"] = max(s["max"], float(out.max()))
        return (out, st, aux)

    def __repr__(self) -> str:
        return f"TAAClamp(x{self.scale}, {self.cfg})"


def _selftest() -> None:
    torch.manual_seed(0)
    (h, w, s) = (10, 12, 2)
    lr = torch.rand(1, 3, h, w)
    jit = torch.tensor([[0.1, -0.2]])
    y = taau.rgb_to_ycocg(lr)
    (mu, half) = minmax_box(y, jit, s)
    hist = torch.randn(1, 3, h * s, w * s)
    got = smp.variance_clip(hist, (mu, half), 1.0, mode="clamp")
    (lo, hi) = (mu - half, mu + half)
    ref = torch.minimum(torch.maximum(hist, lo), hi)
    assert torch.allclose(got, ref, atol=1e-06)
    near = smp.gather_nearest(y, jit, s)
    assert bool(((near >= lo - 1e-06) & (near <= hi + 1e-06)).all())
    m = TAAClamp(s, clamp_config(alpha_min=0.1))
    st = None
    for _ in range(4):
        (out, st, aux) = m.step(
            torch.rand(1, 3, h, w),
            torch.rand(1, 1, h, w) * 0.1 + 0.01,
            torch.randn(1, 2, h, w) * 0.3,
            torch.rand(1, 2) - 0.5,
            st,
        )
        assert bool(torch.isfinite(out).all())
    print("taa_clamp selftest PASS", m)


if __name__ == "__main__":
    _selftest()
