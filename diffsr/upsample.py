from __future__ import annotations
import math
import torch
import torch.nn.functional as F

__all__ = ["lanczos_kernel", "lanczos_upsample"]


def lanczos_kernel(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    x = x.double()
    out = torch.where(
        x.abs() < 1e-12,
        torch.ones_like(x),
        a
        * torch.sin(math.pi * x)
        * torch.sin(math.pi * x / a)
        / (math.pi**2 * x * x).clamp_min(1e-300),
    )
    return torch.where(x.abs() < a, out, torch.zeros_like(x))


def _axis(x: torch.Tensor, scale: int, a: int, dim: int) -> torch.Tensor:
    n = x.shape[dim]
    dev = x.device
    out_idx = torch.arange(n * scale, device=dev, dtype=torch.float64)
    src = (out_idx + 0.5) / scale - 0.5
    base = torch.floor(src)
    taps = torch.arange(-a + 1, a + 1, device=dev, dtype=torch.float64)
    pos = base[:, None] + taps[None, :]
    w = lanczos_kernel(src[:, None] - pos, a)
    w = w / w.sum(dim=1, keepdim=True)
    idx = pos.clamp(0, n - 1).long()
    xt = x.movedim(dim, -1)
    gathered = xt[..., idx]
    y = (gathered * w.to(x.dtype)).sum(-1)
    return y.movedim(-1, dim)


def lanczos_upsample(
    x: torch.Tensor, scale: int, a: int = 3, clamp: bool = True
) -> torch.Tensor:
    xf = x.float()
    y = _axis(_axis(xf, scale, a, -1), scale, a, -2)
    y = y.to(x.dtype)
    return y.clamp(0.0, 1.0) if clamp else y


def _selftest() -> None:
    torch.manual_seed(0)
    x = torch.rand(1, 3, 17, 23)
    for s in (2, 3, 4):
        y = lanczos_upsample(x, s, clamp=False)
        assert y.shape[-2:] == (17 * s, 23 * s)
        c = lanczos_upsample(torch.full_like(x, 0.37), s, clamp=False)
        assert torch.allclose(c, torch.full_like(c, 0.37), atol=1e-06)
        ramp = (
            torch.arange(40, dtype=torch.float32).view(1, 1, 1, 40).expand(1, 1, 5, 40)
        )
        r = lanczos_upsample(ramp, s, clamp=False)[0, 0, 2]
        exp = (torch.arange(40 * s, dtype=torch.float32) + 0.5) / s - 0.5
        err = (r[4 * s : -4 * s] - exp[4 * s : -4 * s]).abs().max()
        assert err < 0.05, err
        print(f"x{s}: ramp max error {float(err):.4f} px (slope 1 px/px)")
        yb = F.interpolate(x, scale_factor=s, mode="bicubic", align_corners=False)
        print(f"x{s}: ok; mean |lanczos - bicubic| {float((y - yb).abs().mean()):.4f}")
    print("upsample selftest PASS")


if __name__ == "__main__":
    _selftest()
