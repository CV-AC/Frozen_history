from __future__ import annotations
from typing import Any
import numpy as np
from .synth import PATTERNS

__all__ = ["halton_jitter", "render_sequence", "lr_nyquist_band"]


def halton_jitter(count: int, bases: tuple[int, int] = (2, 3)) -> np.ndarray:

    def radical(index: int, base: int) -> float:
        (out, fraction) = (0.0, 1.0 / base)
        while index:
            (index, digit) = divmod(index, base)
            out += digit * fraction
            fraction /= base
        return out

    return np.array(
        [
            [radical(i + 1, bases[0]) - 0.5, radical(i + 1, bases[1]) - 0.5]
            for i in range(count)
        ],
        dtype=np.float32,
    )


def render_sequence(
    pattern: str,
    frames: int = 24,
    hr_size: int = 256,
    scale: int = 4,
    ss: int = 8,
    velocity: tuple[float, float] = (0.35, 0.72),
    jitter: np.ndarray | None = None,
    depth_value: float = 0.5,
    lr_filter: str = "point",
    **params: Any,
) -> dict[str, np.ndarray]:
    if hr_size % scale:
        raise ValueError(f"hr_size {hr_size} is not a multiple of {scale}")
    fn = PATTERNS[pattern]
    lr_size = hr_size // scale
    jitter = halton_jitter(frames) if jitter is None else np.asarray(jitter)
    n_hr = hr_size * ss
    hr_step = 1.0 / ss
    hr_coords = (np.arange(n_hr, dtype=np.float64) + 0.5) * hr_step
    block = ss * scale
    hr_out = np.empty((frames, 1, hr_size, hr_size), dtype=np.float32)
    lr_out = np.empty((frames, 1, lr_size, lr_size), dtype=np.float32)
    for t in range(frames):
        shift_y = velocity[0] * t
        shift_x = velocity[1] * t
        field = fn(
            hr_coords - shift_y, hr_coords - shift_x, float(hr_size), **params
        ).astype(np.float64)
        hr_out[t, 0] = field.reshape(hr_size, ss, hr_size, ss).mean(axis=(1, 3))
        jy = float(jitter[t, 0]) * scale
        jx = float(jitter[t, 1]) * scale
        if lr_filter == "box":
            field_lr = fn(
                hr_coords - shift_y + jy,
                hr_coords - shift_x + jx,
                float(hr_size),
                **params,
            ).astype(np.float64)
            lr_out[t, 0] = field_lr.reshape(lr_size, block, lr_size, block).mean(
                axis=(1, 3)
            )
        elif lr_filter == "point":
            centres = (np.arange(lr_size, dtype=np.float64) + 0.5) * scale
            lr_out[t, 0] = fn(
                centres - shift_y + jy, centres - shift_x + jx, float(hr_size), **params
            ).astype(np.float64)
        else:
            raise ValueError(f"lr_filter must be 'point' or 'box', got {lr_filter!r}")
    motion = np.zeros((frames, 2, lr_size, lr_size), dtype=np.float32)
    motion[:, 0] = -velocity[0] / scale
    motion[:, 1] = -velocity[1] / scale
    motion[0] = 0.0
    depth = np.full((frames, 1, lr_size, lr_size), depth_value, dtype=np.float32)
    return {
        "hr": hr_out,
        "lr": lr_out,
        "motion": motion,
        "depth": depth,
        "jitter": jitter.astype(np.float32),
    }


def lr_nyquist_band(scale: int, bands: int) -> float:
    return bands / float(scale)


def oracle_accumulation(
    seq: dict[str, np.ndarray], scale: int, velocity: tuple[float, float]
) -> np.ndarray:
    from scipy.interpolate import griddata

    lr = seq["lr"][:, 0]
    jitter = seq["jitter"]
    (frames, lr_size, _) = lr.shape
    hr_size = lr_size * scale
    index = np.arange(lr_size, dtype=np.float64)
    (points, values) = ([], [])
    for t in range(frames):
        y = (index + 0.5 + jitter[t, 0]) * scale - velocity[0] * t
        x = (index + 0.5 + jitter[t, 1]) * scale - velocity[1] * t
        (gy, gx) = np.meshgrid(y, x, indexing="ij")
        points.append(np.stack((gy.ravel(), gx.ravel()), axis=1))
        values.append(lr[t].ravel())
    points = np.concatenate(points)
    values = np.concatenate(values)
    grid = np.arange(hr_size, dtype=np.float64) + 0.5
    (ty, tx) = np.meshgrid(grid, grid, indexing="ij")
    out = griddata(points, values, (ty, tx), method="linear")
    return out


def _selftest() -> None:
    import torch
    import torch.nn.functional as F
    from .synth import band_error

    (scale, hr_size, frames) = (4, 256, 32)
    velocity = (0.35, 0.72)
    for lr_filter in ("point", "box"):
        _premise(lr_filter, scale, hr_size, frames, velocity, kmax=0.5)


def _premise(lr_filter, scale, hr_size, frames, velocity, kmax=0.5):
    import torch
    import torch.nn.functional as F
    from .synth import band_error

    seq = render_sequence(
        "zoneplate",
        frames=frames,
        hr_size=hr_size,
        scale=scale,
        velocity=velocity,
        kmax=kmax,
        lr_filter=lr_filter,
    )
    reference = seq["hr"][0, 0]
    one = F.interpolate(
        torch.from_numpy(seq["lr"][0:1]),
        scale_factor=scale,
        mode="bicubic",
        align_corners=False,
    )[0, 0].numpy()
    many = oracle_accumulation(seq, scale, velocity)
    filled = ~np.isnan(many)
    coverage = filled.mean()
    many = np.where(filled, many, one)
    valid = filled
    bands = 16
    e_one = band_error(np.where(valid, one, reference), reference, bands)
    e_many = band_error(np.where(valid, many, reference), reference, bands)
    edge = int(lr_nyquist_band(scale, bands))
    print(f"\n=== lr_filter={lr_filter} ===")
    print(
        f"zone plate, x{scale}, {frames} jittered frames, splat coverage {coverage:.1%}"
    )
    print(f"low-resolution Nyquist is band {edge} of {bands}\n")
    top = min(bands, int(round(edge * 2 * kmax / 0.5)))
    above_one = e_one[edge:top].mean()
    above_many = e_many[edge:top].mean()
    print(
        f"  recoverable band = {edge}..{top - 1} (LR Nyquist to the content bandwidth)"
    )
    print(
        f"  mean error there: one frame {above_one:.4f}, accumulated {above_many:.4f}"
    )
    ratio = above_one / max(above_many, 1e-09)
    print(f"  accumulation is {ratio:.2f}x lower in error above the LR Nyquist")
    if lr_filter == "point":
        assert above_many < above_one, (
            "accumulating jittered point samples did not recover super-Nyquist content; the sequence construction is wrong"
        )


if __name__ == "__main__":
    _selftest()
