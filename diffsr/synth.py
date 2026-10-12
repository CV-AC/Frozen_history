from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable
import numpy as np

__all__ = [
    "render_pair",
    "build_suite",
    "load_suite",
    "peak_recovery",
    "local_extremum_violation",
    "edge_position",
    "band_error",
    "PATTERNS",
]


def _normal(angle_deg: float) -> tuple[float, float]:
    theta = math.radians(angle_deg)
    return (math.sin(theta), math.cos(theta))


def _signed_distance(
    y: np.ndarray, x: np.ndarray, angle: float, phase: float, size: float
) -> np.ndarray:
    (ny, nx) = _normal(angle)
    centre = size / 2.0
    return (y[:, None] - centre) * ny + (x[None, :] - centre) * nx - phase


def p_line(y, x, size, angle=0.0, width=1.0, phase=0.0):
    return (np.abs(_signed_distance(y, x, angle, phase, size)) <= width / 2.0).astype(
        np.float32
    )


def p_step(y, x, size, angle=0.0, phase=0.0):
    return (_signed_distance(y, x, angle, phase, size) >= 0.0).astype(np.float32)


def p_bars(y, x, size, period=4.0, angle=0.0, duty=0.5, phase=0.0):
    d = _signed_distance(y, x, angle, phase, size)
    return (d / period % 1.0 < duty).astype(np.float32)


def p_checker(y, x, size, period=4.0, angle=0.0, phase=0.0):
    (ny, nx) = _normal(angle)
    centre = size / 2.0
    u = (y[:, None] - centre) * ny + (x[None, :] - centre) * nx - phase
    v = (y[:, None] - centre) * nx - (x[None, :] - centre) * ny - phase
    return ((np.floor(u / period) + np.floor(v / period)) % 2.0).astype(np.float32)


def p_zoneplate(y, x, size, kmax=1.0):
    centre = size / 2.0
    radius = np.hypot(y[:, None] - centre, x[None, :] - centre)
    phi = np.pi * kmax * radius**2 / (2.0 * centre)
    return (0.5 + 0.5 * np.cos(phi)).astype(np.float32)


def p_concentric(y, x, size, period=6.0):
    centre = size / 2.0
    radius = np.hypot(y[:, None] - centre, x[None, :] - centre)
    return (0.5 + 0.5 * np.cos(2.0 * np.pi * radius / period)).astype(np.float32)


def p_siemens(y, x, size, spokes=32):
    centre = size / 2.0
    theta = np.arctan2(y[:, None] - centre, x[None, :] - centre)
    return (np.cos(spokes * theta) >= 0.0).astype(np.float32)


def p_disk(y, x, size, radius=8.0, phase=0.0):
    centre = size / 2.0
    dist = np.hypot(y[:, None] - centre - phase, x[None, :] - centre - phase)
    return (dist <= radius).astype(np.float32)


def p_corner(y, x, size, angle=0.0, aperture=30.0, phase=0.0):
    first = _signed_distance(y, x, angle, phase, size) >= 0.0
    second = _signed_distance(y, x, angle + aperture, phase, size) < 0.0
    return (first & second).astype(np.float32)


PATTERNS: dict[str, Callable[..., np.ndarray]] = {
    "line": p_line,
    "step": p_step,
    "bars": p_bars,
    "checker": p_checker,
    "zoneplate": p_zoneplate,
    "concentric": p_concentric,
    "siemens": p_siemens,
    "disk": p_disk,
    "corner": p_corner,
}


def render_pair(
    pattern: str, hr_size: int = 256, scale: int = 4, ss: int = 8, **params: Any
) -> tuple[np.ndarray, np.ndarray]:
    if hr_size % scale:
        raise ValueError(f"hr_size {hr_size} is not a multiple of scale {scale}")
    fn = PATTERNS[pattern]
    n = hr_size * ss
    coords = (np.arange(n, dtype=np.float64) + 0.5) / ss
    field = fn(coords, coords, float(hr_size), **params).astype(np.float32)
    hr = field.reshape(hr_size, ss, hr_size, ss).mean(axis=(1, 3))
    lr_size = hr_size // scale
    block = ss * scale
    lr = field.reshape(lr_size, block, lr_size, block).mean(axis=(1, 3))
    return (hr.astype(np.float32), lr.astype(np.float32))


def default_specs(scale: int = 4) -> list[dict[str, Any]]:
    angles = [a * 180.0 / 16.0 for a in range(16)]
    phases = [p / 8.0 for p in range(8)]
    specs: list[dict[str, Any]] = []
    for angle in angles:
        for phase in phases:
            specs.append(dict(pattern="line", angle=angle, phase=phase, width=1.0))
            specs.append(dict(pattern="step", angle=angle, phase=phase))
    for angle in angles[::2]:
        for period in (3.0, 4.0, 6.0, 10.0):
            specs.append(
                dict(pattern="bars", angle=angle, period=period, duty=0.5, phase=0.0)
            )
            specs.append(dict(pattern="checker", angle=angle, period=period, phase=0.0))
    for kmax in (0.75, 1.0):
        specs.append(dict(pattern="zoneplate", kmax=kmax))
    for period in (4.0, 6.0, 10.0):
        specs.append(dict(pattern="concentric", period=period))
    for spokes in (16, 32, 64):
        specs.append(dict(pattern="siemens", spokes=spokes))
    for radius in (6.0, 12.0, 24.0):
        for phase in (0.0, 0.5):
            specs.append(dict(pattern="disk", radius=radius, phase=phase))
    for angle in angles[::4]:
        for aperture in (15.0, 30.0, 60.0):
            specs.append(
                dict(pattern="corner", angle=angle, aperture=aperture, phase=0.0)
            )
    return specs


def _spec_id(index: int, spec: dict[str, Any]) -> str:
    tail = "_".join(
        (
            f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}"
            for (k, v) in sorted(spec.items())
            if k != "pattern"
        )
    )
    return (
        f"{index:04d}_{spec['pattern']}_{tail}"
        if tail
        else f"{index:04d}_{spec['pattern']}"
    )


def build_suite(
    out_dir: str | Path,
    hr_size: int = 256,
    scale: int = 4,
    ss: int = 8,
    specs: list[dict[str, Any]] | None = None,
    png: bool = False,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = specs or default_specs(scale)
    arrays: dict[str, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        params = {k: v for (k, v) in spec.items() if k != "pattern"}
        (hr, lr) = render_pair(spec["pattern"], hr_size, scale, ss, **params)
        key = _spec_id(index, spec)
        arrays[f"{key}/hr"] = hr
        arrays[f"{key}/lr"] = lr
        manifest.append(dict(id=key, **spec))
    path = out / f"suite_x{scale}.npz"
    np.savez_compressed(path, **arrays)
    meta = dict(
        hr_size=hr_size,
        scale=scale,
        supersample=ss,
        count=len(specs),
        items=manifest,
        note="area-averaged from a common continuous field; diagnostic use only, never for tuning",
    )
    (out / f"suite_x{scale}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    if png:
        _write_previews(out / f"preview_x{scale}", arrays)
    return path


def _write_previews(directory: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not available; skipping PNG previews")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        name = key.replace("/", "__") + ".png"
        data = np.clip(value * 255.0 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(data).save(directory / name)


def load_suite(
    path: str | Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    path = Path(path)
    data = np.load(path)
    items: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        (name, field) = key.rsplit("/", 1)
        items.setdefault(name, {})[field] = data[key]
    meta_path = path.with_suffix(".json")
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    )
    return (items, meta)


def _shift_extrema(image: np.ndarray, radius: int, reduce: Callable) -> np.ndarray:
    out = image.copy()
    padded = np.pad(image, radius, mode="edge")
    (height, width) = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            window = padded[
                radius + dy : radius + dy + height, radius + dx : radius + dx + width
            ]
            out = reduce(out, window)
    return out


def peak_recovery(
    pred: np.ndarray, hr: np.ndarray, base: np.ndarray, quantile: float = 1.0
) -> float:

    def top(a: np.ndarray) -> float:
        return float(a.max()) if quantile >= 1.0 else float(np.quantile(a, quantile))

    gain = top(hr) - top(base)
    if abs(gain) < 1e-08:
        return float("nan")
    return (top(pred) - top(base)) / gain


def local_extremum_violation(
    pred: np.ndarray, hr: np.ndarray, radius: int = 1
) -> dict[str, float]:
    upper = _shift_extrema(hr, radius, np.maximum)
    lower = _shift_extrema(hr, radius, np.minimum)
    over = np.maximum(pred - upper, 0.0)
    under = np.maximum(lower - pred, 0.0)
    return {
        "over": float(over.mean()),
        "under": float(under.mean()),
        "lev": float((over + under).mean()),
        "over_max": float(over.max()),
        "violating_fraction": float((over + under > 0.0001).mean()),
    }


def edge_position(image: np.ndarray, angle: float, level: float = 0.5) -> float:
    (height, width) = image.shape
    (ny, nx) = _normal(angle)
    (centre_y, centre_x) = (height / 2.0, width / 2.0)
    rows = np.arange(height) + 0.5 - centre_y
    cols = np.arange(width) + 0.5 - centre_x
    dist = rows[:, None] * ny + cols[None, :] * nx
    limit = float(np.abs(dist).max())
    edges = np.arange(-limit, limit + 0.25, 0.25)
    index = np.clip(np.digitize(dist.ravel(), edges) - 1, 0, len(edges) - 2)
    total = np.bincount(
        index, weights=image.ravel().astype(np.float64), minlength=len(edges) - 1
    )
    count = np.bincount(index, minlength=len(edges) - 1)
    keep = count > 0
    profile = total[keep] / count[keep]
    centres = (0.5 * (edges[:-1] + edges[1:]))[keep]
    (low, high) = (profile.min(), profile.max())
    if high - low < 1e-06:
        return float("nan")
    target = low + level * (high - low)
    crossings = np.where(np.diff(np.sign(profile - target)) != 0)[0]
    if crossings.size == 0:
        return float("nan")
    i = crossings[len(crossings) // 2]
    (p0, p1) = (profile[i], profile[i + 1])
    t = 0.0 if abs(p1 - p0) < 1e-12 else (target - p0) / (p1 - p0)
    return float(centres[i] + t * (centres[i + 1] - centres[i]))


def band_error(pred: np.ndarray, hr: np.ndarray, bands: int = 8) -> np.ndarray:
    error = np.fft.fftshift(
        np.fft.fft2(pred.astype(np.float64) - hr.astype(np.float64))
    )
    power = np.abs(error) ** 2
    (height, width) = power.shape
    rows = (np.arange(height) - height / 2.0) / (height / 2.0)
    cols = (np.arange(width) - width / 2.0) / (width / 2.0)
    radius = np.hypot(rows[:, None], cols[None, :])
    index = np.clip((radius * bands).astype(int), 0, bands - 1)
    out = np.zeros(bands)
    for band in range(bands):
        mask = index == band
        out[band] = power[mask].mean() if mask.any() else 0.0
    return out / max(power.mean(), 1e-30)


def _selftest() -> None:
    failures = []
    (hr, lr) = render_pair("line", 64, 4, 8, angle=0.0, width=1000000000.0, phase=0.0)
    if abs(hr.min() - 1.0) > 1e-06 or abs(lr.min() - 1.0) > 1e-06:
        failures.append("constant field")
    print(f"ok   constant field: hr={hr.min():.6f} lr={lr.min():.6f}")
    (hr, lr) = render_pair("disk", 64, 4, 8, radius=11.0, phase=0.0)
    pooled = hr.reshape(16, 4, 16, 4).mean(axis=(1, 3))
    gap = float(np.abs(pooled - lr).max())
    print(f"ok   LR == 4x4 block mean of HR: max |delta| = {gap:.2e}")
    if gap > 1e-06:
        failures.append("nesting")
    (hr, lr) = render_pair("line", 64, 4, 16, angle=0.0, width=1.0, phase=0.0)
    print(
        f"ok   1px line: hr peak {hr.max():.3f}, lr peak {lr.max():.3f} (attenuation {1.0 - lr.max() / hr.max():.1%})"
    )
    if lr.max() >= hr.max() - 0.001:
        failures.append("line attenuation")
    for phase in (0.0, 0.25, 0.5):
        (hr, _) = render_pair("step", 64, 4, 16, angle=30.0, phase=phase)
        found = edge_position(hr, 30.0)
        print(
            f"ok   edge at phase {phase:.2f} -> measured {found:+.3f} (error {found - phase:+.3f} px)"
        )
        if not math.isfinite(found) or abs(found - phase) > 0.15:
            failures.append(f"edge phase {phase}")
    (hr, _) = render_pair("step", 64, 4, 8, angle=45.0, phase=0.0)
    exact = local_extremum_violation(hr, hr)
    sharp = hr + 0.5 * (
        hr
        - 0.25
        * (
            np.roll(hr, 1, 0)
            + np.roll(hr, -1, 0)
            + np.roll(hr, 1, 1)
            + np.roll(hr, -1, 1)
        )
    )
    rung = local_extremum_violation(sharp, hr)
    print(f"ok   LEV: exact {exact['lev']:.2e}, unsharp-masked {rung['lev']:.4f}")
    if exact["lev"] > 1e-06 or rung["lev"] <= exact["lev"]:
        failures.append("lev")
    print()
    print("all passed" if not failures else f"failed: {failures}")
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--hr", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--ss",
        type=int,
        default=8,
        help="supersamples per HR pixel (8 to measure, 16 for figures)",
    )
    parser.add_argument("--png", action="store_true", help="also write 8-bit previews")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
        return
    path = build_suite(args.out, args.hr, args.scale, args.ss, png=args.png)
    (items, meta) = load_suite(path)
    print(f"wrote {len(items)} pairs to {path}")
    print(
        f"  HR {meta['hr_size']}px, LR {meta['hr_size'] // meta['scale']}px, x{meta['scale']}, {meta['supersample']} supersamples/HR px"
    )


if __name__ == "__main__":
    main()
