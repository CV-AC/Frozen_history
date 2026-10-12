from __future__ import annotations
import numpy as np

__all__ = ["imresize", "cubic"]


def cubic(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    inner = (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
    outer = (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0) * (
        (absx > 1.0) & (absx <= 2.0)
    )
    return inner + outer


def _weights(
    in_length: int,
    out_length: int,
    scale: float,
    kernel_width: float,
    antialiasing: bool,
):
    if scale < 1 and antialiasing:
        kernel_width = kernel_width / scale
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - kernel_width / 2.0)
    taps = int(np.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(taps, dtype=np.float64)[None, :]
    distance = u[:, None] - indices
    weights = (
        scale * cubic(distance * scale)
        if scale < 1 and antialiasing
        else cubic(distance)
    )
    weights = weights / weights.sum(axis=1, keepdims=True)
    indices = indices.astype(np.int64) - 1
    indices = np.where(indices < 0, -indices - 1, indices)
    over = indices > in_length - 1
    indices = np.where(over, 2 * in_length - indices - 1, indices)
    indices = np.clip(indices, 0, in_length - 1)
    keep = ~np.all(weights == 0, axis=0)
    return (indices[:, keep], weights[:, keep])


def imresize(image: np.ndarray, scale: float, antialiasing: bool = True) -> np.ndarray:
    as_uint8 = image.dtype == np.uint8
    work = image.astype(np.float64)
    squeeze = work.ndim == 2
    if squeeze:
        work = work[:, :, None]
    out_h = int(np.ceil(work.shape[0] * scale))
    out_w = int(np.ceil(work.shape[1] * scale))
    (indices, weights) = _weights(work.shape[0], out_h, scale, 4.0, antialiasing)
    work = np.einsum("ijkl,ij->ikl", work[indices], weights)
    if as_uint8:
        work = np.clip(np.round(work), 0, 255)
    (indices, weights) = _weights(work.shape[1], out_w, scale, 4.0, antialiasing)
    work = np.einsum("ijkl,jk->ijl", work[:, indices], weights)
    if squeeze:
        work = work[:, :, 0]
    if as_uint8:
        return np.clip(np.round(work), 0, 255).astype(np.uint8)
    return work


def _verify(limit: int = 12) -> None:
    import json
    from pathlib import Path
    import cv2
    from .paths import DATA_ROOT

    root = DATA_ROOT / "DIV2K" / "train"
    entry = json.loads((root / "pairs.json").read_text())
    worst = {}
    for scale in (2, 3, 4):
        info = entry["scales"][str(scale)]
        (diffs, exact, total) = ([], 0, 0)
        for hr_name, lr_name in info["pairs"][:limit]:
            hr = cv2.imread(str(root / "HR" / hr_name), cv2.IMREAD_COLOR)
            ref = cv2.imread(
                str(root / "LR_bicubic" / f"X{scale}" / lr_name), cv2.IMREAD_COLOR
            )
            hr = hr[: ref.shape[0] * scale, : ref.shape[1] * scale]
            got = imresize(hr, 1.0 / scale)
            if got.shape != ref.shape:
                print(f"  x{scale} {hr_name}: shape {got.shape} vs {ref.shape}")
                continue
            delta = np.abs(got.astype(np.int16) - ref.astype(np.int16))
            diffs.append(delta.mean())
            exact += int((delta == 0).sum())
            total += delta.size
            worst[scale] = max(worst.get(scale, 0), int(delta.max()))
        print(
            f"  x{scale}: mean |delta| = {np.mean(diffs):.4f} LSB, {100.0 * exact / max(total, 1):.2f}% of pixels bit-exact, max |delta| = {worst.get(scale)}"
        )
    print()
    if all((v <= 1 for v in worst.values())):
        print("PASS: reproduces the official DIV2K LR to within one LSB")
    else:
        print(
            "FAIL: this is not MATLAB imresize -- do NOT generate Flickr2K LR with it"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.verify:
        _verify(args.limit)
    else:
        parser.print_help()
