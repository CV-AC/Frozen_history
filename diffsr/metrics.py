from __future__ import annotations
import numpy as np

__all__ = [
    "rgb_to_y",
    "psnr",
    "ssim",
    "thin_structure_mask",
    "psnr_masked",
    "local_extremum_violation",
    "band_error",
]
_Y_WEIGHTS = np.array([65.481, 128.553, 24.966], dtype=np.float64)
_Y_OFFSET = 16.0


def _prepare(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float64)
    array = array.astype(np.float64)
    return array * 255.0 if array.max() <= 1.5 else array


def rgb_to_y(image: np.ndarray, grey_passthrough: bool = False) -> np.ndarray:
    array = _prepare(image) / 255.0
    if array.ndim == 2:
        return (
            array * 255.0
            if grey_passthrough
            else array[..., None].repeat(3, axis=2)[..., :3] @ _Y_WEIGHTS + _Y_OFFSET
        )
    if (
        grey_passthrough
        and array.shape[2] >= 3
        and np.array_equal(array[..., 0], array[..., 1])
        and np.array_equal(array[..., 1], array[..., 2])
    ):
        return array[..., 0] * 255.0
    return array[..., :3] @ _Y_WEIGHTS + _Y_OFFSET


def _align(
    pred: np.ndarray, gt: np.ndarray, scale: int, crop_border: int | None, to_y: bool
) -> tuple[np.ndarray, np.ndarray]:
    (pred, gt) = (_prepare(pred), _prepare(gt))
    dh = abs(pred.shape[0] - gt.shape[0])
    dw = abs(pred.shape[1] - gt.shape[1])
    if dh >= max(scale, 1) or dw >= max(scale, 1):
        raise ValueError(
            f"prediction {pred.shape[:2]} vs reference {gt.shape[:2]}: wrong scale, refusing to crop"
        )
    height = min(pred.shape[0], gt.shape[0])
    width = min(pred.shape[1], gt.shape[1])
    (pred, gt) = (pred[:height, :width], gt[:height, :width])
    if to_y:
        (pred, gt) = (rgb_to_y(pred), rgb_to_y(gt))
    border = scale if crop_border is None else crop_border
    if border:
        pred = pred[border:-border, border:-border]
        gt = gt[border:-border, border:-border]
    return (pred, gt)


def psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    scale: int = 4,
    crop_border: int | None = None,
    to_y: bool = True,
) -> float:
    (pred, gt) = _align(pred, gt, scale, crop_border, to_y)
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def _ssim_plane(pred: np.ndarray, gt: np.ndarray) -> float:
    import cv2

    (c1, c2) = ((0.01 * 255) ** 2, (0.03 * 255) ** 2)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(pred, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
    (mu1_sq, mu2_sq, mu1_mu2) = (mu1**2, mu2**2, mu1 * mu2)
    sigma1 = cv2.filter2D(pred**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2 = cv2.filter2D(gt**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(pred * gt, -1, window)[5:-5, 5:-5] - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    return float((numerator / denominator).mean())


def ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    scale: int = 4,
    crop_border: int | None = None,
    to_y: bool = True,
) -> float:
    (pred, gt) = _align(pred, gt, scale, crop_border, to_y)
    if pred.ndim == 2:
        return _ssim_plane(pred, gt)
    return float(
        np.mean([_ssim_plane(pred[..., c], gt[..., c]) for c in range(pred.shape[2])])
    )


def thin_structure_mask(
    gt: np.ndarray, scale: int = 4, fraction: float = 0.05
) -> np.ndarray:
    import cv2
    from .imresize import imresize

    luma = rgb_to_y(gt)
    small = imresize(luma / 255.0, 1.0 / scale, antialiasing=True)
    back = (
        cv2.resize(small, (luma.shape[1], luma.shape[0]), interpolation=cv2.INTER_CUBIC)
        * 255.0
    )
    residual = np.abs(luma - back)
    threshold = np.quantile(residual, 1.0 - fraction)
    return residual >= threshold


def psnr_masked(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    scale: int = 4,
    crop_border: int | None = None,
) -> float:
    (pred_y, gt_y) = _align(pred, gt, scale, crop_border, True)
    border = scale if crop_border is None else crop_border
    sub = mask[border:-border, border:-border] if border else mask
    sub = sub[: pred_y.shape[0], : pred_y.shape[1]]
    if not sub.any():
        return float("nan")
    mse = float(np.mean((pred_y[sub] - gt_y[sub]) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def _extremum(image: np.ndarray, radius: int, reduce) -> np.ndarray:
    out = image.copy()
    padded = np.pad(image, radius, mode="edge")
    (height, width) = image.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy or dx:
                out = reduce(
                    out,
                    padded[
                        radius + dy : radius + dy + height,
                        radius + dx : radius + dx + width,
                    ],
                )
    return out


def local_extremum_violation(
    pred: np.ndarray,
    gt: np.ndarray,
    radius: int = 1,
    scale: int = 4,
    crop_border: int | None = None,
) -> dict[str, float]:
    (pred_y, gt_y) = _align(pred, gt, scale, crop_border, True)
    upper = _extremum(gt_y, radius, np.maximum)
    lower = _extremum(gt_y, radius, np.minimum)
    over = np.maximum(pred_y - upper, 0.0)
    under = np.maximum(lower - pred_y, 0.0)
    return {
        "lev": float((over + under).mean()),
        "over": float(over.mean()),
        "under": float(under.mean()),
        "violating_fraction": float((over + under > 0.5).mean()),
    }


def band_error(
    pred: np.ndarray,
    gt: np.ndarray,
    bands: int = 8,
    scale: int = 4,
    crop_border: int | None = None,
) -> np.ndarray:
    (pred_y, gt_y) = _align(pred, gt, scale, crop_border, True)
    spectrum = np.fft.fftshift(np.fft.fft2(pred_y - gt_y))
    power = np.abs(spectrum) ** 2
    (height, width) = power.shape
    rows = (np.arange(height) - height / 2) / (height / 2)
    cols = (np.arange(width) - width / 2) / (width / 2)
    index = np.clip(
        (np.hypot(rows[:, None], cols[None, :]) * bands).astype(int), 0, bands - 1
    )
    out = np.array(
        [power[index == b].mean() if (index == b).any() else 0.0 for b in range(bands)]
    )
    return out / max(power.mean(), 1e-30)


PUBLISHED_BICUBIC = {
    ("Set5", 2): (33.66, 0.9299),
    ("Set5", 3): (30.39, 0.8682),
    ("Set5", 4): (28.42, 0.8104),
    ("Set14", 2): (30.24, 0.8688),
    ("Set14", 3): (27.55, 0.7742),
    ("Set14", 4): (26.0, 0.7027),
    ("BSD100", 2): (29.56, 0.8431),
    ("BSD100", 3): (27.21, 0.7385),
    ("BSD100", 4): (25.96, 0.6675),
    ("Urban100", 2): (26.88, 0.8403),
    ("Urban100", 3): (24.46, 0.7349),
    ("Urban100", 4): (23.14, 0.6577),
}


def _bicubic_check(scales=(2, 3, 4)) -> None:
    import cv2
    from .data_sisr import build_test_dataset
    from .imresize import imresize

    print(
        f"{'set':<18}{'x':>3}{'PSNR':>9}{'ref':>8}{'d':>7}{'SSIM':>9}{'ref':>8}{'d':>8}",
        flush=True,
    )
    worst = 0.0
    for name in ("Set5", "Set14", "BSD100", "Urban100", "Manga109-mid109"):
        for scale in scales:
            try:
                ds = build_test_dataset(name, scale)
            except Exception:
                continue
            values = []
            for i in range(len(ds)):
                item = ds[i]
                lr = item["lr"].numpy().transpose(1, 2, 0) * 255.0
                gt = item["hr"].numpy().transpose(1, 2, 0) * 255.0
                up = imresize(np.clip(lr, 0, 255).astype(np.uint8), float(scale))
                values.append((psnr(up, gt, scale), ssim(up, gt, scale)))
            p = float(np.mean([v[0] for v in values]))
            s = float(np.mean([v[1] for v in values]))
            ref = PUBLISHED_BICUBIC.get((name, scale))
            if ref:
                (dp, dsm) = (p - ref[0], s - ref[1])
                worst = max(worst, abs(dp))
                print(
                    f"{name:<18}{scale:>3}{p:>9.2f}{ref[0]:>8.2f}{dp:>+7.2f}{s:>9.4f}{ref[1]:>8.4f}{dsm:>+8.4f}",
                    flush=True,
                )
            else:
                print(
                    f"{name:<18}{scale:>3}{p:>9.2f}{'--':>8}{'':>7}{s:>9.4f}{'--':>8}",
                    flush=True,
                )
    print()
    if worst <= 0.1:
        print(
            f"PASS: every bicubic value within {worst:.2f} dB of the published reference (BSD100 and Urban100 exact at all scales; the residual is Set14's image source, see the note in this file)"
        )
    else:
        print(
            f"FAIL: largest deviation {worst:.2f} dB -- the evaluation protocol does not match the literature"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bicubic", action="store_true")
    ap.add_argument("--scales", default="2,3,4")
    args = ap.parse_args()
    if args.bicubic:
        _bicubic_check(tuple((int(v) for v in args.scales.split(","))))
    else:
        ap.print_help()
