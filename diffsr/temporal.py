from __future__ import annotations
import numpy as np
import torch
from .rrta import warp

__all__ = [
    "static_stability",
    "temporal_gradient_error",
    "is_static",
    "temporal_report",
]


def is_static(motion: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    spread = motion.flatten(2).std(dim=2)
    return spread.max(dim=1).values < threshold


def static_stability(
    prediction: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    pred = prediction.double() * 255.0
    ref = reference.double() * 255.0
    return {
        "pixel_std": float(pred.std(dim=0, unbiased=False).mean()),
        "pixel_std_reference": float(ref.std(dim=0, unbiased=False).mean()),
        "psnr_static": _psnr(pred, ref),
    }


def _psnr(pred: torch.Tensor, ref: torch.Tensor) -> float:
    mse = float(((pred - ref) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse))


def temporal_gradient_error(
    prediction: torch.Tensor, reference: torch.Tensor, motion_hr: torch.Tensor
) -> dict[str, float]:
    if prediction.shape[0] < 2:
        return {
            "tgrad_l1": float("nan"),
            "tpsnr": float("nan"),
            "sse": 0.0,
            "count": 0.0,
        }
    (previous_pred, _) = warp(prediction[:-1], motion_hr[1:])
    (previous_ref, valid) = warp(reference[:-1], motion_hr[1:])
    delta_pred = (prediction[1:] - previous_pred) * valid
    delta_ref = (reference[1:] - previous_ref) * valid
    difference = (delta_pred - delta_ref).double() * 255.0
    weight = valid.double().expand_as(difference).sum().clamp_min(1.0)
    mse = float((difference**2).sum() / weight)
    return {
        "tgrad_l1": float(difference.abs().sum() / weight),
        "tpsnr": float("inf") if mse == 0 else float(10.0 * np.log10(255.0**2 / mse)),
        "sse": float((difference**2).sum()),
        "count": float(weight),
    }


def temporal_report(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    motion_lr: torch.Tensor,
    scale: int,
    warmup: int = 2,
) -> dict[str, float]:
    prediction = prediction[warmup:]
    reference = reference[warmup:]
    motion_lr = motion_lr[warmup:]
    if prediction.shape[0] < 2:
        return {}
    motion_hr = (
        torch.nn.functional.interpolate(motion_lr, scale_factor=scale, mode="nearest")
        * scale
    )
    out = temporal_gradient_error(prediction, reference, motion_hr)
    out.update(static_stability(prediction, reference))
    return out
