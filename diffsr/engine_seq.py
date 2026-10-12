from __future__ import annotations
import argparse
import copy
import inspect
import json
import shutil
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data_seq import QrispClips, build_clip_train, static_positions, test_segments
from .metrics import psnr as psnr_fn, psnr_masked, ssim as ssim_fn, thin_structure_mask
from .rrta import RRTA_SIZES, RRTANet, create_rrta, warp
from . import checkpoint as _ckpt
from .temporal import is_static, temporal_gradient_error

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "runs"


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    return np.ascontiguousarray(array.transpose(1, 2, 0)) * 255.0


def motion_to_hr(motion: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(motion, scale_factor=scale, mode="nearest") * scale


def sequence_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    motion: torch.Tensor,
    scale: int,
    warmup: int,
    temporal_weight: float,
    objective: str = "l1",
    weighting: str = "uniform",
    window: float = 16.0,
    luma_weight: float = 0.0,
    temporal_objective: str = "l1",
    static_temporal: float = 0.0,
    static_threshold: float = 2.0,
    spread: torch.Tensor | None = None,
    temporal_warp: str = "motion",
) -> tuple[torch.Tensor, dict]:
    pred = prediction[:, warmup:]
    ref = reference[:, warmup:]
    weight = None
    if weighting == "converged" and pred.shape[1] > 1:
        ages = (
            torch.arange(
                warmup, warmup + pred.shape[1], device=pred.device, dtype=pred.dtype
            )
            + 1.0
        )
        weight = (ages / (ages + window)).clamp_min(0.001)
        weight = weight / weight.mean()
    residual = pred - ref
    if luma_weight > 0.0:
        weights = (
            torch.tensor(
                [65.481, 128.553, 24.966], device=pred.device, dtype=pred.dtype
            )
            / 255.0
        )
        luma = (residual * weights[None, None, :, None, None]).sum(dim=2, keepdim=True)
    if objective == "l2":
        per_frame = (residual**2).mean(dim=(0, 2, 3, 4))
    elif objective == "charbonnier":
        per_frame = (residual**2 + 1e-06).sqrt().mean(dim=(0, 2, 3, 4))
    else:
        per_frame = residual.abs().mean(dim=(0, 2, 3, 4))
    spatial = (per_frame * weight).mean() if weight is not None else per_frame.mean()
    if luma_weight > 0.0:
        per_luma = (luma**2 if objective == "l2" else luma.abs()).mean(dim=(0, 2, 3, 4))
        spatial = spatial + luma_weight * (
            (per_luma * weight).mean() if weight is not None else per_luma.mean()
        )
    stats = {
        "l1": float(spatial.detach()),
        "unw": float(residual.abs().mean().detach()),
    }
    total = spatial
    if temporal_weight > 0.0 and pred.shape[1] >= 2:
        if temporal_warp == "none":
            previous_pred = pred[:, :-1].flatten(0, 1)
            previous_ref = ref[:, :-1].flatten(0, 1)
            valid = torch.ones_like(previous_pred[:, :1])
        elif temporal_warp == "motion":
            flat_motion = motion[:, warmup + 1 :].flatten(0, 1)
            hr_motion = motion_to_hr(flat_motion, scale)
            (previous_pred, valid) = warp(pred[:, :-1].flatten(0, 1), hr_motion)
            (previous_ref, _) = warp(ref[:, :-1].flatten(0, 1), hr_motion)
        else:
            raise ValueError(
                f"temporal_warp must be 'motion' or 'none', got {temporal_warp!r}"
            )
        delta_pred = pred[:, 1:].flatten(0, 1) - previous_pred
        delta_ref = ref[:, 1:].flatten(0, 1) - previous_ref
        gradient = delta_pred - delta_ref
        error = (gradient**2 if temporal_objective == "l2" else gradient.abs()) * valid
        denominator = valid.expand_as(delta_pred)
        if weight is not None:
            pair = weight[1:].repeat(pred.shape[0]).view(-1, 1, 1, 1)
            temporal = (error * pair).sum() / (denominator * pair).sum().clamp_min(1.0)
        else:
            temporal = error.sum() / denominator.sum().clamp_min(1.0)
        stats["temporal"] = float(temporal.detach())
        total = total + temporal_weight * temporal
        if static_temporal > 0.0:
            if spread is None:
                raise ValueError(
                    "static_temporal needs per-frame spread; the loader found no runs/reference/spread_<split>_x<scale>.npz cache"
                )
            spread = spread.to(error.device)
            later = spread[:, warmup + 1 :]
            static = (later < static_threshold).to(error.dtype)
            static = static.flatten().view(-1, 1, 1, 1)
            static_term = (error * static).sum() / (
                denominator * static
            ).sum().clamp_min(1.0)
            stats["temporal_static"] = float(static_term.detach())
            stats["static_frac"] = float(static.mean().detach())
            total = total + static_temporal * static_term
    return (total, stats)


def candidate_loss(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    warmup: int,
    weighting: str = "uniform",
    window: float = 16.0,
) -> torch.Tensor:
    cand = candidate[:, warmup:]
    ref = reference[:, warmup:]
    per_frame = ((cand - ref) ** 2).mean(dim=(0, 2, 3, 4))
    if weighting == "converged" and cand.shape[1] > 1:
        ages = (
            torch.arange(
                warmup, warmup + cand.shape[1], device=cand.device, dtype=cand.dtype
            )
            + 1.0
        )
        weight = (ages / (ages + window)).clamp_min(0.001)
        weight = weight / weight.mean()
        return (per_frame * weight).mean()
    return per_frame.mean()


def resolve_run_dir(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    if (DEFAULT_OUT / path).is_dir():
        return DEFAULT_OUT / path
    return path.resolve()


def load_teacher(args, device) -> tuple[torch.nn.Module, dict]:
    run = resolve_run_dir(args.distill_teacher)
    if not (run / "config.json").is_file() or not (run / "last.pt").is_file():
        raise SystemExit(
            f"--distill-teacher {args.distill_teacher}: no config.json/last.pt in {run}"
        )
    (teacher, tcfg) = _ckpt.load_run(run, device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    want = _ckpt.input_pipeline(vars(args))
    have = _ckpt.input_pipeline(tcfg)
    if have["scale"] is None:
        have["scale"] = int(teacher.scale)
    bad = {k: (have[k], want[k]) for k in want if have[k] != want[k]}
    if bad:
        raise SystemExit(
            f"--distill-teacher {run.name}: input pipeline differs from the student's (teacher, student): {bad}"
        )
    if args.distill_alpha_weight > 0.0 and teacher.accumulate != args.accumulate:
        raise SystemExit(
            f"--distill-alpha-weight: teacher accumulate={teacher.accumulate!r} vs student {args.accumulate!r}; alpha means different things"
        )
    if args.gt_init_prob > 0.0:
        teacher.track_wbar = True
    info = {
        "run": str(run),
        "ckpt_iteration": tcfg.get("ckpt_iteration"),
        "params_m": sum((p.numel() for p in teacher.parameters())) / 1000000.0,
        "input_pipeline": have,
        "model_kwargs": tcfg.get("model_kwargs"),
        "track_wbar": bool(teacher.track_wbar),
    }
    return (teacher, info)


def distill_terms(
    prediction: torch.Tensor,
    teacher_pred: torch.Tensor,
    auxes: list,
    teacher_auxes: list,
    warmup: int,
    weighting: str,
    window: float,
    alpha: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = candidate_loss(
        prediction.float(), teacher_pred.float(), warmup, weighting, window
    )
    if not alpha:
        return (image, None)
    student_alpha = torch.stack([a["alpha"] for a in auxes], dim=1).float()
    teacher_alpha = torch.stack([a["alpha"] for a in teacher_auxes], dim=1).float()
    return (
        image,
        candidate_loss(student_alpha, teacher_alpha, warmup, weighting, window),
    )


@torch.no_grad()
def disocclusion_mask(
    depth: torch.Tensor,
    previous_depth: torch.Tensor,
    motion: torch.Tensor,
    scale: int,
    tolerance: float = 0.02,
) -> np.ndarray:
    (warped, inside) = warp(previous_depth, motion)
    mismatch = (depth - warped).abs() > tolerance
    bad = mismatch | (inside < 0.5)
    bad = F.interpolate(bad.float(), scale_factor=scale, mode="nearest")
    return bad[0, 0].cpu().numpy() > 0.5


@torch.no_grad()
def evaluate(model, args, device, clips: QrispClips, positions: list[int]) -> dict:
    model.eval()
    scale = args.scale
    warmup = args.eval_warmup
    (rows, temporal_rows, static_rows) = ([], [], [])
    per_scene: dict[str, list[float]] = {}
    for position in positions:
        item = clips[position]
        lr = item["lr"][None].to(device)
        hr = item["hr"][None].to(device)
        depth = item["depth"][None].to(device)
        motion = item["motion"][None].to(device)
        jitter = item["jitter"][None].to(device)
        steps = lr.shape[1]
        state = None
        previous_pred = previous_ref = None
        total = square = ref_total = ref_square = None
        count = 0
        (clip_rows, clip_temporal) = ([], [])
        for t in range(steps):
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                (out, state, _) = model.step(
                    lr[:, t], depth[:, t], motion[:, t], jitter[:, t], state
                )
            out = out.float().clamp(0, 1)
            state = tuple((v.float() if torch.is_tensor(v) else v for v in state))
            if t < warmup:
                (previous_pred, previous_ref) = (out, hr[:, t])
                continue
            reference = to_numpy(hr[0, t])
            prediction = to_numpy(out[0])
            upsampled = to_numpy(
                F.interpolate(
                    lr[:, t, :3],
                    scale_factor=scale,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0, 1)[0]
            )
            row = {
                "psnr": psnr_fn(
                    prediction, reference, scale, crop_border=0, to_y=False
                ),
                "ssim": ssim_fn(
                    prediction, reference, scale, crop_border=0, to_y=False
                ),
                "psnr_bicubic": psnr_fn(
                    upsampled, reference, scale, crop_border=0, to_y=False
                ),
            }
            thin = thin_structure_mask(reference, scale)
            row["psnr_thin"] = psnr_masked(
                prediction, reference, thin, scale, crop_border=0
            )
            row["psnr_thin_bicubic"] = psnr_masked(
                upsampled, reference, thin, scale, crop_border=0
            )
            if t > 0:
                bad = disocclusion_mask(
                    depth[:, t], depth[:, t - 1], motion[:, t], scale
                )
                if bad.any() and (not bad.all()):
                    row["psnr_disocc"] = psnr_masked(
                        prediction, reference, bad, scale, crop_border=0
                    )
                    row["psnr_disocc_bicubic"] = psnr_masked(
                        upsampled, reference, bad, scale, crop_border=0
                    )
                    row["disocc_fraction"] = float(bad.mean())
            clip_rows.append(row)
            if previous_pred is not None and t > warmup:
                hr_motion = motion_to_hr(motion[:, t], scale)
                clip_temporal.append(
                    temporal_gradient_error(
                        torch.cat((previous_pred, out)),
                        torch.cat((previous_ref, hr[:, t])),
                        hr_motion.repeat(2, 1, 1, 1),
                    )
                )
            value = out[0].double() * 255.0
            truth = hr[0, t].double() * 255.0
            total = value if total is None else total + value
            square = value**2 if square is None else square + value**2
            ref_total = truth if ref_total is None else ref_total + truth
            ref_square = truth**2 if ref_square is None else ref_square + truth**2
            count += 1
            (previous_pred, previous_ref) = (out, hr[:, t])
        if not clip_rows:
            continue
        scene = item["scene"]
        mean_psnr = float(np.mean([r["psnr"] for r in clip_rows]))
        keys = set().union(*(r.keys() for r in clip_rows))
        rows.append(
            {k: float(np.nanmean([r[k] for r in clip_rows if k in r])) for k in keys}
        )
        per_scene.setdefault(scene, []).append(mean_psnr)
        if clip_temporal:
            temporal_rows.append(
                {
                    k: float(np.nanmean([r[k] for r in clip_temporal]))
                    for k in clip_temporal[0]
                }
            )
        if count > 1 and bool(is_static(item["motion"]).all()):
            variance = (square / count - (total / count) ** 2).clamp_min(0.0)
            ref_variance = (ref_square / count - (ref_total / count) ** 2).clamp_min(
                0.0
            )
            predicted = float(variance.sqrt().mean())
            reference_std = float(ref_variance.sqrt().mean())
            static_rows.append(
                {
                    "pixel_std": predicted,
                    "pixel_std_reference": reference_std,
                    "pixel_std_excess": predicted - reference_std,
                    "psnr_static": mean_psnr,
                }
            )
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    result = {k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}
    for name, ref in (
        ("", "psnr_bicubic"),
        ("_thin", "psnr_thin_bicubic"),
        ("_disocc", "psnr_disocc_bicubic"),
    ):
        key = f"psnr{name}"
        if key in result and ref in result:
            result[f"gain{name or '_over_bicubic'}"] = result[key] - result[ref]
    if temporal_rows:
        result.update(
            {
                k: float(np.nanmean([r[k] for r in temporal_rows]))
                for k in temporal_rows[0]
            }
        )
    if static_rows:
        result.update(
            {k: float(np.mean([r[k] for r in static_rows])) for k in static_rows[0]}
        )
        result["static_clips"] = len(static_rows)
    if per_scene:
        result["per_scene_psnr"] = {
            s: float(np.mean(v)) for (s, v) in sorted(per_scene.items())
        }
        result["psnr_scene_balanced"] = float(
            np.mean(list(result["per_scene_psnr"].values()))
        )
    model.train()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="t")
    ap.add_argument("--dim", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=0)
    ap.add_argument("--arch", choices=("rrta", "qrisp"), default="rrta")
    ap.add_argument("--qrisp-f", type=int, default=64)
    ap.add_argument("--qrisp-m", type=int, default=5)
    ap.add_argument("--qrisp-hyper-hidden", type=int, default=2048)
    ap.add_argument("--qrisp-hyper-layers", type=int, default=7)
    ap.add_argument("--qrisp-hyper-param", choices=("mup", "plain"), default="mup")
    ap.add_argument("--qrisp-dilation-block", type=int, default=8)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--clip", type=int, default=8)
    ap.add_argument("--patch", type=int, default=66)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--layer-scale", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--eval-warmup", type=int, default=2)
    ap.add_argument("--temporal-weight", type=float, default=0.25)
    ap.add_argument("--phase-weight-init", type=float, default=-2.0)
    ap.add_argument("--wd", type=float, default=0.0001)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--keep-every", type=int, default=0)
    ap.add_argument("--temporal-objective", default="l1", choices=("l1", "l2"))
    ap.add_argument("--temporal-warp", default="motion", choices=("motion", "none"))
    ap.add_argument("--static-gate", type=float, default=0.0)
    ap.add_argument("--accum-alpha-min", type=float, default=0.0)
    ap.add_argument(
        "--first-order", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--static-temporal-weight", type=float, default=0.0)
    ap.add_argument("--static-threshold", type=float, default=2.0)
    ap.add_argument("--motion-lerp", type=float, default=0.0)
    ap.add_argument("--motion-jit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dejitter", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--phase-geometry", choices=("legacy", "image"), default="legacy")
    ap.add_argument(
        "--registered-skip", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--lr-rejitter", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--depth-jit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--fp32-accum", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--skip-dead", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--readout", choices=("accum", "residual", "fresh"), default="accum"
    )
    ap.add_argument("--vcap-max", type=float, default=0.0)
    ap.add_argument("--vcap-min", type=float, default=4.0)
    ap.add_argument("--vcap-v0", type=float, default=2.0)
    ap.add_argument("--feature-bound", type=float, default=0.0)
    ap.add_argument("--consistency-gate", choices=("none", "sample"), default="none")
    ap.add_argument("--gate-shorten", choices=("off", "on"), default="off")
    ap.add_argument("--gate-shorten-init", type=float, default=0.01)
    ap.add_argument("--state-mode", choices=("linear", "gru"), default="linear")
    ap.add_argument("--cand-loss-weight", type=float, default=0.0)
    ap.add_argument("--hist-clip", choices=("none", "var"), default="none")
    ap.add_argument("--gt-init-prob", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument(
        "--global-motion", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--static-cap", type=float, default=64.0)
    ap.add_argument("--gamma-max", type=float, default=2.0)
    ap.add_argument("--alpha-min", type=float, default=1.0 / 16.0)
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--ycocg", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--motion-dilation", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--rectify",
        default="learned",
        choices=("learned", "classic", "moments", "none"),
    )
    ap.add_argument("--recurrent", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--phase-blend", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--band-lowres", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--feature-dim", type=int, default=8)
    ap.add_argument("--norm", default="none", choices=("none", "group"))
    ap.add_argument("--depth-edge", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--stem", default="conv3_film", choices=("jitter1x1", "conv3", "conv3_film")
    )
    ap.add_argument(
        "--jitter-head", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument(
        "--phase-memory", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--block", default="conv3", choices=("dw7", "conv3"))
    ap.add_argument(
        "--body",
        default="plain",
        choices=(
            "plain",
            "conv3",
            "rep3",
            "pconv",
            "mbv2",
            "star",
            "unet",
            "dwmlp",
            "shiftmlp",
            "pconvglu",
            "poolmlp",
            "linattn",
        ),
    )
    ap.add_argument(
        "--body-inner",
        default="conv3",
        choices=("conv3", "rep3", "pconv", "mbv2", "star"),
    )
    ap.add_argument("--body-width-mult", type=float, default=2.0)
    ap.add_argument("--body-unet-levels", type=int, default=1)
    ap.add_argument("--body-unet-blocks", default="")
    ap.add_argument("--body-pconv-ratio", type=int, default=4)
    ap.add_argument("--body-expand", type=float, default=2.0)
    ap.add_argument("--body-rep-branches", default="ecb")
    ap.add_argument("--body-act", default="gelu", choices=("gelu", "relu", "silu"))
    ap.add_argument("--body-dw-kernel", type=int, default=3)
    ap.add_argument("--body-dilations", default="1,2")
    ap.add_argument("--body-attn-idx", default="")
    ap.add_argument("--body-attn-dim", type=int, default=16)
    ap.add_argument("--band-split", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hr-refine", type=int, default=16)
    ap.add_argument("--hr-layers", type=int, default=3)
    ap.add_argument("--frames-in", type=int, default=1)
    ap.add_argument("--squeeze", type=int, default=0)
    ap.add_argument("--accum-cap", type=float, default=0.0)
    ap.add_argument("--kernel", default="gauss", choices=("none", "gauss"))
    ap.add_argument(
        "--reject", default="none", choices=("none", "mismatch", "variance", "ztest")
    )
    ap.add_argument(
        "--accumulate",
        default="blend",
        choices=("blend", "normalized", "samples", "direct"),
    )
    ap.add_argument("--smr-sigma", type=float, default=0.6)
    ap.add_argument("--smr-lambda-init", type=float, default=1.0)
    ap.add_argument("--smr-lambda-eps", type=float, default=0.01)
    ap.add_argument("--smr-input-eps", type=float, default=0.05)
    ap.add_argument("--smr-depth", default="far0.1")
    ap.add_argument(
        "--loss-weighting", default="uniform", choices=("uniform", "converged")
    )
    ap.add_argument("--dual-rate", type=float, default=0.0)
    ap.add_argument(
        "--canonical-phase", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument("--loss", default="l1", choices=("l1", "l2", "charbonnier"))
    ap.add_argument("--luma-weight", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--qrisp-colour", default="MipBiasJit")
    ap.add_argument("--dataset", choices=("qrisp", "stss"), default="qrisp")
    ap.add_argument("--stss-train", nargs="+", default=None)
    ap.add_argument("--stss-test", default=None)
    ap.add_argument("--stss-depth-mode", choices=("linear", "rational"), default=None)
    ap.add_argument("--lr-unshuffle", type=int, default=1)
    ap.add_argument("--distill-teacher", default=None)
    ap.add_argument("--distill-weight", type=float, default=0.0)
    ap.add_argument("--distill-alpha-weight", type=float, default=0.0)
    ap.add_argument("--eval-clip", type=int, default=12)
    ap.add_argument("--eval-per-scene", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    rrta_only = (
        "size",
        "dim",
        "blocks",
        "layer_scale",
        "phase_weight_init",
        "static_gate",
        "accum_alpha_min",
        "first_order",
        "phase_geometry",
        "registered_skip",
        "lr_rejitter",
        "fp32_accum",
        "skip_dead",
        "readout",
        "vcap_max",
        "vcap_min",
        "vcap_v0",
        "feature_bound",
        "consistency_gate",
        "gate_shorten",
        "gate_shorten_init",
        "state_mode",
        "hist_clip",
        "global_motion",
        "static_cap",
        "gamma_max",
        "lam",
        "ycocg",
        "rectify",
        "recurrent",
        "phase_blend",
        "band_lowres",
        "feature_dim",
        "norm",
        "depth_edge",
        "stem",
        "jitter_head",
        "phase_memory",
        "block",
        "body",
        "body_inner",
        "body_width_mult",
        "body_unet_levels",
        "body_unet_blocks",
        "body_pconv_ratio",
        "body_expand",
        "body_rep_branches",
        "body_act",
        "band_split",
        "body_dw_kernel",
        "body_dilations",
        "body_attn_idx",
        "body_attn_dim",
        "hr_refine",
        "hr_layers",
        "frames_in",
        "squeeze",
        "accum_cap",
        "kernel",
        "reject",
        "accumulate",
        "smr_sigma",
        "smr_lambda_init",
        "smr_lambda_eps",
        "smr_input_eps",
        "smr_depth",
        "dual_rate",
        "canonical_phase",
        "lr_unshuffle",
    )
    qrisp_only = (
        "qrisp_f",
        "qrisp_m",
        "qrisp_hyper_hidden",
        "qrisp_hyper_layers",
        "qrisp_hyper_param",
        "qrisp_dilation_block",
    )
    foreign = rrta_only if args.arch == "qrisp" else qrisp_only
    foreign = [k for k in foreign if getattr(args, k) != ap.get_default(k)]
    if foreign:
        raise SystemExit(
            f"--arch {args.arch} ignores {['--' + k.replace('_', '-') for k in foreign]}; drop them"
        )
    if args.lr_rejitter and args.motion_jit and (not args.dejitter):
        raise SystemExit(
            "--lr-rejitter adds the jitter delta to a NON-jittered field; use it with --no-motion-jit (or --dejitter)"
        )
    distilling = args.distill_weight > 0.0 or args.distill_alpha_weight > 0.0
    if args.distill_weight < 0.0 or args.distill_alpha_weight < 0.0:
        raise SystemExit("--distill-weight/--distill-alpha-weight must be >= 0")
    if distilling and (not args.distill_teacher):
        raise SystemExit(
            "--distill-weight/--distill-alpha-weight need --distill-teacher"
        )
    if args.distill_teacher and (not distilling):
        raise SystemExit(
            "--distill-teacher given but both distill weights are 0 (the teacher would cost time for nothing)"
        )
    stss = args.dataset == "stss"
    stss_data = None
    if stss:
        from . import data_stss

        if args.scale != 2:
            raise SystemExit("--dataset stss is 540p -> 1080p: use --scale 2")
        if args.motion_lerp:
            raise SystemExit(
                "--motion-lerp blends toward the OTHER motion field; STSS has exactly one (no jitter)"
            )
        args.stss_train = [
            str(Path(r))
            for r in args.stss_train
            or [data_stss.STSS_PACKED / n for n in data_stss.STSS_TRAIN]
        ]
        args.stss_test = str(
            Path(args.stss_test)
            if args.stss_test
            else data_stss.STSS_PACKED / data_stss.STSS_TEST
        )
        args.stss_depth_mode = args.stss_depth_mode or "linear"
        stss_data = data_stss.stss_data_provenance(
            args.stss_train + ([args.stss_test] if args.eval_every > 0 else [])
        )
    elif (
        args.stss_train is not None
        or args.stss_test is not None
        or args.stss_depth_mode is not None
    ):
        raise SystemExit(
            "--stss-train/--stss-test/--stss-depth-mode need --dataset stss"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if stss:
        train = data_stss.build_stss_train(
            root=args.stss_train,
            length=args.clip,
            patch_lr=args.patch,
            scale=args.scale,
            depth_mode=args.stss_depth_mode,
        )
        print(
            f"  dataset: STSS x2 (jitter 0) train {[Path(r).name for r in args.stss_train]} ({sum((len(s.frames) for s in train.scenes))} frames), depth {args.stss_depth_mode}",
            flush=True,
        )
    else:
        train = build_clip_train(
            scale=args.scale,
            length=args.clip,
            patch_lr=args.patch,
            colour=args.qrisp_colour,
            motion_jit=args.motion_jit,
            motion_lerp=args.motion_lerp,
            depth_jit=args.depth_jit,
        )
    loader = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers else None,
    )
    model_kwargs = dict(
        gamma_max=args.gamma_max,
        alpha_min=args.alpha_min,
        lam=args.lam,
        phase_weight_init=args.phase_weight_init,
        ycocg=args.ycocg,
        motion_dilation=args.motion_dilation,
        rectify=args.rectify,
        recurrent=args.recurrent,
        phase_blend=args.phase_blend,
        band_lowres=args.band_lowres,
        feature_dim=args.feature_dim,
        norm=args.norm,
        depth_edge=args.depth_edge,
        stem=args.stem,
        jitter_head=args.jitter_head,
        phase_memory=args.phase_memory,
        block=args.block,
        band_split=args.band_split,
        hr_refine=args.hr_refine,
        hr_layers=args.hr_layers,
        frames_in=args.frames_in,
        squeeze=args.squeeze,
        accumulate=args.accumulate,
        reject=args.reject,
        kernel=args.kernel,
        accum_cap=args.accum_cap,
        static_gate=args.static_gate,
        global_motion=args.global_motion,
        dejitter=args.dejitter,
        first_order=args.first_order,
        accum_alpha_min=args.accum_alpha_min,
        static_cap=args.static_cap,
        dual_rate=args.dual_rate,
        canonical_phase=args.canonical_phase,
        layer_scale=args.layer_scale,
        phase_geometry=args.phase_geometry,
        registered_skip=args.registered_skip,
        lr_rejitter=args.lr_rejitter,
        depth_jit=args.depth_jit,
        fp32_accum=args.fp32_accum,
        skip_dead=args.skip_dead,
        readout=args.readout,
        vcap_max=args.vcap_max,
        vcap_min=args.vcap_min,
        vcap_v0=args.vcap_v0,
        hist_clip=args.hist_clip,
        feature_bound=args.feature_bound,
        consistency_gate=args.consistency_gate,
        state_mode=args.state_mode,
        gate_shorten=args.gate_shorten,
        gate_shorten_init=args.gate_shorten_init,
    )
    model_kwargs.update(
        smr_sigma=args.smr_sigma,
        smr_lambda_init=args.smr_lambda_init,
        smr_lambda_eps=args.smr_lambda_eps,
        smr_input_eps=args.smr_input_eps,
        smr_depth=args.smr_depth,
    )
    if args.accumulate == "samples" and args.gt_init_prob > 0.0:
        raise SystemExit(
            "--gt-init-prob is not supported with --accumulate samples (it would seed the sample state with GT)"
        )
    if args.accumulate == "direct" and args.gt_init_prob > 0.0:
        raise SystemExit("--gt-init-prob is not supported with --accumulate direct")
    model_kwargs["lr_unshuffle"] = args.lr_unshuffle
    model_kwargs.update(
        body=args.body,
        body_inner=args.body_inner,
        body_width_mult=args.body_width_mult,
        body_unet_levels=args.body_unet_levels,
        body_unet_blocks=args.body_unet_blocks,
        body_pconv_ratio=args.body_pconv_ratio,
        body_expand=args.body_expand,
        body_rep_branches=args.body_rep_branches,
        body_act=args.body_act,
    )
    model_kwargs.update(
        body_dw_kernel=args.body_dw_kernel,
        body_dilations=args.body_dilations,
        body_attn_idx=args.body_attn_idx,
        body_attn_dim=args.body_attn_dim,
    )
    if args.dim:
        model_kwargs["dim"] = args.dim
    if args.blocks:
        model_kwargs["blocks"] = args.blocks
    expected = set(inspect.signature(RRTANet.__init__).parameters) - {
        "self",
        "scale",
        "dim",
        "blocks",
        "mlp_ratio",
    }
    unwired = expected - set(model_kwargs)
    if unwired:
        raise SystemExit(
            f"model options not recorded in config: {sorted(unwired)} -- wire them into model_kwargs"
        )
    resolved = dict(RRTA_SIZES[args.size])
    resolved.update(model_kwargs)
    (resolved["size"], resolved["scale"]) = (args.size, args.scale)
    if args.arch == "qrisp":
        from .qrisp_model import QRISPNet, create_qrisp

        resolved = dict(
            f=args.qrisp_f,
            m=args.qrisp_m,
            hyper_hidden=args.qrisp_hyper_hidden,
            hyper_layers=args.qrisp_hyper_layers,
            hyper_param=args.qrisp_hyper_param,
            motion_dilation=args.motion_dilation,
            dilation_block=args.qrisp_dilation_block,
            dejitter=args.dejitter,
            scale=args.scale,
            arch="qrisp",
        )
        unwired = (
            set(inspect.signature(QRISPNet.__init__).parameters)
            - {"self"}
            - set(resolved)
        )
        if unwired:
            raise SystemExit(f"QRISP options not recorded in config: {sorted(unwired)}")
    (teacher, teacher_info) = (None, None)
    if distilling:
        (teacher, teacher_info) = load_teacher(args, device)
        print(
            f"  distill: teacher {teacher_info['run']} (it {teacher_info['ckpt_iteration']}, {teacher_info['params_m']:.3f}M params), weight {args.distill_weight}, alpha weight {args.distill_alpha_weight}",
            flush=True,
        )
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "model_kwargs": resolved,
                **({"stss_data": stss_data} if stss else {}),
                **({"distill_teacher_info": teacher_info} if teacher_info else {}),
            },
            indent=1,
        )
    )
    if args.arch == "qrisp":
        model = create_qrisp(**resolved).to(device)
        params = sum((p.numel() for p in model.parameters())) / 1000000.0
        counts = model.param_counts()
        print(
            f"{args.name}: QRISP f{model.f}-m{model.m} x{args.scale}, {params:.3f}M trainable params (kernel MLP {counts['hypernet'] / 1000000.0:.3f}M, deployed convs {counts['deployed_convs'] / 1000000.0:.4f}M), {len(train)} clips of {args.clip}, hyper {model.hyper_param}, dilation {model.motion_dilation}, temporal_weight {args.temporal_weight}",
            flush=True,
        )
    else:
        model = create_rrta(args.size, args.scale, **model_kwargs).to(device)
        params = sum((p.numel() for p in model.parameters())) / 1000000.0
        print(
            f"{args.name}: RRTA-{args.size} x{args.scale}, {params:.3f}M params, {len(train)} clips of {args.clip}, rectify={args.rectify}, recurrent={args.recurrent}, temporal_weight={args.temporal_weight}",
            flush=True,
        )
    if args.arch == "qrisp":
        groups = model.optimizer_groups(args.lr)
        optimizer = torch.optim.AdamW(
            groups, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2)
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in groups],
            total_steps=args.iters,
            pct_start=0.05,
            final_div_factor=200.0,
        )
        print(
            f"  optimizer groups: "
            + ", ".join(
                (
                    f"{sum((p.numel() for p in g['params'])) / 1000000.0:.3f}M @ {g['lr']:.3g}"
                    for g in groups
                )
            ),
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
        )
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.iters,
            pct_start=0.05,
            final_div_factor=200.0,
        )
    (start, ckpt) = (0, None)
    if args.resume and (out_dir / "last.pt").is_file():
        ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("live", ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        schedule.load_state_dict(ckpt["schedule"])
        start = int(ckpt["iteration"])
        print(f"  resumed at {start}", flush=True)
    evaluating = args.eval_every > 0
    if evaluating and stss:
        eval_clips = data_stss.build_stss_clip_test(
            args.stss_test,
            length=args.eval_clip,
            scale=args.scale,
            depth_mode=args.stss_depth_mode,
        )
        positions = data_stss.stss_test_segments(eval_clips, args.eval_per_scene)
        stationary = data_stss.stss_static_positions(eval_clips)
        positions = sorted(set(positions) | set(stationary))
        print(
            f"  evaluation: {len(positions)} STSS {Path(args.stss_test).name} clips of {args.eval_clip} frames ({len(stationary)} of them inside a stationary-camera run)",
            flush=True,
        )
    elif evaluating:
        eval_clips = QrispClips(
            "test",
            scale=args.scale,
            length=args.eval_clip,
            colour=args.qrisp_colour,
            tiles=None,
            flip=False,
            patch_lr=0,
            motion_jit=args.motion_jit,
            motion_lerp=args.motion_lerp,
            depth_jit=args.depth_jit,
        )
        positions = test_segments(
            scale=args.scale,
            colour=args.qrisp_colour,
            length=args.eval_clip,
            per_scene=args.eval_per_scene,
        )
        stationary = static_positions(eval_clips)
        positions = sorted(set(positions) | set(stationary))
        print(
            f"  evaluation: {len(positions)} clips of {args.eval_clip} frames ({len(stationary)} of them inside a stationary-camera run)",
            flush=True,
        )
    else:
        (eval_clips, positions) = (None, [])
        print("  evaluation: DISABLED (--eval-every 0)", flush=True)
    gt_generator = torch.Generator().manual_seed(args.seed + 90125)
    want_diag = (
        args.fp32_accum
        or args.vcap_max > 0.0
        or args.hist_clip != "none"
        or (args.readout != "accum")
        or (args.accumulate == "samples")
        or (args.consistency_gate != "none")
        or (args.state_mode != "linear")
    )
    need_cand = args.cand_loss_weight > 0.0
    need_alpha = args.distill_alpha_weight > 0.0
    (teacher_ms, teacher_events) = ([], None)
    ema_model = None
    if args.ema > 0.0:
        import copy

        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        if ckpt is not None:
            ema_model.load_state_dict(ckpt["model"])
            print("  restored EMA weights from the checkpoint", flush=True)

    def ema_update() -> None:
        with torch.no_grad():
            for live, shadow in zip(
                model.state_dict().values(), ema_model.state_dict().values()
            ):
                if shadow.dtype.is_floating_point:
                    shadow.mul_(args.ema).add_(live.detach(), alpha=1 - args.ema)
                else:
                    shadow.copy_(live)

    log = (out_dir / "log.jsonl").open("a")
    iterator = iter(loader)
    model.train()
    (running, tick) = ([], time.time())
    for iteration in range(start + 1, args.iters + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        lr_frames = batch["lr"].to(device, non_blocking=True)
        hr_frames = batch["hr"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        motion = batch["motion"].to(device, non_blocking=True)
        jitter = batch["jitter"].to(device, non_blocking=True)
        init_hr = init_mask = None
        if args.gt_init_prob > 0.0:
            draw = torch.rand(lr_frames.shape[0], generator=gt_generator)
            if bool((draw < args.gt_init_prob).any()):
                init_mask = (draw < args.gt_init_prob).to(device)
                init_hr = hr_frames[:, 0]
        logging_now = iteration % args.log_every == 0
        auxes = []
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            if logging_now and want_diag or need_cand or need_alpha:
                (prediction, _, auxes) = model(
                    lr_frames,
                    depth,
                    motion,
                    jitter,
                    return_aux=True,
                    init_hr=init_hr,
                    init_mask=init_mask,
                )
            elif init_hr is not None:
                (prediction, _) = model(
                    lr_frames,
                    depth,
                    motion,
                    jitter,
                    init_hr=init_hr,
                    init_mask=init_mask,
                )
            else:
                (prediction, _) = model(lr_frames, depth, motion, jitter)
            if teacher is not None:
                if device.type == "cuda":
                    teacher_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    teacher_events[0].record()
                with torch.no_grad():
                    teacher_out = teacher(
                        lr_frames,
                        depth,
                        motion,
                        jitter,
                        return_aux=need_alpha,
                        init_hr=init_hr,
                        init_mask=init_mask,
                    )
                if teacher_events is not None:
                    teacher_events[1].record()
        (loss, stats) = sequence_loss(
            prediction.float(),
            hr_frames,
            motion,
            args.scale,
            args.warmup,
            args.temporal_weight,
            args.loss,
            args.loss_weighting,
            1.0 / args.alpha_min,
            luma_weight=args.luma_weight,
            temporal_objective=args.temporal_objective,
            static_temporal=args.static_temporal_weight,
            static_threshold=args.static_threshold,
            spread=batch.get("spread"),
            temporal_warp=args.temporal_warp,
        )
        if need_cand:
            cand_term = candidate_loss(
                torch.stack([a["candidate"] for a in auxes], dim=1).float(),
                hr_frames,
                args.warmup,
                args.loss_weighting,
                1.0 / args.alpha_min,
            )
            stats["cand"] = float(cand_term.detach())
            loss = loss + args.cand_loss_weight * cand_term
        if teacher is not None:
            (d_image, d_alpha) = distill_terms(
                prediction,
                teacher_out[0],
                auxes,
                teacher_out[2] if need_alpha else [],
                args.warmup,
                args.loss_weighting,
                1.0 / args.alpha_min,
                need_alpha,
            )
            stats["distill"] = float(d_image.detach())
            if args.distill_weight > 0.0:
                loss = loss + args.distill_weight * d_image
            if d_alpha is not None:
                stats["distill_alpha"] = float(d_alpha.detach())
                loss = loss + args.distill_alpha_weight * d_alpha
            del teacher_out
            if teacher_events is not None:
                teacher_ms.append(teacher_events[0].elapsed_time(teacher_events[1]))
        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"  !! skipped step {iteration}: non-finite grad norm", flush=True
                )
        else:
            print(f"  !! skipped step {iteration}: non-finite loss", flush=True)
        schedule.step()
        if ema_model is not None:
            ema_update()
        running.append(stats)
        if logging_now:
            rate = args.log_every * args.batch * args.clip / (time.time() - tick)
            means = {
                k: float(np.mean([r[k] for r in running if k in r]))
                for k in running[-1]
            }
            text = "  ".join((f"{k} {v:.5f}" for (k, v) in means.items()))
            print(
                f"  {args.name} {iteration}/{args.iters} {text} {rate:.0f} frames/s lr {schedule.get_last_lr()[0]:.2e}"
                + (
                    f" teacher_ms/it {float(np.mean(teacher_ms)):.1f} it/s {rate / (args.batch * args.clip):.2f}"
                    if teacher_ms
                    else ""
                ),
                flush=True,
            )
            teacher_ms = []
            diags = [a["diag"] for a in auxes if "diag" in a]
            if diags:
                summary = {
                    k: float(
                        torch.stack([d[k].float() for d in diags if k in d]).mean()
                    )
                    for k in diags[-1]
                }
                print(
                    f"  {args.name} {iteration} diag "
                    + "  ".join((f"{k} {v:.4f}" for (k, v) in sorted(summary.items())))
                    + f"  it/s {rate / (args.batch * args.clip):.2f}"
                    + (
                        f"  gt_init {int(init_mask.sum())}/{lr_frames.shape[0]}"
                        if init_mask is not None
                        else ""
                    ),
                    flush=True,
                )
            (running, tick) = ([], time.time())
        final = iteration == args.iters
        if evaluating and (iteration % args.eval_every == 0 or final):
            metrics = evaluate(
                ema_model if ema_model is not None else model,
                args,
                device,
                eval_clips,
                positions,
            )
            log.write(json.dumps({"iteration": iteration, "metrics": metrics}) + "\n")
            log.flush()
            head = (
                f"psnr {metrics.get('psnr', float('nan')):.3f} (bicubic {metrics.get('psnr_bicubic', float('nan')):.3f}, gain {metrics.get('gain_over_bicubic', float('nan')):+.3f}; "
                + ("STSS Arena clips, RGB) " if stss else "paper gain +3.00) ")
                + f"tpsnr {metrics.get('tpsnr', float('nan')):.3f} pixel_std {metrics.get('pixel_std', float('nan')):.3f}"
            )
            print(f"  {args.name} @{iteration} {head}", flush=True)
        if iteration % args.ckpt_every == 0 or final:
            torch.save(
                {
                    "model": (
                        ema_model if ema_model is not None else model
                    ).state_dict(),
                    "live": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "iteration": iteration,
                },
                out_dir / "last.pt",
            )
            if args.keep_every and iteration % args.keep_every == 0:
                shutil.copyfile(out_dir / "last.pt", out_dir / f"ckpt_{iteration}.pt")
    if not evaluating:
        print(f"{args.name}: done (evaluation disabled, no result.json)", flush=True)
        return
    metrics = evaluate(
        ema_model if ema_model is not None else model,
        args,
        device,
        eval_clips,
        positions,
    )
    split = f"STSS-{Path(args.stss_test).name}-clips" if stss else "QRISP-test"
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "model": args.arch,
                "size": args.size,
                "params_m": params,
                "iteration": args.iters,
                "metrics": {split: metrics},
                "config": vars(args),
            },
            indent=1,
        )
    )
    print(f"{args.name}: done", flush=True)


if __name__ == "__main__":
    main()
