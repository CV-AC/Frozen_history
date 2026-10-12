from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsr.paths import (
    REPOSITORY,
    DATA_ROOT,
    QRISP_RAW,
    QRISP_LMDB,
    STSS_PACKED,
    ARM_TEST,
    STSSNET_UTILS,
)
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT = REPOSITORY
CURRENT = REPOSITORY
TEST = QRISP_RAW / "TestSet"
OUTDIR = REPOSITORY / "results" / "graphics"
PER = OUTDIR / "per_method"
BASELINES = ("DLSS2", "FSR", "bicubic", "GT")
WARMUP = 8
FPS = 60.0
DISPLAY = "standard_fhd"
STATIC_SEGMENTS = {
    "AbandonedSchool/0001": (274, 292),
    "SeaPort/0000#a": (20, 38),
    "SeaPort/0000#b": (205, 223),
}
AS_STOP = ("AbandonedSchool/0001", 275, 292)


def md5(path: Path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def resolve_code(method, override):
    return (Path(override).resolve() if override else REPOSITORY, "release source")


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", help="run name under runs/, or DLSS2 / bicubic / GT")
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument(
        "--code", default=None, help="override the code directory (default: repository)"
    )
    ap.add_argument(
        "--cvvdp-frames",
        type=int,
        default=0,
        help="score only the first N scored frames of each run with CVVDP (0 = all 292); applied identically to every method",
    )
    ap.add_argument("--display", default=DISPLAY)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="SMOKE ONLY: first N frames of each run",
    )
    ap.add_argument("--no-cvvdp", action="store_true")
    ap.add_argument(
        "--sanity",
        action="store_true",
        help="GT vs GT: FLIP must be 0 and JOD the maximum (10)",
    )
    ap.add_argument(
        "--collect",
        action="store_true",
        help="merge per-method JSONs into graphics_metrics.json and graphics_metrics.md",
    )
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def score(args) -> None:
    (code, code_why) = resolve_code(args.method, args.code)
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "scripts"))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffsr.checkpoint import load_run
    from diffsr.metrics import psnr
    from diffsr.data_qrisp import QrispDataset, QRISP_LMDB, _Lmdb
    from protocol_full import input_pipeline, dataset_for, frame_orders, camera_static
    import flip_evaluator as flip
    import pycvvdp
    from pycvvdp.video_source import video_source_array

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (method, scale) = (args.method, args.scale)
    model = cfg = pipe = None
    if method in BASELINES:
        if scale is None:
            raise SystemExit("--scale is required for a baseline")
        if method in ("DLSS2", "FSR") and scale != 2:
            print(
                f"  NOTE: shipped {method} frames are 540p -> 1080p (x2); --scale {scale} only selects the LR/motion variant",
                flush=True,
            )
        data = QrispDataset(
            "test",
            scale=scale,
            colour="MipBias",
            with_depth=False,
            with_motion=False,
            patch_lr=0,
            flip=False,
            tiles=None,
        )
    else:
        (model, cfg) = load_run(PROJECT / "runs" / method, dev, args.ckpt)
        pipe = input_pipeline(cfg)
        if scale is None:
            scale = pipe["scale"]
        data = dataset_for(pipe, scale)
    field = _Lmdb(QRISP_LMDB / "QRISP_test_Motion270.lmdb")
    metric = (
        None
        if args.no_cvvdp
        else pycvvdp.cvvdp(
            display_name=args.display, heatmap=None, device=dev, quiet=True
        )
    )
    print(
        f"{method} x{scale}  code={code} ({code_why})"
        + (f"  [{args.ckpt} @ {cfg.get('ckpt_iteration')}]" if cfg else "")
        + (f"  input {pipe}" if pipe else ""),
        flush=True,
    )
    if metric is not None:
        print(
            f"  CVVDP display '{args.display}': ppd {float(metric.pix_per_deg):.2f}",
            flush=True,
        )
    grouping = frame_orders(data, True)
    (per_frame, per_run, nonfinite) = ([], {}, [])
    seg_frames: dict[str, list] = {k: [] for k in STATIC_SEGMENTS}
    seg_refs: dict[str, list] = {k: [] for k in STATIC_SEGMENTS}
    flip_params = None
    t_flip = t_cvvdp = t_model = 0.0
    missing = 0
    for scene, orders in sorted(grouping.items()):
        for order in orders:
            if args.max_frames:
                order = order[: args.max_frames]
            meta0 = data.frames[order[0]]
            run_name = f"{scene}/{meta0['sequence']}"
            state = None
            (vid_test, vid_ref) = ([], [])
            run_rows = []
            for n, i in enumerate(order):
                meta = data.frames[i]
                item = data[i]
                t0 = time.time()
                if model is not None:
                    lr = item["lr"][None, :3].to(dev)
                    depth = item["depth"][None].to(dev)
                    motion = item["motion"][None].to(dev)
                    jit = item.get("jitter")
                    jitter = (jit[None] if jit is not None else torch.zeros(1, 2)).to(
                        dev
                    )
                    with torch.inference_mode():
                        (out, state, _) = model.step(lr, depth, motion, jitter, state)
                elif method in ("DLSS2", "FSR"):
                    png = (
                        TEST
                        / scene
                        / "1080p"
                        / method
                        / meta["sequence"]
                        / f"{meta['frame']}.png"
                    )
                    if not png.is_file():
                        missing += 1
                        out = None
                    else:
                        arr = (
                            np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
                            / 255.0
                        )
                        out = torch.from_numpy(arr).permute(2, 0, 1)[None]
                elif method == "bicubic":
                    out = F.interpolate(
                        item["lr"][None, :3],
                        scale_factor=scale,
                        mode="bicubic",
                        align_corners=False,
                    ).clamp(0, 1)
                else:
                    out = item["hr"][None]
                t_model += time.time() - t0
                if n < WARMUP:
                    continue
                if out is None:
                    raise SystemExit(f"missing {method} frame {meta['key']}")
                finite = bool(torch.isfinite(out).all())
                if not finite:
                    nonfinite.append(meta["key"])
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                pred = out[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255.0
                ref = item["hr"].permute(1, 2, 0).float().numpy() * 255.0
                p_float = psnr(pred, ref, scale, crop_border=0, to_y=True)
                pred_u8 = np.round(pred).clip(0, 255).astype(np.uint8)
                ref_u8 = np.round(ref).clip(0, 255).astype(np.uint8)
                p_u8 = psnr(pred_u8, ref_u8, scale, crop_border=0, to_y=True)
                t0 = time.time()
                (_, fl, flip_params) = flip.evaluate(
                    ref_u8.astype(np.float32) / 255.0,
                    pred_u8.astype(np.float32) / 255.0,
                    "LDR",
                    inputsRGB=True,
                    applyMagma=False,
                    computeMeanError=True,
                    parameters={},
                )
                t_flip += time.time() - t0
                m270 = torch.from_numpy(
                    np.ascontiguousarray(field.array(data.samples[i][1]))
                ).permute(2, 0, 1)[None]
                static = bool(camera_static(m270, 4))
                fr = int(meta["frame"])
                row = {
                    "run": run_name,
                    "frame": fr,
                    "flip": float(fl),
                    "psnr_y_float": float(p_float),
                    "psnr_y_u8": float(p_u8),
                    "static": static,
                }
                per_frame.append(row)
                run_rows.append(row)
                vid_test.append(pred_u8)
                vid_ref.append(ref_u8)
                for seg, (a, b) in STATIC_SEGMENTS.items():
                    if seg.split("#")[0] == run_name and a <= fr <= b:
                        seg_frames[seg].append(pred_u8)
                        seg_refs[seg].append(ref_u8)
            jod = None
            cv_n = len(vid_test)
            if metric is not None and vid_test:
                if args.cvvdp_frames:
                    cv_n = min(cv_n, args.cvvdp_frames)
                t0 = time.time()
                with torch.inference_mode():
                    vs = video_source_array(
                        np.stack(vid_test[:cv_n]),
                        np.stack(vid_ref[:cv_n]),
                        FPS,
                        dim_order="FHWC",
                        display_photometry=metric.display_photometry,
                    )
                    (q, _) = metric.predict_video_source(vs)
                jod = float(q)
                t_cvvdp += time.time() - t0
            per_run[run_name] = {
                "frames_scored": len(run_rows),
                "cvvdp_frames": cv_n,
                "flip": float(np.mean([r["flip"] for r in run_rows])),
                "psnr_y_float": float(np.mean([r["psnr_y_float"] for r in run_rows])),
                "psnr_y_u8": float(np.mean([r["psnr_y_u8"] for r in run_rows])),
                "jod": jod,
            }
            print(
                f"  {run_name:<22}{len(run_rows):>4} frames  PSNR-Y {per_run[run_name]['psnr_y_float']:7.3f} (u8 {per_run[run_name]['psnr_y_u8']:7.3f})  FLIP {per_run[run_name]['flip']:.4f}  JOD {(jod if jod is None else round(jod, 4))} over {cv_n}  [model {t_model:.0f}s flip {t_flip:.0f}s cvvdp {t_cvvdp:.0f}s]",
                flush=True,
            )
            del vid_test, vid_ref
    segs = {}
    for seg, frs in seg_frames.items():
        if not frs:
            continue
        arr = np.stack(frs).astype(np.float64).transpose(0, 3, 1, 2)
        refa = np.stack(seg_refs[seg]).astype(np.float64).transpose(0, 3, 1, 2)
        jod = None
        if metric is not None:
            with torch.inference_mode():
                vs = video_source_array(
                    np.stack(frs),
                    np.stack(seg_refs[seg]),
                    FPS,
                    dim_order="FHWC",
                    display_photometry=metric.display_photometry,
                )
                (q, _) = metric.predict_video_source(vs)
            jod = float(q)
        (a, b) = STATIC_SEGMENTS[seg]
        segs[seg] = {
            "frames": [a, b],
            "n": len(frs),
            "pixel_std_u8": float(arr.std(axis=0, ddof=0).mean()),
            "pixel_std_reference": float(refa.std(axis=0, ddof=0).mean()),
            "jod": jod,
            "flip": float(
                np.mean(
                    [
                        r["flip"]
                        for r in per_frame
                        if r["run"] == seg.split("#")[0] and a <= r["frame"] <= b
                    ]
                )
            ),
        }
    stop = [
        r
        for r in per_frame
        if r["run"] == AS_STOP[0] and AS_STOP[1] <= r["frame"] <= AS_STOP[2]
    ]
    stop_std = None
    if seg_frames.get("AbandonedSchool/0001"):
        a0 = STATIC_SEGMENTS["AbandonedSchool/0001"][0]
        frs = seg_frames["AbandonedSchool/0001"][AS_STOP[1] - a0 : AS_STOP[2] - a0 + 1]
        if len(frs) == AS_STOP[2] - AS_STOP[1] + 1:
            stop_std = float(
                np.stack(frs)
                .astype(np.float64)
                .transpose(0, 3, 1, 2)
                .std(axis=0, ddof=0)
                .mean()
            )

    def mean_by(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else None

    scenes = sorted({r["run"].split("/")[0] for r in per_frame})
    per_scene = {
        s: {
            k: mean_by([r for r in per_frame if r["run"].startswith(s + "/")], k)
            for k in ("flip", "psnr_y_float", "psnr_y_u8")
        }
        for s in scenes
    }
    jods = [v["jod"] for v in per_run.values() if v["jod"] is not None]
    result = {
        "method": method,
        "scale": scale,
        "kind": "baseline" if method in BASELINES else "model",
        "code_resolution": code_why,
        "ckpt": None if cfg is None else args.ckpt,
        "ckpt_iteration": None if cfg is None else cfg.get("ckpt_iteration"),
        "input_pipeline": pipe,
        "protocol": {
            "reset_runs": True,
            "warmup": WARMUP,
            "stride": 1,
            "max_frames": args.max_frames,
            "cvvdp_frames": args.cvvdp_frames or "all",
            "scored_output": "uint8 round(clamp(out)*255) for FLIP and CVVDP; psnr_y_float = protocol_full check",
        },
        "flip": {
            "scene_mean": float(np.mean([v["flip"] for v in per_scene.values()])),
            "frame_mean": mean_by(per_frame, "flip"),
            "static_frames": mean_by([r for r in per_frame if r["static"]], "flip"),
            "moving_frames": mean_by([r for r in per_frame if not r["static"]], "flip"),
            "n_static": sum((r["static"] for r in per_frame)),
            "parameters": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for (k, v) in (flip_params or {}).items()
            },
            "per_scene": {s: v["flip"] for (s, v) in per_scene.items()},
        },
        "cvvdp": {
            "display": args.display,
            "ppd": None if metric is None else float(metric.pix_per_deg),
            "fps": FPS,
            "mean_over_runs": float(np.mean(jods)) if jods else None,
            "scene_mean": float(
                np.mean(
                    [
                        np.mean(
                            [
                                v["jod"]
                                for (k, v) in per_run.items()
                                if k.startswith(s + "/")
                            ]
                        )
                        for s in scenes
                    ]
                )
            )
            if jods
            else None,
            "per_run": {k: v["jod"] for (k, v) in per_run.items()},
            "static_segments": {k: v["jod"] for (k, v) in segs.items()},
        },
        "psnr_y_float": float(np.mean([v["psnr_y_float"] for v in per_scene.values()])),
        "psnr_y_u8": float(np.mean([v["psnr_y_u8"] for v in per_scene.values()])),
        "per_scene": per_scene,
        "per_run": per_run,
        "static_segments": segs,
        "as_stop_275_292": {
            "pixel_std_u8": stop_std,
            "psnr_y_float": mean_by(stop, "psnr_y_float"),
            "flip": mean_by(stop, "flip"),
            "frames": len(stop),
        },
        "nonfinite_frames": nonfinite,
        "missing_frames": missing,
        "per_frame": per_frame,
        "timing_s": {
            "model_or_io": round(t_model, 1),
            "flip": round(t_flip, 1),
            "cvvdp": round(t_cvvdp, 1),
        },
        "provenance": {
            "argv": sys.argv,
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "torch": torch.__version__,
            "python": sys.executable,
            "flip_evaluator": _version("flip-evaluator"),
            "cvvdp": _version("cvvdp"),
            "md5": {
                "scripts/metrics_graphics.py": md5(Path(__file__)),
                "protocol_full.py": md5(code / "scripts/protocol_full.py"),
                "diffsr/rrta.py": md5(code / "diffsr/rrta.py"),
                "diffsr/checkpoint.py": md5(code / "diffsr/checkpoint.py"),
                "diffsr/data_qrisp.py": md5(code / "diffsr/data_qrisp.py"),
                "diffsr/metrics.py": md5(code / "diffsr/metrics.py"),
            },
        },
    }
    ref = reference_psnr(method, scale)
    if ref is not None:
        result["reference_protocol_psnr"] = ref
        result["reference_delta"] = result["psnr_y_float"] - ref["psnr_y"]
    out = Path(args.out) if args.out else PER / f"{method}_x{scale}.json"
    if args.max_frames:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        f"  scene-mean PSNR-Y {result['psnr_y_float']:.3f}"
        + (
            f" (reference {ref['psnr_y']:.3f}, delta {result['reference_delta']:+.4f})"
            if ref
            else ""
        )
        + f"  FLIP {result['flip']['scene_mean']:.4f}  JOD {result['cvvdp']['scene_mean']}  AS-stop pixel std (u8) {stop_std}",
        flush=True,
    )
    print(f"  wrote {out}")


def _version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def reference_psnr(method: str, scale: int) -> dict | None:
    ref_dir = PROJECT / "results"
    cands = sorted(ref_dir.glob(f"*_{method}_protocol.json"))
    for c in cands:
        try:
            d = json.loads(c.read_text())
            row = d["table"][method]
            if (
                row.get("stride") == 1
                and row.get("reset_runs") is True
                and (row.get("warmup") == WARMUP)
                and (not row.get("tta"))
                and (not row.get("max_frames"))
                and (d.get("scale") == scale)
            ):
                return {
                    "file": str(c),
                    "psnr_y": row["psnr_y"],
                    "per_run_y": row.get("per_run_y"),
                }
        except Exception:
            continue
    if method == "DLSS2":
        for c in sorted(ref_dir.glob("*ship*.json")):
            try:
                d = json.loads(c.read_text())
                row = d.get("DLSS2")
                if row and row.get("stride") == 1 and row.get("reset_runs"):
                    return {"file": str(c), "psnr_y": row["psnr_y"]}
            except Exception:
                continue
    return None


def sanity(args) -> None:
    sys.path.insert(0, str(CURRENT))
    sys.path.insert(0, str(CURRENT / "scripts"))
    import numpy as np
    import torch
    from diffsr.data_qrisp import QrispDataset
    import flip_evaluator as flip
    import pycvvdp
    from pycvvdp.video_source import video_source_array

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = QrispDataset(
        "test",
        scale=2,
        colour="MipBias",
        with_depth=False,
        with_motion=False,
        patch_lr=0,
        flip=False,
        tiles=None,
    )
    idx = list(range(575, 575 + 40))
    (gts, lrs) = ([], [])
    t0 = time.time()
    for i in idx:
        item = data[i]
        gts.append(np.round(item["hr"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        up = torch.nn.functional.interpolate(
            item["lr"][None, :3], scale_factor=2, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        lrs.append(np.round(up[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    print(f"loaded {len(idx)} frames in {time.time() - t0:.1f}s", flush=True)
    rep = {}
    t0 = time.time()
    fl_gt = [
        flip.evaluate(
            g.astype(np.float32) / 255,
            g.astype(np.float32) / 255,
            "LDR",
            applyMagma=False,
        )[1]
        for g in gts[:8]
    ]
    t1 = time.time()
    fl_bic = [
        flip.evaluate(
            g.astype(np.float32) / 255,
            b.astype(np.float32) / 255,
            "LDR",
            applyMagma=False,
        )
        for (g, b) in zip(gts[:8], lrs[:8])
    ]
    rep["flip_gt_vs_gt"] = [float(v) for v in fl_gt]
    rep["flip_bicubic_x2_vs_gt"] = [float(v[1]) for v in fl_bic]
    rep["flip_parameters"] = {
        k: float(v) if isinstance(v, (int, float)) else str(v)
        for (k, v) in fl_bic[0][2].items()
    }
    rep["flip_s_per_frame"] = (time.time() - t1) / 8
    rep["flip_s_per_frame_identical"] = (t1 - t0) / 8
    metric = pycvvdp.cvvdp(
        display_name=args.display, heatmap=None, device=dev, quiet=True
    )
    rep["cvvdp_display"] = args.display
    rep["cvvdp_ppd"] = float(metric.pix_per_deg)
    dm = metric.display_photometry
    rep["cvvdp_display_photometry"] = {
        k: float(getattr(dm, k))
        if isinstance(getattr(dm, k, None), (int, float))
        else str(getattr(dm, k, None))
        for k in ("Y_peak", "contrast", "E_ambient", "k_refl", "EOTF")
    }
    for name, test in (("gt_vs_gt", gts), ("bicubic_x2_vs_gt", lrs)):
        t0 = time.time()
        vs = video_source_array(
            np.stack(test), np.stack(gts), FPS, dim_order="FHWC", display_photometry=dm
        )
        with torch.inference_mode():
            (q, _) = metric.predict_video_source(vs)
        rep[f"jod_{name}_40f"] = float(q)
        rep[f"cvvdp_s_40f_{name}"] = time.time() - t0
    with torch.inference_mode():
        (q_img, _) = metric.predict(gts[0], gts[0], dim_order="HWC")
    rep["jod_gt_vs_gt_image"] = float(q_img)
    ok = (
        max(rep["flip_gt_vs_gt"]) == 0.0
        and abs(rep["jod_gt_vs_gt_40f"] - 10.0) < 0.0001
        and (abs(rep["jod_gt_vs_gt_image"] - 10.0) < 0.0001)
    )
    rep["PASS"] = bool(ok)
    rep["versions"] = {
        "flip_evaluator": _version("flip-evaluator"),
        "cvvdp": _version("cvvdp"),
        "torch": torch.__version__,
        "python": sys.executable,
    }
    out = OUTDIR / "sanity_gt_vs_gt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(f"wrote {out}")


ORDER = [
    ("qrisp_x2", 2),
    ("DLSS2", 2),
    ("bicubic", 2),
    ("qrisp_x3", 3),
    ("bicubic", 3),
    ("qrisp_x4", 4),
    ("bicubic", 4),
]


def collect(args) -> None:
    import numpy as np

    rows = []
    files = sorted(PER.glob("*_x[234].json"))
    by_key = {}
    for f in files:
        d = json.loads(f.read_text())
        by_key[d["method"], d["scale"]] = d
    keys = [k for k in ORDER if k in by_key] + sorted(
        (k for k in by_key if k not in ORDER)
    )
    table = {}
    for method, scale in keys:
        d = by_key[method, scale]
        segs = d.get("static_segments", {})
        table[f"{method}@x{scale}"] = {
            "method": method,
            "scale": scale,
            "kind": d["kind"],
            "ckpt_iteration": d.get("ckpt_iteration"),
            "psnr_y_float": d["psnr_y_float"],
            "psnr_y_u8": d["psnr_y_u8"],
            "reference_protocol_psnr": (d.get("reference_protocol_psnr") or {}).get(
                "psnr_y"
            ),
            "flip_scene_mean": d["flip"]["scene_mean"],
            "flip_static_frames": d["flip"]["static_frames"],
            "flip_moving_frames": d["flip"]["moving_frames"],
            "flip_per_scene": d["flip"]["per_scene"],
            "jod_mean_over_runs": d["cvvdp"]["mean_over_runs"],
            "jod_scene_mean": d["cvvdp"]["scene_mean"],
            "jod_per_run": d["cvvdp"]["per_run"],
            "jod_static_segments": d["cvvdp"]["static_segments"],
            "jod_static_mean": float(
                np.mean(
                    [v for v in d["cvvdp"]["static_segments"].values() if v is not None]
                )
            )
            if d["cvvdp"]["static_segments"]
            else None,
            "static_pixel_std_u8": {k: v["pixel_std_u8"] for (k, v) in segs.items()},
            "as_stop_275_292": d.get("as_stop_275_292"),
            "cvvdp_frames": d["protocol"]["cvvdp_frames"],
            "nonfinite_frames": len(d.get("nonfinite_frames", [])),
            "per_method_file": str(PER / f"{method}_x{scale}.json"),
        }
        rows.append(table[f"{method}@x{scale}"])
    sanity_file = OUTDIR / "sanity_gt_vs_gt.json"
    sanity_rep = json.loads(sanity_file.read_text()) if sanity_file.is_file() else None
    any_d = next(iter(by_key.values())) if by_key else {}
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "protocol": "scripts/protocol_full.py: every test frame, reset at each camera run, warm-up 8, stride 1; FLIP/CVVDP on the 8-bit output vs 8-bit Enhanced GT",
        "flip": "LDR-FLIP (flip-evaluator), default 67 ppd; mean per frame, averaged per scene then over scenes",
        "cvvdp": f"ColorVideoVDP (cvvdp), display '{DISPLAY}' (24-inch 1920x1080 sRGB, 200 cd/m^2, 1000:1, 250 lux, 0.6 m), 60 fps; one JOD per camera run; scene_mean averages the runs of a scene then the scenes; static = three stationary segments",
        "cvvdp_ppd": (any_d.get("cvvdp") or {}).get("ppd"),
        "sanity": sanity_rep,
        "table": table,
    }
    (OUTDIR / "graphics_metrics.json").write_text(json.dumps(out, indent=1))

    def f(v, nd=4):
        return "--" if v is None else f"{v:.{nd}f}"

    lines = [
        "| method | scale | PSNR-Y (float / u8) | FLIP (lower better) | FLIP static / moving frames | JOD per-run mean (higher better) | JOD static segments | static pixel std u8 (AS stop) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        stop = (r["as_stop_275_292"] or {}).get("pixel_std_u8")
        lines.append(
            f"| {r['method']} | x{r['scale']} | {f(r['psnr_y_float'], 3)} / {f(r['psnr_y_u8'], 3)} | {f(r['flip_scene_mean'])} | {f(r['flip_static_frames'])} / {f(r['flip_moving_frames'])} | {f(r['jod_scene_mean'])} | {f(r['jod_static_mean'])} | {f(stop, 3)} |"
        )
    lines += [
        "",
        "Per camera run (JOD / FLIP):",
        "",
        "| method | scale | "
        + " | ".join(["AS/0000", "AS/0001", "SeaPort/0000", "SSD/0000", "SSD/0001"])
        + " |",
        "|---|---|---|---|---|---|---|",
    ]
    runs_order = [
        "AbandonedSchool/0000",
        "AbandonedSchool/0001",
        "SeaPort/0000",
        "SpaceShipDemo/0000",
        "SpaceShipDemo/0001",
    ]
    for r in rows:
        d = by_key[r["method"], r["scale"]]
        cells = []
        for rn in runs_order:
            pr = d["per_run"].get(rn)
            cells.append("--" if pr is None else f"{f(pr['jod'], 3)} / {f(pr['flip'])}")
        lines.append(f"| {r['method']} | x{r['scale']} | " + " | ".join(cells) + " |")
    note = f"\nProtocol: protocol_full (reset per camera run, warm-up 8, stride 1); FLIP/CVVDP on the 8-bit output. CVVDP display '{DISPLAY}' (24-inch 1080p sRGB, 200 cd/m^2, 250 lux, 0.6 m). FLIP: LDR, default 67 ppd. CVVDP frames per run: {(rows[0]['cvvdp_frames'] if rows else '--')}."
    (OUTDIR / "graphics_metrics.md").write_text("\n".join(lines) + "\n" + note + "\n")
    print("\n".join(lines) + note)
    print(
        f"wrote {OUTDIR / 'graphics_metrics.json'} and {OUTDIR / 'graphics_metrics.md'}"
    )


def main() -> None:
    args = parse_args()
    if args.sanity:
        sanity(args)
    elif args.collect:
        collect(args)
    else:
        if not args.method:
            raise SystemExit("--method is required")
        score(args)


if __name__ == "__main__":
    main()
