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
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

MINE = Path(__file__).resolve().parent.parent
PROJECT = REPOSITORY
OUT_DIR = REPOSITORY / "results" / "benchmarks"
FRAMES_DIR = REPOSITORY / "results" / "frames"
BASELINES = ("bicubic", "regbicubic", "lanczos3", "taau", "taa_clamp", "GT")
STSS_PAPER_RANGE = {
    "Arena_test": (500, 1500),
    "Lewis_test": (5, 1000),
    "SunTemple_test": (5, 1000),
    "Subway_test": (6, 1000),
}
TAAU_SELECT = REPOSITORY / "configs" / "baselines" / "taau_x2.json"
TAAC_SELECT = REPOSITORY / "configs" / "baselines" / "taa_clamp_x2.json"
TPSNR_CAP = 60.0


def md5(path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_code(method):
    return (REPOSITORY, "release source")


def nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    keep = arr[~np.isnan(arr)]
    return float(keep.mean()) if keep.size else float("nan")


def camera_static(m_lr, factor: int, tol: float = 0.02) -> bool:
    med = m_lr.flatten(2).abs().median(dim=2).values.amax()
    return bool(med * factor / 4.0 < tol)


def aggregate_tpsnr(pairs: list[dict]) -> dict:

    def pooled(ps):
        if not ps:
            return float("nan")
        sse = sum((p["sse"] for p in ps))
        cnt = sum((p["count"] for p in ps))
        if not np.isfinite(sse):
            return float("nan")
        return (
            float("inf") if sse == 0 else float(10.0 * np.log10(255.0**2 * cnt / sse))
        )

    db = np.array([p["tpsnr"] for p in pairs], dtype=np.float64)
    nan = float("nan")
    with np.errstate(invalid="ignore"):
        f = db[~np.isnan(db)]
        return {
            "meandb": float(np.mean(f)) if len(f) else nan,
            "cap60": float(np.mean(np.minimum(f, TPSNR_CAP))) if len(f) else nan,
            "pooled": pooled(pairs),
            "pooled_static": pooled([p for p in pairs if p["static"]]),
            "pooled_moving": pooled([p for p in pairs if not p["static"]]),
            "pairs": len(pairs),
            "static_pairs": int(sum((bool(p["static"]) for p in pairs))),
            "nan_pairs": int(np.sum(np.isnan(db))),
        }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=("arm", "stss"), required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument(
        "--scenes", nargs="+", default=["Arena_test"], help="STSS packed scene names"
    )
    ap.add_argument("--fps-mode", type=int, choices=(60, 30), default=60)
    ap.add_argument(
        "--no-paper-range",
        action="store_true",
        help="STSS: score every packed frame, not the STSSNet test range",
    )
    ap.add_argument("--exposure", type=float, default=2.0)
    ap.add_argument("--encode", choices=("none", "srgb"), default="none")
    ap.add_argument(
        "--jitter-variant",
        default="loader",
        choices=("loader", "ymirror", "zero"),
        help="Arm VALIDATION only: feed a wrong jitter",
    )
    ap.add_argument(
        "--motion-variant",
        default="loader",
        choices=("loader", "negate", "jitfield"),
        help="Arm: 'negate' = VALIDATION (wrong sign); 'jitfield' = emulate QRISP's JITTERED motion field for arms trained on it (motion_jit=True): m + (o_t - o_t-1), o = image offset of the samples (measured on QRISP: MJit - M = d(-jy, +jx), results/benchmarks/qrisp_jitfield_relation.json)",
    )
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--no-cvvdp", action="store_true")
    ap.add_argument(
        "--save-frames",
        nargs="*",
        type=int,
        default=None,
        help="dataset frame numbers to save as PNG candidates",
    )
    ap.add_argument("--max-frames", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    (code, code_why) = resolve_code(args.method)
    sys.path.insert(0, str(code))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from diffsr.metrics import psnr, ssim
    from diffsr.temporal import temporal_gradient_error
    from diffsr import samples as smp

    ups = load_file_module("release_upsample", MINE / "diffsr/upsample.py")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = 2
    (method, model, cfg, base) = (args.method, None, None, None)
    if method not in BASELINES:
        from diffsr.checkpoint import load_run

        (model, cfg) = load_run(PROJECT / "runs" / method, dev, args.ckpt)
        sc = cfg.get("scale") or cfg.get("model_kwargs", {}).get("scale")
        if int(sc) != S:
            raise SystemExit(f"{method} is x{sc}; benchmarks are x2")
    elif method in ("taau", "taa_clamp"):
        from diffsr import taau

        sel_path = TAAU_SELECT if method == "taau" else TAAC_SELECT
        sel = json.loads(sel_path.read_text())
        tcfg = taau.TAAUConfig(**sel["chosen_config"])
        if method == "taau":
            base = taau.TAAU(S, tcfg)
        else:
            from diffsr import taa_clamp

            base = taa_clamp.TAAClamp(S, tcfg)
        base_info = {
            "selection_file": str(sel_path),
            "chosen": sel["chosen"],
            "config": sel["chosen_config"],
            "train_psnr_y_qrisp_x2": sel["chosen_psnr_y"],
            "retuned_on_this_dataset": False,
        }
    if args.dataset == "arm":
        dmod = load_file_module("release_data_arm", MINE / "diffsr/data_arm.py")
        data = dmod.ArmSequence(
            exposure=args.exposure, encode=args.encode, max_frames=args.max_frames
        )
        (fps, factor) = (30.0, S)
        orders = data.runs()
        data_prov = data.provenance()
    else:
        from diffsr.data_stss import StssSequence, stss_pipeline

        depth_mode = stss_pipeline(cfg)["depth_mode"] if cfg else "linear"
        data = StssSequence(
            [str(STSS_PACKED / s) for s in args.scenes], scale=S, depth_mode=depth_mode
        )
        (fps, factor) = (float(args.fps_mode), S)
        orders = {}
        for scene, runs_ in data.runs().items():
            (lo, hi) = STSS_PAPER_RANGE.get(scene, (-1, 10**9))
            if args.no_paper_range:
                (lo, hi) = (-1, 10**9)
            for o in runs_:
                o = [i for i in o if lo <= data.frames[i]["n"] <= hi]
                if args.fps_mode == 30:
                    (ev, cur) = ([], [])
                    idx_of = {data.frames[i]["n"]: i for i in o}
                    for i in o:
                        n = data.frames[i]["n"]
                        if n % 2:
                            continue
                        if cur and (
                            not (n - 1 in idx_of and data.frames[cur[-1]]["n"] == n - 2)
                        ):
                            ev.append(cur)
                            cur = []
                        cur.append(i)
                    if cur:
                        ev.append(cur)
                    orders.setdefault(scene, []).extend(ev)
                elif o:
                    orders.setdefault(scene, []).append(o)
            if args.max_frames:
                orders[scene] = [o[: args.max_frames] for o in orders[scene]]
        n_index = {
            (data.frames[i]["scene"], data.frames[i]["n"]): i for i in range(len(data))
        }
        data_prov = {
            name: {
                "root": str(sc.root),
                "frames": len(sc.frames),
                "frames_npy_md5": md5(sc.root / "frames.npy"),
                "cuts": sorted(sc.cuts),
            }
            for (name, sc) in data.scenes.items()
        }
        data_prov["depth_mode"] = depth_mode
    if args.dataset == "stss":
        stm = load_file_module(
            "stssnet_matlab_metric", STSSNET_UTILS / "matlab_metric.py"
        )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lp_std = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        .to(dev)
        .eval()
    )
    lp_raw = (
        LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
        .to(dev)
        .eval()
        if args.dataset == "stss"
        else None
    )
    flip = None
    if not args.no_flip:
        import flip_evaluator as flip
    cv = None
    if not args.no_cvvdp:
        import pycvvdp
        from pycvvdp.video_source import video_source_array

        cv = pycvvdp.cvvdp(
            display_name="standard_fhd", heatmap=None, device=dev, quiet=True
        )
    print(
        f"{args.dataset} x2 {method}  code={code} ({code_why})"
        + (f" [{args.ckpt} @ {cfg.get('ckpt_iteration')}]" if cfg else "")
        + f"  fps={fps} runs={ {k: [len(o) for o in v] for (k, v) in orders.items()} }",
        flush=True,
    )
    save_set = set(args.save_frames or [])

    def step(state, lr, depth, motion, jitter, prev_lr):
        if model is not None:
            with torch.inference_mode():
                (out, state, _) = model.step(lr, depth, motion, jitter, state)
            return (out, state)
        if base is not None:
            with torch.inference_mode():
                (out, state, _) = base.step(lr, depth, motion, jitter, state)
            return (out, state)
        if method == "bicubic":
            return (
                F.interpolate(
                    lr, scale_factor=S, mode="bicubic", align_corners=False
                ).clamp(0, 1),
                None,
            )
        if method == "regbicubic":
            return (smp.registered_bicubic(lr, jitter, S), None)
        if method == "lanczos3":
            return (ups.lanczos_upsample(lr, S, 3), None)
        raise ValueError(method)

    (per_frame, per_scene, nonfinite) = ([], {}, [])
    t0 = time.time()
    t_flip = t_cv = 0.0
    for scene, runs_ in sorted(orders.items()):
        (tpairs, cv_runs) = ([], [])
        for r, order in enumerate(runs_):
            state = None
            prev = None
            (vid_t, vid_r) = ([], [])
            for k, i in enumerate(order):
                item = data[i]
                n = data.frames[i]["n"] if args.dataset == "stss" else i
                lr = item["lr"][None, :3].to(dev).float()
                depth = item["depth"][None].to(dev).float()
                motion = item["motion"][None].to(dev).float()
                if args.dataset == "arm":
                    jitter = item["jitter"][None].to(dev).float()
                    if args.jitter_variant == "ymirror":
                        jitter = jitter * torch.tensor([-1.0, 1.0], device=dev)
                    elif args.jitter_variant == "zero":
                        jitter = torch.zeros_like(jitter)
                    m_hr = item["motion_hr"][None].to(dev).float()
                    if args.motion_variant == "negate":
                        motion = -motion
                    elif args.motion_variant == "jitfield":
                        jr = item["jitter_raw"].to(dev).float()
                        if prev is not None:
                            motion = motion + (jr - prev["jr"]).view(1, 2, 1, 1)
                else:
                    jitter = torch.zeros(1, 2, device=dev)
                    if args.fps_mode == 30:
                        j = n_index.get((scene, n - 1))
                        if j is not None:
                            m_prev = data[j]["motion"][None].to(dev).float()
                            (back, _) = smp.warp_bilinear(m_prev, motion)
                            motion = motion + back
                    m_hr = smp.motion_to_hr(motion, S)
                if method == "GT":
                    out = item["hr"][None].to(dev).float()
                else:
                    (out, state) = step(
                        state,
                        lr,
                        depth,
                        motion,
                        jitter,
                        None if prev is None else prev["lr"],
                    )
                finite = bool(torch.isfinite(out).all())
                if not finite:
                    nonfinite.append(item["key"])
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
                out = out.clamp(0, 1).float()
                hr = item["hr"][None].to(dev).float()
                row = {"scene": scene, "run": r, "n": int(n), "k": k, "finite": finite}
                if prev is not None:
                    tv = temporal_gradient_error(
                        torch.cat((prev["out"], out), 0),
                        torch.cat((prev["hr"], hr), 0),
                        torch.cat((prev["m_hr"], m_hr), 0),
                    )
                    tv["static"] = camera_static(motion, factor)
                    row.update(
                        tpsnr=tv["tpsnr"],
                        t_sse=tv["sse"],
                        t_count=tv["count"],
                        static=tv["static"],
                    )
                    if k >= args.warmup:
                        tpairs.append(tv)
                prev = {
                    "out": out,
                    "hr": hr,
                    "m_hr": m_hr,
                    "lr": lr,
                    "jr": item["jitter_raw"].to(dev).float()
                    if args.dataset == "arm"
                    else None,
                }
                pred = out[0].permute(1, 2, 0).cpu().numpy() * 255.0
                ref = hr[0].permute(1, 2, 0).cpu().numpy() * 255.0
                row["psnr_y"] = psnr(pred, ref, S, crop_border=0, to_y=True)
                row["ssim_y"] = ssim(pred, ref, S, crop_border=0, to_y=True)
                with torch.inference_mode():
                    row["lpips"] = float(lp_std(out, hr))
                    if lp_raw is not None:
                        row["lpips_stssnet"] = float(lp_raw(out, hr))
                if args.dataset == "stss":
                    (p_s, s_s) = stm.calc_metrics(
                        pred / 255.0, ref / 255.0, 0, norm=True
                    )
                    (row["psnr_y_stssnet"], row["ssim_y_stssnet"]) = (
                        float(p_s),
                        float(s_s),
                    )
                pu8 = np.round(pred).clip(0, 255).astype(np.uint8)
                ru8 = np.round(ref).clip(0, 255).astype(np.uint8)
                row["psnr_y_u8"] = psnr(pu8, ru8, S, crop_border=0, to_y=True)
                if flip is not None and k >= args.warmup:
                    tf = time.time()
                    (_, fl, _) = flip.evaluate(
                        ru8.astype(np.float32) / 255.0,
                        pu8.astype(np.float32) / 255.0,
                        "LDR",
                        inputsRGB=True,
                        applyMagma=False,
                        computeMeanError=True,
                        parameters={},
                    )
                    row["flip"] = float(fl)
                    t_flip += time.time() - tf
                if cv is not None and k >= args.warmup:
                    vid_t.append(pu8)
                    vid_r.append(ru8)
                if n in save_set:
                    d = (
                        FRAMES_DIR
                        / args.dataset
                        / scene
                        / (
                            method
                            + (f"_fps{args.fps_mode}" if args.dataset == "stss" else "")
                            + (f"_{args.tag}" if args.tag else "")
                        )
                    )
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(pu8).save(d / f"{n:05d}.png")
                    g = FRAMES_DIR / args.dataset / scene / "reference"
                    g.mkdir(parents=True, exist_ok=True)
                    if not (g / f"{n:05d}.png").is_file():
                        Image.fromarray(ru8).save(g / f"{n:05d}.png")
                per_frame.append(row)
            jod = None
            if cv is not None and vid_t:
                tc = time.time()
                with torch.inference_mode():
                    vs = video_source_array(
                        np.stack(vid_t),
                        np.stack(vid_r),
                        fps,
                        dim_order="FHWC",
                        display_photometry=cv.display_photometry,
                    )
                    (q, _) = cv.predict_video_source(vs)
                jod = float(q)
                t_cv += time.time() - tc
            cv_runs.append({"run": r, "frames": len(vid_t), "jod": jod})
            del vid_t, vid_r
        rows = [x for x in per_frame if x["scene"] == scene]
        ours = [x for x in rows if x["k"] >= args.warmup]
        agg = {
            "frames_stepped": len(rows),
            "frames_scored": len(ours),
            "runs": len(runs_),
            "psnr_y": float(np.mean([x["psnr_y"] for x in ours])),
            "ssim_y": float(np.mean([x["ssim_y"] for x in ours])),
            "lpips": float(np.mean([x["lpips"] for x in ours])),
            "psnr_y_u8": float(np.mean([x["psnr_y_u8"] for x in ours])),
            "flip": float(np.mean([x["flip"] for x in ours]))
            if ours and "flip" in ours[0]
            else None,
            "tpsnr": aggregate_tpsnr(tpairs),
            "jod_per_run": cv_runs,
            "jod": float(np.mean([c["jod"] for c in cv_runs if c["jod"] is not None]))
            if any((c["jod"] is not None for c in cv_runs))
            else None,
        }
        if args.dataset == "stss":
            sf = [x for x in rows if x["n"] % 2 == 0]
            agg["stss_sf"] = {
                "frames": len(sf),
                "first_n": min((x["n"] for x in sf)) if sf else None,
                "last_n": max((x["n"] for x in sf)) if sf else None,
                "psnr_y": float(np.mean([x["psnr_y_stssnet"] for x in sf])),
                "ssim_y": float(np.mean([x["ssim_y_stssnet"] for x in sf])),
                "lpips_stssnet": float(np.mean([x["lpips_stssnet"] for x in sf])),
                "lpips": float(np.mean([x["lpips"] for x in sf])),
                "psnr_y_float": float(np.mean([x["psnr_y"] for x in sf])),
            }
        per_scene[scene] = agg
        t = agg["tpsnr"]
        print(
            f"  {scene:<16} stepped {agg['frames_stepped']:>5} scored {agg['frames_scored']:>5}  PSNR-Y {agg['psnr_y']:.3f}  SSIM {agg['ssim_y']:.4f}  LPIPS {agg['lpips']:.4f}  FLIP {agg['flip']}  JOD {agg['jod']}  tcap {t['cap60']:.3f} tpool {t['pooled']:.3f}"
            + (
                f"  | SF(STSSNet metric) {agg['stss_sf']['psnr_y']:.3f} / {agg['stss_sf']['ssim_y']:.4f} over {agg['stss_sf']['frames']}"
                if "stss_sf" in agg
                else ""
            ),
            flush=True,
        )

    def smean(key, sub=None):
        vals = []
        for a in per_scene.values():
            v = a[key] if sub is None else (a[key] or {}).get(sub)
            vals.append(float("nan") if v is None else v)
        return nanmean(vals)

    table = {
        "psnr_y": smean("psnr_y"),
        "ssim_y": smean("ssim_y"),
        "lpips": smean("lpips"),
        "psnr_y_u8": smean("psnr_y_u8"),
        "flip": smean("flip"),
        "jod": smean("jod"),
        "tpsnr_cap60": smean("tpsnr", "cap60"),
        "tpsnr_pooled": smean("tpsnr", "pooled"),
        "tpsnr_pooled_static": smean("tpsnr", "pooled_static"),
        "tpsnr_pooled_moving": smean("tpsnr", "pooled_moving"),
    }
    if args.dataset == "stss":
        for key in ("psnr_y", "ssim_y", "lpips_stssnet", "lpips"):
            table[f"stss_sf_{key}"] = smean("stss_sf", key)
    result = {
        "dataset": args.dataset,
        "method": method,
        "scale": S,
        "fps_mode": args.fps_mode if args.dataset == "stss" else 30,
        "table": table,
        "per_scene": per_scene,
        "nonfinite_frames": nonfinite,
        "per_frame": per_frame,
        "method_info": base_info
        if base is not None
        else {
            "ckpt": args.ckpt,
            "ckpt_iteration": cfg.get("ckpt_iteration"),
            "trained": {
                k: cfg.get(k)
                for k in (
                    "dataset",
                    "qrisp_colour",
                    "motion_jit",
                    "registered_skip",
                    "depth_jit",
                    "temporal_weight",
                    "iters",
                    "dim",
                    "blocks",
                    "recurrent",
                    "stss_train",
                )
            },
        }
        if cfg
        else {"kind": method},
        "protocol": {
            "warmup": args.warmup,
            "stride": 1,
            "reset": "per camera run (STSS: numbering gaps / cuts)",
            "paper_range": not args.no_paper_range if args.dataset == "stss" else None,
            "fps_for_cvvdp": fps,
            "cvvdp_display": "standard_fhd",
            "flip": "LDR, defaults (67 ppd), 8-bit round",
            "exposure_log": args.exposure if args.dataset == "arm" else None,
            "encode": args.encode if args.dataset == "arm" else None,
            "jitter_variant": args.jitter_variant,
            "motion_variant": args.motion_variant,
            "tpsnr_field": "Arm output-resolution jitter-free 'motion'"
            if args.dataset == "arm"
            else "STSS native field, nearest x2"
            + (" (composed over two 60 fps steps)" if args.fps_mode == 30 else ""),
            "max_frames": args.max_frames,
        },
        "timing_s": {
            "total": round(time.time() - t0, 1),
            "flip": round(t_flip, 1),
            "cvvdp": round(t_cv, 1),
        },
        "provenance": {
            "argv": sys.argv,
            "code_resolution": code_why,
            "bench_snapshot": str(MINE),
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "torch": torch.__version__,
            "python": sys.executable,
            "dataset": data_prov,
            "md5": {
                "scripts/protocol_bench.py": md5(__file__),
                "diffsr/upsample.py": md5(MINE / "diffsr/upsample.py"),
                "diffsr/data_arm.py": md5(MINE / "diffsr/data_arm.py"),
                "diffsr/taau.py": md5(MINE / "diffsr/taau.py"),
                "diffsr/taa_clamp.py": md5(MINE / "diffsr/taa_clamp.py"),
                "code/diffsr/rrta.py": md5(code / "diffsr/rrta.py"),
                "code/diffsr/checkpoint.py": md5(code / "diffsr/checkpoint.py"),
                "code/diffsr/metrics.py": md5(code / "diffsr/metrics.py"),
                "code/diffsr/temporal.py": md5(code / "diffsr/temporal.py"),
                "code/diffsr/data_stss.py": md5(code / "diffsr/data_stss.py"),
                "stssnet/utils/matlab_metric.py": md5(
                    STSSNET_UTILS / "matlab_metric.py"
                ),
            },
        },
    }
    tag = args.tag or ""
    if args.out:
        out = Path(args.out)
    else:
        name = f"{args.dataset}_{method}"
        if args.dataset == "stss":
            name += f"_fps{args.fps_mode}_" + "+".join(
                (s.replace("_test", "") for s in args.scenes)
            )
        if args.encode != "none":
            name += f"_{args.encode}"
        if args.jitter_variant != "loader":
            name += f"_jit-{args.jitter_variant}"
        if args.motion_variant != "loader":
            name += f"_mot-{args.motion_variant}"
        if tag:
            name += f"_{tag}"
        if args.max_frames:
            name += "_SMOKE"
        out = OUT_DIR / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(
        f"  MEAN  PSNR-Y {table['psnr_y']:.3f}  SSIM {table['ssim_y']:.4f}  LPIPS {table['lpips']:.4f}  FLIP {table['flip']:.4f}  JOD {table['jod']:.4f}  tPSNR60 {table['tpsnr_cap60']:.3f}"
        + (
            f"  | SF {table['stss_sf_psnr_y']:.3f} / {table['stss_sf_ssim_y']:.4f}"
            if args.dataset == "stss"
            else ""
        )
        + f"   [{result['timing_s']}]\n  wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
