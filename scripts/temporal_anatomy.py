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
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffsr.checkpoint import load_run
from diffsr.rrta import warp
from protocol_full import (
    scenes,
    frame_orders,
    add_reset_flags,
    MetricField,
    TPSNR_FIELDS,
    TPSNR_CAP,
    input_pipeline,
    dataset_for,
    provenance,
)

TEST = QRISP_RAW / "TestSet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--methods", nargs="+", default=["DLSS2", "FSR"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--tpsnr-field", choices=TPSNR_FIELDS, default="nonjit")
    add_reset_flags(ap)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="SMOKE TESTS ONLY: first N frames of each pass",
    )
    ap.add_argument(
        "--out", default=None, help="JSON; default results/tanat_<run>_local.json"
    )
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (model, cfg) = load_run(ROOT / "runs" / args.run, dev, args.ckpt)
    print(
        f"{args.run} [{args.ckpt} @ {cfg.get('ckpt_iteration')}] vs {', '.join(args.methods)}",
        flush=True,
    )
    pipe = input_pipeline(cfg)
    data = dataset_for(pipe, args.scale)
    field = MetricField(args.tpsnr_field, args.scale)
    if pipe["motion_lerp"] and field.db is None:
        raise SystemExit(
            "motion_lerp arm with --tpsnr-field input would put eps into the METRIC warp; use a nonjit field"
        )
    print(
        f"  input motion_jit={pipe['motion_jit']} lerp={pipe['motion_lerp']} depth_jit={pipe['depth_jit']}; metric field {field.name}; reset_runs={args.reset_runs}\n",
        flush=True,
    )
    cols = ["ours"] + args.methods
    acc = {
        s: {c: dict(pp=0.0, rr=0.0, pr=0.0, n=0.0, err=[], jd=[]) for c in cols}
        for s in scenes(data)
    }
    for scene, orders in sorted(frame_orders(data, args.reset_runs).items()):
        for order in orders:
            if args.max_frames:
                order = order[: args.max_frames]
            state = None
            prev = {c: None for c in cols}
            prev_ref = prev_jit = None
            for n, i in enumerate(order):
                item = data[i]
                lr = item["lr"][None, :3].to(dev)
                depth = item["depth"][None].to(dev)
                motion = item["motion"][None].to(dev)
                jit = item.get("jitter")
                jitter = (jit[None] if jit is not None else torch.zeros(1, 2)).to(dev)
                with torch.inference_mode():
                    (out, state, _) = model.step(lr, depth, motion, jitter, state)
                cur = {"ours": out.clamp(0, 1)}
                meta = data.frames[i]
                for m in args.methods:
                    png = (
                        TEST
                        / scene
                        / "1080p"
                        / m
                        / meta["sequence"]
                        / f"{meta['frame']}.png"
                    )
                    cur[m] = (
                        torch.from_numpy(
                            np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
                            / 255.0
                        )
                        .permute(2, 0, 1)[None]
                        .to(dev)
                        if png.is_file()
                        else None
                    )
                ref = item["hr"][None].to(dev)
                if prev_ref is not None and n >= args.warmup:
                    mhr = field.hr(field.lr(data, i, motion))
                    (wref, valid) = warp(prev_ref, mhr)
                    dR = (ref - wref) * valid * 255.0
                    jd = (
                        float((jitter - prev_jit).abs().sum())
                        if prev_jit is not None
                        else 0.0
                    )
                    for c in cols:
                        if cur[c] is None or prev[c] is None:
                            continue
                        (wp, _) = warp(prev[c], mhr)
                        dP = (cur[c] - wp) * valid * 255.0
                        a = acc[scene][c]
                        a["pp"] += float((dP * dP).sum())
                        a["rr"] += float((dR * dR).sum())
                        a["pr"] += float((dP * dR).sum())
                        a["n"] += float(valid.expand_as(dP).sum())
                        a["err"].append(
                            float(
                                ((dP - dR) ** 2).sum()
                                / valid.expand_as(dP).sum().clamp_min(1)
                            )
                        )
                        a["jd"].append(jd)
                prev = {
                    c: v.clone() if v is not None else None for (c, v) in cur.items()
                }
                (prev_ref, prev_jit) = (ref, jitter.clone())
        print(f"  {scene} done", flush=True)
    print(
        f"\n  {'scene':<17}{'method':<8}{'A=rms(dP)':>11}{'B=rms(dR)':>11}{'A/B':>7}{'rho':>7}{'tPSNR':>8}{'tcap60':>8}{'corr(err,|dJ|)':>16}"
    )
    print("  " + "-" * 94)
    rows = {}
    for scene in sorted(acc):
        B = None
        for c in cols:
            a = acc[scene][c]
            if a["n"] == 0:
                continue
            A = (a["pp"] / a["n"]) ** 0.5
            B = (a["rr"] / a["n"]) ** 0.5
            rho = a["pr"] / a["n"] / (A * B) if A > 0 and B > 0 else float("nan")
            mse = A * A + B * B - 2 * rho * A * B
            tp = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
            (e, j) = (np.array(a["err"]), np.array(a["jd"]))
            with np.errstate(divide="ignore"):
                db = np.where(
                    e > 0, 10 * np.log10(255.0**2 / np.maximum(e, 1e-300)), np.inf
                )
            tcap = (
                float(np.nanmean(np.minimum(db, TPSNR_CAP)))
                if len(db)
                else float("nan")
            )
            cj = (
                np.corrcoef(e, j)[0, 1]
                if j.std() > 1e-09 and e.std() > 1e-09
                else float("nan")
            )
            rows.setdefault(scene, {})[c] = {
                "A": A,
                "B": B,
                "A_over_B": A / B if B else float("nan"),
                "rho": rho,
                "tpsnr_pooled": float(tp),
                "tpsnr_cap60": tcap,
                "corr_err_jitter": float(cj),
                "pairs": len(e),
            }
            print(
                f"  {scene:<17}{c:<8}{A:>11.3f}{B:>11.3f}{A / B:>7.2f}{rho:>7.3f}{tp:>8.3f}{tcap:>8.3f}{cj:>16.3f}"
            )
        print()
    out = args.out or f"results/tanat_{args.run}_{'local'}.json"
    path = ROOT / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run": args.run,
                "ckpt": args.ckpt,
                "ckpt_iteration": cfg.get("ckpt_iteration"),
                "rows": rows,
                "metric_field": field.name,
                "tpsnr_field": args.tpsnr_field,
                "reset_runs": args.reset_runs,
                "warmup": args.warmup,
                "scale": args.scale,
                "input_pipeline": pipe,
                "max_frames": args.max_frames,
                "provenance": provenance(
                    scale=args.scale,
                    warmup=args.warmup,
                    reset_runs=args.reset_runs,
                    tpsnr_field=args.tpsnr_field,
                ),
            },
            indent=1,
        )
    )
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
