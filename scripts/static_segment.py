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
from diffsr.metrics import psnr
from protocol_full import input_pipeline, dataset_for, add_reset_flags, provenance

TEST = QRISP_RAW / "TestSet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--methods", nargs="+", default=["DLSS2", "FSR"])
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--scene", default="AbandonedSchool")
    ap.add_argument("--sequence", default="0001")
    ap.add_argument("--first", type=int, default=275)
    ap.add_argument("--last", type=int, default=292)
    ap.add_argument(
        "--static-gate",
        type=float,
        default=0.0,
        help="inference-time override: motion-spread threshold below which the accumulator runs as an exact mean",
    )
    ap.add_argument("--static-cap", type=float, default=64.0)
    ap.add_argument(
        "--dejitter",
        action="store_true",
        help="inference-time override: strip the jitter delta from the motion field before reprojection",
    )
    add_reset_flags(ap)
    ap.add_argument(
        "--out", default=None, help="JSON; default results/static_<run>_local.json"
    )
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (model, cfg) = load_run(ROOT / "runs" / args.run, dev, args.ckpt)
    if args.dejitter:
        model.dejitter = True
    if args.static_gate > 0.0:
        model.static_gate = float(args.static_gate)
        model.static_cap = float(args.static_cap)
    pipe = input_pipeline(cfg)
    data = dataset_for(pipe, args.scale)
    run_idx = [
        i
        for i in range(len(data))
        if data.frames[i]["scene"] == args.scene
        and data.frames[i].get("sequence") == args.sequence
    ]
    seg = [
        i for i in run_idx if args.first <= int(data.frames[i]["frame"]) <= args.last
    ]
    if not seg:
        raise SystemExit("empty segment")
    if args.reset_runs:
        warm_idx = run_idx
    else:
        warm_idx = [
            i
            for i in range(len(data))
            if data.frames[i]["scene"] == args.scene and i <= seg[-1]
        ]
    print(f"{args.run} [{args.ckpt} @ {cfg.get('ckpt_iteration')}]")
    print(
        f"  input colour={pipe['colour']} motion_jit={pipe['motion_jit']} lerp={pipe['motion_lerp']} depth_jit={pipe['depth_jit']} x{args.scale}; reset_runs={args.reset_runs}"
    )
    print(
        f"{args.scene} seq {args.sequence} frames {args.first}-{args.last}: {len(seg)} frames, accumulator warmed over the preceding {len([i for i in warm_idx if i < seg[0]])}\n",
        flush=True,
    )
    (ours, refs) = ([], [])
    (fired, in_seg_spread) = ([], [])
    first_nonfinite = None
    state = None
    seg_set = set(seg)
    for i in warm_idx:
        if i > seg[-1]:
            break
        item = data[i]
        with torch.inference_mode():
            (out, state, _) = model.step(
                item["lr"][None, :3].to(dev),
                item["depth"][None].to(dev),
                item["motion"][None].to(dev),
                (
                    item["jitter"][None]
                    if item.get("jitter") is not None
                    else torch.zeros(1, 2)
                ).to(dev),
                state,
            )
        if first_nonfinite is None and (not bool(torch.isfinite(out).all())):
            first_nonfinite = i
            print(
                f"  !! first non-finite output at dataset index {i} (frame {data.frames[i]['frame']})",
                flush=True,
            )
        spread = float(item["motion"].flatten(1).std(dim=1).max())
        fired.append(spread < args.static_gate if args.static_gate > 0 else False)
        if i in seg_set:
            in_seg_spread.append(spread)
            ours.append(out[0].clamp(0, 1).float().cpu().numpy() * 255.0)
            refs.append(item["hr"].float().numpy() * 255.0)
    if args.static_gate > 0.0:
        print(
            f"  static gate tau={args.static_gate} cap={args.static_cap}: fired on {sum(fired)}/{len(fired)} frames of the run; scored-segment spread {min(in_seg_spread):.3f}-{max(in_seg_spread):.3f}\n",
            flush=True,
        )
    stacks = {"ours": np.stack(ours), "reference": np.stack(refs)}
    if not np.isfinite(stacks["ours"]).all():
        print("  !! non-finite output on the segment", flush=True)
    stacks["ours_uint8"] = np.round(stacks["ours"])
    for m in args.methods:
        fr = []
        for i in seg:
            meta = data.frames[i]
            png = (
                TEST
                / args.scene
                / "1080p"
                / m
                / meta["sequence"]
                / f"{meta['frame']}.png"
            )
            if png.is_file():
                fr.append(
                    np.asarray(
                        Image.open(png).convert("RGB"), dtype=np.float64
                    ).transpose(2, 0, 1)
                )
        if len(fr) == len(seg):
            stacks[m] = np.stack(fr)
    ref = stacks["reference"]
    rows = {}
    print(f"  {'method':<12}{'pixel std':>11}{'PSNR(static)':>14}   notes")
    print("  " + "-" * 56)
    for name, arr in stacks.items():
        std = float(arr.std(axis=0, ddof=0).mean())
        if name == "reference":
            rows[name] = {"pixel_std": std}
            print(f"  {name:<12}{std:>11.3f}{'--':>14}   reference variation sets the floor")
            continue
        p = float(
            np.mean(
                [
                    psnr(
                        arr[t].transpose(1, 2, 0),
                        ref[t].transpose(1, 2, 0),
                        args.scale,
                        crop_border=0,
                        to_y=True,
                    )
                    for t in range(len(arr))
                ]
            )
        )
        rows[name] = {"pixel_std": std, "psnr_static": p}
        note = {
            "ours": "float: only vs our own arms",
            "ours_uint8": "8-bit: the row to put next to PNG baselines",
        }.get(name, "")
        print(f"  {name:<12}{std:>11.3f}{p:>14.3f}   {note}")
    print(f"\n  This run uses x{args.scale}. QRISP Table 3 reports x2 ablations.")
    print("  Direct comparison requires an x2 run.")
    out = args.out or f"results/static_{args.run}_{'local'}.json"
    out_path = ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "run": args.run,
                "ckpt": args.ckpt,
                "ckpt_iteration": cfg.get("ckpt_iteration"),
                "segment": {
                    "scene": args.scene,
                    "sequence": args.sequence,
                    "first": args.first,
                    "last": args.last,
                    "frames": len(seg),
                },
                "scale": args.scale,
                "reset_runs": args.reset_runs,
                "input_pipeline": pipe,
                "first_nonfinite": first_nonfinite,
                "dejitter_override": args.dejitter,
                "static_gate_override": args.static_gate,
                "rows": rows,
                "provenance": provenance(scale=args.scale, reset_runs=args.reset_runs),
            },
            indent=1,
        )
    )
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
