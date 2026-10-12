import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from diffsr.checkpoint import load_run
from diffsr.engine_seq import to_numpy
from diffsr.fast_blocks import fuse_model
from diffsr.metrics import psnr
from protocol_full import dataset_for, frame_orders, input_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="last.pt")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--opts", default="{}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Deployment validation requires CUDA")
    from diffsr.fast.resolve_triton3 import FusedResolve3, GraphedStep

    model, configuration = load_run(Path(args.run), "cuda", args.ckpt)
    deployed = copy.deepcopy(model)
    folded = fuse_model(deployed)
    deployed = deployed.to(memory_format=torch.channels_last)
    amp = torch.bfloat16 if args.precision == "bf16" else None
    runner = FusedResolve3(deployed, amp=amp, opts=json.loads(args.opts))
    step = GraphedStep(runner).step
    data = dataset_for(input_pipeline(configuration), model.scale)
    records = []
    failures = []
    with torch.inference_mode():
        for scene, runs in frame_orders(data, True).items():
            for order in runs:
                state_reference = state_deployed = None
                if args.max_frames:
                    order = order[: args.max_frames]
                for position, index in enumerate(order):
                    item = data[index]
                    rgb, depth, motion, jitter = [
                        item[key][None].cuda()
                        for key in ("lr", "depth", "motion", "jitter")
                    ]
                    reference, state_reference, _ = model.step(
                        rgb, depth, motion, jitter, state_reference
                    )
                    output, state_deployed, _ = step(
                        rgb.to(memory_format=torch.channels_last),
                        depth,
                        motion,
                        jitter,
                        state_deployed,
                    )
                    if not bool(torch.isfinite(output).all()):
                        failures.append(item["key"])
                    target = to_numpy(item["hr"])
                    ref_score = psnr(
                        to_numpy(reference[0]),
                        target,
                        model.scale,
                        crop_border=0,
                        to_y=True,
                    )
                    score = psnr(
                        to_numpy(output[0]),
                        target,
                        model.scale,
                        crop_border=0,
                        to_y=True,
                    )
                    records.append(
                        {
                            "key": item["key"],
                            "scene": scene,
                            "scored": position >= 8,
                            "psnr_reference": ref_score,
                            "psnr_deployed": score,
                            "delta_db": score - ref_score,
                            "max_abs_error": float(
                                (output.float() - reference.float()).abs().max()
                            ),
                        }
                    )
    report = {
        "precision": args.precision,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "folded_blocks": folded,
        "independent_recurrent_states": True,
        "truncated": bool(args.max_frames),
        "nonfinite_frames": failures,
        "per_frame": records,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Nonfinite deployed outputs; see the report")
    print(f"Validated {len(records)} frames; report: {destination}")


if __name__ == "__main__":
    main()
