import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from diffsr.checkpoint import build_model, load_run
from diffsr.fast_blocks import fuse_model


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run")
    source.add_argument("--config")
    parser.add_argument("--mode", choices=("eager", "fused", "graph"), default="graph")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--opts", default="{}")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Benchmarking requires a CUDA GPU")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats positive")
    torch.manual_seed(0)
    if args.run:
        model, config = load_run(Path(args.run), "cuda")
    else:
        config = json.loads(Path(args.config).read_text())
        model = build_model(config, "cuda").eval()
    fused = fuse_model(model)
    scale = model.scale
    if args.height % scale or args.width % scale:
        raise ValueError("Output dimensions must divide by the model scale")
    h, w = args.height // scale, args.width // scale
    rgb = torch.rand(1, 3, h, w, device="cuda")
    depth = torch.full((1, 1, h, w), 0.5, device="cuda")
    motion = torch.zeros(1, 2, h, w, device="cuda")
    jitter = torch.tensor([[-0.25, 1 / 6]], device="cuda")
    amp = torch.bfloat16 if args.precision == "bf16" else None
    model = model.to(memory_format=torch.channels_last)
    rgb = rgb.to(memory_format=torch.channels_last)
    state = None
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp is not None),
    ):
        for _ in range(8):
            _, state, _ = model.step(rgb, depth, motion, jitter, state)
        reference = model.step(rgb, depth, motion, jitter, state)[0].clone()
        if args.mode == "eager":
            one = lambda: model.step(rgb, depth, motion, jitter, state)[0]
        else:
            from diffsr.fast.resolve_triton3 import FusedResolve3, FixedGraph

            runner = FusedResolve3(model, amp=amp, opts=json.loads(args.opts))
            if args.mode == "graph":
                one = FixedGraph(runner, rgb, depth, motion, jitter, state)
            else:
                one = lambda: runner.step(rgb, depth, motion, jitter, state)[0]
        for _ in range(args.warmup):
            one()
        torch.cuda.synchronize()
        prediction = one().clone()
        error = float((prediction.float() - reference.float()).abs().max())
        latencies = []
        for _ in range(args.repeats):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            one()
            end.record()
            end.synchronize()
            latencies.append(start.elapsed_time(end))
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mode": args.mode,
        "precision": args.precision,
        "random_weights": args.config is not None,
        "random_inputs": True,
        "folded_blocks": fused,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "ms_mean": float(np.mean(latencies)),
        "ms_p50": float(np.median(latencies)),
        "ms_p90": float(np.percentile(latencies, 90)),
        "max_abs_error": error,
        "finite": bool(torch.isfinite(prediction).all()),
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
