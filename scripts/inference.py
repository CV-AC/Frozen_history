import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from diffsr.checkpoint import load_run
from protocol_full import dataset_for, frame_orders, input_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--ckpt", default="last.pt")
    parser.add_argument("--dataset", choices=("qrisp", "stss", "arm"), default="qrisp")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model, config = load_run(Path(args.run), device, args.ckpt)
    if args.dataset == "qrisp":
        data = dataset_for(input_pipeline(config), model.scale)
        orders = frame_orders(data, True)
    elif args.dataset == "stss":
        from diffsr.data_stss import STSS_PACKED, STSS_TEST, StssSequence

        if model.scale != 2:
            raise ValueError("STSS inference requires a model at scale 2")
        from diffsr.data_stss import stss_pipeline

        data = StssSequence(
            [STSS_PACKED / STSS_TEST], depth_mode=stss_pipeline(config)["depth_mode"]
        )
        orders = data.runs()
    else:
        from diffsr.data_arm import ArmSequence

        if model.scale != 2:
            raise ValueError("Arm inference requires a model at scale 2")
        data = ArmSequence()
        orders = data.runs()
    output = Path(args.out)
    count = 0
    with torch.inference_mode():
        for runs in orders.values():
            for order in runs:
                state = None
                for index in order:
                    item = data[index]
                    inputs = [
                        item[key][None].to(device)
                        for key in ("lr", "depth", "motion", "jitter")
                    ]
                    prediction, state, _ = model.step(*inputs, state)
                    pixels = (
                        prediction[0]
                        .clamp(0, 1)
                        .mul(255)
                        .round()
                        .byte()
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy()
                    )
                    destination = output / (str(item["key"]) + ".png")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.ascontiguousarray(pixels)).save(destination)
                    count += 1
                    if args.max_frames and count >= args.max_frames:
                        print(f"Saved {count} frames")
                        return
    print(f"Saved {count} frames")


if __name__ == "__main__":
    main()
