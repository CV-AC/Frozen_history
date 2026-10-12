import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import lmdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def synthetic_databases(root, split, tiled):
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    rng = np.random.default_rng(0)
    for i in range(4):
        sequence = "0000" if split == "train" else f"{i // 2:04d}"
        frame = i if split == "train" else i % 2
        frames.append(
            {
                "key": f"Synthetic/{sequence}/{frame:04d}",
                "scene": "Synthetic",
                "sequence": sequence,
                "frame": f"{frame:04d}",
                "jitter_yx": [0.125, -0.25],
            }
        )
    (root / f"QRISP_{split}_index.json").write_text(
        json.dumps({"frames": frames, "count": 4, "split": split})
    )
    suffix = "_t2x4" if tiled else ""
    for name, channels, dtype in [
        ("GT1080Enhanced", 3, "png"),
        ("LR540MipBiasJit", 3, "png"),
        ("Depth540", 1, "f16"),
        ("Motion540", 2, "f16"),
    ]:
        path = root / f"QRISP_{split}_{name}{suffix}.lmdb"
        env = lmdb.open(str(path), map_size=1 << 24)
        metadata = []
        with env.begin(write=True) as transaction:
            for frame in frames:
                for tile in range(8 if tiled else 1):
                    key = frame["key"] + (f"/{tile}" if tiled else "")
                    height, width = (6, 8) if tiled else (12, 16)
                    if name.startswith("GT"):
                        height, width = height * 2, width * 2
                    if dtype == "png":
                        array = rng.integers(
                            32, 224, size=(height, width, 3), dtype=np.uint8
                        )
                        payload = cv2.imencode(".png", array)[1].tobytes()
                        shape = "-"
                    else:
                        shape_tuple = (
                            (height, width)
                            if channels == 1
                            else (height, width, channels)
                        )
                        array = np.full(
                            shape_tuple, 0.5 if channels == 1 else 0, dtype=np.float16
                        )
                        payload = array.tobytes()
                        shape = "x".join(map(str, shape_tuple))
                    transaction.put(key.encode(), payload)
                    metadata.append(f"{key} {dtype} {shape}")
        env.close()
        (path / "meta_info.txt").write_text("\n".join(metadata) + "\n")


def test_synthetic_training_inference_and_protocol(tmp_path):
    data = tmp_path / "lmdb"
    synthetic_databases(data, "train", True)
    synthetic_databases(data, "test", False)
    config = json.loads((ROOT / "configs/train/qrisp_x2.json").read_text())
    config.update(
        name="synthetic",
        out=str(tmp_path / "runs"),
        dim=8,
        blocks=1,
        patch=6,
        batch=1,
        clip=4,
        warmup=1,
        iters=4,
        eval_every=0,
        ckpt_every=4,
        workers=0,
        log_every=1,
    )
    configuration = tmp_path / "config.json"
    configuration.write_text(json.dumps(config))
    environment = {
        **os.environ,
        "QRISP_LMDB": str(data),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def execute(*arguments):
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    execute("scripts/train.py", "--config", str(configuration))
    run = tmp_path / "runs/synthetic"
    assert (run / "last.pt").is_file()
    predictions = tmp_path / "predictions"
    execute(
        "scripts/inference.py", "--run", str(run), "--out", str(predictions), "--cpu"
    )
    assert len(list(predictions.rglob("*.png"))) == 4
    report = tmp_path / "protocol.json"
    execute(
        "scripts/protocol_full.py",
        str(run),
        "--scale",
        "2",
        "--warmup",
        "0",
        "--out",
        str(report),
    )
    result = json.loads(report.read_text())
    assert result["table"]
