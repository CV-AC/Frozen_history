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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsr.qrisp import decode_depth, decode_motion, read_camera

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
cv2.setNumThreads(1)
ROOT = REPOSITORY
SCENES = QRISP_RAW
OUT = QRISP_LMDB
TRAIN = (
    "CBApocalypse",
    "FloodedGrounds",
    "FloodedGroundsBridges",
    "ScifiBase",
    "ScifiBaseNightStartStop",
    "ScifiBaseStartStop",
    "ScifiFacility",
    "SunTemple",
    "SunTempleBush",
    "SunTempleLamps",
)
TEST = ("AbandonedSchool", "SeaPort", "SpaceShipDemo")
EXPECTED_FRAMES = {"train": 7260, "test": 1500}
MODALITIES: dict[str, tuple[str, str, str]] = {
    "GT1080Enhanced": ("1080p", "Enhanced", "raw"),
    "GT1080Native": ("1080p", "Native", "raw"),
    "LR270Native": ("270p", "Native", "raw"),
    "LR270MipBias": ("270p", "MipBiasMinus2", "raw"),
    "LR270MipBiasJit": ("270p", "MipBiasMinus2Jittered", "raw"),
    "Depth270": ("270p", "DepthMipBiasMinus2", "depth"),
    "Depth270Jit": ("270p", "DepthMipBiasMinus2Jittered", "depth"),
    "Motion270": ("270p", "MotionVectorsMipBiasMinus2", "motion"),
    "Motion270Jit": ("270p", "MotionVectorsMipBiasMinus2Jittered", "motion"),
    "LR540Native": ("540p", "Native", "raw"),
    "LR540MipBias": ("540p", "MipBiasMinus1", "raw"),
    "LR540MipBiasJit": ("540p", "MipBiasMinus1Jittered", "raw"),
    "Depth540": ("540p", "DepthMipBiasMinus1", "depth"),
    "Depth540Jit": ("540p", "DepthMipBiasMinus1Jittered", "depth"),
    "Motion540": ("540p", "MotionVectorsMipBiasMinus1", "motion"),
    "Motion540Jit": ("540p", "MotionVectorsMipBiasMinus1Jittered", "motion"),
    "LR360Native": ("360p", "Native", "raw"),
    "LR360MipBias": ("360p", "MipBiasMinus1.58", "raw"),
    "LR360MipBiasJit": ("360p", "MipBiasMinus1.58Jittered", "raw"),
    "Depth360": ("360p", "DepthMipBiasMinus1.58", "depth"),
    "Depth360Jit": ("360p", "DepthMipBiasMinus1.58Jittered", "depth"),
    "Motion360": ("360p", "MotionVectorsMipBiasMinus1.58", "motion"),
    "Motion360Jit": ("360p", "MotionVectorsMipBiasMinus1.58Jittered", "motion"),
}
DEFAULT_SET = list(MODALITIES)


def log(*args) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def scene_dir(scene: str) -> Path:
    direct = SCENES / scene
    return direct if direct.is_dir() else SCENES / "TestSet" / scene


def list_frames(split: str) -> list[tuple[str, str, str]]:
    scenes = TRAIN if split == "train" else TEST
    out = []
    for scene in scenes:
        base = scene_dir(scene) / "1080p" / "Enhanced"
        if not base.is_dir():
            log(f"  WARNING {scene}: no 1080p/Enhanced, skipping")
            continue
        for sequence in sorted((p for p in base.iterdir() if p.is_dir())):
            for frame in sorted(sequence.iterdir()):
                out.append((scene, sequence.name, frame.stem))
    return out


def _tiles(array: np.ndarray, grid: tuple[int, int]):
    (rows, cols) = grid
    (height, width) = array.shape[:2]
    if height % rows or width % cols:
        raise ValueError(
            f"{array.shape[:2]} does not divide by {rows}x{cols} (rows x cols). QRISP is 16:9 -- use 2x4, not 4x2."
        )
    (th, tw) = (height // rows, width // cols)
    for r in range(rows):
        for c in range(cols):
            yield (r * cols + c, array[r * th : (r + 1) * th, c * tw : (c + 1) * tw])


def _encode(args):
    (scene, sequence, frame, res, modality, kind, grid) = args
    base = scene_dir(scene) / res / modality / sequence
    stem = f"{scene}/{sequence}/{frame}"
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if kind == "raw":
        path = base / f"{frame}.png"
        if not path.is_file():
            return None
        if grid is None:
            return [(stem, path.read_bytes(), "png", None)]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return [
            (
                f"{stem}/{i}",
                cv2.imencode(".png", t, png_params)[1].tobytes(),
                "png",
                None,
            )
            for (i, t) in _tiles(image, grid)
        ]
    if kind == "depth":
        packed = cv2.imread(str(base / f"{frame}.png"), cv2.IMREAD_UNCHANGED)
        if packed is None:
            return None
        value = np.clip(decode_depth(packed), 0.0, 1.0).astype(np.float16)
    else:
        exr = cv2.imread(str(base / f"{frame}.exr"), cv2.IMREAD_UNCHANGED)
        if exr is None:
            return None
        value = decode_motion(exr).astype(np.float16)
    if grid is None:
        return [(stem, value.tobytes(), "f16", value.shape)]
    return [
        (f"{stem}/{i}", np.ascontiguousarray(t).tobytes(), "f16", t.shape)
        for (i, t) in _tiles(value, grid)
    ]


def build(
    split: str, names: list[str], workers: int, frames, grid: tuple[int, int] | None
) -> dict:
    import lmdb

    report = {}
    suffix = "" if grid is None else f"_t{grid[0]}x{grid[1]}"
    for name in names:
        (res, modality, kind) = MODALITIES[name]
        target = OUT / f"QRISP_{split}_{name}{suffix}.lmdb"
        if (target / "data.mdb").is_file():
            log(f"{split}/{name}: already built, skipping")
            continue
        target.mkdir(parents=True, exist_ok=True)
        jobs = [(s, q, f, res, modality, kind, grid) for (s, q, f) in frames]
        env = lmdb.open(str(target), map_size=1 << 41)
        (meta, written, missing) = ([], 0, 0)
        start = time.time()
        with ProcessPoolExecutor(workers) as pool:
            txn = env.begin(write=True)
            for done, result in enumerate(pool.map(_encode, jobs, chunksize=8), 1):
                if result is None:
                    missing += 1
                    continue
                for key, payload, dtype, shape in result:
                    txn.put(key.encode(), payload)
                    meta.append(
                        f"{key} {dtype} {('x'.join(map(str, shape)) if shape else '-')}"
                    )
                    written += 1
                if done % 2000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    log(
                        f"  {split}/{name}: {done}/{len(jobs)} {time.time() - start:.0f}s"
                    )
            txn.commit()
        (target / "meta_info.txt").write_text("\n".join(meta) + "\n")
        env.close()
        if missing:
            raise RuntimeError(
                f"{split}/{name}: {missing} missing modality frames; remove the incomplete database and supply the missing files"
            )
        size = sum((f.stat().st_size for f in target.iterdir()))
        log(
            f"{split}/{name}{suffix}: {written} entries"
            + (f", {missing} MISSING" if missing else "")
            + f", {size / 2**30:.2f} GiB, {time.time() - start:.0f}s"
        )
        report[name] = {"frames": written, "missing": missing, "gib": size / 2**30}
    return report


def build_index(split: str, frames) -> None:
    entries = []
    for scene, sequence, frame in frames:
        path = scene_dir(scene) / "270p" / "CameraData" / sequence / f"{frame}.json"
        camera = read_camera(path) if path.is_file() else {}
        entries.append(
            {
                "key": f"{scene}/{sequence}/{frame}",
                "scene": scene,
                "sequence": sequence,
                "frame": frame,
                "jitter_yx": camera.get("jitter_yx", [0.0, 0.0]),
                "near": camera.get("near"),
                "far": camera.get("far"),
                "fov": camera.get("fov"),
            }
        )
    out = OUT / f"QRISP_{split}_index.json"
    out.write_text(
        json.dumps({"split": split, "count": len(entries), "frames": entries})
    )
    log(f"{split}: index with {len(entries)} frames -> {out.name}")


def main() -> None:
    global SCENES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", default=str(QRISP_RAW))
    ap.add_argument("--out", default=str(QRISP_LMDB))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--modalities", default=",".join(DEFAULT_SET))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--tile-grid",
        default="",
        help="rows x cols, e.g. 2x4 -- the only grid that divides every QRISP resolution. Splits each frame into a non-overlapping grid; empty keeps whole frames. Training wants tiles (8x less decode per crop), evaluation wants whole frames.",
    )
    args = ap.parse_args()
    SCENES, OUT = Path(args.source_root), Path(args.out)
    grid = None
    if args.tile_grid:
        (rows, cols) = (int(v) for v in args.tile_grid.lower().split("x"))
        grid = (rows, cols)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in [s for s in args.splits.split(",") if s]:
        frames = list_frames(split)
        want = EXPECTED_FRAMES[split]
        if len(frames) != want and not args.allow_incomplete:
            raise ValueError(
                f"{split}: expected {want} frames, found {len(frames)}; use --allow-incomplete only for diagnostics"
            )
        flag = "OK" if len(frames) == want else f"!! paper says {want}"
        log(f"{split}: {len(frames)} frames  {flag}")
        build_index(split, frames)
        summary[split] = build(
            split,
            [m for m in args.modalities.split(",") if m],
            args.workers,
            frames,
            grid,
        )
    name = (
        "pack_report.json" if grid is None else f"pack_report_t{grid[0]}x{grid[1]}.json"
    )
    (OUT / name).write_text(json.dumps(summary, indent=1))
    log("done")


if __name__ == "__main__":
    main()
