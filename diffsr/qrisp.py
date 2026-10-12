from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
__all__ = [
    "DEPTH_WEIGHTS",
    "MOTION_SIGN",
    "VERTICAL_INDEX",
    "HORIZONTAL_INDEX",
    "TRAIN_SCENES",
    "TEST_SCENES",
    "SCALE_DIRS",
    "read_color",
    "decode_depth",
    "read_depth",
    "decode_motion",
    "read_motion",
    "read_camera",
    "warp_by_motion",
    "iter_frames",
]
DEPTH_WEIGHTS = np.array(
    [1.0 / 255.0, 1.0 / 255.0**2, 1.0 / 255.0**3, 1.0 / 255.0**4], dtype=np.float64
)
VERTICAL_INDEX = 1
HORIZONTAL_INDEX = 2
MOTION_SIGN: tuple[float, float] = (1.0, -1.0)
TRAIN_SCENES = (
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
TEST_SCENES = ("AbandonedSchool", "SeaPort", "SpaceShipDemo")
SCALE_DIRS = {
    "540p": (2, "MipBiasMinus1"),
    "360p": (3, "MipBiasMinus1.58"),
    "270p": (4, "MipBiasMinus2"),
}


def read_color(path: str | Path, keep_alpha: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_UNCHANGED if keep_alpha else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return image


def decode_depth(packed: np.ndarray) -> np.ndarray:
    if packed.ndim != 3 or packed.shape[2] != 4:
        raise ValueError(f"depth must be RGBA8, got {packed.shape}")
    rgba = packed[..., [2, 1, 0, 3]].astype(np.float64)
    return (rgba * DEPTH_WEIGHTS).sum(-1).astype(np.float32)


def read_depth(path: str | Path) -> np.ndarray:
    packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if packed is None:
        raise FileNotFoundError(path)
    return decode_depth(packed)


def decode_motion(
    exr: np.ndarray, sign: tuple[float, float] | None = None
) -> np.ndarray:
    if exr.ndim != 3 or exr.shape[2] < 3:
        raise ValueError(f"motion must have >=3 channels, got {exr.shape}")
    (sy, sx) = MOTION_SIGN if sign is None else sign
    (height, width) = exr.shape[:2]
    vertical = exr[..., VERTICAL_INDEX].astype(np.float32) * height * sy
    horizontal = exr[..., HORIZONTAL_INDEX].astype(np.float32) * width * sx
    return np.stack([vertical, horizontal], axis=-1)


def read_motion(
    path: str | Path, sign: tuple[float, float] | None = None
) -> np.ndarray:
    exr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if exr is None:
        raise FileNotFoundError(f"{path} (is OPENCV_IO_ENABLE_OPENEXR set?)")
    return decode_motion(exr, sign)


def read_camera(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        camera = json.load(handle)
    jitter = camera.get("jitter_offset") or {}
    camera["jitter_yx"] = (float(jitter.get("y", 0.0)), float(jitter.get("x", 0.0)))
    return camera


def warp_by_motion(previous: np.ndarray, motion: np.ndarray) -> np.ndarray:
    (height, width) = motion.shape[:2]
    (grid_y, grid_x) = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    map_y = grid_y + motion[..., 0]
    map_x = grid_x + motion[..., 1]
    return cv2.remap(
        previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def iter_frames(
    root: str | Path, scene: str, resolution: str, modality: str
) -> list[tuple[str, str, Path]]:
    base = Path(root) / scene / resolution / modality
    out = []
    for sequence in sorted((p for p in base.iterdir() if p.is_dir())):
        for frame in sorted(sequence.iterdir()):
            out.append((sequence.name, frame.stem, frame))
    return out
