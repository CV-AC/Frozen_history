from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_qrisp import QRISP_LMDB, RES_FOR_SCALE, _Lmdb
from .data_sisr import to_tensor

__all__ = [
    "QrispClips",
    "build_clip_train",
    "build_clip_test",
    "test_segments",
    "static_positions",
]


class QrispClips(Dataset):
    def __init__(
        self,
        split: str = "train",
        scale: int = 4,
        length: int = 8,
        patch_lr: int | None = None,
        colour: str = "MipBiasJit",
        target: str = "Enhanced",
        flip: bool = True,
        tiles: tuple[int, int] | None = (2, 4),
        scenes: tuple[str, ...] | None = None,
        root: Path | str = QRISP_LMDB,
        motion_jit: bool = True,
        motion_lerp: float = 0.0,
        depth_jit: bool = True,
    ) -> None:
        if not colour.startswith("MipBias"):
            raise ValueError(
                "recurrent training needs depth and motion, which QRISP ships only for the mip-biased renders"
            )
        self.root = Path(root)
        (self.split, self.scale, self.length) = (split, int(scale), int(length))
        (self.colour, self.flip) = (colour, bool(flip))
        self.motion_jit = bool(motion_jit)
        self.motion_lerp = float(motion_lerp)
        self.depth_jit = bool(depth_jit)
        self._spread = None
        self.jittered = colour.endswith("Jit")
        self.patch_lr = int(264 // self.scale if patch_lr is None else patch_lr)
        res = RES_FOR_SCALE[self.scale]
        suffix = "Jit" if self.jittered else ""
        motion_suffix = suffix if motion_jit else ""
        depth_suffix = suffix if self.depth_jit else ""
        self.tiles = tuple(tiles) if tiles else None
        tag = "" if self.tiles is None else f"_t{self.tiles[0]}x{self.tiles[1]}"
        self.hr = _Lmdb(self.root / f"QRISP_{split}_GT1080{target}{tag}.lmdb")
        self.lr = _Lmdb(self.root / f"QRISP_{split}_LR{res}{colour}{tag}.lmdb")
        self.depth = _Lmdb(
            self.root / f"QRISP_{split}_Depth{res}{depth_suffix}{tag}.lmdb"
        )
        self.motion = _Lmdb(
            self.root / f"QRISP_{split}_Motion{res}{motion_suffix}{tag}.lmdb"
        )
        self.motion_other = (
            _Lmdb(
                self.root
                / f"QRISP_{split}_Motion{res}{('Jit' if not motion_jit else '')}{tag}.lmdb"
            )
            if self.motion_lerp
            else None
        )
        index = json.loads((self.root / f"QRISP_{split}_index.json").read_text())
        frames = index["frames"]
        cache = (
            Path(__file__).resolve().parents[1]
            / "runs"
            / "reference"
            / f"spread_{split}_x{self.scale}.npz"
        )
        if cache.is_file():
            self._spread = np.load(cache)["spread"]
            if len(self._spread) != len(frames):
                self._spread = None
        if scenes:
            frames = [f for f in frames if f["scene"] in scenes]
        self.frames = frames
        run = 0
        chains: list[int] = []
        for i, frame in enumerate(frames):
            contiguous = i > 0 and (
                frames[i - 1]["scene"],
                frames[i - 1]["sequence"],
            ) == (frame["scene"], frame["sequence"])
            if contiguous:
                try:
                    contiguous = int(frame["frame"]) == int(frames[i - 1]["frame"]) + 1
                except ValueError:
                    contiguous = False
            run = run + 1 if contiguous else 1
            chains.append(run)
        self.anchors = [i for (i, r) in enumerate(chains) if r >= self.length]
        if not self.anchors:
            raise RuntimeError(f"no contiguous run of {self.length} frames in {split}")
        count = 1 if self.tiles is None else self.tiles[0] * self.tiles[1]
        self.samples = [(a, t) for a in self.anchors for t in range(count)]

    def __len__(self) -> int:
        return len(self.samples)

    def scene_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, (anchor, _) in enumerate(self.samples):
            out.setdefault(self.frames[anchor]["scene"], []).append(i)
        return out

    def _key(self, frame_index: int, tile: int) -> str:
        key = self.frames[frame_index]["key"]
        return key if self.tiles is None else f"{key}/{tile}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        (anchor, tile) = self.samples[index]
        order = list(range(anchor - self.length + 1, anchor + 1))
        (lrs, hrs, depths, motions, jitters) = ([], [], [], [], [])
        top = left = None
        for frame_index in order:
            key = self._key(frame_index, tile)
            lr = self.lr.image(key)
            hr = self.hr.image(key)
            depth = self.depth.array(key)
            motion = self.motion.array(key)
            if self.motion_other is not None:
                motion = motion + self.motion_lerp * (
                    self.motion_other.array(key) - motion
                )
            size = self.patch_lr
            if size and size < min(lr.shape[:2]):
                if top is None:
                    top = random.randint(0, lr.shape[0] - size)
                    left = random.randint(0, lr.shape[1] - size)
                lr = lr[top : top + size, left : left + size]
                hr = hr[
                    top * self.scale : (top + size) * self.scale,
                    left * self.scale : (left + size) * self.scale,
                ]
                depth = depth[top : top + size, left : left + size]
                motion = motion[top : top + size, left : left + size]
            meta = self.frames[frame_index]
            lrs.append(lr)
            hrs.append(hr)
            depths.append(depth)
            motions.append(motion)
            jitters.append(list(meta["jitter_yx"]) if self.jittered else [0.0, 0.0])
        if self.flip:
            if random.random() < 0.5:
                lrs = [a[:, ::-1] for a in lrs]
                hrs = [a[:, ::-1] for a in hrs]
                depths = [a[:, ::-1] for a in depths]
                motions = [a[:, ::-1].copy() for a in motions]
                for m in motions:
                    m[..., 1] *= -1.0
                for j in jitters:
                    j[1] = -j[1]
            if random.random() < 0.5:
                lrs = [a[::-1] for a in lrs]
                hrs = [a[::-1] for a in hrs]
                depths = [a[::-1] for a in depths]
                motions = [a[::-1].copy() for a in motions]
                for m in motions:
                    m[..., 0] *= -1.0
                for j in jitters:
                    j[0] = -j[0]
        return {
            "lr": torch.stack([to_tensor(np.ascontiguousarray(a)) for a in lrs]),
            "hr": torch.stack([to_tensor(np.ascontiguousarray(a)) for a in hrs]),
            "depth": torch.stack(
                [torch.from_numpy(np.ascontiguousarray(a))[None] for a in depths]
            ),
            "motion": torch.stack(
                [
                    torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
                    for a in motions
                ]
            ),
            "jitter": torch.tensor(jitters, dtype=torch.float32),
            "spread": torch.tensor(
                [float(self._spread[i]) for i in order]
                if self._spread is not None
                else [-1.0] * len(order),
                dtype=torch.float32,
            ),
            "scene": self.frames[anchor]["scene"],
            "key": self._key(anchor, tile),
        }


def build_clip_train(scale: int = 4, **kwargs: Any) -> QrispClips:
    return QrispClips("train", scale=scale, **kwargs)


def build_clip_test(scale: int = 4, **kwargs: Any) -> QrispClips:
    kwargs.setdefault("tiles", None)
    kwargs.setdefault("flip", False)
    kwargs.setdefault("patch_lr", 0)
    return QrispClips("test", scale=scale, **kwargs)


def test_segments(
    scale: int = 4,
    colour: str = "MipBiasJit",
    length: int = 16,
    per_scene: int = 4,
    root: Path | str = QRISP_LMDB,
) -> list[list[int]]:
    clips = QrispClips(
        "test",
        scale=scale,
        length=length,
        colour=colour,
        tiles=None,
        flip=False,
        patch_lr=0,
        root=root,
    )
    by_scene: dict[str, list[int]] = {}
    for position, (anchor, _) in enumerate(clips.samples):
        by_scene.setdefault(clips.frames[anchor]["scene"], []).append(position)
    picked = []
    for scene in sorted(by_scene):
        pool = by_scene[scene]
        take = (np.arange(per_scene) + 0.5) / per_scene * len(pool)
        picked.extend((int(pool[int(min(t, len(pool) - 1))]) for t in take))
    return sorted(set(picked))


def static_positions(
    clips: QrispClips, segments: Path | str = "runs/reference/static_segments.json"
) -> list[int]:
    path = Path(segments)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        return []
    runs = json.loads(path.read_text())["segments"]
    anchor_at = {
        anchor: position for (position, (anchor, _)) in enumerate(clips.samples)
    }
    picked = []
    for run in runs:
        for anchor in range(run["start"] + clips.length - 1, run["end"] + 1):
            if anchor in anchor_at:
                picked.append(anchor_at[anchor])
    return sorted(set(picked))
