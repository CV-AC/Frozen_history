from __future__ import annotations
from diffsr.paths import (
    REPOSITORY,
    DATA_ROOT,
    QRISP_RAW,
    QRISP_LMDB,
    STSS_PACKED,
    ARM_TEST,
    STSSNET_UTILS,
)
import json
import random
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_sisr import to_tensor

RES_FOR_SCALE = {2: "540", 3: "360", 4: "270"}
COLOUR_VARIANTS = ("Native", "MipBias", "MipBiasJit")
__all__ = ["QrispDataset", "QRISP_LMDB", "build_qrisp_train", "build_qrisp_test"]
_ENV_CACHE: dict[str, Any] = {}


class _Lmdb:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not (self.path / "data.mdb").is_file():
            raise FileNotFoundError(self.path)
        first = (self.path / "meta_info.txt").read_text().split("\n", 1)[0].split()
        self.dtype = first[1]
        self.shape = (
            tuple((int(v) for v in first[2].split("x"))) if first[2] != "-" else None
        )
        self._env = None

    def env(self):
        if self._env is None:
            key = str(self.path.resolve())
            cached = _ENV_CACHE.get(key)
            if cached is not None:
                self._env = cached
                return self._env
            import lmdb

            self._env = lmdb.open(
                str(self.path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
            )
            _ENV_CACHE[key] = self._env
        return self._env

    def get(self, key: str) -> bytes:
        with self.env().begin(write=False) as txn:
            buf = txn.get(key.encode())
        if buf is None:
            raise KeyError(f"{key} not in {self.path.name}")
        return buf

    def image(self, key: str) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.get(key), np.uint8), cv2.IMREAD_COLOR)

    def array(self, key: str) -> np.ndarray:
        return (
            np.frombuffer(self.get(key), np.float16)
            .reshape(self.shape)
            .astype(np.float32)
        )


class QrispDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        scale: int = 4,
        patch_lr: int | None = None,
        colour: str = "MipBias",
        with_depth: bool = True,
        with_motion: bool = False,
        with_history: bool = False,
        motion_jit: bool = True,
        motion_lerp: float = 0.0,
        depth_jit: bool = True,
        target: str = "Enhanced",
        flip: bool = True,
        scenes: tuple[str, ...] | None = None,
        tiles: tuple[int, int] | None = (2, 4),
        root: Path | str = QRISP_LMDB,
    ) -> None:
        if colour not in COLOUR_VARIANTS:
            raise ValueError(f"colour must be one of {COLOUR_VARIANTS}")
        if colour == "Native" and (with_depth or with_motion or with_history):
            raise ValueError(
                "QRISP ships depth and motion only for the mip-biased renders; use colour='MipBias' or turn both off"
            )
        self.root = Path(root)
        (self.split, self.scale) = (split, int(scale))
        (self.colour, self.flip) = (colour, bool(flip))
        self.motion_jit = bool(motion_jit)
        self.motion_lerp = float(motion_lerp)
        self.depth_jit = bool(depth_jit)
        self.with_history = bool(with_history)
        if self.with_history and (not with_depth):
            raise ValueError("history reconstruction requires current depth")
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
        self.depth = (
            _Lmdb(self.root / f"QRISP_{split}_Depth{res}{depth_suffix}{tag}.lmdb")
            if with_depth
            else None
        )
        self.motion = (
            _Lmdb(self.root / f"QRISP_{split}_Motion{res}{motion_suffix}{tag}.lmdb")
            if with_motion or self.with_history
            else None
        )
        other = "Jit" if not motion_jit else ""
        self.motion_other = (
            _Lmdb(self.root / f"QRISP_{split}_Motion{res}{other}{tag}.lmdb")
            if self.motion_lerp and self.motion
            else None
        )
        self.return_motion = bool(with_motion)
        index = json.loads((self.root / f"QRISP_{split}_index.json").read_text())
        frames = index["frames"]
        if scenes:
            frames = [f for f in frames if f["scene"] in scenes]
        self.frames = frames
        self.previous = []
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
            self.previous.append(i - 1 if contiguous else None)
        if self.tiles is None:
            self.samples = [(i, f["key"]) for (i, f) in enumerate(frames)]
        else:
            count = self.tiles[0] * self.tiles[1]
            self.samples = [
                (i, f"{f['key']}/{t}")
                for (i, f) in enumerate(frames)
                for t in range(count)
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def scene_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, (frame_index, _) in enumerate(self.samples):
            out.setdefault(self.frames[frame_index]["scene"], []).append(i)
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        (frame_index, key) = self.samples[index]
        meta = self.frames[frame_index]
        lr = self.lr.image(key)
        hr = self.hr.image(key)
        depth = self.depth.array(key) if self.depth else None
        motion = self.motion.array(key) if self.motion else None
        if self.motion_other is not None and motion is not None:
            motion = motion + self.motion_lerp * (self.motion_other.array(key) - motion)
        history_lr = history_depth = history_valid = None
        history_weight = history_footprint_offset = None
        history_jitter = [0.0, 0.0]
        if self.with_history:
            previous_index = self.previous[frame_index]
            if previous_index is None:
                history_lr = np.repeat(lr[None], 4, axis=0)
                history_depth = np.repeat(depth[None], 4, axis=0)
                history_valid = np.zeros((4, *lr.shape[:2]), dtype=np.float32)
                history_weight = np.zeros_like(history_valid)
                history_footprint_offset = np.zeros(
                    (4, *lr.shape[:2], 2), dtype=np.float32
                )
                history_jitter = list(meta["jitter_yx"])
            else:
                previous = self.frames[previous_index]
                suffix_key = key[len(meta["key"]) :]
                previous_key = previous["key"] + suffix_key
                previous_lr = self.lr.image(previous_key)
                previous_depth = self.depth.array(previous_key)
                (height, width) = lr.shape[:2]
                (rows, cols) = np.meshgrid(
                    np.arange(height, dtype=np.float32),
                    np.arange(width, dtype=np.float32),
                    indexing="ij",
                )
                map_y = rows + motion[..., 0]
                map_x = cols + motion[..., 1]
                (y0, x0) = (np.floor(map_y), np.floor(map_x))
                (fy, fx) = (map_y - y0, map_x - x0)
                tap_y = np.stack((y0, y0, y0 + 1.0, y0 + 1.0))
                tap_x = np.stack((x0, x0 + 1.0, x0, x0 + 1.0))
                history_weight = np.stack(
                    ((1.0 - fy) * (1.0 - fx), (1.0 - fy) * fx, fy * (1.0 - fx), fy * fx)
                ).astype(np.float32)
                history_valid = (
                    (tap_y >= 0.0)
                    & (tap_y <= height - 1.0)
                    & (tap_x >= 0.0)
                    & (tap_x <= width - 1.0)
                ).astype(np.float32)
                iy = np.clip(tap_y, 0, height - 1).astype(np.int64)
                ix = np.clip(tap_x, 0, width - 1).astype(np.int64)
                history_lr = previous_lr[iy, ix]
                history_depth = previous_depth[iy, ix]
                history_footprint_offset = np.stack(
                    (tap_y - map_y[None], tap_x - map_x[None]), axis=-1
                ).astype(np.float32)
                history_jitter = list(previous["jitter_yx"])
        size = self.patch_lr
        if size and size < min(lr.shape[:2]):
            top = random.randint(0, lr.shape[0] - size)
            left = random.randint(0, lr.shape[1] - size)
            lr = lr[top : top + size, left : left + size]
            hr = hr[
                top * self.scale : (top + size) * self.scale,
                left * self.scale : (left + size) * self.scale,
            ]
            if depth is not None:
                depth = depth[top : top + size, left : left + size]
            if motion is not None:
                motion = motion[top : top + size, left : left + size]
            if history_lr is not None:
                history_lr = history_lr[:, top : top + size, left : left + size]
                history_depth = history_depth[:, top : top + size, left : left + size]
                history_valid = history_valid[:, top : top + size, left : left + size]
                history_weight = history_weight[:, top : top + size, left : left + size]
                history_footprint_offset = history_footprint_offset[
                    :, top : top + size, left : left + size
                ]
        jitter = list(meta["jitter_yx"]) if self.jittered else [0.0, 0.0]
        if self.flip:
            if random.random() < 0.5:
                (lr, hr) = (lr[:, ::-1], hr[:, ::-1])
                if depth is not None:
                    depth = depth[:, ::-1]
                if motion is not None:
                    motion = motion[:, ::-1].copy()
                    motion[..., 1] *= -1.0
                if history_lr is not None:
                    history_lr = history_lr[:, :, ::-1]
                    history_depth = history_depth[:, :, ::-1]
                    history_valid = history_valid[:, :, ::-1]
                    history_weight = history_weight[:, :, ::-1]
                    history_footprint_offset = history_footprint_offset[
                        :, :, ::-1
                    ].copy()
                    history_footprint_offset[..., 1] *= -1.0
                    history_jitter[1] = -history_jitter[1]
                jitter[1] = -jitter[1]
            if random.random() < 0.5:
                (lr, hr) = (lr[::-1], hr[::-1])
                if depth is not None:
                    depth = depth[::-1]
                if motion is not None:
                    motion = motion[::-1].copy()
                    motion[..., 0] *= -1.0
                if history_lr is not None:
                    history_lr = history_lr[:, ::-1]
                    history_depth = history_depth[:, ::-1]
                    history_valid = history_valid[:, ::-1]
                    history_weight = history_weight[:, ::-1]
                    history_footprint_offset = history_footprint_offset[:, ::-1].copy()
                    history_footprint_offset[..., 0] *= -1.0
                    history_jitter[0] = -history_jitter[0]
                jitter[0] = -jitter[0]
        sample = {
            "lr": to_tensor(np.ascontiguousarray(lr)),
            "hr": to_tensor(np.ascontiguousarray(hr)),
            "jitter": torch.tensor(jitter, dtype=torch.float32),
            "key": key,
            "scene": meta["scene"],
        }
        if depth is not None:
            sample["depth"] = torch.from_numpy(np.ascontiguousarray(depth))[None]
        if motion is not None and self.return_motion:
            sample["motion"] = torch.from_numpy(np.ascontiguousarray(motion)).permute(
                2, 0, 1
            )
        if history_lr is not None:
            sample["history_lr"] = (
                torch.from_numpy(np.ascontiguousarray(history_lr[..., ::-1]))
                .permute(0, 3, 1, 2)
                .float()
                .div_(255.0)
            )
            sample["history_depth"] = torch.from_numpy(
                np.ascontiguousarray(history_depth)
            )[:, None]
            sample["history_valid"] = torch.from_numpy(
                np.ascontiguousarray(history_valid)
            )[:, None]
            sample["history_weight"] = torch.from_numpy(
                np.ascontiguousarray(history_weight)
            )[:, None]
            sample["history_footprint_offset"] = torch.from_numpy(
                np.ascontiguousarray(history_footprint_offset)
            ).permute(0, 3, 1, 2)
            sample["history_jitter"] = torch.tensor(history_jitter, dtype=torch.float32)
        return sample


def build_qrisp_train(scale: int = 4, **kwargs: Any) -> QrispDataset:
    return QrispDataset("train", scale=scale, **kwargs)


def build_qrisp_test(scale: int = 4, **kwargs: Any) -> QrispDataset:
    kwargs.setdefault("patch_lr", 0)
    kwargs.setdefault("flip", False)
    kwargs.setdefault("tiles", None)
    return QrispDataset("test", scale=scale, **kwargs)


def _selftest() -> None:
    import time

    for scale in (4, 3, 2):
        print(f"=== x{scale} ===", flush=True)
        for colour, depth, motion in (
            ("MipBias", True, True),
            ("MipBiasJit", True, True),
            ("Native", False, False),
        ):
            try:
                ds = QrispDataset(
                    "train",
                    scale=scale,
                    colour=colour,
                    with_depth=depth,
                    with_motion=motion,
                )
            except Exception as exc:
                print(
                    f"  {colour:<12} unavailable ({type(exc).__name__}: {str(exc)[:70]})"
                )
                continue
            s = ds[0]
            ok = (
                s["hr"].shape[1] == s["lr"].shape[1] * scale
                and s["lr"].shape[1] == ds.patch_lr
            )
            extra = " ".join(
                (f"{k}{tuple(s[k].shape)}" for k in ("depth", "motion") if k in s)
            )
            print(
                f"  {colour:<12} {len(ds):>6} frames  lr{tuple(s['lr'].shape)} hr{tuple(s['hr'].shape)} {extra} jitter={s['jitter'].tolist()} {('OK' if ok else 'BAD')}"
            )
            order = random.sample(range(len(ds)), min(200, len(ds)))
            ds[order[0]]
            start = time.time()
            for i in order:
                ds[i]
            print(
                f"               {len(order) / (time.time() - start):7.1f} samples/s single process"
            )
    test = build_qrisp_test(scale=4, colour="MipBias", with_motion=True)
    s = test[0]
    print(
        f"=== test set: {len(test)} frames, { {k: len(v) for (k, v) in test.scene_indices().items()} }"
    )
    print(
        f"  full frame lr{tuple(s['lr'].shape)} hr{tuple(s['hr'].shape)} depth{tuple(s['depth'].shape)} motion{tuple(s['motion'].shape)}"
    )


if __name__ == "__main__":
    _selftest()
