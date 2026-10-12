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

__all__ = [
    "DATA_ROOT",
    "LmdbPairedDataset",
    "FolderPairedDataset",
    "build_train_dataset",
    "build_test_dataset",
    "to_tensor",
]


def to_tensor(image: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        .float()
        .div_(255.0)
    )


def _augment(
    lr: np.ndarray, hr: np.ndarray, flip: bool, rot: bool
) -> tuple[np.ndarray, np.ndarray]:
    if flip and random.random() < 0.5:
        (lr, hr) = (lr[:, ::-1], hr[:, ::-1])
    if flip and random.random() < 0.5:
        (lr, hr) = (lr[::-1], hr[::-1])
    if rot and random.random() < 0.5:
        (lr, hr) = (lr.transpose(1, 0, 2), hr.transpose(1, 0, 2))
    return (np.ascontiguousarray(lr), np.ascontiguousarray(hr))


class LmdbPairedDataset(Dataset):
    def __init__(
        self,
        hr_lmdb: str | Path,
        lr_lmdb: str | Path,
        scale: int,
        patch_lr: int = 64,
        flip: bool = True,
        rot: bool = True,
        repeat: int = 1,
    ) -> None:
        (self.hr_path, self.lr_path) = (Path(hr_lmdb), Path(lr_lmdb))
        self.scale = int(scale)
        self.patch_lr = int(patch_lr)
        (self.flip, self.rot) = (bool(flip), bool(rot))
        self.repeat = max(1, int(repeat))
        meta = (self.lr_path / "meta_info.txt").read_text().strip().splitlines()
        self.keys = [line.split()[0] for line in meta]
        if not self.keys:
            raise RuntimeError(f"{self.lr_path} has an empty meta_info.txt")
        self._env: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.keys) * self.repeat

    def _envs(self):
        if not self._env:
            import lmdb

            opts = dict(
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=512,
            )
            self._env = {
                "hr": lmdb.open(str(self.hr_path), **opts),
                "lr": lmdb.open(str(self.lr_path), **opts),
            }
        return (self._env["hr"], self._env["lr"])

    @staticmethod
    def _decode(env, key: bytes) -> np.ndarray:
        with env.begin(write=False) as txn:
            buf = txn.get(key)
        if buf is None:
            raise KeyError(key)
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self.keys[index % len(self.keys)].encode()
        (hr_env, lr_env) = self._envs()
        lr = self._decode(lr_env, key)
        hr = self._decode(hr_env, key)
        size = self.patch_lr
        top = random.randint(0, lr.shape[0] - size)
        left = random.randint(0, lr.shape[1] - size)
        lr = lr[top : top + size, left : left + size]
        hr = hr[
            top * self.scale : (top + size) * self.scale,
            left * self.scale : (left + size) * self.scale,
        ]
        (lr, hr) = _augment(lr, hr, self.flip, self.rot)
        return {
            "lr": to_tensor(lr),
            "hr": to_tensor(hr),
            "key": self.keys[index % len(self.keys)],
        }


class FolderPairedDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        scale: int,
        crop_hr: bool = True,
        limit: int | None = None,
    ) -> None:
        self.dir = Path(split_dir)
        self.scale = int(scale)
        self.crop_hr = bool(crop_hr)
        entry = json.loads((self.dir / "pairs.json").read_text())
        info = entry["scales"][str(self.scale)]
        if not info["count"]:
            raise RuntimeError(f"{self.dir} has no x{scale} pairs")
        self.pairs = info["pairs"][:limit] if limit else info["pairs"]
        self.lr_dir = self.dir / "LR_bicubic" / f"X{self.scale}"
        self.hr_dir = self.dir / "HR"

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        (hr_name, lr_name) = self.pairs[index]
        hr = cv2.imread(str(self.hr_dir / hr_name), cv2.IMREAD_COLOR)
        lr = cv2.imread(str(self.lr_dir / lr_name), cv2.IMREAD_COLOR)
        if self.crop_hr:
            hr = hr[: lr.shape[0] * self.scale, : lr.shape[1] * self.scale]
        return {"lr": to_tensor(lr), "hr": to_tensor(hr), "key": Path(hr_name).stem}


TRAIN_SETS = {"DIV2K": "DIV2K_train", "Flickr2K": "Flickr2K_train"}
TEST_SETS = ("Set5", "Set14", "BSD100", "Urban100", "Manga109-mid109")


def build_train_dataset(
    name: str = "DIV2K", scale: int = 4, patch_lr: int = 64, **kwargs: Any
):
    if name.upper() == "DF2K":
        from torch.utils.data import ConcatDataset

        parts = [
            build_train_dataset(n, scale, patch_lr, **kwargs)
            for n in ("DIV2K", "Flickr2K")
        ]
        combined = ConcatDataset(parts)
        (combined.scale, combined.patch_lr) = (scale, patch_lr)
        return combined
    tag = TRAIN_SETS.get(name, name)
    return LmdbPairedDataset(
        DATA_ROOT / "lmdb" / f"{tag}_HR_sub_x{scale}.lmdb",
        DATA_ROOT / "lmdb" / f"{tag}_LR_bicubicX{scale}_sub.lmdb",
        scale=scale,
        patch_lr=patch_lr,
        **kwargs,
    )


def build_test_dataset(
    name: str = "Set5", scale: int = 4, **kwargs: Any
) -> FolderPairedDataset:
    if name == "DIV2K_valid":
        return FolderPairedDataset(DATA_ROOT / "DIV2K" / "valid", scale, **kwargs)
    return FolderPairedDataset(DATA_ROOT / "benchmark" / name, scale, **kwargs)


def _selftest(scale: int = 4) -> None:
    import time

    print("=== folder test sets ===", flush=True)
    for name in TEST_SETS:
        try:
            ds = build_test_dataset(name, scale)
        except Exception as exc:
            print(f"  {name:<10} unavailable ({type(exc).__name__})")
            continue
        sample = ds[0]
        (lr, hr) = (sample["lr"], sample["hr"])
        ok = hr.shape[1] == lr.shape[1] * scale and hr.shape[2] == lr.shape[2] * scale
        print(
            f"  {name:<10} {len(ds):>4} pairs   lr{tuple(lr.shape)} hr{tuple(hr.shape)}  {('OK' if ok else 'MISALIGNED')}"
        )
    print("=== lmdb train sets ===", flush=True)
    for name in list(TRAIN_SETS) + ["DF2K"]:
        try:
            ds = build_train_dataset(name, scale)
        except Exception as exc:
            print(f"  {name:<10} unavailable ({type(exc).__name__}: {exc})")
            continue
        sample = ds[0]
        (lr, hr) = (sample["lr"], sample["hr"])
        ok = (
            hr.shape[1] == lr.shape[1] * scale
            and hr.shape[2] == lr.shape[2] * scale
            and (lr.shape[1] == ds.patch_lr)
        )
        print(
            f"  {name:<10} {len(ds):>7} crops  lr{tuple(lr.shape)} hr{tuple(hr.shape)}  {('OK' if ok else 'BAD')}"
        )
        if not isinstance(ds, LmdbPairedDataset):
            continue
        order = random.sample(range(len(ds)), min(600, len(ds)))
        _ = ds[order[0]]
        start = time.time()
        for i in order:
            ds[i]
        rate = len(order) / (time.time() - start)
        print(
            f"             random reads: {rate:8.1f} samples/s (single process, cold-ish cache)"
        )
        gaps = []
        for i in random.sample(range(len(ds)), 8):
            item = ds[i]
            hr_np = item["hr"].numpy().transpose(1, 2, 0)
            lr_np = item["lr"].numpy().transpose(1, 2, 0)
            down = cv2.resize(
                hr_np, lr_np.shape[:2][::-1], interpolation=cv2.INTER_AREA
            )
            gaps.append(float(np.abs(down - lr_np).mean()))
        print(
            f"             |area-downsampled HR - LR| = {np.mean(gaps):.4f} (max {np.max(gaps):.4f})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=4)
    _selftest(parser.parse_args().scale)
