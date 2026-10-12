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
from typing import Any, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "STSS_PACKED",
    "STSS_TRAIN",
    "STSS_TEST",
    "DEPTH_FAR",
    "DEPTH_MODES",
    "StssClips",
    "StssSequence",
    "build_stss_train",
    "build_stss_test",
    "build_stss_clip_test",
    "stss_test_segments",
    "stss_static_positions",
    "stss_pipeline",
    "stss_data_provenance",
    "stss_depth",
]
STSS_TRAIN = ("Lewis_train1",)
STSS_TEST = "Arena_test"
DEPTH_FAR = 16.0
DEPTH_MODES = ("linear", "rational")


def stss_depth(raw: np.ndarray, mode: str = "linear") -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if mode == "linear":
        return np.clip(raw / DEPTH_FAR, 0.0, 1.0)
    if mode == "rational":
        return raw / (1.0 + raw)
    raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")


class _Scene:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.name = self.root.name
        meta_path = self.root / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        order = self.meta.get("layout", {}).get("colour_order", "BGR")
        if order not in ("BGR", "RGB"):
            raise ValueError(f"{self.root}: unknown colour_order {order}")
        self.bgr = order == "BGR"
        on_disk = {
            int(p.stem)
            for p in (self.root / "lr").glob("[0-9]*.npy")
            if (self.root / "hr" / p.name).is_file()
        }
        manifest = self.root / "frames.npy"
        listed = (
            [int(x) for x in np.load(manifest)]
            if manifest.is_file()
            else sorted(on_disk)
        )
        self.frames = [n for n in listed if n in on_disk]
        if not self.frames:
            raise RuntimeError(f"no packed frames in {self.root}")
        self.spread: dict[int, float] = {}
        self.medmotion: dict[int, float] = {}
        for attr, fname in (("spread", "spread.npy"), ("medmotion", "medmotion.npy")):
            path = self.root / fname
            if manifest.is_file() and path.is_file():
                values = np.load(path)
                if len(values) == len(listed):
                    setattr(self, attr, {n: float(v) for (n, v) in zip(listed, values)})
        index = set(self.frames)
        self.cuts = {int(c) for c in self.meta.get("cuts", [])}
        self.run: dict[int, int] = {}
        for n in self.frames:
            chained = n - 1 in index and n not in self.cuts
            self.run[n] = self.run.get(n - 1, 0) + 1 if chained else 1

    def lr(self, n: int) -> np.ndarray:
        return np.load(self.root / "lr" / f"{n:05d}.npy", mmap_mode="r")

    def hr(self, n: int) -> np.ndarray:
        return np.load(self.root / "hr" / f"{n:05d}.npy", mmap_mode="r")

    @staticmethod
    def _rows(path: Path, top: int | None, h: int | None) -> np.ndarray:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            header = {
                (1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0,
            }.get(version)
            shape = fortran = dtype = None
            if header is not None:
                (shape, fortran, dtype) = header(f)
            if header is None or fortran or dtype.hasobject or (not shape):
                array = np.load(path, mmap_mode="r")
                return array if top is None else array[top : top + h]
            row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
            if top is None:
                (top, h) = (0, shape[0])
            if top < 0 or h < 0 or top + h > shape[0]:
                raise ValueError(f"{path}: rows [{top}, {top + h}) outside {shape[0]}")
            f.seek(f.tell() + top * row)
            buf = f.read(h * row)
        if len(buf) != h * row:
            raise OSError(f"{path}: short read ({len(buf)} of {h * row} B)")
        return np.frombuffer(buf, dtype=dtype).reshape((h,) + tuple(shape[1:]))

    def read(
        self, n: int, box: tuple[int, int, int, int] | None, scale: int, depth_mode: str
    ) -> tuple[np.ndarray, ...]:
        lr_path = self.root / "lr" / f"{n:05d}.npy"
        hr_path = self.root / "hr" / f"{n:05d}.npy"
        if box is not None:
            (top, left, h, w) = box
            lo = self._rows(lr_path, top, h)[:, left : left + w]
            hi = self._rows(hr_path, top * scale, h * scale)[
                :, left * scale : (left + w) * scale
            ]
        else:
            (lo, hi) = (
                self._rows(lr_path, None, None),
                self._rows(hr_path, None, None),
            )
        lo = np.asarray(lo, dtype=np.float32)
        hi = np.asarray(hi, dtype=np.float32)
        (colour, hr) = (lo[:, :, 0:3], hi)
        if self.bgr:
            (colour, hr) = (colour[:, :, ::-1], hr[:, :, ::-1])
        colour = np.clip(colour, 0.0, 1.0)
        hr = np.clip(hr, 0.0, 1.0)
        depth = stss_depth(lo[:, :, 3], depth_mode)
        motion = np.stack((lo[:, :, 5], -lo[:, :, 6]), axis=2)
        return (colour, hr, depth, motion)


def _roots(
    root: str | Path | Sequence[str | Path] | None, default: Sequence[str]
) -> list[Path]:
    if root is None:
        return [STSS_PACKED / name for name in default]
    if isinstance(root, (str, Path)):
        return [Path(root)]
    return [Path(r) for r in root]


def _chw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)


class StssClips(Dataset):
    def __init__(
        self,
        root: str | Path | Sequence[str | Path] | None = None,
        length: int = 16,
        patch_lr: int | None = 66,
        flip: bool = True,
        scale: int = 2,
        depth_mode: str = "linear",
    ) -> None:
        if int(scale) != 2:
            raise ValueError("STSS is 540p -> 1080p, i.e. x2 only")
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")
        self.scenes = [_Scene(r) for r in _roots(root, STSS_TRAIN)]
        (self.length, self.scale) = (int(length), int(scale))
        self.patch_lr = int(patch_lr or 0)
        (self.flip, self.depth_mode) = (bool(flip), depth_mode)
        self.samples = [
            (s, n)
            for (s, scene) in enumerate(self.scenes)
            for n in scene.frames
            if scene.run[n] >= self.length
        ]
        if not self.samples:
            raise RuntimeError(
                f"no contiguous run of {self.length} frames in {[s.root for s in self.scenes]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def scene_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, (s, _) in enumerate(self.samples):
            out.setdefault(self.scenes[s].name, []).append(i)
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        (s, anchor) = self.samples[index]
        scene = self.scenes[s]
        order = list(range(anchor - self.length + 1, anchor + 1))
        (height, width) = scene.lr(anchor).shape[:2]
        size = self.patch_lr
        box = None
        if size and size < min(height, width):
            box = (
                random.randint(0, height - size),
                random.randint(0, width - size),
                size,
                size,
            )
        (lrs, hrs, depths, motions) = ([], [], [], [])
        for n in order:
            (colour, hr, depth, motion) = scene.read(
                n, box, self.scale, self.depth_mode
            )
            lrs.append(colour)
            hrs.append(hr)
            depths.append(depth[:, :, None])
            motions.append(motion)
        if self.flip:
            if random.random() < 0.5:
                lrs = [a[:, ::-1] for a in lrs]
                hrs = [a[:, ::-1] for a in hrs]
                depths = [a[:, ::-1] for a in depths]
                motions = [a[:, ::-1].copy() for a in motions]
                for m in motions:
                    m[..., 1] *= -1.0
            if random.random() < 0.5:
                lrs = [a[::-1] for a in lrs]
                hrs = [a[::-1] for a in hrs]
                depths = [a[::-1] for a in depths]
                motions = [a[::-1].copy() for a in motions]
                for m in motions:
                    m[..., 0] *= -1.0
        spread = [scene.spread.get(n, -1.0) if scene.spread else -1.0 for n in order]
        return {
            "lr": torch.stack([_chw(a) for a in lrs]),
            "hr": torch.stack([_chw(a) for a in hrs]),
            "depth": torch.stack([_chw(a) for a in depths]),
            "motion": torch.stack([_chw(a) for a in motions]),
            "jitter": torch.zeros(len(order), 2, dtype=torch.float32),
            "spread": torch.tensor(spread, dtype=torch.float32),
            "scene": scene.name,
            "key": f"{scene.name}/{anchor:05d}",
        }


class StssSequence(Dataset):
    def __init__(
        self,
        root: str | Path | Sequence[str | Path] | None = None,
        scale: int = 2,
        depth_mode: str = "linear",
        crop: tuple[int, int, int, int] | None = None,
        max_frames: int = 0,
    ) -> None:
        if int(scale) != 2:
            raise ValueError("STSS is 540p -> 1080p, i.e. x2 only")
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"depth_mode must be one of {DEPTH_MODES}")
        (self.scale, self.depth_mode) = (int(scale), depth_mode)
        self.crop = tuple((int(v) for v in crop)) if crop else None
        self.scenes = {}
        self.frames: list[dict[str, Any]] = []
        for path in _roots(root, (STSS_TEST,)):
            scene = _Scene(path)
            self.scenes[scene.name] = scene
            numbers = scene.frames[:max_frames] if max_frames else scene.frames
            run = -1
            for n in numbers:
                if scene.run[n] == 1:
                    run += 1
                seq = f"{run:04d}"
                self.frames.append(
                    {
                        "key": f"{scene.name}/{seq}/{n:05d}",
                        "scene": scene.name,
                        "sequence": seq,
                        "frame": f"{n:05d}",
                        "n": n,
                        "jitter_yx": [0.0, 0.0],
                        "spread": scene.spread.get(n, -1.0) if scene.spread else -1.0,
                        "medmotion": scene.medmotion.get(n, -1.0)
                        if scene.medmotion
                        else -1.0,
                    }
                )
        self.previous: list[int | None] = []
        for i, f in enumerate(self.frames):
            p = self.frames[i - 1] if i else None
            self.previous.append(
                i - 1
                if p is not None
                and (p["scene"], p["sequence"]) == (f["scene"], f["sequence"])
                else None
            )
        self.samples = [(i, f["key"]) for (i, f) in enumerate(self.frames)]

    def __len__(self) -> int:
        return len(self.samples)

    def scene_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, f in enumerate(self.frames):
            out.setdefault(f["scene"], []).append(i)
        return out

    def runs(self) -> dict[str, list[list[int]]]:
        out: dict[str, list[list[int]]] = {}
        last = None
        for i, f in enumerate(self.frames):
            key = (f["scene"], f["sequence"])
            if key != last:
                out.setdefault(f["scene"], []).append([])
                last = key
            out[f["scene"]][-1].append(i)
        return out

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.frames[index]
        scene = self.scenes[meta["scene"]]
        (colour, hr, depth, motion) = scene.read(
            meta["n"], self.crop, self.scale, self.depth_mode
        )
        return {
            "lr": _chw(colour),
            "hr": _chw(hr),
            "jitter": torch.zeros(2, dtype=torch.float32),
            "key": meta["key"],
            "scene": meta["scene"],
            "depth": torch.from_numpy(np.ascontiguousarray(depth))[None],
            "motion": _chw(motion),
        }


def build_stss_train(
    root=None, length: int = 16, patch_lr: int | None = 66, **kwargs: Any
) -> StssClips:
    return StssClips(root, length=length, patch_lr=patch_lr, **kwargs)


def build_stss_clip_test(root=None, length: int = 16, **kwargs: Any) -> StssClips:
    kwargs.setdefault("flip", False)
    kwargs.setdefault("patch_lr", 0)
    return StssClips(
        root if root is not None else STSS_PACKED / STSS_TEST, length=length, **kwargs
    )


def build_stss_test(root=None, **kwargs: Any) -> StssSequence:
    return StssSequence(root, **kwargs)


def stss_test_segments(clips: StssClips, per_scene: int = 4) -> list[int]:
    picked = []
    for _, pool in sorted(clips.scene_indices().items()):
        take = (np.arange(per_scene) + 0.5) / per_scene * len(pool)
        picked.extend((int(pool[int(min(t, len(pool) - 1))]) for t in take))
    return sorted(set(picked))


def stss_static_positions(clips: StssClips, threshold: float = 0.25) -> list[int]:
    picked = []
    for position, (s, anchor) in enumerate(clips.samples):
        scene = clips.scenes[s]
        if not scene.spread:
            continue
        values = [
            scene.spread.get(n, -1.0)
            for n in range(anchor - clips.length + 1, anchor + 1)
        ]
        if all((0.0 <= v < threshold for v in values)):
            picked.append(position)
    return picked


def stss_pipeline(cfg: dict) -> dict:
    dataset = cfg.get("dataset") or "qrisp"
    depth_mode = cfg.get("stss_depth_mode") or "linear"
    if depth_mode not in DEPTH_MODES:
        raise ValueError(f"config stss_depth_mode {depth_mode!r} not in {DEPTH_MODES}")
    return {
        "dataset": dataset,
        "depth_mode": depth_mode,
        "train_roots": cfg.get("stss_train"),
        "test_root": cfg.get("stss_test"),
        "scale": int(cfg.get("scale") or 2),
    }


def stss_data_provenance(roots: Sequence[str | Path]) -> dict:
    import hashlib

    out = {}
    for r in roots:
        scene = _Scene(r)
        manifest = scene.root / "frames.npy"
        out[scene.name] = {
            "root": str(scene.root),
            "frames": len(scene.frames),
            "cuts": sorted(scene.cuts),
            "frames_npy_md5": hashlib.md5(manifest.read_bytes()).hexdigest()
            if manifest.is_file()
            else None,
            "colour_order_stored": "BGR" if scene.bgr else "RGB",
            "meta_provenance": scene.meta.get("provenance"),
        }
    return out


def _selftest() -> None:
    from .data_seq import QrispClips
    from .data_qrisp import QrispDataset
    from .rrta import warp
    import torch.nn.functional as F

    ok = True

    def check(cond: bool, what: str) -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{('PASS' if cond else 'FAIL')}] {what}", flush=True)

    random.seed(0)
    q = QrispClips(
        "train",
        scale=2,
        length=4,
        patch_lr=66,
        colour="MipBiasJit",
        motion_jit=False,
        depth_jit=False,
    )[0]
    clips = StssClips(length=4, patch_lr=66)
    s = clips[len(clips) // 2]
    check(set(q) == set(s), f"StssClips keys == QrispClips keys {sorted(s)}")
    for k, v in q.items():
        if torch.is_tensor(v):
            check(
                s[k].dtype == v.dtype and s[k].shape == v.shape,
                f"{k}: {tuple(s[k].shape)} {s[k].dtype} vs QrispClips {tuple(v.shape)} {v.dtype}",
            )
        else:
            check(type(s[k]) is type(v), f"{k}: {type(s[k]).__name__}")
    check(bool((s["jitter"] == 0).all()), "jitter is zero")
    scene = clips.scenes[0]
    n_last = int(s["key"].split("/")[-1])
    want = [scene.spread[n] for n in range(n_last - 3, n_last + 1)]
    check(
        np.allclose(s["spread"].numpy(), want),
        f"spread is the FULL-FRAME value {np.round(want, 3).tolist()}",
    )
    check(
        float(s["lr"].min()) >= 0
        and float(s["lr"].max()) <= 1
        and (float(s["hr"].max()) <= 1),
        "colour clamped to [0,1]",
    )
    missing = [
        n
        for n in range(scene.frames[0], scene.frames[-1])
        if n not in set(scene.frames)
    ]
    crossing = [
        n
        for (_, n) in clips.samples
        if any((n - k in missing for k in range(clips.length)))
        or any((n - k in scene.cuts for k in range(clips.length - 1)))
    ]
    check(
        not crossing,
        f"no clip crosses a gap (missing {missing}) or a reset (cuts {sorted(scene.cuts)})",
    )
    whole = StssClips(length=2, patch_lr=0, flip=False)
    moving = [
        i
        for (i, (_, n)) in enumerate(whole.samples)
        if whole.scenes[0].medmotion.get(n, 0) > 1.0
    ]
    base = whole[moving[len(moving) // 2]]

    def residual(c: dict) -> tuple[float, float]:
        m = F.interpolate(c["motion"][1:2], scale_factor=2, mode="nearest") * 2
        (w, inside) = warp(c["hr"][0:1], m)
        r = float(((w - c["hr"][1:2]).abs() * inside).sum() / inside.expand_as(w).sum())
        return (r, float((c["hr"][0] - c["hr"][1]).abs().mean()))

    (r0, ident) = residual(base)
    check(r0 < 0.5 * ident, f"reprojection {r0:.5f} << identity {ident:.5f}")
    for h, v in ((True, False), (False, True), (True, True)):
        f = dict(base)
        for key in ("lr", "hr", "depth", "motion"):
            x = base[key]
            x = x.flip(-1) if h else x
            f[key] = x.flip(-2) if v else x
        f["motion"] = f["motion"].clone()
        if h:
            f["motion"][:, 1] *= -1
        if v:
            f["motion"][:, 0] *= -1
        (r, _) = residual(f)
        check(
            abs(r - r0) < 0.001 * max(1.0, r0) + 0.0002,
            f"flip h={h} v={v}: residual {r:.5f} == {r0:.5f}",
        )
    random.seed(3)
    flipped = StssClips(length=2, patch_lr=0, flip=True)
    hits = 0
    for trial in range(6):
        c = flipped[moving[len(moving) // 2]]
        (r, _) = residual(c)
        hits += abs(r - r0) < 0.001 * max(1.0, r0) + 0.0002
    check(hits == 6, f"StssClips random flips keep the residual ({hits}/6)")
    seq = StssSequence()
    item = seq[len(seq) // 2]
    sky = item["depth"][0] >= 1.0
    (b, r) = (float(item["lr"][2][sky].mean()), float(item["lr"][0][sky].mean()))
    check(
        sky.float().mean() > 0.05 and b > r,
        f"RGB: sky B {b:.3f} > R {r:.3f} ({float(sky.float().mean()):.2f} of the frame is sky)",
    )
    qd = QrispDataset(
        "test",
        scale=2,
        colour="MipBiasJit",
        with_depth=True,
        with_motion=True,
        patch_lr=0,
        flip=False,
        tiles=None,
        motion_jit=False,
        depth_jit=False,
    )
    qi = qd[0]
    check(
        set(qi) == set(item), f"StssSequence item keys == QrispDataset {sorted(item)}"
    )
    for k, v in qi.items():
        if torch.is_tensor(v):
            check(
                item[k].dtype == v.dtype and item[k].ndim == v.ndim,
                f"{k}: {tuple(item[k].shape)} {item[k].dtype}",
            )
    check(
        item["lr"].shape == (3, 540, 960) and item["hr"].shape == (3, 1080, 1920),
        "whole 540p / 1080p frames",
    )
    runs = seq.runs()
    check(
        len(seq) == 1001 and [len(o) for o in runs["Arena_test"]] == [1001],
        f"Arena: {len(seq)} frames in runs {[len(o) for o in runs['Arena_test']]}",
    )
    check(
        seq.previous[0] is None
        and all((seq.previous[i] == i - 1 for i in range(1, len(seq)))),
        "previous chain unbroken",
    )
    check(
        all(
            (
                set(f) >= {"key", "scene", "sequence", "frame", "jitter_yx"}
                for f in seq.frames
            )
        ),
        "frames carry protocol_full's fields",
    )
    d = item["depth"]
    check(
        float(d.min()) > 0 and float(d.max()) == 1.0,
        f"linear depth in (0,1], near {float(d.min()):.4f}",
    )
    rng = np.random.default_rng(0)
    (exact, tried) = (True, 0)
    for sc in (clips.scenes[0], next(iter(seq.scenes.values()))):
        for n in rng.choice(sc.frames, 4, replace=False):
            n = int(n)
            for kind, s in (("lr", 1), ("hr", 2)):
                ref = getattr(sc, kind)(n)
                path = sc.root / kind / f"{n:05d}.npy"
                top = int(rng.integers(0, ref.shape[0] // s - 66))
                exact &= np.array_equal(
                    sc._rows(path, top * s, 66 * s),
                    np.asarray(ref[top * s : (top + 66) * s]),
                )
                exact &= np.array_equal(sc._rows(path, None, None), np.asarray(ref))
                tried += 2
    check(
        exact,
        f"one-read row-band loader == mmap ({tried} reads, Lewis + Arena, crops and whole frames)",
    )
    print("ALL PASS" if ok else "SOME CHECKS FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _selftest()
