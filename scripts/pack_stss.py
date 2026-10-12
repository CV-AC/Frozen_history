from __future__ import annotations
import argparse
import datetime
import hashlib
import itertools
import json
import os
import random
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

LR_RE = re.compile("(?:^|/)NPY/compressed\\.(\\d+)\\.NoWarp\\.npy$")
HR_RE = re.compile("(?:^|/)HR/compressedHR\\.(\\d+)\\.npy$")
MOTION_AT = {33: 30, 35: 31}
(LR_SHAPE, HR_SHAPE) = ((540, 960), (1080, 1920, 3))
LAYOUT = {
    "channels": 7,
    "dtype": "float16",
    "colour": [0, 3],
    "colour_order": "BGR",
    "depth": 3,
    "depth_units": "raw (linear view depth, source units)",
    "motion_raw": [4, 7],
    "motion_convention": "(dy, dx) = (+ch5, -ch6), LR pixels, backward (frame t -> t-1); ch4 is identically zero",
    "hr": "source array unchanged: BGR fp16 1080x1920x3",
}


class Source:
    def __init__(self, path: str, kind: str) -> None:
        (self.path, self.kind) = (str(path), kind)
        self._zip = None

    def names(self) -> list[str]:
        if self.kind == "zip":
            with zipfile.ZipFile(self.path) as z:
                return [i.filename for i in z.infolist() if not i.is_dir()]
        root = Path(self.path)
        return [
            str(p.relative_to(root))
            for sub in ("NPY", "HR")
            for p in (root / sub).glob("*.npy")
        ]

    def sizes(self) -> dict[str, int]:
        if self.kind == "zip":
            with zipfile.ZipFile(self.path) as z:
                return {i.filename: i.file_size for i in z.infolist()}
        root = Path(self.path)
        return {n: (root / n).stat().st_size for n in self.names()}

    def read(self, name: str) -> np.ndarray:
        if self.kind == "zip":
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.path)
            with self._zip.open(name) as f:
                return np.lib.format.read_array(f)
        return np.load(Path(self.path) / name)


def listing(src: Source) -> tuple[dict[int, str], dict[int, str], list[str]]:
    (lr, hr, other) = ({}, {}, [])
    for name in src.names():
        m = LR_RE.search(name)
        if m:
            lr[int(m.group(1))] = name
            continue
        m = HR_RE.search(name)
        if m:
            hr[int(m.group(1))] = name
            continue
        if not name.endswith(".Warp.npy"):
            other.append(name)
    return (lr, hr, other)


def motion_yx(comp: np.ndarray) -> np.ndarray:
    return np.stack(
        (comp[:, :, 5].astype(np.float32), -comp[:, :, 6].astype(np.float32)), axis=2
    )


def frame_stats(comp: np.ndarray) -> dict:
    m = motion_yx(comp)
    flat = m.reshape(-1, 2)
    return {
        "spread": float(flat.std(axis=0).max()),
        "medmotion": float(np.median(np.abs(flat), axis=0).max()),
        "motion_absmax": float(np.abs(flat).max()),
        "triplet0_absmax": float(np.abs(comp[:, :, 4].astype(np.float32)).max()),
        "nonfinite": int((~np.isfinite(comp.astype(np.float32))).sum()),
    }


def compact(t: np.ndarray) -> np.ndarray:
    if t.ndim != 3 or t.shape[:2] != LR_SHAPE or t.shape[2] not in MOTION_AT:
        raise ValueError(f"unexpected LR shape {t.shape}")
    m = MOTION_AT[t.shape[2]]
    return np.concatenate(
        [
            np.asarray(t[:, :, 0:3]),
            np.asarray(t[:, :, 26:27]),
            np.asarray(t[:, :, m : m + 3]),
        ],
        axis=2,
    ).astype(np.float16)


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _valid_npy(path: Path, shape: tuple) -> bool:
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape == shape and a.dtype == np.float16
    except Exception:
        return False


_SRC: Source | None = None


def _init(path: str, kind: str) -> None:
    global _SRC
    _SRC = Source(path, kind)


def pack_one(n: int, lr_name: str, hr_name: str, dst: str, overwrite: bool) -> dict:
    out_lr = Path(dst) / "lr" / f"{n:05d}.npy"
    out_hr = Path(dst) / "hr" / f"{n:05d}.npy"
    try:
        channels = None
        if (
            not overwrite
            and _valid_npy(out_lr, (*LR_SHAPE, 7))
            and _valid_npy(out_hr, HR_SHAPE)
        ):
            comp = np.load(out_lr)
            reused = True
        else:
            t = _SRC.read(lr_name)
            channels = int(t.shape[2]) if t.ndim == 3 else None
            comp = compact(t)
            del t
            h = _SRC.read(hr_name)
            if h.shape != HR_SHAPE:
                raise ValueError(f"unexpected HR shape {h.shape}")
            _atomic_save(out_lr, comp)
            _atomic_save(out_hr, np.asarray(h).astype(np.float16))
            reused = False
        return {
            "n": n,
            "ok": True,
            "reused": reused,
            "src_channels": channels,
            **frame_stats(comp),
        }
    except Exception as exc:
        return {"n": n, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


def verify_one(n: int, lr_name: str, hr_name: str, dst: str) -> dict:
    try:
        comp = np.load(Path(dst) / "lr" / f"{n:05d}.npy")
        hr = np.load(Path(dst) / "hr" / f"{n:05d}.npy")
        ref = compact(_SRC.read(lr_name))
        ref_hr = _SRC.read(hr_name).astype(np.float16)
        return {
            "n": n,
            "lr_equal": bool(np.array_equal(comp, ref)),
            "hr_equal": bool(np.array_equal(hr, ref_hr)),
        }
    except Exception as exc:
        return {
            "n": n,
            "lr_equal": False,
            "hr_equal": False,
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }


def runs_of(frames: list[int], cuts: list[int] = ()) -> list[dict]:
    (out, cut) = ([], set(cuts))
    for n in frames:
        if out and out[-1]["end"] == n - 1 and (n not in cut):
            (out[-1]["end"], out[-1]["length"]) = (n, out[-1]["length"] + 1)
        else:
            out.append({"start": n, "end": n, "length": 1})
    return out


(CUT_MOTION, CUT_DELTA) = (0.001, 0.01)
CUT_RULE = f"max |motion| < {CUT_MOTION} LR px over the whole frame AND mean |colour_t - colour_(t-1)| > {CUT_DELTA} (frame t-1 present)"


def find_cuts(dst: Path, ok: list[dict]) -> list[dict]:
    present = {r["n"] for r in ok}
    out = []
    for r in ok:
        n = r["n"]
        if r.get("motion_absmax", 1.0) >= CUT_MOTION or n - 1 not in present:
            continue
        cur = np.load(dst / "lr" / f"{n:05d}.npy")[:, :, :3].astype(np.float32)
        prv = np.load(dst / "lr" / f"{n - 1:05d}.npy")[:, :, :3].astype(np.float32)
        delta = float(np.abs(np.clip(cur, 0, 1) - np.clip(prv, 0, 1)).mean())
        out.append(
            {
                "n": n,
                "motion_absmax": r["motion_absmax"],
                "colour_delta": delta,
                "cut": delta > CUT_DELTA,
            }
        )
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_meta(dst: Path, results: list[dict], extra: dict) -> dict:
    ok = sorted((r for r in results if r["ok"]), key=lambda r: r["n"])
    frames = [r["n"] for r in ok]
    candidates = find_cuts(dst, ok)
    cuts = [c["n"] for c in candidates if c["cut"]]
    np.save(dst / "frames.npy", np.array(frames, dtype=np.int32))
    np.save(dst / "spread.npy", np.array([r["spread"] for r in ok], np.float32))
    np.save(dst / "medmotion.npy", np.array([r["medmotion"] for r in ok], np.float32))
    runs = runs_of(frames, cuts)
    spread = np.array([r["spread"] for r in ok], np.float64)
    med = np.array([r["medmotion"] for r in ok], np.float64)
    chans = {}
    for r in ok:
        if r.get("src_channels") is not None:
            chans[str(r["src_channels"])] = chans.get(str(r["src_channels"]), 0) + 1
    meta = {
        "scene": dst.name,
        "layout": LAYOUT,
        "frames": len(frames),
        "first": frames[0] if frames else None,
        "last": frames[-1] if frames else None,
        "runs": runs,
        "longest_run": max((r["length"] for r in runs), default=0),
        "cuts": cuts,
        "cut_rule": CUT_RULE,
        "zero_motion_frames": candidates,
        "skipped": [{"n": r["n"], "error": r["error"]} for r in results if not r["ok"]],
        "src_channels": chans,
        "triplet0_absmax": float(
            max((r["triplet0_absmax"] for r in ok), default=float("nan"))
        ),
        "nonfinite_values": int(sum((r["nonfinite"] for r in ok))),
        "spread_percentiles_1_5_25_50_75_99": np.percentile(
            spread, [1, 5, 25, 50, 75, 99]
        )
        .round(4)
        .tolist()
        if len(spread)
        else [],
        "static_frames_medmotion_lt_0.01": int((med < 0.01).sum()),
        "provenance": {
            "argv": sys.argv,
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "packer_md5": _md5(Path(__file__)),
        },
        **extra,
    }
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pack STSS source frames with verified color, depth and motion layout"
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--src", help="unpacked <Scene> directory with NPY/ and HR/")
    g.add_argument("--zip", help="the scene archive itself")
    ap.add_argument("--dst", required=True, help="packed/<Scene> directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N frames only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--verify",
        type=int,
        default=24,
        help="re-read N random frames from the source and require bit-exact equality (0 = skip)",
    )
    ap.add_argument(
        "--backfill",
        action="store_true",
        help="existing packed dir: write frames/spread/medmotion/meta from the packed arrays only",
    )
    ap.add_argument(
        "--src-channels",
        type=int,
        default=0,
        help="--backfill: channel count of the source files (recorded only)",
    )
    a = ap.parse_args()
    dst = Path(a.dst)
    t0 = time.time()
    if a.backfill:
        files = sorted((dst / "lr").glob("[0-9]*.npy"))
        frames = [int(p.stem) for p in files if (dst / "hr" / p.name).is_file()]
        results = []
        for k, n in enumerate(frames):
            try:
                comp = np.load(dst / "lr" / f"{n:05d}.npy")
                if comp.shape != (*LR_SHAPE, 7):
                    raise ValueError(f"packed shape {comp.shape}")
                if not _valid_npy(dst / "hr" / f"{n:05d}.npy", HR_SHAPE):
                    raise ValueError("bad packed HR")
                results.append(
                    {
                        "n": n,
                        "ok": True,
                        "src_channels": a.src_channels or None,
                        **frame_stats(comp),
                    }
                )
            except Exception as exc:
                results.append(
                    {"n": n, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
                )
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(frames)}", flush=True)
        old = (
            json.loads((dst / "meta.json").read_text())
            if (dst / "meta.json").is_file()
            else {}
        )
        keep = {k: old[k] for k in ("source", "verify", "seconds") if k in old}
        keep.setdefault("source", "backfill from packed arrays")
        keep.setdefault("verify", None)
        if old.get("src_channels") and (not a.src_channels):
            keep["src_channels_at_pack"] = old["src_channels"]
        keep["backfilled_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        meta = write_meta(dst, results, keep)
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in ("frames", "runs", "cuts", "skipped", "longest_run")
                },
                indent=1,
            )
        )
        return
    if not (a.src or a.zip):
        ap.error("need --src or --zip (or --backfill)")
    (kind, path) = ("zip", a.zip) if a.zip else ("dir", a.src)
    src = Source(path, kind)
    (lr, hr, other) = listing(src)
    both = sorted(set(lr) & set(hr))
    if a.limit:
        both = both[: a.limit]
    print(
        f"{kind} {path}: {len(lr)} NoWarp, {len(hr)} HR, {len(both)} with both; LR-only {sorted(set(lr) - set(hr))[:10]}; non-frame members {other[:5]}",
        flush=True,
    )
    (dst / "lr").mkdir(parents=True, exist_ok=True)
    (dst / "hr").mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(
        a.workers, initializer=_init, initargs=(path, kind)
    ) as pool:
        futs = [
            pool.submit(pack_one, n, lr[n], hr[n], str(dst), a.overwrite) for n in both
        ]
        for k, fut in enumerate(as_completed(futs)):
            r = fut.result()
            results.append(r)
            if not r["ok"]:
                print(f"  !! frame {r['n']}: {r['error']}", flush=True)
            if (k + 1) % 100 == 0:
                rate = (k + 1) / (time.time() - t0)
                print(f"  {k + 1}/{len(both)}  {rate:.2f} frames/s", flush=True)
    verify = None
    good = sorted((r["n"] for r in results if r["ok"]))
    if a.verify and good:
        pick = sorted(random.Random(0).sample(good, min(a.verify, len(good))))
        with ProcessPoolExecutor(
            min(a.workers, len(pick)), initializer=_init, initargs=(path, kind)
        ) as pool:
            checks = list(
                pool.map(
                    verify_one,
                    pick,
                    [lr[n] for n in pick],
                    [hr[n] for n in pick],
                    [str(dst)] * len(pick),
                )
            )
        verify = {
            "frames": pick,
            "all_equal": all((c["lr_equal"] and c["hr_equal"] for c in checks)),
            "failures": [c for c in checks if not (c["lr_equal"] and c["hr_equal"])],
        }
        print(
            f"  verify {len(pick)} frames bit-exact vs source: {verify['all_equal']}",
            flush=True,
        )
    sizes = src.sizes()
    meta = write_meta(
        dst,
        results,
        {
            "source": {
                "kind": kind,
                "path": str(path),
                "archive_bytes": os.path.getsize(path) if kind == "zip" else None,
                "nowarp_members": len(lr),
                "hr_members": len(hr),
                "lr_without_hr": sorted(set(lr) - set(hr)),
                "hr_without_lr": sorted(set(hr) - set(lr)),
                "non_frame_members": other,
                "odd_sized_members": sorted(
                    (
                        (n, s)
                        for (n, s) in sizes.items()
                        if (LR_RE.search(n) or HR_RE.search(n))
                        and s not in (34214528, 36288128, 12441728)
                    )
                ),
            },
            "verify": verify,
            "seconds": round(time.time() - t0, 1),
        },
    )
    print(f"packed {meta['frames']} frames -> {dst} in {time.time() - t0:.0f}s")
    print(
        f"  {len(meta['runs'])} contiguous run(s): {meta['runs']}; cuts (zero motion field) {meta['cuts']}"
    )
    print(f"  skipped: {meta['skipped']}")
    print(
        f"  src channels {meta['src_channels']}, triplet ch4 |max| {meta['triplet0_absmax']}, non-finite values {meta['nonfinite_values']}"
    )
    if verify is not None and (not verify["all_equal"]):
        sys.exit("VERIFY FAILED: packed arrays differ from the source")


if __name__ == "__main__":
    main()
