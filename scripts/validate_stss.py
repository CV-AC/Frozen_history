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
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsr.rrta import registered_upsample, warp

STSS = STSS_PACKED


def load(root: Path, n: int):
    lo = np.load(root / "lr" / f"{n:05d}.npy").astype(np.float32)
    hi = np.load(root / "hr" / f"{n:05d}.npy").astype(np.float32)
    return (lo, hi)


def t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None]


def masked_l1(a, b, ok) -> float:
    return float(((a - b).abs() * ok).sum() / ok.expand_as(a).sum().clamp_min(1))


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def pick_pairs(
    frames: list[int], med: np.ndarray, k: int, moving: bool = True
) -> list[tuple[int, int]]:
    idx = {n: i for (i, n) in enumerate(frames)}
    cands = [
        (n - 1, n) for n in frames if n - 1 in idx and (not moving or med[idx[n]] > 0.5)
    ]
    if not cands:
        return []
    take = np.linspace(0, len(cands) - 1, min(k, len(cands))).round().astype(int)
    return [cands[i] for i in sorted(set(take.tolist()))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(STSS / "Arena_test"))
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument(
        "--frames",
        type=int,
        default=6,
        help="frames for the colour / registration checks",
    )
    ap.add_argument(
        "--no-scan", action="store_true", help="skip the whole-scene contiguity scan"
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = Path(a.root)
    torch.set_grad_enabled(False)
    frames = [int(x) for x in np.load(root / "frames.npy")]
    med = (
        np.load(root / "medmotion.npy")
        if (root / "medmotion.npy").is_file()
        else np.full(len(frames), 1.0)
    )
    report: dict = {
        "root": str(root),
        "frames": len(frames),
        "first": frames[0],
        "last": frames[-1],
    }
    gaps = [(p, q) for (p, q) in zip(frames, frames[1:]) if q != p + 1]
    report["numbering_gaps"] = gaps
    print(
        f"{root.name}: {len(frames)} frames {frames[0]}..{frames[-1]}, numbering gaps {gaps}",
        flush=True,
    )
    pairs = pick_pairs(frames, med, a.pairs) or pick_pairs(
        frames, med, a.pairs, moving=False
    )
    rows: dict[str, list[float]] = {}
    cache = {}
    for p, n in pairs:
        (lo, hi) = load(root, n)
        (_, hp) = load(root, p)
        (cur, prv) = (t(hi), t(hp))
        mv = lo[:, :, 4:7]
        cache[p, n] = (mv, cur, prv)
        rows.setdefault("identity", []).append(
            masked_l1(prv, cur, torch.ones_like(cur[:, :1]))
        )
        for (i, j), sy, sx in itertools.product(
            [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)], (1, -1), (1, -1)
        ):
            m = t(np.stack((mv[:, :, i] * sy, mv[:, :, j] * sx), 2))
            m = F.interpolate(m, size=cur.shape[-2:], mode="nearest") * 2.0
            (w, ok) = warp(prv, m)
            key = f"ch{4 + i}{('+' if sy > 0 else '-')},ch{4 + j}{('+' if sx > 0 else '-')}"
            rows.setdefault(key, []).append(masked_l1(w, cur, ok))
    ranked = sorted(
        ((float(np.mean(v)), k) for (k, v) in rows.items() if k != "identity")
    )
    report["convention"] = {
        "pairs": pairs,
        "ranked": ranked[:6],
        "identity": float(np.mean(rows["identity"])),
        "best": ranked[0][1],
        "best_over_identity": float(np.mean(rows["identity"]) / ranked[0][0]),
        "best_over_runner_up": float(ranked[1][0] / ranked[0][0]),
    }
    print("\n[1] convention (mean |HR residual| over moving pairs)")
    for v, k in ranked[:6]:
        print(f"    {k:<14}{v:.5f}")
    print(f"    identity      {np.mean(rows['identity']):.5f}", flush=True)
    ks = [0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0]
    unit = {}
    for k in ks:
        vals = []
        for (p, n), (mv, cur, prv) in cache.items():
            m = t(np.stack((mv[:, :, 1], -mv[:, :, 2]), 2)) * k
            m = F.interpolate(m, size=cur.shape[-2:], mode="nearest") * 2.0
            (w, ok) = warp(prv, m)
            vals.append(masked_l1(w, cur, ok))
        unit[k] = float(np.mean(vals))
    kbest = min(unit, key=unit.get)
    report["units"] = {
        "residual_by_k": unit,
        "k_best": kbest,
        "reading": "k=1 -> LR (540p) pixels; 0.5 -> HR pixels",
    }
    print("\n[2] units: residual of k * (+ch5, -ch6) in LR px")
    print("    " + "  ".join((f"{k}:{v:.5f}" for (k, v) in unit.items())))
    print(f"    best k = {kbest}", flush=True)
    sel = np.linspace(0, len(frames) - 1, a.frames).round().astype(int)
    (same, swapped, sky, ground) = ([], [], [], [])
    offs = [x / 8.0 for x in range(-4, 5)]
    reg = {}
    for i in sel:
        n = frames[int(i)]
        (lo, hi) = load(root, n)
        (lr, hr) = (t(np.clip(lo[:, :, :3], 0, 1)), t(np.clip(hi, 0, 1)))
        up = F.interpolate(
            lr, scale_factor=2, mode="bicubic", align_corners=False
        ).clamp(0, 1)
        same.append(psnr01(up, hr))
        swapped.append(psnr01(up.flip(1), hr))
        d = lo[:, :, 3]
        far = d > 10.0
        if far.sum() > 1000:
            sky.append(lo[:, :, :3][far].mean(0).tolist())
        ground.append(lo[:, :, :3][d < 1.0].mean(0).tolist())
        for oy, ox in itertools.product(offs, offs):
            jit = torch.tensor([[-oy, ox]], dtype=torch.float32)
            reg.setdefault((oy, ox), []).append(
                psnr01(registered_upsample(lr, jit, 2), hr)
            )
        reg.setdefault("plain_bicubic", []).append(same[-1])
    sky_m = np.mean(sky, 0).tolist() if sky else None
    report["colour_order"] = {
        "frames": [frames[int(i)] for i in sel],
        "psnr_bicubic_lr_vs_hr_same_order": float(np.mean(same)),
        "psnr_bicubic_lr_vs_hr_lr_reversed": float(np.mean(swapped)),
        "lr_hr_same_order": bool(np.mean(same) > np.mean(swapped) + 3.0),
        "far_plane_mean_stored_ch012": sky_m,
        "near_geometry_mean_stored_ch012": np.mean(ground, 0).tolist(),
        "far_plane_frames": len(sky),
        "stored_order_is_BGR_by_sky": bool(sky_m[0] > sky_m[2]) if sky_m else None,
    }
    grid = {
        f"{k[0]:+.3f},{k[1]:+.3f}": float(np.mean(v))
        for (k, v) in reg.items()
        if isinstance(k, tuple)
    }
    obest = max(grid, key=grid.get)
    report["registration"] = {
        "psnr_by_offset_lr_px": grid,
        "best_offset": obest,
        "psnr_best": grid[obest],
        "psnr_zero_offset": grid["+0.000,+0.000"],
        "psnr_plain_bicubic": float(np.mean(reg["plain_bicubic"])),
    }
    print(
        f"\n[3] colour: bicubic(LR) vs HR  same order {np.mean(same):.3f} dB, LR reversed {np.mean(swapped):.3f} dB; far-plane stored ch0/1/2 {sky_m} -> BGR={report['colour_order']['stored_order_is_BGR_by_sky']}"
    )
    print(
        f"[4] registration: best offset {obest} {grid[obest]:.4f} dB, zero {grid['+0.000,+0.000']:.4f}, plain bicubic {np.mean(reg['plain_bicubic']):.4f}",
        flush=True,
    )
    if not a.no_scan:
        t0 = time.time()
        scan = []
        prev = None
        edges = np.concatenate(([0.0], np.logspace(-2, 3, 26)))
        dhist = np.zeros(len(edges) - 1, np.int64)
        (geo_max, sky_min, gap_px, gap_frames) = (0.0, float("inf"), 0, [])
        for n in frames:
            lo = np.load(root / "lr" / f"{n:05d}.npy").astype(np.float32)
            col = t(np.clip(lo[:, :, :3], 0, 1))
            d = lo[:, :, 3]
            dhist += np.histogram(d, bins=edges)[0]
            geo = d[d < 100.0]
            if geo.size:
                geo_max = max(geo_max, float(geo.max()))
            if (d >= 100.0).any():
                sky_min = min(sky_min, float(d[d >= 100.0].min()))
            g = int(((d >= 16.0) & (d < 100.0)).sum())
            gap_px += g
            if g:
                gap_frames.append((n, g))
            if prev is not None and prev[0] == n - 1:
                m = t(np.stack((lo[:, :, 5], -lo[:, :, 6]), 2))
                (w, ok) = warp(prev[1], m)
                ident = float((prev[1] - col).abs().mean())
                res = masked_l1(w, col, ok)
                scan.append((n, res, ident, float(ok.mean())))
            prev = (n, col)
        arr = np.array([(r, i) for (_, r, i, _) in scan])
        suspect = [s for s in scan if s[2] > 0.03 and s[1] > 0.8 * s[2]]
        worst = sorted(scan, key=lambda s: -s[1])[:10]
        report["contiguity"] = {
            "pairs": len(scan),
            "warp_residual_mean": float(arr[:, 0].mean()),
            "identity_residual_mean": float(arr[:, 1].mean()),
            "suspected_cuts": suspect,
            "worst_warp_residual": worst,
            "rule": "cut if identity > 0.03 and warp > 0.8*identity",
            "seconds": round(time.time() - t0, 1),
        }
        report["depth"] = {
            "raw_min": float(edges[np.nonzero(dhist)[0][0]]),
            "geometry_max_below_100": geo_max,
            "far_plane_min_at_or_above_100": sky_min,
            "pixels_in_16_to_100": gap_px,
            "frames_with_pixels_in_16_to_100": gap_frames[:50],
            "log_histogram_edges": edges.tolist(),
            "log_histogram_counts": dhist.tolist(),
        }
        print(
            f"[6] depth: geometry max {geo_max:.3f}, far-plane min {sky_min:.3f}, pixels in [16,100) {gap_px} ({len(gap_frames)} frames)",
            flush=True,
        )
        print(
            f"\n[5] contiguity: {len(scan)} pairs, warp {arr[:, 0].mean():.5f} vs identity {arr[:, 1].mean():.5f}; suspected cuts {suspect[:10]}; worst {[(w[0], round(w[1], 4), round(w[2], 4)) for w in worst[:5]]}",
            flush=True,
        )
    out = Path(
        a.out
        or Path(__file__).resolve().parents[1]
        / "runs"
        / "reference"
        / f"stss_validate_{root.name}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
