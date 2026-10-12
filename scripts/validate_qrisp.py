from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffsr.paths import QRISP_RAW
from diffsr.qrisp import (
    decode_motion,
    read_camera,
    read_color,
    read_depth,
    warp_by_motion,
)

cv2.setNumThreads(4)


def check_motion(
    root: Path, scene: str, res: str, tag: str, sequences: int, frames: int
) -> dict:
    import itertools

    colour_dir = root / scene / res / tag
    motion_dir = root / scene / res / f"MotionVectors{tag}"
    seqs = sorted((p for p in colour_dir.iterdir() if p.is_dir()))[:sequences]

    def colour(seq, n):
        return cv2.imread(str(colour_dir / seq / f"{n}.png"), cv2.IMREAD_COLOR)

    def raw_motion(seq, n):
        return cv2.imread(str(motion_dir / seq / f"{n}.exr"), cv2.IMREAD_UNCHANGED)

    pairs = []
    for seq in seqs:
        names = sorted((p.stem for p in (colour_dir / seq.name).iterdir()))
        pairs += [(seq.name, a, b) for (a, b) in zip(names, names[1:])][:frames]
    baseline = float(
        np.mean(
            [
                np.abs(
                    colour(s, b).astype(np.float32) - colour(s, a).astype(np.float32)
                ).mean()
                for (s, a, b) in pairs
            ]
        )
    )
    print(f"  no warp at all (frame difference)  {baseline:8.3f}")
    (height, width) = colour(pairs[0][0], pairs[0][1]).shape[:2]
    scales = {
        "H,W": (height, width),
        "H/2,W/2": (height / 2, width / 2),
        "W,W": (width, width),
        "H,H": (height, height),
    }
    results = []
    for vi, hi in ((1, 2), (2, 1)):
        for scale_name, (sy, sx) in scales.items():
            for gy, gx in itertools.product((1, -1), repeat=2):
                for source in ("t", "t-1"):
                    errors = []
                    for seq, a, b in pairs:
                        exr = raw_motion(seq, b if source == "t" else a)
                        if exr is None:
                            continue
                        dy = exr[..., vi].astype(np.float32) * sy * gy
                        dx = exr[..., hi].astype(np.float32) * sx * gx
                        margin = (
                            int(np.ceil(max(np.abs(dy).max(), np.abs(dx).max()))) + 2
                        )
                        if margin * 2 >= min(height, width):
                            continue
                        warped = warp_by_motion(
                            colour(seq, a), np.stack([dy, dx], -1)
                        ).astype(np.float32)
                        cut = (slice(margin, -margin), slice(margin, -margin))
                        errors.append(
                            float(
                                np.abs(
                                    warped[cut] - colour(seq, b).astype(np.float32)[cut]
                                ).mean()
                            )
                        )
                    if errors:
                        results.append(
                            {
                                "error": float(np.mean(errors)),
                                "vertical_index": vi,
                                "horizontal_index": hi,
                                "scale": scale_name,
                                "sign": (gy, gx),
                                "source": source,
                            }
                        )
    results.sort(key=lambda r: r["error"])
    for r in results[:6]:
        print(
            f"  {r['error']:8.3f}  vert=ch{r['vertical_index']} horiz=ch{r['horizontal_index']} scale={r['scale']:<8} sign=({r['sign'][0]:+d},{r['sign'][1]:+d}) mv@{r['source']:<4} ({baseline / r['error']:.2f}x vs no warp)"
        )
    best = results[0]
    print(
        f"\n  BEST  vertical=index {best['vertical_index']}, horizontal=index {best['horizontal_index']}, scale {best['scale']}, sign {best['sign']}, motion read at frame {best['source']}"
    )
    print(
        f"        {baseline / best['error']:.2f}x better than not warping; runner-up {results[1]['error']:.3f}"
    )
    if baseline / best["error"] < 1.3:
        print(
            "        WARNING: no convention beats not-warping by much. Do not use the motion vectors until this is understood."
        )
    return best


def check_depth(root: Path, scene: str, res: str, tag: str) -> None:
    depth_dir = root / scene / res / f"Depth{tag}"
    camera_dir = root / scene / res / "CameraData"
    seq = sorted((p for p in depth_dir.iterdir() if p.is_dir()))[0]
    for name in sorted((p.stem for p in seq.iterdir()))[:3]:
        depth = read_depth(depth_dir / seq.name / f"{name}.png")
        camera = read_camera(camera_dir / seq.name / f"{name}.json")
        (near, far) = (camera["near"], camera["far"])
        metres = depth * (far - near)
        print(
            f"  {name}: decoded [{depth.min():.6f}, {depth.max():.6f}]  median {np.median(depth):.4f}  near={near:.3f} far={far:.1f}  -> distance [{metres.min():.3f}, {metres.max():.1f}] units"
        )
    out_of_range = float(((depth < 0) | (depth > 1.0000001)).mean())
    print(
        f"  fraction outside [0, 1]: {out_of_range:.6%}  (a non-zero but tiny value is 8-bit quantisation at the far plane)"
    )
    at_far = float((depth > 0.999).mean())
    print(
        f"  fraction at the far plane (>0.999): {at_far:.2%}  = sky / background, which is what it should be"
    )


def check_alpha(root: Path, scene: str) -> None:
    for res, modality in (("1080p", "Enhanced"), ("270p", "Native")):
        base = root / scene / res / modality
        seq = sorted((p for p in base.iterdir() if p.is_dir()))[0]
        name = sorted((p.stem for p in seq.iterdir()))[1]
        rgba = read_color(base / seq.name / f"{name}.png", keep_alpha=True)
        if rgba.shape[2] != 4:
            print(f"  {res}/{modality}: 3 channels, no alpha")
            continue
        bgr = rgba[..., :3].astype(np.float64)
        alpha = rgba[..., 3].astype(np.float64)
        luma = 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]
        corr = float(np.corrcoef(alpha.ravel(), luma.ravel())[0, 1])
        print(
            f"  {res}/{modality}: alpha range [{alpha.min():.0f}, {alpha.max():.0f}], {len(np.unique(alpha))} distinct, std {alpha.std():.1f}, corr with luma {corr:+.3f}"
        )
        for i, ch in enumerate("BGR"):
            c = float(np.corrcoef(alpha.ravel(), bgr[..., i].ravel())[0, 1])
            print(f"      corr with {ch}: {c:+.3f}")


def check_pairing(root: Path, scene: str) -> None:
    for res, scale in (("540p", 2), ("360p", 3), ("270p", 4)):
        lr_dir = root / scene / res / "Native"
        gt_dir = root / scene / "1080p" / "Enhanced"
        seq = sorted((p for p in lr_dir.iterdir() if p.is_dir()))[0]
        name = sorted((p.stem for p in seq.iterdir()))[2]
        lr = read_color(lr_dir / seq.name / f"{name}.png")
        gt = read_color(gt_dir / seq.name / f"{name}.png")
        ok = gt.shape[0] == lr.shape[0] * scale and gt.shape[1] == lr.shape[1] * scale
        down = cv2.resize(
            gt, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        gap = float(np.abs(down - lr.astype(np.float32)).mean())
        print(
            f"  {res}: lr{lr.shape[:2]} gt{gt.shape[:2]} x{scale} {('OK' if ok else 'MISMATCH')}  |area-downsampled GT - LR| = {gap:.2f}/255"
        )
    native = read_color(root / scene / "1080p" / "Native" / seq.name / f"{name}.png")
    enh = read_color(gt_dir / seq.name / f"{name}.png")
    print(
        f"  1080p Native vs Enhanced: mean |delta| = {np.abs(native.astype(np.float32) - enh.astype(np.float32)).mean():.2f}/255  (they are different images -- use Enhanced)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(QRISP_RAW))
    ap.add_argument("--scene", default="CBApocalypse")
    ap.add_argument("--res", default="270p")
    ap.add_argument("--tag", default="MipBiasMinus2")
    ap.add_argument("--sequences", type=int, default=3)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.root)
    print(f"=== {args.scene} / {args.res} ===\n")
    print("--- LR/GT pairing ---")
    check_pairing(root, args.scene)
    print("\n--- colour alpha channel (undocumented) ---")
    check_alpha(root, args.scene)
    print("\n--- depth (supplement eq. 4) ---")
    check_depth(root, args.scene, args.res, args.tag)
    print("\n--- motion vector convention, by reprojection residual ---")
    best = check_motion(
        root, args.scene, args.res, args.tag, args.sequences, args.frames
    )
    print(
        f"\n  diffsr/qrisp.py should carry: VERTICAL_INDEX={best['vertical_index']}, HORIZONTAL_INDEX={best['horizontal_index']}, MOTION_SIGN={best['sign']}"
    )


if __name__ == "__main__":
    main()
