from __future__ import annotations
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diffsr import taau
from diffsr.data_qrisp import QrispDataset
from diffsr.metrics import psnr as np_psnr
from protocol_full import provenance, runs

ALPHAS = (1 / 16, 1 / 10, 1 / 8)
GAMMAS = (1.0, 1.25, 1.5)
LUMA = (False, True)
BLENDS = ("decay", "floor")
DILATE = (True, False)
VELOCITY = (False, True)
_YW = (65.481, 128.553, 24.966)


def rhos_for(scale: int) -> tuple[float, ...]:
    return tuple(sorted({1.0, max(1.0, scale / 2.0)}))


def grid(scale: int, preset: str = "full") -> list[taau.TAAUConfig]:
    if preset == "smoke":
        return [
            taau.TAAUConfig(blend=b, rho=r, alpha_min=0.1, gamma=1.25, dilate=True)
            for b in BLENDS
            for r in rhos_for(scale)
        ]
    return [
        taau.TAAUConfig(
            alpha_min=a, gamma=g, luma_weight=lw, blend=b, rho=r, dilate=d, velocity=v
        )
        for (b, r, d, v, a, g, lw) in itertools.product(
            BLENDS, rhos_for(scale), DILATE, VELOCITY, ALPHAS, GAMMAS, LUMA
        )
    ]


def psnr_y(pred: torch.Tensor, ref: torch.Tensor) -> float:
    w = torch.tensor(_YW, dtype=torch.float64, device=pred.device).view(1, 3, 1, 1)
    yp = (pred.double() * w).sum(1) + 16.0
    yr = (ref.double() * w).sum(1) + 16.0
    mse = float(((yp - yr) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def pick_runs(data, per_scene: int) -> list[tuple[str, str, list[int]]]:
    out = []
    for scene, orders in sorted(runs(data).items()):
        n = len(orders)
        idx = (
            sorted(
                {round(k * (n - 1) / max(per_scene - 1, 1)) for k in range(per_scene)}
            )
            if per_scene > 1
            else [0]
        )
        for k in idx:
            order = orders[k]
            out.append((scene, data.frames[order[0]]["sequence"], order))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--runs-per-scene", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--grid", default="full", choices=("full", "smoke"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--chunk", type=int, default=36, help="configs stepped per batched update"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = args.scale
    data = QrispDataset(
        "train",
        scale=s,
        colour="MipBiasJit",
        with_depth=True,
        with_motion=True,
        motion_jit=False,
        depth_jit=False,
        patch_lr=0,
        flip=False,
        tiles=None,
    )
    chosen_runs = pick_runs(data, args.runs_per_scene)
    cfgs = grid(s, args.grid)
    names = [c.name() for c in cfgs]
    assert len(set(names)) == len(names)
    rhos = rhos_for(s)
    print(
        f"TAAU selection x{s} on TRAIN: {len(chosen_runs)} runs ({args.runs_per_scene}/scene), {len(cfgs)} configs, warm-up {args.warmup}, device {dev}",
        flush=True,
    )
    for sc, sq, o in chosen_runs:
        print(f"  {sc}/{sq}: {len(o)} frames", flush=True)
    order_all = []
    for sc, sq, o in chosen_runs:
        o = o[: args.max_frames] if args.max_frames else o
        order_all += [(sc, sq, n, i) for (n, i) in enumerate(o)]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, [t[3] for t in order_all]),
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        prefetch_factor=4 if args.workers else None,
        persistent_workers=False,
    )
    groups = {}
    for c in cfgs:
        groups.setdefault(c.group(), []).append(c)
    batches = []
    for g, members in groups.items():
        for k0 in range(0, len(members), args.chunk):
            batches.append(members[k0 : k0 + args.chunk])
    print(f"  {len(groups)} (rho, dilate) groups, {len(batches)} batches", flush=True)
    scores = {nm: {} for nm in names}
    states = [None] * len(batches)
    prev_depth = None
    check_np = []
    t0 = time.time()
    nonfinite = {nm: 0 for nm in names}
    alpha_log = {nm: [] for nm in names}
    for k, ((sc, sq, n, i), item) in enumerate(zip(order_all, loader)):
        run_key = f"{sc}/{sq}"
        if n == 0:
            states = [None] * len(batches)
            prev_depth = None
        lr = item["lr"][None, :3].to(dev)
        dep = item["depth"][None].to(dev)
        mot = item["motion"][None].to(dev)
        jit = item["jitter"][None].to(dev)
        hr = item["hr"][None].to(dev)
        with torch.inference_mode():
            ctx = taau.frame_context(
                lr, dep, mot, jit, prev_depth, s, rhos=rhos, dilate_opts=DILATE
            )
            for b, batch in enumerate(batches):
                (out, states[b], aux) = taau.taau_update(batch, ctx, states[b])
                for j, c in enumerate(batch):
                    nm = c.name()
                    o = out[j : j + 1]
                    if not bool(torch.isfinite(o).all()):
                        nonfinite[nm] += 1
                    if n >= args.warmup:
                        p = psnr_y(o, hr)
                        scores[nm].setdefault(run_key, []).append((n, p))
                        alpha_log[nm].append(aux["alpha_mean"][j])
                        if len(check_np) < 3 and b == 0 and (j == 0):
                            pred = (
                                o[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
                                * 255.0
                            )
                            ref = item["hr"].permute(1, 2, 0).float().numpy() * 255.0
                            check_np.append(
                                (p, np_psnr(pred, ref, s, crop_border=0, to_y=True))
                            )
                del out
        prev_depth = dep
        if (k + 1) % 50 == 0:
            print(
                f"  [{k + 1}/{len(order_all)} frames, {time.time() - t0:.0f}s]",
                flush=True,
            )

    def aggregate(nm, warm):
        per_run = {
            rk: float(np.mean([p for (n, p) in v if n >= warm]))
            for (rk, v) in scores[nm].items()
        }
        per_scene = {}
        for rk, v in per_run.items():
            per_scene.setdefault(rk.split("/")[0], []).append(v)
        per_scene = {sc: float(np.mean(v)) for (sc, v) in per_scene.items()}
        return (float(np.mean(list(per_scene.values()))), per_scene, per_run)

    table = {}
    for c, nm in zip(cfgs, names):
        (m8, ps8, pr8) = aggregate(nm, args.warmup)
        (m16, ps16, _) = aggregate(nm, max(16, args.warmup))
        table[nm] = {
            "config": c.to_dict(),
            "psnr_y": m8,
            "per_scene": ps8,
            "per_run": pr8,
            "psnr_y_warmup16": m16,
            "per_scene_warmup16": ps16,
            "alpha_mean": float(np.mean(alpha_log[nm])) if alpha_log[nm] else None,
            "nonfinite_frames": nonfinite[nm],
        }
    ranked = sorted(names, key=lambda nm: -table[nm]["psnr_y"])
    best = ranked[0]
    struct = {}
    for nm in names:
        c = table[nm]["config"]
        key = f"{c['blend']}_r{c['rho']:g}_{('dil' if c['dilate'] else 'nodil')}_{('vel' if c['velocity'] else 'novel')}"
        if key not in struct or table[nm]["psnr_y"] > table[struct[key]]["psnr_y"]:
            struct[key] = nm
    print(f"\n{'config':<50}{'PSNR-Y w8':>10}{'w16':>9}{'alpha':>8}")
    for nm in ranked[:30]:
        r = table[nm]
        print(
            f"{nm:<50}{r['psnr_y']:>10.3f}{r['psnr_y_warmup16']:>9.3f}{r['alpha_mean']:>8.3f}"
        )
    print("\nbest per structural combination:")
    for key, nm in sorted(struct.items(), key=lambda kv: -table[kv[1]]["psnr_y"]):
        print(f"  {key:<28} {nm:<50} {table[nm]['psnr_y']:.3f}")
    print(
        f"\nCHOSEN x{s}: {best}  PSNR-Y {table[best]['psnr_y']:.3f} (warm-up 16: {table[best]['psnr_y_warmup16']:.3f})"
    )
    print(f"GPU PSNR vs diffsr.metrics.psnr: {check_np}")
    out = {
        "what": "TAAU hyperparameter selection on QRISP TRAIN runs only",
        "scale": s,
        "warmup": args.warmup,
        "stride": 1,
        "metric": "PSNR-Y (MATLAB BT.601 luma, float output clamped, no crop), mean per run -> per scene -> over scenes",
        "train_runs": [f"{sc}/{sq}" for (sc, sq, _) in chosen_runs],
        "frames_per_run": [len(o) for (_, _, o) in chosen_runs],
        "grid": {
            "alpha_min": ALPHAS,
            "gamma": GAMMAS,
            "luma_weight": LUMA,
            "blend": BLENDS,
            "rho": rhos,
            "dilate": DILATE,
            "velocity": VELOCITY,
        },
        "n_configs": len(cfgs),
        "chosen": best,
        "chosen_config": table[best]["config"],
        "chosen_psnr_y": table[best]["psnr_y"],
        "ranked": ranked,
        "best_per_structure": struct,
        "gpu_psnr_check": check_np,
        "taau_md5": taau.code_md5(),
        "table": table,
        "seconds": round(time.time() - t0, 1),
        "provenance": provenance(
            scale=s,
            warmup=args.warmup,
            runs_per_scene=args.runs_per_scene,
            grid=args.grid,
            max_frames=args.max_frames,
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
