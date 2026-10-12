from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diffsr import samples as smp
from diffsr.data_arm import ArmSequence
from diffsr.rrta import warp


def sample_at(img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    (H, W) = img.shape[-2:]
    gy = 2.0 * pos[:, 0] / (H - 1) - 1.0
    gx = 2.0 * pos[:, 1] / (W - 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def masked_l1(a, b, mask=None) -> float:
    d = (a - b).abs()
    if mask is None:
        return float(d.mean())
    m = mask.expand_as(d)
    return float((d * m).sum() / m.sum().clamp_min(1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--exposure", type=float, default=2.0)
    ap.add_argument("--out", default="results/arm_validation.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ArmSequence(exposure=args.exposure, max_frames=args.frames)
    s = 2
    t0 = time.time()
    dp = data.depth_params.double()
    zn = data.z_near.double()
    v4a = {
        "depth_params_unique": torch.unique(dp, dim=0).tolist()[:4],
        "p0_is_1": bool(torch.allclose(dp[:, 0], torch.ones_like(dp[:, 0]))),
        "p1_is_minus_zNear": bool(torch.allclose(dp[:, 1], -zn)),
        "zNear": zn[:1].tolist(),
    }
    jit_alts = {
        "loader (-dy,+dx)": lambda j: torch.stack((-j[0], j[1])),
        "y mirrored (+dy,+dx)": lambda j: torch.stack((j[0], j[1])),
        "x mirrored (-dy,-dx)": lambda j: torch.stack((-j[0], -j[1])),
        "both mirrored (+dy,-dx)": lambda j: torch.stack((j[0], -j[1])),
        "axes swapped (-dx,+dy)": lambda j: torch.stack((-j[1], j[0])),
        "zero": lambda j: torch.zeros(2),
    }
    v1 = {k: [] for k in jit_alts}
    mot_alts = [
        "identity",
        "loader HR field",
        "loader LR field (nearest x2)",
        "HR field negated",
        "HR field y negated",
        "HR field x negated",
        "HR field axes swapped",
    ]
    v2 = {k: [] for k in mot_alts}
    v3 = {
        k: []
        for k in (
            "identity",
            "LR field, no lattice correction",
            "LR field + d(o) (two-lattice)",
            "LR field - d(o)",
        )
    }
    v4b = {"correct motion": [], "negated motion": []}
    depth_rng = []
    prev = None
    for i in range(0, len(data), args.step):
        it = data[i]
        lr = it["lr"][None].to(dev)
        hr = it["hr"][None].to(dev)
        jraw = it["jitter_raw"]
        (h, w) = lr.shape[-2:]
        for name, fn in jit_alts.items():
            jy = fn(jraw)[None].to(dev)
            pos = smp.registered_sample_positions(s, jy, h, w)
            v1[name].append(masked_l1(sample_at(hr, pos), lr))
        dep = it["depth"][None].to(dev)
        depth_rng.append(
            [
                float(dep.min()),
                float(dep.median()),
                float(dep.max()),
                float((dep >= 1.0).float().mean()),
            ]
        )
        m_lr = it["motion"][None].to(dev)
        m_hr = it["motion_hr"][None].to(dev)
        if prev is not None and i - prev["i"] == 1:
            fields = {
                "identity": torch.zeros_like(m_hr),
                "loader HR field": m_hr,
                "loader LR field (nearest x2)": smp.motion_to_hr(m_lr, s),
                "HR field negated": -m_hr,
                "HR field y negated": m_hr
                * torch.tensor([-1.0, 1.0], device=dev).view(1, 2, 1, 1),
                "HR field x negated": m_hr
                * torch.tensor([1.0, -1.0], device=dev).view(1, 2, 1, 1),
                "HR field axes swapped": m_hr.flip(1),
            }
            for name, f in fields.items():
                (wp, inside) = warp(prev["hr"], f)
                v2[name].append(masked_l1(wp, hr, inside))
            do = (jraw - prev["jraw"]).to(dev).view(1, 2, 1, 1)
            for name, f in (
                ("identity", torch.zeros_like(m_lr)),
                ("LR field, no lattice correction", m_lr),
                ("LR field + d(o) (two-lattice)", m_lr + do),
                ("LR field - d(o)", m_lr - do),
            ):
                (wp, inside) = warp(prev["lr"], f)
                v3[name].append(masked_l1(wp, lr, inside))
            for name, f in (("correct motion", m_lr), ("negated motion", -m_lr)):
                (valid, _) = smp.depth_test(dep, prev["dep"], f, rel=0.1, mode="far")
                (_, inside) = warp(prev["dep"], f)
                v4b[name].append(float(1.0 - valid.sum() / inside.sum().clamp_min(1)))
        prev = {"i": i, "hr": hr, "lr": lr, "jraw": jraw, "dep": dep}
    summ = lambda d: {
        k: {"mean": float(np.mean(v)), "n": len(v)} for (k, v) in d.items() if v
    }

    def best(d):
        m = {k: np.mean(v) for (k, v) in d.items() if v}
        return min(m, key=m.get)

    out = {
        "what": "model-free convention checks, Arm NGD (diffsr/data_arm.py)",
        "display_space": f"Reinhard(exp({args.exposure})*x)",
        "V1_jitter_lr_vs_gt_at_registered_positions": summ(v1),
        "V1_best": best(v1),
        "V2_motion_gt_warp_residual": summ(v2),
        "V2_best": best(v2),
        "V3_two_lattice_lr_warp_residual": summ(v3),
        "V3_best": best(v3),
        "V4a_depth_linearisation": v4a,
        "V4b_far0.1_rejected_fraction": summ(v4b),
        "V4c_depth01_min_median_max_skyfrac_mean": np.mean(depth_rng, axis=0).tolist(),
        "V1_per_frame_loader": v1["loader (-dy,+dx)"],
        "V1_per_frame_zero": v1["zero"],
        "provenance": {
            "argv": sys.argv,
            "dataset": data.provenance(),
            "seconds": round(time.time() - t0, 1),
        },
    }
    ok = (
        out["V1_best"].startswith("loader")
        and out["V2_best"] in ("loader HR field", "loader LR field (nearest x2)")
        and out["V3_best"].startswith("LR field + d(o)")
        and v4a["p0_is_1"]
        and v4a["p1_is_minus_zNear"]
    )
    out["PASS"] = bool(ok)
    p = Path(args.out)
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for key in (
        "V1_jitter_lr_vs_gt_at_registered_positions",
        "V2_motion_gt_warp_residual",
        "V3_two_lattice_lr_warp_residual",
        "V4b_far0.1_rejected_fraction",
    ):
        print(key)
        for k, v in out[key].items():
            print(f"   {k:<36} {v['mean']:.5f}  (n={v['n']})")
    print("V4a", v4a)
    print(
        "V4c depth01 [min, median, max, sky fraction]",
        out["V4c_depth01_min_median_max_skyfrac_mean"],
    )
    print("PASS" if ok else "FAIL", f"-> {p}")


if __name__ == "__main__":
    main()
