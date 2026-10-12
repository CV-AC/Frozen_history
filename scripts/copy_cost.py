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
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = REPOSITORY
SNAP = REPOSITORY
sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(SNAP / "scripts"))
from diffsr.data_qrisp import QRISP_LMDB, RES_FOR_SCALE, _Lmdb
from diffsr.rrta import warp
from diffsr.samples import depth_test
from diffsr.engine_seq import motion_to_hr
from protocol_full import camera_static

OUT = REPOSITORY / "results" / "copy_cost"
KS = (1, 2, 4, 8, 16, 30)
T0 = 30
CAP = 60.0
VALID_EPS = 0.999
Y_W = (65.481, 128.553, 24.966)
AS_STOP = ("AbandonedSchool", "0001", 275, 292)
HR_FIELDS = ("nonjit", "jit", "identity")
LR_FIELDS = ("rejit", "jit", "nonjit", "identity")
MASKS = ("hold", "valid", "inb")


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def luma(x: torch.Tensor) -> torch.Tensor:
    return 16.0 + Y_W[0] * x[:, 0] + Y_W[1] * x[:, 1] + Y_W[2] * x[:, 2]


def masked_stats(
    pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor
) -> tuple[float, float, float, float]:
    m = mask.to(torch.float64)
    count = float(m.sum())
    if count == 0:
        return (0.0, 0.0, 0.0, 0.0)
    dy = luma(pred).double() - luma(ref).double()
    sse_y = float((dy * dy * m).sum())
    d = pred.double() - ref.double()
    l1 = float((d.abs() * m[:, None]).sum())
    sse = float((d * d * m[:, None]).sum())
    return (sse_y, count, l1, sse)


def run_groups(frames: list[dict]) -> list[list[int]]:
    (out, last) = ([], None)
    for i, f in enumerate(frames):
        key = (f["scene"], f["sequence"])
        if key != last:
            out.append([])
            last = key
        out[-1].append(i)
    return out


class Store:
    def __init__(self, dev):
        self.dev = dev
        self.dbs: dict[str, _Lmdb] = {}

    def db(self, name: str) -> _Lmdb:
        if name not in self.dbs:
            self.dbs[name] = _Lmdb(QRISP_LMDB / f"QRISP_test_{name}.lmdb")
        return self.dbs[name]

    def image(self, name: str, key: str) -> torch.Tensor:
        bgr = self.db(name).image(key)
        rgb = np.ascontiguousarray(bgr[..., ::-1].transpose(2, 0, 1))
        return torch.from_numpy(rgb).to(self.dev)

    def array(self, name: str, key: str) -> torch.Tensor:
        arr = self.db(name).array(key)
        if arr.ndim == 2:
            arr = arr[..., None]
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).to(
            self.dev
        )


def jitter_delta_field(jit_t, jit_prev, like: torch.Tensor) -> torch.Tensor:
    dy = -(float(jit_t[0]) - float(jit_prev[0]))
    dx = float(jit_t[1]) - float(jit_prev[1])
    return torch.tensor([dy, dx], device=like.device, dtype=like.dtype).view(1, 2, 1, 1)


def chain_pass(
    images: torch.Tensor,
    fields: dict,
    geo: str,
    dtest: torch.Tensor,
    fac: int,
    n: int,
    T0_: int,
    common: dict | None,
    emit=None,
) -> None:
    (H, W) = images.shape[-2:]
    names = list(fields)
    first = common is None
    if first:
        common = {}
    for s in range(0, n - 1):
        src = images[s : s + 1].float() / 255.0
        ones = torch.ones(1, 1, H, W, device=images.device)
        xs = {}
        for f in names:
            parts = ([src] if not first else []) + [ones]
            if f == geo:
                parts.append(ones)
            xs[f] = torch.cat(parts, 1)
        c0 = 0 if first else 3
        for j in range(1, min(max(KS), n - 1 - s) + 1):
            t = s + j
            dt = (
                F.interpolate(dtest[t : t + 1], size=(H, W), mode="nearest")
                if fac > 1
                else dtest[t : t + 1]
            )
            for f in names:
                m = fields[f][t : t + 1]
                m = motion_to_hr(m, fac) if fac > 1 else m
                (xs[f], inside) = warp(xs[f], m)
                xs[f][:, c0 : c0 + 1] *= inside
                if f == geo:
                    xs[f][:, c0 + 1 : c0 + 2] *= inside * dt
            if j not in KS or t < T0_:
                continue
            inb = torch.ones(1, H, W, dtype=torch.bool, device=images.device)
            for f in names:
                inb &= xs[f][:, c0] > VALID_EPS
            valid = inb & (xs[geo][:, c0 + 1] > VALID_EPS)
            if first:
                common[t] = valid if t not in common else common[t] & valid
            else:
                preds = {f: xs[f][:, :3] for f in names}
                emit(
                    t,
                    j,
                    src,
                    preds,
                    {"hold": common[t] & valid, "valid": valid, "inb": inb},
                )
    return common


def process_run(
    store: Store,
    frames: list[dict],
    order: list[int],
    scales: tuple[int, ...],
    records: list,
    checks: dict,
    dev,
) -> None:
    n = len(order)
    keys = [frames[i]["key"] for i in order]
    meta0 = frames[order[0]]
    (scene, seq) = (meta0["scene"], meta0["sequence"])
    run_name = f"{scene}/{seq}"
    fnum = [int(frames[i]["frame"]) for i in order]
    jit = [frames[i]["jitter_yx"] for i in order]
    t_start = time.time()
    static_step = np.zeros(n, dtype=bool)
    for t in range(1, n):
        m270 = store.array("Motion270", keys[t])[None]
        static_step[t] = camera_static(m270, 4)
    gt = torch.stack([store.image("GT1080Enhanced", k) for k in keys])

    def chain_state(t: int, k: int) -> int:
        steps = static_step[max(t - k + 1, 1) : t + 1]
        return 0 if steps.all() else 1 if not steps.any() else 2

    for scale in scales:
        res = RES_FOR_SCALE[scale]
        fac = 1080 // int(res)
        lr = torch.stack([store.image(f"LR{res}MipBiasJit", k) for k in keys])
        dep = torch.stack([store.array(f"Depth{res}", k) for k in keys])
        mnj = torch.stack([store.array(f"Motion{res}", k) for k in keys])
        mj = torch.stack([store.array(f"Motion{res}Jit", k) for k in keys])
        dtest = torch.zeros(n, 1, *dep.shape[-2:], device=dev)
        mrj = torch.zeros_like(mnj)
        (chk_med, chk_frac) = ([], [])
        for t in range(1, n):
            (ok, _) = depth_test(
                dep[t : t + 1], dep[t - 1 : t], mnj[t : t + 1], rel=0.1, mode="far"
            )
            dtest[t] = ok[0]
            dj = jitter_delta_field(jit[t], jit[t - 1], mnj)
            mrj[t] = (mnj[t : t + 1] + dj)[0]
            diff = mj[t] - mrj[t]
            chk_med.append(diff.flatten(1).median(dim=1).values.abs().max().item())
            chk_frac.append(float((diff.abs().amax(0) < 0.01).float().mean()))
        c = checks.setdefault(
            f"x{scale}", {"median_abs_dev": [], "frac_within_0.01": []}
        )
        c["median_abs_dev"].extend(chk_med)
        c["frac_within_0.01"].extend(chk_frac)
        for lattice, images, fields, geo, f_ in (
            ("hr", gt, {"nonjit": mnj, "jit": mj}, "nonjit", fac),
            ("lr", lr, {"rejit": mrj, "jit": mj, "nonjit": mnj}, "nonjit", 1),
        ):
            npix = float(images.shape[-2] * images.shape[-1])
            common = chain_pass(images, fields, geo, dtest, f_, n, T0, None)

            def emit(
                t, k, src, preds, masks, lattice=lattice, images=images, npix=npix
            ):
                ref = images[t : t + 1].float() / 255.0
                st = chain_state(t, k)
                st30 = chain_state(t, 30)
                st16 = chain_state(t, 16)
                for fname, pred in list(preds.items()) + [("identity", src)]:
                    for mname, mask in masks.items():
                        (sse_y, cnt, l1, sse) = masked_stats(pred, ref, mask)
                        records.append(
                            (
                                lattice,
                                scale,
                                fname,
                                k,
                                mname,
                                run_name,
                                fnum[t],
                                st,
                                st30,
                                st16,
                                sse_y,
                                cnt,
                                l1,
                                sse,
                                npix,
                            )
                        )

            chain_pass(images, fields, geo, dtest, f_, n, T0, common, emit)
            del common
        del lr, dep, mnj, mj, dtest, mrj
        print(
            f"    {run_name} x{scale} done ({time.time() - t_start:.0f}s)", flush=True
        )
    checks.setdefault("static_steps", {})[run_name] = int(static_step[1:].sum())
    del gt
    torch.cuda.empty_cache()


def aggregate(records: list) -> dict:
    cols = list(zip(*records))
    lat = np.array(cols[0])
    sc = np.array(cols[1])
    fld = np.array(cols[2])
    kk = np.array(cols[3])
    msk = np.array(cols[4])
    run = np.array(cols[5])
    fr = np.array(cols[6])
    st = np.array(cols[7])
    st30 = np.array(cols[8])
    st16 = np.array(cols[9])
    sse_y = np.array(cols[10], dtype=np.float64)
    cnt = np.array(cols[11], dtype=np.float64)
    l1 = np.array(cols[12], dtype=np.float64)
    sse = np.array(cols[13], dtype=np.float64)
    npix = np.array(cols[14], dtype=np.float64)
    scene = np.array([r.split("/")[0] for r in run])
    seqs = np.array([r.split("/")[1] for r in run])
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = np.where(sse_y > 0, 10 * np.log10(255.0**2 * cnt / sse_y), CAP)
    pf = np.minimum(pf, CAP)
    pf[cnt == 0] = np.nan
    as_stop = (
        (scene == AS_STOP[0])
        & (seqs == AS_STOP[1])
        & (fr >= AS_STOP[2])
        & (fr <= AS_STOP[3])
    )

    def summ(sel: np.ndarray) -> dict:
        sel = sel & (cnt > 0)
        if not sel.any():
            return {"frames": 0}
        (c, e) = (cnt[sel].sum(), sse_y[sel].sum())
        return {
            "frames": int(sel.sum()),
            "psnr": float(np.nanmean(pf[sel])),
            "psnr_pooled": float(10 * np.log10(255.0**2 * c / e)) if e > 0 else CAP,
            "l1_rgb": float(l1[sel].sum() / (3 * c)),
            "mse_rgb": float(sse[sel].sum() / (3 * c)),
            "valid_fraction": float(c / npix[sel].sum()),
        }

    scenes = sorted(set(scene.tolist()))
    fixed = {
        "ALL": np.ones_like(st30, dtype=bool),
        "moving": st30 == 1,
        "static": st16 == 0,
        "as_stop": as_stop,
    }
    chain = {"chain_static": st == 0, "chain_moving": st == 1, "chain_mixed": st == 2}
    out: dict = {}
    for L in sorted(set(lat.tolist())):
        for s_ in sorted(set(sc[lat == L].tolist())):
            for f_ in sorted(set(fld[(lat == L) & (sc == s_)].tolist())):
                for k_ in KS:
                    for m_ in MASKS:
                        base = (
                            (lat == L)
                            & (sc == s_)
                            & (fld == f_)
                            & (kk == k_)
                            & (msk == m_)
                        )
                        node = {}
                        for bn, bsel in {**fixed, **chain}.items():
                            node[bn] = summ(base & bsel)
                        for scn in scenes:
                            for bn in ("ALL", "moving", "static"):
                                key = scn if bn == "ALL" else f"{scn}/{bn}"
                                node[key] = summ(base & fixed[bn] & (scene == scn))
                        for bn in ("ALL", "moving", "static"):
                            vals = [
                                node[scn if bn == "ALL" else f"{scn}/{bn}"].get("psnr")
                                for scn in scenes
                            ]
                            vals = [v for v in vals if v is not None]
                            node[f"scene_mean/{bn}"] = {
                                "psnr": float(np.mean(vals)) if vals else None,
                                "scenes": len(vals),
                            }
                        if k_ > 16:
                            node["static"]["note"] = (
                                "k > 16: these chains include moving steps (no stop longer than 19 frames on the test split)"
                            )
                        out.setdefault(L, {}).setdefault(f"x{s_}", {}).setdefault(
                            f_, {}
                        ).setdefault(f"k{k_}", {})[m_] = node
    return out


def plot(result: dict, out_stem: Path, scale: str = "x4", mask: str = "hold") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (blue, orange, grey) = ("#2a78d6", "#eb6834", "#898781")
    (ink, ink2, grid) = ("#0b0b0b", "#52514e", "#e1e0d9")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": ink2,
            "xtick.color": ink2,
            "ytick.color": ink2,
            "text.color": ink,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    (fig, axes) = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    panels = (
        (
            "hr",
            "1080p history (accumulator lattice)",
            (
                ("nonjit", "registered (non-jittered MV)", blue),
                ("jit", "jittered MV (old pipeline)", orange),
            ),
        ),
        (
            "lr",
            "LR jittered-sample lattice",
            (
                ("rejit", "registered (MV + jitter delta)", blue),
                ("nonjit", "non-jittered MV (misregistered here)", orange),
            ),
        ),
    )
    for ax, (lat, title, series) in zip(axes, panels):
        node = result["results"][lat][scale]
        for fname, label, colour in series:
            for bucket, ls, mk, suffix in (
                ("moving", "-", "o", "moving camera"),
                ("static", "--", "s", "static camera"),
            ):
                (ks, ys) = ([], [])
                for k in KS:
                    if bucket == "static" and k > 16:
                        continue
                    v = node[fname][f"k{k}"][mask][bucket]
                    if v.get("frames", 0) >= 5:
                        ks.append(k)
                        ys.append(v["psnr"])
                if not ks:
                    continue
                ax.plot(
                    ks,
                    ys,
                    ls=ls,
                    marker=mk,
                    ms=4.5,
                    lw=1.6,
                    color=colour,
                    mec="white",
                    mew=0.8,
                    label=f"{label}, {suffix}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("frames the copy is held (k)")
        ax.set_title(title, color=ink, loc="left")
        ax.grid(True, color=grid, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("PSNR-Y of held copy vs GT$_t$ (dB)")
    (handles, labels) = ([], [])
    for ax in axes:
        (h_, l_) = ax.get_legend_handles_labels()
        handles += h_
        labels += l_
    for ax in axes:
        ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 3, 2])
    ap.add_argument(
        "--runs", type=int, default=0, help="smoke test: only the first N camera runs"
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="smoke test: first N frames of each run",
    )
    ap.add_argument("--out", default=str(OUT / "copy_cost.json"))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(json.loads(out.read_text()), out.with_suffix(""))
        print(f"wrote {out.with_suffix('.pdf')}")
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = json.loads((QRISP_LMDB / "QRISP_test_index.json").read_text())["frames"]
    groups = run_groups(frames)
    if args.runs:
        groups = groups[: args.runs]
    if args.max_frames:
        groups = [g[: args.max_frames] for g in groups]
    store = Store(dev)
    records: list = []
    checks: dict = {}
    t0 = time.time()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for g in groups:
            process_run(store, frames, g, tuple(args.scales), records, checks, dev)
    for key, c in list(checks.items()):
        if key.startswith("x"):
            checks[key] = {
                "pairs": len(c["median_abs_dev"]),
                "median_abs_dev_mean": float(np.mean(c["median_abs_dev"])),
                "median_abs_dev_max": float(np.max(c["median_abs_dev"])),
                "frac_within_0.01_mean": float(np.mean(c["frac_within_0.01"])),
                "frac_within_0.01_min": float(np.min(c["frac_within_0.01"])),
                "note": "Jit field minus (nonJit + (-dJy,+dJx)), LR px: per-pair median |dev| and fraction of pixels < 0.01",
            }
    result = {
        "results": aggregate(records),
        "checks": checks,
        "conventions": {
            "ks": list(KS),
            "targets": f"t = {T0}..last of every camera run (identical target set for every k)",
            "warp": "diffsr.rrta.warp: backward, bilinear, border, align_corners=True; LR field upsampled nearest x(1080/res) for HR (engine_seq.motion_to_hr)",
            "hr_fields": {
                "nonjit": "Motion{res} (registered HR)",
                "jit": "Motion{res}Jit (pre-fix arms)",
                "identity": "no warp",
            },
            "lr_fields": {
                "rejit": "Motion{res} + (-dJy,+dJx) (REG lr_rejitter; registered on the jittered lattice)",
                "jit": "Motion{res}Jit (shipped; pre-fix arms)",
                "nonjit": "Motion{res} (misregistered on the jittered lattice; mj_* arms)",
                "identity": "no warp",
            },
            "lr_colour": "LR{res}MipBiasJit",
            "gt": "GT1080Enhanced",
            "masks": {
                "hold": "PRIMARY. AND over k in KS of 'valid' for the same target: one pixel set per target, identical for every k and every field",
                "valid": "in-bounds for every field's chain AND depth test (samples.depth_test far 0.10, Depth{res} non-jittered + Motion{res}) at every step of THIS k's chain (shrinks with k)",
                "inb": "in-bounds for every field's chain of this k (the training loss's mask)",
            },
            "buckets": {
                "ALL": "all targets",
                "moving": "targets whose preceding 30 steps are all moving (same set for every k)",
                "static": "targets whose preceding 16 steps are all static (same set for every k; k=30 chains include moving steps)",
                "as_stop": "AbandonedSchool/0001 targets 275..292",
                "chain_*": "classified by the chain's own k steps (set varies with k)",
            },
            "valid_threshold": VALID_EPS,
            "camera_state": "protocol_full.camera_static on Motion270 (non-jittered) per step t-1 -> t (median |MV| < 0.02 270p px)",
            "as_stop": "AbandonedSchool/0001 targets 275..292 (QRISP fn. 3)",
            "psnr": f"per-frame PSNR-Y (MATLAB luma, 0-255) on the mask, capped at {CAP} dB, averaged; psnr_pooled = pooled SSE",
            "l1_rgb/mse_rgb": "[0,1] units; k=1 inb l1_rgb = temporal-loss value of a perfectly frozen output",
            "scales": args.scales,
        },
        "provenance": {
            "argv": sys.argv,
            "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "torch": torch.__version__,
            "code_snapshot": str(SNAP),
            "md5": {
                "scripts/copy_cost.py": md5(Path(__file__)),
                "diffsr/rrta.py": md5(SNAP / "diffsr/rrta.py"),
                "diffsr/samples.py": md5(SNAP / "diffsr/samples.py"),
                "diffsr/data_qrisp.py": md5(SNAP / "diffsr/data_qrisp.py"),
                "scripts/protocol_full.py": md5(SNAP / "scripts/protocol_full.py"),
            },
            "runtime_s": round(time.time() - t0, 1),
            "smoke": bool(args.runs or args.max_frames),
        },
    }
    out.write_text(json.dumps(result, indent=1))
    rec = np.array(records, dtype=object)
    np.savez_compressed(
        out.with_name(out.stem + "_frames.npz"),
        records=rec,
        columns=np.array(
            [
                "lattice",
                "scale",
                "field",
                "k",
                "mask",
                "run",
                "frame",
                "chain_state_k(0s,1m,2x)",
                "state_30steps",
                "state_16steps",
                "sse_y",
                "count",
                "l1_rgb_sum",
                "sse_rgb_sum",
                "pixels",
            ]
        ),
    )
    print(f"wrote {out} ({len(records)} records, {time.time() - t0:.0f}s)")
    if 4 in args.scales:
        plot(result, out.with_suffix(""))
        print(f"wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
