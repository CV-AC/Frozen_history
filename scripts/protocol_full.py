from __future__ import annotations
import argparse
import datetime
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diffsr.checkpoint import load_run
from diffsr.data_qrisp import QrispDataset, QRISP_LMDB, RES_FOR_SCALE, _Lmdb
from diffsr.metrics import psnr, ssim

_LPIPS = None


def _lpips_fn(device):
    global _LPIPS
    if _LPIPS is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        _LPIPS = (
            LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
            .to(device)
            .eval()
        )
    return _LPIPS


from diffsr.engine_seq import motion_to_hr
from diffsr.temporal import temporal_gradient_error

TPSNR_FIELDS = ("nonjit", "nonjit540", "input")
TPSNR_CAP = 60.0


class MetricField:
    def __init__(self, kind: str, scale: int):
        if kind not in TPSNR_FIELDS:
            raise ValueError(f"tpsnr field must be one of {TPSNR_FIELDS}")
        self.kind = kind
        if kind == "input":
            (self.db, self.factor, self.name) = (None, int(scale), "input(arm field)")
        else:
            res = "540" if kind == "nonjit540" else RES_FOR_SCALE[int(scale)]
            self.factor = 1080 // int(res)
            self.db = _Lmdb(QRISP_LMDB / f"QRISP_test_Motion{res}.lmdb")
            self.name = self.db.path.name

    def lr(self, data, i: int, input_motion: torch.Tensor) -> torch.Tensor:
        if self.db is None:
            return input_motion
        arr = self.db.array(data.samples[i][1])
        return (
            torch.from_numpy(np.ascontiguousarray(arr))
            .permute(2, 0, 1)[None]
            .to(input_motion.device)
        )

    def hr(self, pair: torch.Tensor) -> torch.Tensor:
        return motion_to_hr(pair, self.factor)


def nanmean(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    keep = arr[~np.isnan(arr)]
    return float(keep.mean()) if keep.size else float("nan")


def camera_static(m_lr: torch.Tensor, factor: int, tol: float = 0.02) -> bool:
    med = m_lr.flatten(2).abs().median(dim=2).values.amax()
    return bool(med * factor / 4.0 < tol)


def aggregate_tpsnr(pairs: list[dict]) -> dict:

    def pooled(ps):
        if not ps:
            return float("nan")
        sse = sum((p["sse"] for p in ps))
        cnt = sum((p["count"] for p in ps))
        if not np.isfinite(sse):
            return float("nan")
        return (
            float("inf") if sse == 0 else float(10.0 * np.log10(255.0**2 * cnt / sse))
        )

    db = np.array([p["tpsnr"] for p in pairs], dtype=np.float64)
    nan = float("nan")
    with np.errstate(invalid="ignore"):
        finite_or_inf = db[~np.isnan(db)]
        return {
            "meandb": float(np.mean(finite_or_inf)) if len(finite_or_inf) else nan,
            "cap60": float(np.mean(np.minimum(finite_or_inf, TPSNR_CAP)))
            if len(finite_or_inf)
            else nan,
            "pooled": pooled(pairs),
            "pooled_static": pooled([p for p in pairs if p["static"]]),
            "pooled_moving": pooled([p for p in pairs if not p["static"]]),
            "pairs": len(pairs),
            "static_pairs": int(sum((bool(p["static"]) for p in pairs))),
            "pairs_above_40db": int(np.sum(finite_or_inf > 40.0)),
            "nan_pairs": int(np.sum(np.isnan(db))),
        }


_DATASET_PARAMS = set(inspect.signature(QrispDataset.__init__).parameters)


def _cfg(cfg: dict, key: str, default):
    value = cfg.get(key)
    return default if value is None else value


def input_pipeline(cfg: dict) -> dict:
    scale = cfg.get("scale") or cfg.get("model_kwargs", {}).get("scale")
    return {
        "colour": cfg["qrisp_colour"],
        "motion_jit": bool(_cfg(cfg, "motion_jit", True)),
        "motion_lerp": float(_cfg(cfg, "motion_lerp", 0.0)),
        "depth_jit": bool(_cfg(cfg, "depth_jit", True)),
        "scale": int(scale) if scale is not None else None,
    }


def dataset_for(pipe: dict, scale: int, split: str = "test") -> QrispDataset:
    if pipe["scale"] is not None and pipe["scale"] != scale:
        raise SystemExit(f"run trained at x{pipe['scale']}, asked to score x{scale}")
    kw = dict(
        scale=scale,
        colour=pipe["colour"],
        motion_jit=pipe["motion_jit"],
        motion_lerp=pipe["motion_lerp"],
        with_depth=True,
        with_motion=True,
        patch_lr=0,
        flip=False,
        tiles=None,
    )
    if "depth_jit" in _DATASET_PARAMS:
        kw["depth_jit"] = pipe["depth_jit"]
    elif not pipe["depth_jit"]:
        raise SystemExit(
            "run trained with depth_jit=False but this QrispDataset has no depth_jit option"
        )
    return QrispDataset(split, **kw)


def _md5(path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def code_md5() -> dict:
    mods = {}
    for name in (
        "diffsr.rrta",
        "diffsr.temporal",
        "diffsr.checkpoint",
        "diffsr.metrics",
        "diffsr.data_qrisp",
        "diffsr.engine_seq",
        "diffsr.qrisp_model",
    ):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__file__", None):
            mods[name.replace(".", "/") + ".py"] = _md5(mod.__file__)
    mods["scripts/protocol_full.py"] = _md5(__file__)
    main = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if main is not None and main.is_file():
        mods[f"scripts/{main.name}"] = _md5(main)
    return mods


def provenance(**extra) -> dict:
    return {
        "argv": list(sys.argv),
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "torch": torch.__version__,
        "md5": code_md5(),
        **extra,
    }


PUBLISHED = {
    4: {"bicubic": 26.42, "Xiao et al.": 29.02, "QRISP Ours-L": 29.42},
    3: {"bicubic": 27.61, "Xiao et al.": 30.24, "QRISP Ours-L": 30.67},
    2: {"bicubic": 29.51, "Xiao et al.": 31.89, "QRISP Ours-L": 32.21},
}
PER_SCENE = {4: {"AbandonedSchool": 25.61, "SeaPort": 23.59, "SpaceShipDemo": 30.06}}


def scenes(data) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i in range(len(data)):
        out.setdefault(data.frames[i]["scene"], []).append(i)
    return out


def runs(data) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    last: tuple | None = None
    for i in range(len(data)):
        f = data.frames[i]
        key = (f["scene"], f.get("sequence"))
        if key != last:
            out.setdefault(f["scene"], []).append([])
            last = key
        out[f["scene"]][-1].append(i)
    return out


def frame_orders(data, reset_runs: bool) -> dict[str, list[list[int]]]:
    return runs(data) if reset_runs else {k: [v] for (k, v) in scenes(data).items()}


def add_reset_flags(ap: argparse.ArgumentParser, legacy_help: str = "") -> None:
    ap.add_argument(
        "--reset-runs",
        dest="reset_runs",
        action="store_true",
        default=True,
        help="start a fresh accumulator at each camera cut, and\nre-apply the warmup there.  Scenes hold two runs;\nthe shipped baselines were rendered per run, so\nNOT resetting charges only the recurrent method.\nBaselines must be rescored with the same flag.\nEnabled by default.",
    )
    ap.add_argument(
        "--no-reset-runs",
        dest="reset_runs",
        action="store_false",
        help="legacy: carry state across cuts (old logs only)" + legacy_help,
    )


FLIPS = ((False, False), (True, False), (False, True), (True, True))
DIHEDRAL = tuple(((t, h, v) for t in (False, True) for (h, v) in FLIPS))


def apply_flip(lr, depth, motion, jitter, h, v):
    if h:
        (lr, depth, motion) = (lr.flip(-1), depth.flip(-1), motion.flip(-1))
        motion = motion.clone()
        motion[:, 1] *= -1.0
        jitter = jitter.clone()
        jitter[:, 1] *= -1.0
    if v:
        (lr, depth, motion) = (lr.flip(-2), depth.flip(-2), motion.flip(-2))
        motion = motion.clone()
        motion[:, 0] *= -1.0
        jitter = jitter.clone()
        jitter[:, 0] *= -1.0
    return (lr, depth, motion, jitter)


def unflip(out, h, v):
    if h:
        out = out.flip(-1)
    if v:
        out = out.flip(-2)
    return out


def apply_dihedral(lr, depth, motion, jitter, t, h, v):
    if t:
        (lr, depth) = (lr.transpose(-2, -1), depth.transpose(-2, -1))
        motion = motion.transpose(-2, -1).flip(1)
        jitter = -jitter.flip(1)
    return apply_flip(lr, depth, motion, jitter, h, v)


def undihedral(out, t, h, v):
    out = unflip(out, h, v)
    return out.transpose(-2, -1) if t else out


def run_model(
    names,
    scale: int,
    warmup: int,
    device,
    stride: int,
    ckpt_name: str = "last.pt",
    tta: int = 0,
    reset_runs: bool = True,
    lpips: bool = False,
    static_gate: float = 0.0,
    static_cap: float = 64.0,
    dejitter: bool = False,
    tpsnr_field: str = "nonjit",
    quantize: bool = False,
    max_frames: int = 0,
) -> dict:
    if isinstance(names, str):
        names = [names]
    used = (
        DIHEDRAL
        if tta == 8
        else tuple(((False, h, v) for (h, v) in (FLIPS if tta else FLIPS[:1])))
    )
    (models, cfgs) = ([], [])
    for name in names:
        (m, c) = load_run(ROOT / "runs" / name, device, ckpt_name)
        if dejitter:
            m.dejitter = True
        if static_gate > 0.0:
            m.static_gate = float(static_gate)
            m.static_cap = float(static_cap)
        models.append(m)
        cfgs.append(c)
        print(
            f"    [{name}: {ckpt_name} @ iteration {c.get('ckpt_iteration')}{(f', TTA x{len(used)}' if tta else '')}]",
            flush=True,
        )
    pipes = {tuple(sorted(input_pipeline(c).items())) for c in cfgs}
    if len(pipes) != 1:
        raise SystemExit(
            f"ensemble members disagree on the input pipeline (colour, motion_jit, motion_lerp, depth_jit, scale): {pipes}"
        )
    pipe = dict(pipes.pop())
    field = MetricField(tpsnr_field, scale)
    if pipe["motion_lerp"] and field.db is None:
        raise SystemExit(
            "motion_lerp arm with --tpsnr-field input would put eps into the METRIC warp; use a nonjit field"
        )
    data = dataset_for(pipe, scale)
    print(
        f"    [input colour={pipe['colour']} motion_jit={pipe['motion_jit']} lerp={pipe['motion_lerp']} depth_jit={pipe['depth_jit']} x{scale}; metric field {field.name}; reset_runs={reset_runs}; quantize={quantize}]",
        flush=True,
    )
    lp = None
    if lpips:
        try:
            lp = _lpips_fn(device)
        except Exception as exc:
            print(f"    (LPIPS unavailable: {str(exc)[:60]})", flush=True)
    (y, s, per, tg, lps) = ({}, {}, {}, {}, {})
    (tagg, nonfinite, per_run, per_frame) = ({}, [], {}, {})
    grouping = frame_orders(data, reset_runs)
    for scene, orders in sorted(grouping.items()):
        (vals, svals, tvals_scene, lvals) = ([], [], [], [])
        frames_scene = []
        for order in orders:
            if max_frames:
                order = order[:max_frames]
            states = [[None] * len(used) for _ in models]
            prev_pred = prev_ref = prev_mm = None
            tvals = []
            for n, i in enumerate(order):
                item = data[i]
                lr = item["lr"][None, :3].to(device)
                depth = item["depth"][None].to(device)
                motion = item["motion"][None].to(device)
                jit = item.get("jitter")
                jitter = (jit[None] if jit is not None else torch.zeros(1, 2)).to(
                    device
                )
                with torch.inference_mode():
                    outs = []
                    for mi, model in enumerate(models):
                        for k, (t, h, v) in enumerate(used):
                            (flr, fdep, fmot, fjit) = apply_dihedral(
                                lr, depth, motion, jitter, t, h, v
                            )
                            (o, states[mi][k], _) = model.step(
                                flr, fdep, fmot, fjit, states[mi][k]
                            )
                            outs.append(undihedral(o, t, h, v))
                    out = torch.stack(outs).mean(0)
                finite = bool(torch.isfinite(out).all())
                if not finite:
                    meta = data.frames[i]
                    if not nonfinite:
                        print(
                            f"    !! first non-finite output at dataset index {i} ({meta['scene']}/{meta.get('sequence')}/{meta.get('frame')}, run frame {n})",
                            flush=True,
                        )
                    nonfinite.append(i)
                if quantize:
                    out = (out.clamp(0, 1) * 255.0).round() / 255.0
                hr_ref = item["hr"][None].to(device)
                mm = field.lr(data, i, motion)
                if prev_pred is not None and n >= warmup:
                    pair_p = torch.cat((prev_pred, out.clamp(0, 1)), dim=0)
                    pair_r = torch.cat((prev_ref, hr_ref), dim=0)
                    tv = temporal_gradient_error(
                        pair_p, pair_r, field.hr(torch.cat((prev_mm, mm), 0))
                    )
                    tv["static"] = camera_static(mm, field.factor)
                    tv["index"] = i
                    tvals.append(tv)
                (prev_pred, prev_ref, prev_mm) = (out.clamp(0, 1), hr_ref, mm)
                if n < warmup or (n - warmup) % stride:
                    continue
                pred = out[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255.0
                ref = item["hr"].permute(1, 2, 0).float().numpy() * 255.0
                vals.append(psnr(pred, ref, scale, crop_border=0, to_y=True))
                svals.append(ssim(pred, ref, scale, crop_border=0, to_y=True))
                meta = data.frames[i]
                frames_scene.append((i, meta.get("sequence"), vals[-1], svals[-1]))
                if lp is not None and finite:
                    with torch.inference_mode():
                        lvals.append(
                            float(
                                lp(
                                    out.clamp(0, 1).float(),
                                    item["hr"][None].to(device).float(),
                                )
                            )
                        )
            tvals_scene.extend(tvals)
        per[scene] = float(np.mean(vals))
        (y[scene], s[scene]) = (vals, svals)
        lps[scene] = float(np.mean(lvals)) if lvals else float("nan")
        tagg[scene] = aggregate_tpsnr(tvals_scene)
        tagg[scene]["per_pair"] = [
            (t["index"], t["tpsnr"], t["static"]) for t in tvals_scene
        ]
        tg[scene] = tagg[scene]["meandb"]
        per_frame[scene] = frames_scene
        for seq in sorted({f[1] for f in frames_scene}):
            per_run[f"{scene}/{seq}"] = float(
                np.mean([f[2] for f in frames_scene if f[1] == seq])
            )
        total = sum((min(len(o), max_frames) if max_frames else len(o) for o in orders))
        print(
            f"    {scene:<18}{total:>5} frames, {len(vals):>4} scored  {per[scene]:>8.3f}  tpsnr {tg[scene]:>7.3f}{('' if lp is None else f'  lpips {lps[scene]:.4f}')}{('  [' + str(len(orders)) + ' runs]' if len(orders) > 1 else '')}  | tpool {tagg[scene]['pooled']:>7.3f}  tcap {tagg[scene]['cap60']:>7.3f}  [static {tagg[scene]['pooled_static']:.2f} over {tagg[scene]['static_pairs']}p, moving {tagg[scene]['pooled_moving']:.2f}; {tagg[scene]['pairs_above_40db']}p >40dB]",
            flush=True,
        )
    del models
    torch.cuda.empty_cache()
    if nonfinite:
        print(
            f"    !! {len(nonfinite)} non-finite output frames, first at dataset index {nonfinite[0]}",
            flush=True,
        )

    def scene_mean(key):
        return nanmean((a[key] for a in tagg.values()))

    return {
        "psnr_y": float(np.mean([np.mean(v) for v in y.values()])),
        "ssim_y": float(np.mean([np.mean(v) for v in s.values()])),
        "lpips": nanmean(lps.values()),
        "tpsnr": nanmean(tg.values()),
        "tpsnr_pooled": scene_mean("pooled"),
        "tpsnr_cap60": scene_mean("cap60"),
        "tpsnr_pooled_static": scene_mean("pooled_static"),
        "tpsnr_pooled_moving": scene_mean("pooled_moving"),
        "per_scene_y": per,
        "per_scene_tpsnr": tg,
        "per_scene_ssim": {k: float(np.mean(v)) for (k, v) in s.items()},
        "per_scene_lpips": lps,
        "per_run_y": per_run,
        "per_scene_tpsnr_all": tagg,
        "per_frame": per_frame,
        "nonfinite_frames": nonfinite,
        "metric_field": field.name,
        "tpsnr_field": tpsnr_field,
        "reset_runs": reset_runs,
        "stride": stride,
        "warmup": warmup,
        "scale": scale,
        "quantize": quantize,
        "tta": tta,
        "max_frames": max_frames,
        "ckpt": ckpt_name,
        "ckpt_iteration": [c.get("ckpt_iteration") for c in cfgs],
        "input_pipeline": pipe,
        "input_motion_jit": pipe["motion_jit"],
        "input_motion_lerp": pipe["motion_lerp"],
        "dejitter_override": dejitter,
        "static_gate_override": static_gate,
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument(
        "--tta",
        nargs="?",
        type=int,
        const=4,
        default=0,
        choices=(4, 8),
        help="average geometric transforms, each with its own\nrecurrent state.  4 = the flips; 8 adds the\ntransposed half of the dihedral group.  Costs one\nfull pass each, so report it beside the\nuntransformed number, never instead of it",
    )
    ap.add_argument(
        "--lpips",
        action="store_true",
        help="also compute LPIPS, the third metric Mercier et al. publish (Ours-L is 0.238 at x4)",
    )
    add_reset_flags(ap)
    ap.add_argument(
        "--tpsnr-field",
        choices=TPSNR_FIELDS,
        default="nonjit",
        help="warp for every temporal metric, FIXED for all\nmethods.  nonjit = Motion{res} at this scale;\nnonjit540 = Motion540 x2 at every scale (cross-\nscale tables); input = legacy arm field",
    )
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="round outputs to 8 bit before scoring, like the\nshipped PNG baselines",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="SMOKE TESTS ONLY: score the first N frames of each\npass.  Recorded; exclude truncated runs from paper comparisons",
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="average the named runs into ONE model instead of\nscoring them separately; costs one full pass per\nmember, so report it as its own row with that cost",
    )
    ap.add_argument(
        "--static-gate",
        type=float,
        default=0.0,
        help="inference-time override; see static_segment.py",
    )
    ap.add_argument("--static-cap", type=float, default=64.0)
    ap.add_argument(
        "--dejitter",
        action="store_true",
        help="inference-time override: strip the jitter delta from the motion field before reprojection",
    )
    ap.add_argument("--out", default="runs/reference/protocol_full.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pub = PUBLISHED[args.scale]
    print(
        f"QRISP x{args.scale}, luma, one unbroken pass per {('camera run' if args.reset_runs else 'scene')}, {args.warmup} warmup frames, every {args.stride}th frame scored, reset_runs={args.reset_runs}, tpsnr field {args.tpsnr_field}, quantize={args.quantize}{(f', max_frames={args.max_frames}' if args.max_frames else '')}\n"
    )
    table = {}
    groups = [args.runs] if args.ensemble else [[n] for n in args.runs]
    protocol = provenance(
        stride=args.stride,
        warmup=args.warmup,
        reset_runs=args.reset_runs,
        tpsnr_field=args.tpsnr_field,
        quantize=args.quantize,
        tta=args.tta,
        max_frames=args.max_frames,
        ckpt=args.ckpt,
        scale=args.scale,
        lpips=args.lpips,
    )
    for group in groups:
        name = " + ".join(group)
        print(f"  {name}", flush=True)
        r = run_model(
            group,
            args.scale,
            args.warmup,
            device,
            args.stride,
            args.ckpt,
            args.tta,
            args.reset_runs,
            args.lpips,
            args.static_gate,
            args.static_cap,
            args.dejitter,
            args.tpsnr_field,
            args.quantize,
            args.max_frames,
        )
        table[name] = r
        best = pub["QRISP Ours-L"]
        lp_txt = (
            ""
            if np.isnan(r.get("lpips", float("nan")))
            else f"  lpips {r['lpips']:.4f}"
        )
        print(
            f"    {'scene mean':<18}{'':>5}        {'':>4}       {r['psnr_y']:>8.3f}  tpsnr {r['tpsnr']:>7.3f}  ssim {r['ssim_y']:.4f}{lp_txt}   vs Ours-L {best:.2f} {r['psnr_y'] - best:+.3f}",
            flush=True,
        )
        print(
            f"    {'scene mean':<18}  tpool {r['tpsnr_pooled']:>7.3f}  tcap {r['tpsnr_cap60']:>7.3f}  static {r['tpsnr_pooled_static']:.3f}  moving {r['tpsnr_pooled_moving']:.3f}  field {r['metric_field']}  nonfinite {len(r['nonfinite_frames'])}\n",
            flush=True,
        )
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "scale": args.scale,
                    "published": pub,
                    "protocol": protocol,
                    "table": table,
                },
                indent=1,
            )
        )
    print("  published: " + ", ".join((f"{k} {v}" for (k, v) in pub.items())))
    print(f"  wrote {ROOT / args.out}")


if __name__ == "__main__":
    main()
