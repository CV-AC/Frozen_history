from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import protocol_full as pf
from diffsr.checkpoint import load_run
from diffsr.data_stss import STSS_PACKED, STSS_TEST, DEPTH_MODES, StssSequence
from diffsr.metrics import psnr, ssim
from diffsr.rrta import registered_upsample
from diffsr.temporal import temporal_gradient_error

SCALE = 2
BASELINES = ("bicubic", "regbicubic")


def motion_to_hr(motion: torch.Tensor, scale: int) -> torch.Tensor:
    return F.interpolate(motion, scale_factor=scale, mode="nearest") * scale


def _md5(path) -> str | None:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def provenance(data: StssSequence, **extra) -> dict:
    prov = pf.provenance(**extra)
    for name in ("diffsr.data_stss",):
        mod = sys.modules.get(name)
        if mod is not None:
            prov["md5"][name.replace(".", "/") + ".py"] = _md5(mod.__file__)
    prov["dataset"] = {
        name: {
            "root": str(s.root),
            "frames": len(s.frames),
            "frames_npy_md5": _md5(s.root / "frames.npy"),
            "meta_provenance": s.meta.get("provenance"),
            "colour_order_stored": "BGR" if s.bgr else "RGB",
            "runs": s.meta.get("runs"),
        }
        for (name, s) in data.scenes.items()
    }
    return prov


class Method:
    def __init__(
        self, name: str, device, ckpt: str, tta: int, reg_offset: tuple[float, float]
    ) -> None:
        (self.name, self.device) = (name, device)
        (self.cfgs, self.models) = ([], [])
        self.used = pf.FLIPS if tta else pf.FLIPS[:1]
        self.reg_offset = reg_offset
        if name in BASELINES:
            self.kind = name
            return
        self.kind = "model"
        (model, cfg) = load_run(ROOT / "runs" / name, device, ckpt)
        scale = cfg.get("scale") or cfg.get("model_kwargs", {}).get("scale")
        if int(scale) != SCALE:
            raise SystemExit(f"{name} was trained at x{scale}; STSS is x2")
        (self.models, self.cfgs) = ([model], [cfg])
        print(
            f"    [{name}: {ckpt} @ iteration {cfg.get('ckpt_iteration')}{(', TTA x4' if tta else '')}; trained on {cfg.get('qrisp_colour')} motion_jit={cfg.get('motion_jit')} lerp={cfg.get('motion_lerp')} -- STSS feeds its only field, jitter 0]",
            flush=True,
        )

    def reset(self) -> None:
        self.states = [[None] * len(self.used) for _ in self.models]

    @torch.inference_mode()
    def __call__(self, lr, depth, motion) -> torch.Tensor:
        if self.kind == "bicubic":
            return F.interpolate(
                lr, scale_factor=SCALE, mode="bicubic", align_corners=False
            ).clamp(0, 1)
        if self.kind == "regbicubic":
            (oy, ox) = self.reg_offset
            jit = torch.tensor([[-oy, ox]], dtype=torch.float32, device=lr.device)
            return registered_upsample(lr, jit, SCALE)
        jitter = torch.zeros(lr.shape[0], 2, device=lr.device)
        outs = []
        for mi, model in enumerate(self.models):
            for k, (h, v) in enumerate(self.used):
                (flr, fdep, fmot, fjit) = pf.apply_flip(lr, depth, motion, jitter, h, v)
                (o, self.states[mi][k], _) = model.step(
                    flr, fdep, fmot, fjit, self.states[mi][k]
                )
                outs.append(pf.unflip(o, h, v))
        return torch.stack(outs).mean(0)


def score(
    method: Method,
    data: StssSequence,
    warmup: int,
    stride: int,
    lpips: bool,
    swap_colour: bool,
) -> dict:
    device = method.device
    lp = None
    if lpips:
        try:
            lp = pf._lpips_fn(device)
        except Exception as exc:
            print(f"    (LPIPS unavailable: {str(exc)[:60]})", flush=True)
    (per_scene, curve, nonfinite) = ({}, [], [])
    for scene, orders in sorted(data.runs().items()):
        (vals, svals, lvals, tvals) = ([], [], [], [])
        for order in orders:
            method.reset()
            prev_pred = prev_ref = prev_mm = None
            for n, i in enumerate(order):
                item = data[i]
                lr = item["lr"][None].to(device)
                depth = item["depth"][None].to(device)
                motion = item["motion"][None].to(device)
                if swap_colour:
                    lr = lr.flip(1)
                out = method(lr, depth, motion)
                if swap_colour:
                    out = out.flip(1)
                finite = bool(torch.isfinite(out).all())
                if not finite:
                    if not nonfinite:
                        print(
                            f"    !! first non-finite output at {item['key']} (run frame {n})",
                            flush=True,
                        )
                    nonfinite.append(item["key"])
                out = out.clamp(0, 1).float()
                hr = item["hr"][None].to(device)
                row = {"index": i, "frame": data.frames[i]["n"], "run_frame": n}
                if prev_pred is not None and n >= warmup:
                    tv = temporal_gradient_error(
                        torch.cat((prev_pred, out), 0),
                        torch.cat((prev_ref, hr), 0),
                        motion_to_hr(torch.cat((prev_mm, motion), 0), SCALE),
                    )
                    tv["static"] = pf.camera_static(motion, SCALE)
                    tv["index"] = i
                    tvals.append(tv)
                    row.update(tpsnr=tv["tpsnr"], static=tv["static"])
                (prev_pred, prev_ref, prev_mm) = (out, hr, motion)
                if n < warmup or (n - warmup) % stride:
                    continue
                pred = out[0].permute(1, 2, 0).cpu().numpy() * 255.0
                ref = item["hr"].permute(1, 2, 0).float().numpy() * 255.0
                vals.append(psnr(pred, ref, SCALE, crop_border=0, to_y=True))
                svals.append(ssim(pred, ref, SCALE, crop_border=0, to_y=True))
                row.update(psnr=vals[-1], ssim=svals[-1])
                if lp is not None and finite:
                    with torch.inference_mode():
                        lvals.append(float(lp(out, hr.float())))
                    row["lpips"] = lvals[-1]
                curve.append(row)
        agg = pf.aggregate_tpsnr(tvals)
        per_scene[scene] = {
            "psnr_y": float(np.mean(vals)) if vals else float("nan"),
            "ssim_y": float(np.mean(svals)) if svals else float("nan"),
            "lpips": float(np.mean(lvals)) if lvals else float("nan"),
            "scored_frames": len(vals),
            "runs": len(orders),
            "tpsnr": agg,
        }
        a = per_scene[scene]
        lp_txt = "" if lp is None else f"  LPIPS {a['lpips']:.4f}"
        print(
            f"    {scene:<14}{sum((len(o) for o in orders)):>5} frames, {len(vals):>4} scored  PSNR-Y {a['psnr_y']:.3f}  SSIM {a['ssim_y']:.4f}{lp_txt}  | tpool {agg['pooled']:.3f}  tcap {agg['cap60']:.3f}  [static {agg['pooled_static']:.2f} over {agg['static_pairs']}p, moving {agg['pooled_moving']:.2f}]",
            flush=True,
        )

    def mean(key, sub=None):
        return pf.nanmean(
            (v["tpsnr"][sub] if sub else v[key] for v in per_scene.values())
        )

    return {
        "psnr_y": mean("psnr_y"),
        "ssim_y": mean("ssim_y"),
        "lpips": mean("lpips"),
        "tpsnr_pooled": mean(None, "pooled"),
        "tpsnr_cap60": mean(None, "cap60"),
        "tpsnr_meandb": mean(None, "meandb"),
        "tpsnr_pooled_static": mean(None, "pooled_static"),
        "tpsnr_pooled_moving": mean(None, "pooled_moving"),
        "per_scene": per_scene,
        "per_frame": curve,
        "nonfinite_frames": nonfinite,
        "kind": method.kind,
        "ckpt_iteration": [c.get("ckpt_iteration") for c in method.cfgs],
        "trained_input_pipeline": [
            {
                k: c.get(k)
                for k in (
                    "qrisp_colour",
                    "motion_jit",
                    "motion_lerp",
                    "depth_jit",
                    "scale",
                )
            }
            for c in method.cfgs
        ],
        "swap_colour_diagnostic": swap_colour,
    }


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names under runs/")
    ap.add_argument("--baselines", nargs="*", default=[], choices=BASELINES)
    ap.add_argument(
        "--root",
        nargs="+",
        default=[str(STSS_PACKED / STSS_TEST)],
        help="packed scene dir(s); default Arena_test",
    )
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument(
        "--tta",
        nargs="?",
        type=int,
        const=4,
        default=0,
        choices=(4,),
        help="four flips, one state each",
    )
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--depth-mode", choices=DEPTH_MODES, default="linear")
    ap.add_argument(
        "--reg-offset",
        nargs=2,
        type=float,
        default=(0.0, 0.0),
        metavar=("DY", "DX"),
        help="regbicubic: global image offset of the LR samples,\nLR px (0 0 = STSS's nominal pixel centres)",
    )
    ap.add_argument(
        "--swap-colour",
        action="store_true",
        help="DIAGNOSTIC: feed BGR (the pre-fix loader's order)\nand swap the output back",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="SMOKE/UNIT TESTS ONLY: first N frames of each scene",
    )
    ap.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("TOP", "LEFT", "H", "W"),
        help="UNIT TESTS ONLY: LR crop of every frame",
    )
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="runs/reference/stss_protocol.json")
    args = ap.parse_args()
    if not args.runs and (not args.baselines):
        ap.error("nothing to score: give run names and/or --baselines")
    device = torch.device(
        "cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"
    )
    data = StssSequence(
        args.root,
        scale=SCALE,
        depth_mode=args.depth_mode,
        crop=args.crop,
        max_frames=args.max_frames,
    )
    test_only = bool(args.max_frames or args.crop)
    print(
        f"STSS x2 {', '.join(data.scenes)}: {len(data)} frames, runs { {k: [len(o) for o in v] for (k, v) in data.runs().items()} }, warmup {args.warmup}, stride {args.stride}, depth {args.depth_mode}, device {device}{('  [UNIT/SMOKE TEST ROW -- not a result]' if test_only else '')}\n",
        flush=True,
    )
    protocol = provenance(
        data,
        dataset="STSS",
        scale=SCALE,
        warmup=args.warmup,
        stride=args.stride,
        ckpt=args.ckpt,
        tta=args.tta,
        lpips=args.lpips,
        depth_mode=args.depth_mode,
        reg_offset=list(args.reg_offset),
        max_frames=args.max_frames,
        crop=args.crop,
        swap_colour=args.swap_colour,
        test_only=test_only,
        metric_field="STSS native motion (no jitter: the only field)",
        jitter="zero",
        colour="RGB (stored BGR, reversed at load)",
    )
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    table = {}
    for name in list(args.baselines) + list(args.runs):
        print(f"  {name}", flush=True)
        method = Method(name, device, args.ckpt, args.tta, tuple(args.reg_offset))
        r = score(method, data, args.warmup, args.stride, args.lpips, args.swap_colour)
        table[name] = r
        print(
            f"    {'mean':<14}  PSNR-Y {r['psnr_y']:.3f}  SSIM {r['ssim_y']:.4f}  LPIPS {r['lpips']:.4f}  tpool {r['tpsnr_pooled']:.3f}  tcap {r['tpsnr_cap60']:.3f}  nonfinite {len(r['nonfinite_frames'])}\n",
            flush=True,
        )
        del method
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "dataset": "STSS",
                    "scale": SCALE,
                    "protocol": protocol,
                    "table": table,
                },
                indent=1,
            )
        )
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
