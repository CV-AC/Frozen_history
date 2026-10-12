from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diffsr.checkpoint import load_run
import torch.nn.functional as F
from diffsr.rrta import RRTANet, dilate_motion, registered_upsample, warp
from protocol_full import MetricField, input_pipeline, dataset_for, provenance

HALF = 0.5 / 255.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--scene", default="AbandonedSchool")
    ap.add_argument("--sequence", default="0001")
    ap.add_argument("--first", type=int, default=275)
    ap.add_argument("--last", type=int, default=292)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (model, cfg) = load_run(ROOT / "runs" / a.run, dev, a.ckpt)
    pipe = input_pipeline(cfg)
    data = dataset_for(pipe, a.scale)
    field = MetricField("nonjit", a.scale)
    run_idx = [
        i
        for i in range(len(data))
        if data.frames[i]["scene"] == a.scene
        and data.frames[i].get("sequence") == a.sequence
    ]
    seg = [i for i in run_idx if a.first <= int(data.frames[i]["frame"]) <= a.last]
    if not seg or seg[0] == run_idx[0]:
        raise SystemExit("segment empty or without a predecessor frame")
    accumulate = cfg.get("model_kwargs", {}).get("accumulate", "?")
    print(
        f"{a.run} [{a.ckpt} @ {cfg.get('ckpt_iteration')}] accumulate={accumulate} x{a.scale}; field {field.name}; segment {a.scene}/{a.sequence} {a.first}-{a.last} ({len(seg)} frames)",
        flush=True,
    )
    keys = ("valid", "static")
    acc = {
        k: {
            "num": 0.0,
            "den": 0.0,
            "copy": 0.0,
            "ref_copy": 0.0,
            "sq": 0.0,
            "ref_sq": 0.0,
            "n": 0.0,
            "gnum": 0.0,
            "gden": 0.0,
        }
        for k in keys
    }
    rrta = isinstance(model, RRTANet)
    if rrta and (model.dejitter or model.lr_unshuffle != 1):
        rrta = False
    gen = torch.Generator(device=dev).manual_seed(12345)
    (per_frame, outs, alphas) = ([], [], [])
    state = prev_out = prev_ref = None
    nonfinite = []
    seg_set = set(seg)
    for i in run_idx:
        if i > seg[-1]:
            break
        item = data[i]
        lr = item["lr"][None, :3].to(dev)
        jit = item.get("jitter")
        jitter = (jit[None] if jit is not None else torch.zeros(1, 2)).to(dev)
        depth_t = item["depth"][None].to(dev)
        motion_t = item["motion"][None].to(dev)
        state_in = state
        with torch.inference_mode():
            (out, state, aux) = model.step(lr, depth_t, motion_t, jitter, state_in)
            gain_pair = None
            if rrta and i in seg_set and (state_in is not None):
                prev_c = state_in[0]
                delta = 0.001 * torch.randn(
                    prev_c.shape, generator=gen, device=dev, dtype=prev_c.dtype
                )
                (out_p, _, _) = model.step(
                    lr,
                    depth_t,
                    motion_t,
                    jitter,
                    (prev_c + delta,) + tuple(state_in[1:]),
                )
                m = (
                    dilate_motion(motion_t, depth_t)
                    if model.motion_dilation
                    else motion_t
                )
                m_hr = F.interpolate(m, size=out.shape[-2:], mode="nearest") * a.scale
                (r_delta, inside) = warp(delta, m_hr)
                gain_pair = (
                    out_p.float() - out.float(),
                    r_delta.float(),
                    inside.float(),
                )
        if not bool(torch.isfinite(out).all()):
            nonfinite.append(int(data.frames[i]["frame"]))
        y = out.float().clamp(0, 1)
        ref = item["hr"][None].to(dev).float()
        if i in seg_set:
            mm = field.hr(field.lr(data, i, item["motion"][None].to(dev)))
            (h, valid) = warp(prev_out, mm)
            (r, _) = warp(prev_ref, mm)
            c = registered_upsample(lr.float(), jitter.float(), a.scale)
            valid = valid.float()
            ref_static = ((ref - r).abs().amax(1, keepdim=True) < HALF).float() * valid
            (dy, dc, dr) = (y - h, c - h, ref - r)
            copy = (dy.abs().amax(1, keepdim=True) < HALF).float()
            rcopy = (dr.abs().amax(1, keepdim=True) < HALF).float()
            row = {"frame": int(data.frames[i]["frame"])}
            for k, m in (("valid", valid), ("static", ref_static)):
                m3 = m.expand_as(dy)
                num = float((dy * dc * m3).sum())
                den = float((dc * dc * m3).sum())
                s = acc[k]
                if gain_pair is not None:
                    (d_out, r_d, ins) = gain_pair
                    mg = (m * ins).expand_as(d_out)
                    gn = float((d_out * r_d * mg).sum())
                    gd = float((r_d * r_d * mg).sum())
                    s["gnum"] += gn
                    s["gden"] += gd
                    row[f"alpha_hist_{k}"] = 1.0 - gn / gd if gd > 0 else None
                s["num"] += num
                s["den"] += den
                s["copy"] += float((copy * m).sum())
                s["ref_copy"] += float((rcopy * m).sum())
                s["sq"] += float((dy * dy * m3).sum())
                s["ref_sq"] += float((dr * dr * m3).sum())
                s["n"] += float(m.sum())
                row[f"alpha_eff_bicubic_{k}"] = num / den if den > 0 else None
                row[f"copy_frac_{k}"] = float((copy * m).sum() / m.sum().clamp_min(1.0))
            row["static_px_frac"] = float(ref_static.mean())
            if torch.is_tensor(aux.get("alpha")):
                al = aux["alpha"].float()
                if bool(torch.isfinite(al).all()):
                    q = torch.quantile(
                        al.flatten()[::97],
                        torch.tensor([0.1, 0.5, 0.9], device=al.device),
                    )
                    alphas.append((float(al.mean()), *[float(v) for v in q]))
            per_frame.append(row)
            outs.append(y[0].cpu().numpy() * 255.0)
        (prev_out, prev_ref) = (y, ref)
    res = {}
    for k in keys:
        s = acc[k]
        n3 = 3.0 * max(s["n"], 1.0)
        res[k] = {
            "alpha_hist": 1.0 - s["gnum"] / s["gden"] if s["gden"] > 0 else None,
            "history_gain": s["gnum"] / s["gden"] if s["gden"] > 0 else None,
            "alpha_eff_bicubic": s["num"] / s["den"] if s["den"] > 0 else None,
            "copy_frac": s["copy"] / max(s["n"], 1.0),
            "ref_copy_frac": s["ref_copy"] / max(s["n"], 1.0),
            "copy_rms_u8": 255.0 * float(np.sqrt(s["sq"] / n3)),
            "ref_rms_u8": 255.0 * float(np.sqrt(s["ref_sq"] / n3)),
            "pixels": s["n"],
        }
    arr = np.stack(outs).astype(np.float64)
    std_f = float(arr.std(axis=0, ddof=0).mean())
    std_u8 = float(np.round(arr).std(axis=0, ddof=0).mean())
    alpha_model = None
    if alphas:
        al = np.array(alphas)
        alpha_model = {
            "mean": float(al[:, 0].mean()),
            "p10": float(al[:, 1].mean()),
            "median": float(al[:, 2].mean()),
            "p90": float(al[:, 3].mean()),
        }
    for k in keys:
        r = res[k]
        ah = r["alpha_hist"]
        print(
            f"  [{k:>6}] alpha_hist {(ah if ah is None else round(ah, 5))}  alpha_eff_bicubic {r['alpha_eff_bicubic']:.4f}  copy_frac {r['copy_frac']:.4f} (reference {r['ref_copy_frac']:.4f})  copy_rms {r['copy_rms_u8']:.3f} (reference {r['ref_rms_u8']:.3f}) grey levels over {r['pixels']:.0f} px"
        )
    print(
        f"  pixel std float {std_f:.3f}  uint8 {std_u8:.3f}  model alpha {alpha_model}  nonfinite frames {nonfinite}"
    )
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "run": a.run,
                "ckpt": a.ckpt,
                "ckpt_iteration": cfg.get("ckpt_iteration"),
                "accumulate": accumulate,
                "scale": a.scale,
                "segment": {
                    "scene": a.scene,
                    "sequence": a.sequence,
                    "first": a.first,
                    "last": a.last,
                    "frames": len(seg),
                },
                "metric_field": field.name,
                "input_pipeline": pipe,
                "definitions": {
                    "alpha_hist": "1 - g, g = sum dy . R_m[delta] / sum R_m[delta]^2, dy = output response to delta ~ N(0, 1e-3^2) i.i.d. added to the carried colour state, R_m = the model's own HR history warp; = alpha for a scalar blend",
                    "alpha_eff_bicubic": "sum (y_t - h_t)(c_t - h_t) / sum (c_t - h_t)^2, h_t = R[y_{t-1}] (non-jittered metric field), c_t = registered bicubic; NOT a freeze indicator",
                    "copy_frac": "share of mask pixels with max_c |y_t - h_t| < 0.5/255",
                    "ref_copy_frac": "same for the reference, max_c |gt_t - R[gt_{t-1}]| < 0.5/255",
                    "copy_rms_u8": "RMS (y_t - h_t) x 255",
                    "ref_rms_u8": "RMS (gt_t - R[gt_{t-1}]) x 255",
                    "masks": "valid = in-bounds; static = valid and reference unchanged (max_c |gt_t - R[gt_{t-1}]| < 0.5/255)",
                },
                "result": res,
                "pixel_std_float": std_f,
                "pixel_std_u8": std_u8,
                "model_alpha": alpha_model,
                "nonfinite_frames": nonfinite,
                "per_frame": per_frame,
                "provenance": provenance(scale=a.scale),
            },
            indent=1,
        )
    )
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
