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
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from diffsr import taau

SELECT = REPOSITORY / "configs" / "baselines" / "taau_x{scale}.json"


def load_choice(scale: int, path: str | None = None) -> tuple[taau.TAAUConfig, dict]:
    p = Path(path) if path else Path(str(SELECT).format(scale=scale))
    sel = json.loads(p.read_text())
    if int(sel["scale"]) != int(scale):
        raise SystemExit(f"{p} is a x{sel['scale']} selection, not x{scale}")
    cfg = taau.TAAUConfig(**sel["chosen_config"])
    info = {
        "selection_file": str(p),
        "chosen": sel["chosen"],
        "train_psnr_y": sel["chosen_psnr_y"],
        "train_runs": sel["train_runs"],
        "n_configs": sel["n_configs"],
        "selection_md5": sel.get("taau_md5"),
    }
    return (cfg, info)


class TAAULoader:
    def __init__(self, scale: int, cfg: taau.TAAUConfig):
        (self.scale, self.cfg) = (int(scale), cfg)
        self.models: list[taau.TAAU] = []

    def __call__(self, path, device=None, ckpt_name="last.pt"):
        name = Path(path).name
        allowed = {f"taau_x{self.scale}", "taau"}
        if name not in allowed:
            raise SystemExit(
                f"TAAU loader asked for {name!r}; expected one of {sorted(allowed)}"
            )
        model = taau.TAAU(self.scale, self.cfg)
        self.models.append(model)
        cfg = dict(
            taau.PIPELINE,
            scale=self.scale,
            ckpt_iteration=None,
            model="classical TAAU (diffsr/taau.py), no learned parameters",
            taau_config=self.cfg.to_dict(),
        )
        return (model, cfg)

    def output_stats(self) -> dict:
        st = [m.stats for m in self.models]
        return {
            "steps": sum((s["steps"] for s in st)),
            "nonfinite_steps": sum((s["nonfinite_steps"] for s in st)),
            "min": min((s["min"] for s in st)) if st else None,
            "max": max((s["max"] for s in st)) if st else None,
        }


def taau_block(cfg, info, loader) -> dict:
    return {
        "method": "classical TAAU baseline (Karis 2014 / Salvi 2016 / FSR 2 design), zero learned parameters",
        "config": cfg.to_dict(),
        "config_name": cfg.name(),
        **info,
        "input_pipeline": dict(taau.PIPELINE),
        "output_stats": loader.output_stats(),
        "md5": {**taau.code_md5(), "scripts/protocol_taau.py": _md5(Path(__file__))},
    }


def _md5(p: Path) -> str:
    import hashlib

    return hashlib.md5(p.read_bytes()).hexdigest()


def run_protocol(scale: int, cfg, info, out: Path, smoke: int = 0) -> None:
    import protocol_full as pf

    loader = TAAULoader(scale, cfg)
    pf.load_run = loader
    argv = sys.argv
    sys.argv = [
        str(Path(__file__).resolve()),
        f"taau_x{scale}",
        "--scale",
        str(scale),
        "--stride",
        "1",
        "--warmup",
        "8",
        "--lpips",
        "--out",
        str(out),
    ] + (["--max-frames", str(smoke)] if smoke else [])
    try:
        pf.main()
    finally:
        sys.argv = argv
    d = json.loads(out.read_text())
    d["protocol"]["taau"] = taau_block(cfg, info, loader)
    out.write_text(json.dumps(d, indent=1))
    print(
        f"  added taau provenance to {out}; output stats {d['protocol']['taau']['output_stats']}",
        flush=True,
    )


def run_static(scale: int, cfg, info, out: Path) -> None:
    import static_segment as ss

    loader = TAAULoader(scale, cfg)
    ss.load_run = loader
    argv = sys.argv
    sys.argv = [
        str(Path(__file__).resolve()),
        f"taau_x{scale}",
        "--scale",
        str(scale),
        "--out",
        str(out),
    ]
    try:
        ss.main()
    finally:
        sys.argv = argv
    d = json.loads(out.read_text())
    d["taau"] = taau_block(cfg, info, loader)
    out.write_text(json.dumps(d, indent=1))
    print(f"  added taau provenance to {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--what", default="both", choices=("protocol", "static", "both"))
    ap.add_argument(
        "--select",
        default=None,
        help="selection JSON (default configs/baselines/taau_x{s}.json)",
    )
    ap.add_argument("--tag", default="taau")
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="SMOKE ONLY: protocol_full --max-frames N; outputs go to results/taau_smoke",
    )
    args = ap.parse_args()
    (cfg, info) = load_choice(args.scale, args.select)
    print(
        f"TAAU x{args.scale}: {cfg.name()}  (train PSNR-Y {info['train_psnr_y']:.3f}, {info['selection_file']})",
        flush=True,
    )
    ref = ROOT / ("results/taau_smoke" if args.smoke else "results")
    if args.what in ("protocol", "both"):
        run_protocol(
            args.scale,
            cfg,
            info,
            ref / f"{args.tag}_taau_x{args.scale}_protocol.json",
            args.smoke,
        )
    if args.what in ("static", "both"):
        run_static(
            args.scale, cfg, info, ref / f"{args.tag}_taau_x{args.scale}_static.json"
        )


if __name__ == "__main__":
    main()
