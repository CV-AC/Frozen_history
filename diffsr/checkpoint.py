from __future__ import annotations
import inspect
import json
from pathlib import Path
import torch
from .rrta import RRTA_SIZES, RRTANet, create_rrta

__all__ = [
    "load_run",
    "build_model",
    "load_state_dict_compat",
    "LEGACY",
    "input_pipeline",
]
LEGACY: dict[str, object] = {
    "rectify": "learned",
    "recurrent": True,
    "phase_blend": False,
    "band_lowres": False,
    "feature_dim": 0,
    "norm": "group",
    "depth_edge": False,
    "stem": "jitter1x1",
    "jitter_head": False,
    "phase_memory": False,
    "block": "dw7",
    "band_split": False,
    "hr_refine": 0,
    "hr_layers": 3,
    "frames_in": 1,
    "squeeze": 0,
    "accumulate": "blend",
    "reject": "none",
    "kernel": "gauss",
    "accum_cap": 0.0,
    "static_gate": 0.0,
    "global_motion": False,
    "dejitter": False,
    "first_order": False,
    "accum_alpha_min": 0.0,
    "static_cap": 64.0,
    "dual_rate": 0.0,
    "canonical_phase": False,
    "layer_scale": 0.1,
    "gamma_max": 1.0,
    "alpha_min": 1.0 / 16.0,
    "phase_weight_init": -2.0,
    "lam": 4.0,
    "ycocg": True,
    "motion_dilation": True,
    "phase_geometry": "legacy",
    "registered_skip": False,
    "lr_rejitter": False,
    "depth_jit": True,
    "fp32_accum": False,
    "skip_dead": False,
    "readout": "accum",
    "vcap_max": 0.0,
    "vcap_min": 4.0,
    "vcap_v0": 2.0,
    "hist_clip": "none",
    "smr_sigma": 0.6,
    "smr_lambda_init": 1.0,
    "smr_lambda_eps": 0.01,
    "smr_input_eps": 0.05,
    "smr_depth": "far0.1",
    "feature_bound": 0.0,
    "consistency_gate": "none",
    "state_mode": "linear",
    "gate_shorten": "off",
    "gate_shorten_init": 0.01,
    "lr_unshuffle": 1,
    "body": "plain",
    "body_inner": "conv3",
    "body_width_mult": 2.0,
    "body_unet_levels": 1,
    "body_unet_blocks": "",
    "body_pconv_ratio": 4,
    "body_expand": 2.0,
    "body_rep_branches": "ecb",
    "body_act": "gelu",
    "body_dw_kernel": 3,
    "body_dilations": "1,2",
    "body_attn_idx": "",
    "body_attn_dim": 16,
}
_POST_RECORD = (
    "phase_geometry",
    "registered_skip",
    "lr_rejitter",
    "depth_jit",
    "fp32_accum",
    "skip_dead",
    "readout",
    "vcap_max",
    "vcap_min",
    "vcap_v0",
    "hist_clip",
    "smr_sigma",
    "smr_lambda_init",
    "smr_lambda_eps",
    "smr_input_eps",
    "smr_depth",
    "feature_bound",
    "consistency_gate",
    "state_mode",
    "gate_shorten",
    "gate_shorten_init",
    "lr_unshuffle",
    "body",
    "body_inner",
    "body_width_mult",
    "body_unet_levels",
    "body_unet_blocks",
    "body_pconv_ratio",
    "body_expand",
    "body_rep_branches",
    "body_act",
    "body_dw_kernel",
    "body_dilations",
    "body_attn_idx",
    "body_attn_dim",
)
AMBIGUOUS = ("phase_memory", "band_split", "block", "stem", "depth_edge")
RENAMES = (("dw.", "spatial."), ("mlp.", "mix."))
_OPTIONAL = {
    "rectifier.phase_weight": "phase_blend",
    "rectifier.advantage_weight": "phase_memory",
    "rectifier.band_offset": "band_split",
}


def _missing_options() -> set[str]:
    names = set(inspect.signature(RRTANet.__init__).parameters)
    return names - {"self", "scale", "dim", "blocks", "mlp_ratio"} - set(LEGACY)


def remap(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        if key.startswith("body.") or ".body." in key:
            for old, new_ in RENAMES:
                if old in key:
                    key = key.replace(old, new_)
                    break
        out[key] = value
    return out


def load_state_dict_compat(model, state: dict):
    state = remap(state)
    (missing, unexpected) = model.load_state_dict(state, strict=False)
    bad = []
    for key in missing:
        gate = _OPTIONAL.get(key)
        if gate is None or getattr(model.rectifier, gate, True):
            bad.append(key)
    if bad or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing {bad}, unexpected {list(unexpected)}"
        )
    return missing


def build_model(cfg: dict, device=None, **override) -> torch.nn.Module:
    if cfg.get("arch", "rrta") == "qrisp":
        from .qrisp_model import create_qrisp

        kwargs = dict(cfg["model_kwargs"])
        kwargs.update(override)
        model = create_qrisp(**kwargs)
        return model.to(device) if device is not None else model
    unanswered = _missing_options()
    if unanswered:
        raise RuntimeError(
            f"model options with no legacy default: {sorted(unanswered)} -- add them to diffsr.checkpoint.LEGACY"
        )
    if "model_kwargs" in cfg:
        kwargs = dict(cfg["model_kwargs"])
        (size, scale) = (kwargs.pop("size"), kwargs.pop("scale"))
        for key in _POST_RECORD:
            kwargs.setdefault(key, LEGACY[key])
    else:
        (size, scale) = (cfg["size"], cfg["scale"])
        kwargs = {k: cfg.get(k, d) for (k, d) in LEGACY.items()}
        for key in ("dim", "blocks"):
            if cfg.get(key):
                kwargs[key] = cfg[key]
    kwargs.update(override)
    model = create_rrta(size, scale, **kwargs)
    return model.to(device) if device is not None else model


def input_pipeline(cfg: dict) -> dict:

    def get(key, default):
        value = cfg.get(key)
        return default if value is None else value

    scale = cfg.get("scale")
    if scale is None:
        scale = cfg.get("model_kwargs", {}).get("scale")
    return {
        "scale": int(scale) if scale is not None else None,
        "dataset": str(get("dataset", "qrisp")),
        "colour": get("qrisp_colour", None),
        "motion_jit": bool(get("motion_jit", True)),
        "motion_lerp": float(get("motion_lerp", 0.0)),
        "depth_jit": bool(get("depth_jit", True)),
    }


def load_run(
    run: Path, device=None, ckpt_name: str = "last.pt"
) -> tuple[torch.nn.Module, dict]:
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    ckpt = torch.load(run / ckpt_name, map_location="cpu", weights_only=False)
    if cfg.get("arch", "rrta") == "qrisp":
        model = build_model(cfg, device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        return (model, {**cfg, "ckpt_iteration": ckpt.get("iteration")})
    state = remap(ckpt["model"])
    if "model_kwargs" in cfg:
        model = build_model(cfg, device)
        load_state_dict_compat(model, state)
        model.eval()
        cfg = {**cfg, "ckpt_iteration": ckpt.get("iteration")}
        return (model, cfg)
    decided = {}
    if "band_split" not in cfg and "rectifier.band_offset" in state:
        decided["band_split"] = float(state["rectifier.band_offset"]) != 0.0
    candidates = [decided]
    for key in AMBIGUOUS:
        if key in cfg or key in decided:
            continue
        values = {
            "phase_memory": [False, True],
            "band_split": [False, True],
            "depth_edge": [False, True],
            "block": ["dw7", "conv3"],
            "stem": ["jitter1x1", "conv3", "conv3_film"],
        }[key]
        candidates = [{**c, key: v} for c in candidates for v in values]
    errors = []
    for trial in candidates:
        model = build_model(cfg, device, **trial)
        try:
            load_state_dict_compat(model, state)
        except RuntimeError as exc:
            errors.append((trial, str(exc)[:160]))
            continue
        model.eval()
        cfg = {**cfg, **trial, "_inferred": sorted(trial)}
        model.eval()
        return (model, cfg)
    raise RuntimeError(
        f"{run.name}: no option combination loads this checkpoint; closest of {len(errors)} tried: {(errors[0][1] if errors else 'none')}"
    )
