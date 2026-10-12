from __future__ import annotations
import contextlib
import torch
import torch.nn.functional as F

SECTIONS = (
    "lr_dilate",
    "hr_warp",
    "lr_warp",
    "proj",
    "lr_edges_cat",
    "trunk",
    "heads",
    "skip",
    "accum",
    "aux",
    "guard",
)


def check_config(m, allow_fd0: bool = False) -> None:
    want = dict(
        accumulate="normalized",
        registered_skip=True,
        lr_rejitter=True,
        depth_jit=False,
        fp32_accum=True,
        skip_dead=True,
        readout="accum",
        kernel="gauss",
        reject="none",
        hist_clip="none",
        phase_geometry="image",
        motion_dilation=True,
        recurrent=True,
        global_motion=False,
        dejitter=False,
        first_order=False,
        canonical_phase=False,
        depth_edge=True,
        phase_memory=True,
        frames_in=1,
        needs_m2=False,
        state_guard=True,
    )
    bad = {k: getattr(m, k) for (k, v) in want.items() if getattr(m, k) != v}
    for k in (
        "vcap_max",
        "static_gate",
        "accum_alpha_min",
        "dual_rate",
        "accum_cap",
        "feature_bound",
    ):
        if float(getattr(m, k)) != 0.0:
            bad[k] = getattr(m, k)
    for k in (
        "head_film",
        "refine",
        "head_fast",
        "head_delta",
        "head_mu",
        "head_clip",
        "head_lambda",
        "reject_gain",
    ):
        if getattr(m, k, None) is not None:
            bad[k] = "present"
    if m.feature_dim <= 0 and (not allow_fd0) or m.head_weight is None:
        bad["feature_dim/head_weight"] = (m.feature_dim, m.head_weight)
    if bad:
        raise ValueError(f"lean_step does not cover this configuration: {bad}")


def _hi(t):
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


def lean_step(
    m, rgb, depth, motion, jitter, state, *, aux: bool = True, sections: bool = False
):
    import diffsr.rrta as R

    if state is None:
        return type(m).step(m, rgb, depth, motion, jitter, None)
    sec = (
        torch.profiler.record_function
        if sections
        else lambda name: contextlib.nullcontext()
    )
    (batch, _, height, width) = rgb.shape
    scale = m.scale
    hr_size = (height * scale, width * scale)
    with sec("lr_dilate"):
        motion = R.dilate_motion(motion, depth)
        (
            previous_colour,
            previous_confidence,
            previous_depth,
            previous_feature,
            previous_reach,
            previous_rgb,
            previous_jitter,
            previous_fast,
            previous_m2,
            previous_jitter_full,
            previous_m1,
        ) = state
        motion_lr = m._lr_motion(motion, jitter, previous_jitter_full)
    with sec("hr_warp"):
        motion_hr = F.interpolate(motion, size=hr_size, mode="nearest") * scale
        (warped, inside) = R.warp(
            torch.cat(
                (
                    previous_colour,
                    previous_confidence,
                    previous_reach,
                    previous_m2,
                    previous_m1,
                ),
                dim=1,
            ),
            motion_hr,
        )
        (colour_hist, confidence) = (warped[:, :3], warped[:, 3:4] * inside)
        m2_hist = warped[:, :0]
        m1_hist = warped[:, :0]
        reach_hist = warped[:, 4:5] * inside + (1.0 - inside)
        valid = inside
    with sec("lr_warp"):
        (warped_depth, depth_inside) = R.warp(previous_depth, motion)
        (warped_feat, feat_inside) = R.warp(previous_feature, motion_lr)
        feature_hist = warped_feat * feat_inside
        disocclusion = (depth - warped_depth).abs() * depth_inside
    with sec("proj"):
        history_lr = m.history_proj(F.pixel_unshuffle(colour_hist, scale))
        confidence_lr = m.confidence_proj(F.pixel_unshuffle(confidence, scale))
        reach_lr = m.reach_proj(F.pixel_unshuffle(reach_hist, scale))
    with sec("lr_edges_cat"):
        parts = [
            rgb,
            depth,
            motion,
            disocclusion,
            history_lr,
            confidence_lr,
            reach_lr,
            feature_hist,
            R.depth_edges(depth),
        ]
        stem_in = torch.cat(parts, dim=1)
    with sec("trunk"):
        x = m.body(m.stem(stem_in, jitter))
    with sec("heads"):
        colour = m.head_colour(x)
        candidate = F.pixel_shuffle(colour, scale)
    with sec("skip"):
        skip = m._skip(rgb, jitter, hr_size)
    with sec("accum"):
        candidate = candidate + skip
        phase = R.sample_phase(
            scale, jitter.to(rgb.dtype), *hr_size, geometry=m.phase_geometry
        )
        candidate = _hi(candidate)
        colour_hist = _hi(colour_hist)
        confidence = _hi(confidence)
        valid = _hi(valid)
        phase = _hi(phase)
    with sec("heads"):
        stat = F.pixel_shuffle(m._head(m.head_weight, x), scale)
    with sec("accum"):
        reliability = torch.sigmoid(stat[:, 0:1])
        retention_logit = stat[:, 1:2]
        retention = torch.sigmoid(retention_logit)
        radius = phase.pow(2).sum(1, keepdim=True).sqrt() / scale
        width_k = F.softplus(m.kernel_width) + 0.25
        weight = torch.exp(-(radius / width_k).pow(2) * 0.5) * reliability
        s_hist = colour_hist * confidence
        w_hist = confidence
        cap = 1.0 / m.rectifier.alpha_min
        keep = retention * valid
        s_new = keep * s_hist + weight * candidate
        w_new = (keep * w_hist + weight).clamp_min(0.0001)
        excess = (w_new / cap).clamp_min(1.0)
        s_new = s_new / excess
        w_new = w_new / excess
        out = s_new / w_new
        new_confidence = w_new
    diag = {}
    if aux:
        with sec("aux"):
            alpha = weight / w_new
            diag["w_mean"] = w_new.mean().detach()
            diag["alpha_mean"] = alpha.mean().detach()
            diag["ret_sat"] = (retention >= 0.999).to(w_new.dtype).mean().detach()
            disagreement = torch.zeros_like(w_new)
    new_reach = radius
    state_colour = out
    with sec("heads"):
        new_feature = m.head_feature(x)
    new_m2 = out[:, :0]
    m1_new = m1_hist
    new_fast = state_colour
    carried_jitter = jitter.float()
    with sec("guard"):
        (
            state_colour,
            new_confidence,
            new_feature,
            new_reach,
            new_fast,
            new_m2,
            m1_new,
            guard_px,
        ) = m._guard_state(
            state_colour,
            new_confidence,
            new_feature,
            new_reach,
            new_fast,
            new_m2,
            m1_new,
            skip,
            scale,
        )
    new_state = (
        state_colour,
        new_confidence,
        depth,
        new_feature,
        new_reach,
        None,
        None,
        new_fast,
        new_m2,
        carried_jitter,
        m1_new,
    )
    if not aux:
        return (out, new_state, {})
    a = {
        "alpha": alpha,
        "disagreement": disagreement,
        "candidate": candidate,
        "phase": phase,
        "diag": diag,
        "guard_px": guard_px,
    }
    return (out, new_state, a)
