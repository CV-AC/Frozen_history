import copy
import json
from pathlib import Path

import pytest
import torch

from diffsr.checkpoint import build_model, load_run
from diffsr.fast_blocks import fuse_model
from diffsr.rrta import registered_upsample

ROOT = Path(__file__).resolve().parents[1]


def small_config(scale=4, accumulate="normalized", body="plain"):
    config = json.loads((ROOT / "configs/models/qrisp_x4.json").read_text())
    config["scale"] = scale
    config["model_kwargs"].update(
        dim=8, blocks=1, scale=scale, accumulate=accumulate, body=body
    )
    return config


def inputs(scale):
    torch.manual_seed(8)
    return (
        torch.rand(1, 3, 4, 6),
        torch.full((1, 1, 4, 6), 0.5),
        torch.zeros(1, 2, 4, 6),
        torch.tensor([[0.25, -0.125]]),
    )


@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("accumulate", ["normalized", "direct"])
def test_recurrent_sequence_backward_and_checkpoint(tmp_path, scale, accumulate):
    config = small_config(scale, accumulate)
    model = build_model(config)
    values = inputs(scale)
    out, state, aux = model.step(*values, None)
    assert out.shape == (1, 3, 4 * scale, 6 * scale)
    assert bool(torch.isfinite(out).all())
    if accumulate == "direct":
        torch.testing.assert_close(
            out, registered_upsample(values[0], values[3], scale), rtol=0, atol=0
        )
        assert bool(torch.isnan(aux["alpha"]).all())
    else:
        assert bool(torch.isfinite(aux["alpha"]).all())
        assert bool((aux["alpha"] >= 0).all())
    later, _, _ = model.step(*values, state)
    later.square().mean().backward()
    assert model.head_colour.weight.grad is not None
    assert bool(torch.isfinite(model.head_colour.weight.grad).all())
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save({"model": model.state_dict(), "iteration": 3}, tmp_path / "last.pt")
    restored, metadata = load_run(tmp_path, "cpu")
    assert metadata["ckpt_iteration"] == 3
    torch.testing.assert_close(restored.step(*values, None)[0], out, rtol=0, atol=0)


def test_rep_convolution_folding_preserves_sequence():
    model = build_model(small_config(body="rep3")).eval()
    with torch.no_grad():
        model.head_colour.weight.normal_(0, 0.02)
    folded = copy.deepcopy(model)
    assert fuse_model(folded) > 0
    values = inputs(4)
    state_a = state_b = None
    with torch.inference_mode():
        for _ in range(4):
            a, state_a, _ = model.step(*values, state_a)
            b, state_b, _ = folded.step(*values, state_b)
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_direct_rejects_accumulator_only_options():
    config = small_config(accumulate="direct")
    config["model_kwargs"]["accum_cap"] = 16
    with pytest.raises(ValueError, match="direct"):
        build_model(config)
