import json
from pathlib import Path

import pytest
import torch

from diffsr import taau
from protocol_taau import load_choice

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_baseline_selection_metadata_and_step(scale):
    configuration, metadata = load_choice(scale)
    assert metadata["chosen"]
    assert len(metadata["train_runs"]) == 30
    model = taau.TAAU(scale, configuration)
    rgb = torch.full((1, 3, 4, 6), 0.5)
    depth = torch.full((1, 1, 4, 6), 0.5)
    motion = torch.zeros(1, 2, 4, 6)
    jitter = torch.zeros(1, 2)
    state = None
    for _ in range(3):
        result, state, _ = model.step(rgb, depth, motion, jitter, state)
        assert result.shape == (1, 3, 4 * scale, 6 * scale)
        torch.testing.assert_close(
            result, torch.full_like(result, 0.5), atol=1e-5, rtol=0
        )
    clamp = json.loads(
        (ROOT / f"configs/baselines/taa_clamp_x{scale}.json").read_text()
    )
    taau.TAAUConfig(**clamp["chosen_config"])
