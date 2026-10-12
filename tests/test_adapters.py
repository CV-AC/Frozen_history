import numpy as np
import pytest
import torch

from diffsr.data_arm import ArmSequence, tonemap
from diffsr.data_stss import stss_depth
import pack_stss


@pytest.mark.parametrize("channels,motion_at", [(33, 30), (35, 31)])
def test_stss_source_layout_and_geometry(monkeypatch, channels, motion_at):
    monkeypatch.setattr(pack_stss, "LR_SHAPE", (4, 6))
    source = np.zeros((4, 6, channels), dtype=np.float16)
    source[..., :3] = [0.1, 0.2, 0.3]
    source[..., 26] = 8
    source[..., motion_at : motion_at + 3] = [0, 1, 2]
    packed = pack_stss.compact(source)
    assert packed.shape == (4, 6, 7)
    np.testing.assert_array_equal(packed[..., :3], source[..., :3])
    np.testing.assert_array_equal(
        pack_stss.motion_yx(packed), np.broadcast_to([1, -2], (4, 6, 2))
    )
    np.testing.assert_array_equal(stss_depth(packed[..., 3]), np.full((4, 6), 0.5))


def test_arm_capture_adapter_and_tonemap(tmp_path):
    from safetensors.torch import save_file

    tensors = {
        "colour_linear": torch.ones(2, 3, 4, 6),
        "ground_truth_linear": torch.ones(2, 3, 8, 12),
        "depth": torch.full((2, 1, 4, 6), 0.5),
        "motion_lr": torch.ones(2, 2, 4, 6),
        "motion": torch.ones(2, 2, 8, 12),
        "jitter": torch.tensor([[0.25, -0.125], [0.25, -0.125]]),
        "depth_params": torch.zeros(2, 4),
        "zNear": torch.full((2,), 0.0013),
        "exposure": torch.full((2,), 2),
        "viewProj": torch.eye(4).repeat(2, 1, 1),
    }
    path = tmp_path / "synthetic.safetensors"
    save_file(tensors, path)
    sequence = ArmSequence(path)
    frame = sequence[0]
    torch.testing.assert_close(frame["jitter"], torch.tensor([-0.25, -0.125]))
    torch.testing.assert_close(frame["motion"], -tensors["motion_lr"][0])
    torch.testing.assert_close(frame["depth"], torch.full((1, 4, 6), 0.0026))
    torch.testing.assert_close(frame["lr"], tonemap(tensors["colour_linear"][0]))
    assert len(sequence) == 2
