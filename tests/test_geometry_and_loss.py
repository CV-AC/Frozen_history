import pytest
import torch
import torch.nn.functional as F

from diffsr.engine_seq import motion_to_hr, sequence_loss
from diffsr.rrta import jitter_image_offset, registered_upsample, warp


def test_backward_motion_reads_previous_pixel_and_masks_boundary():
    image = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    motion = torch.zeros(1, 2, 4, 5)
    motion[:, 1] = 1
    result, valid = warp(image, motion)
    torch.testing.assert_close(result[..., :-1], image[..., 1:])
    assert bool((valid[..., :-1] == 1).all())
    assert bool((valid[..., -1] == 0).all())


def test_jitter_convention_and_registered_zero_offset():
    jitter = torch.tensor([[0.25, -0.125]])
    torch.testing.assert_close(
        jitter_image_offset(jitter), torch.tensor([[-0.25, -0.125]])
    )
    rgb = torch.linspace(0.1, 0.9, 60).reshape(1, 3, 4, 5)
    actual = registered_upsample(rgb, torch.zeros(1, 2), 3)
    expected = F.interpolate(
        rgb, scale_factor=3, mode="bicubic", align_corners=False
    ).clamp(0, 1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("temporal_warp", ["motion", "none"])
@pytest.mark.parametrize("temporal_objective", ["l1", "l2"])
def test_derivative_loss_matches_definition_and_gradients(
    temporal_warp, temporal_objective
):
    generator = torch.Generator().manual_seed(7)
    pred = torch.rand(1, 4, 3, 8, 10, generator=generator, requires_grad=True)
    ref = torch.rand(1, 4, 3, 8, 10, generator=generator)
    motion = torch.full((1, 4, 2, 4, 5), 0.25)
    actual, stats = sequence_loss(
        pred,
        ref,
        motion,
        2,
        1,
        0.25,
        objective="l2",
        temporal_objective=temporal_objective,
        temporal_warp=temporal_warp,
    )
    p, r = pred[:, 1:], ref[:, 1:]
    prev_p, prev_r = p[:, :-1].flatten(0, 1), r[:, :-1].flatten(0, 1)
    if temporal_warp == "motion":
        field = motion_to_hr(motion[:, 2:].flatten(0, 1), 2)
        prev_p, mask = warp(prev_p, field)
        prev_r, _ = warp(prev_r, field)
    else:
        mask = torch.ones_like(prev_p[:, :1])
    delta = (p[:, 1:].flatten(0, 1) - prev_p) - (r[:, 1:].flatten(0, 1) - prev_r)
    error = delta.square() if temporal_objective == "l2" else delta.abs()
    expected = (p - r).square().mean() + 0.25 * (error * mask).sum() / mask.expand_as(
        error
    ).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred, retain_graph=True)[0],
        torch.autograd.grad(expected, pred)[0],
    )
    assert stats["temporal"] >= 0
