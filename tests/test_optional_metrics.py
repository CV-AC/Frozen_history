import numpy as np
import pytest
import torch


def test_flip_identity_has_zero_error():
    flip = pytest.importorskip("flip_evaluator")
    image = np.full((32, 48, 3), 0.5, dtype=np.float32)
    _, error, _ = flip.evaluate(
        image,
        image,
        "LDR",
        inputsRGB=True,
        applyMagma=False,
        computeMeanError=True,
        parameters={"ppd": 67},
    )
    assert float(error) == 0


def test_colorvideovdp_identity_has_maximum_jod():
    pycvvdp = pytest.importorskip("pycvvdp")
    from pycvvdp.video_source import video_source_array

    metric = pycvvdp.cvvdp(
        display_name="standard_fhd", device=torch.device("cpu"), quiet=True
    )
    video = np.full((3, 32, 48, 3), 128, dtype=np.uint8)
    source = video_source_array(
        video, video, 60, dim_order="FHWC", display_photometry=metric.display_photometry
    )
    value, _ = metric.predict_video_source(source)
    assert float(value) == pytest.approx(10)
