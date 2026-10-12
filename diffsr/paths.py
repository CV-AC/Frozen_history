import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FROZEN_HISTORY_DATA", REPOSITORY / "data"))
QRISP_RAW = Path(os.environ.get("QRISP_RAW", DATA_ROOT / "qrisp" / "raw"))
QRISP_LMDB = Path(os.environ.get("QRISP_LMDB", DATA_ROOT / "qrisp" / "lmdb"))
STSS_PACKED = Path(os.environ.get("STSS_PACKED", DATA_ROOT / "stss" / "packed"))
ARM_TEST = Path(
    os.environ.get(
        "ARM_TEST", DATA_ROOT / "arm" / "nss_test_full_resolution_sample.safetensors"
    )
)
STSSNET_UTILS = Path(
    os.environ.get("STSSNET_UTILS", REPOSITORY / "external" / "STSSNet" / "utils")
)
