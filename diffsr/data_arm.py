from __future__ import annotations
from diffsr.paths import (
    REPOSITORY,
    DATA_ROOT,
    QRISP_RAW,
    QRISP_LMDB,
    STSS_PACKED,
    ARM_TEST,
    STSSNET_UTILS,
)
import hashlib
import math
from pathlib import Path
import numpy as np
import torch

__all__ = [
    "ARM_TEST",
    "DEPTH_NEAR01",
    "ArmSequence",
    "tonemap",
    "srgb_oetf",
    "arm_depth01",
]
DEPTH_NEAR01 = 0.0013
HDR_MAX = 65504.0
SCENE = "Bistro"


def srgb_oetf(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(
        x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12).pow(1.0 / 2.4) - 0.055
    )


def tonemap(
    x: torch.Tensor, exposure: float = 2.0, encode: str = "none"
) -> torch.Tensor:
    e = math.exp(float(exposure))
    y = x.float().clamp(0.0, HDR_MAX / e) * e
    y = y / (1.0 + y)
    if encode == "srgb":
        y = srgb_oetf(y)
    elif encode != "none":
        raise ValueError(f"encode must be 'none' or 'srgb', not {encode!r}")
    return y.clamp(0.0, 1.0)


def arm_depth01(d: torch.Tensor) -> torch.Tensor:
    return (DEPTH_NEAR01 / (1.0 - d.float()).clamp_min(1e-12)).clamp(0.0, 1.0)


class ArmSequence:
    scale = 2

    def __init__(
        self,
        path: str | Path = ARM_TEST,
        exposure: float = 2.0,
        encode: str = "none",
        max_frames: int = 0,
        crop: tuple[int, int, int, int] | None = None,
        motion_source: str = "lr",
    ) -> None:
        from safetensors import safe_open

        self.path = Path(path)
        (self.exposure, self.encode) = (float(exposure), encode)
        if motion_source not in ("lr", "hr_centre"):
            raise ValueError("motion_source must be 'lr' or 'hr_centre'")
        self.motion_source = motion_source
        self._f = safe_open(str(self.path), framework="pt")
        self.meta = self._f.metadata() or {}
        n = int(self._f.get_slice("colour_linear").get_shape()[0])
        if max_frames:
            n = min(n, int(max_frames))
        self.n = n
        self.crop = crop
        self.jitter_raw = self._f.get_tensor("jitter").reshape(-1, 2)[:n].float()
        self.depth_params = self._f.get_tensor("depth_params").reshape(-1, 4)[:n]
        self.z_near = self._f.get_tensor("zNear").reshape(-1)[:n]
        self.exposure_file = self._f.get_tensor("exposure").reshape(-1)[:n]
        self.view_proj = self._f.get_tensor("viewProj").reshape(-1, 4, 4)[:n]
        self.frames = [
            {
                "scene": SCENE,
                "sequence": "0000",
                "frame": k,
                "n": k,
                "key": f"{SCENE}/0000/{k:04d}",
            }
            for k in range(n)
        ]

    def __len__(self) -> int:
        return self.n

    def runs(self) -> dict[str, list[list[int]]]:
        return {SCENE: [list(range(self.n))]}

    def jitter_yx(self, i: int) -> torch.Tensor:
        (dy, dx) = self.jitter_raw[i]
        return torch.stack((-dy, dx))

    def _get(self, key: str, i: int) -> torch.Tensor:
        return self._f.get_slice(key)[i : i + 1][0]

    def _crop(self, t: torch.Tensor, s: int) -> torch.Tensor:
        if self.crop is None:
            return t
        (top, left, h, w) = self.crop
        return t[..., top * s : (top + h) * s, left * s : (left + w) * s]

    def __getitem__(self, i: int) -> dict:
        lr = tonemap(self._get("colour_linear", i), self.exposure, self.encode)
        hr = tonemap(self._get("ground_truth_linear", i), self.exposure, self.encode)
        depth = arm_depth01(self._get("depth", i))
        m_hr = -self._get("motion", i).float()
        if self.motion_source == "lr":
            m_lr = -self._get("motion_lr", i).float()
        else:
            m_lr = m_hr[:, ::2, ::2] / 2.0
        return {
            "lr": self._crop(lr, 1),
            "hr": self._crop(hr, 2),
            "depth": self._crop(depth, 1),
            "motion": self._crop(m_lr, 1),
            "motion_hr": self._crop(m_hr, 2),
            "jitter": self.jitter_yx(i),
            "jitter_raw": self.jitter_raw[i],
            "key": self.frames[i]["key"],
            "scene": SCENE,
        }

    def provenance(self) -> dict:
        st = self.path.stat()
        head = self.path.open("rb").read(1 << 20)
        return {
            "file": str(self.path),
            "bytes": st.st_size,
            "md5_first_MiB": hashlib.md5(head).hexdigest(),
            "metadata": self.meta,
            "frames": self.n,
            "exposure_log": self.exposure,
            "encode": self.encode,
            "exposure_in_file": float(self.exposure_file[0]),
            "motion_source": self.motion_source,
            "depth_near01": DEPTH_NEAR01,
            "depth_params_frame0": self.depth_params[0].tolist(),
            "tonemap": "Reinhard(exp(E)*x) (ng-model-gym process_nss_data)"
            + (" + sRGB OETF" if self.encode == "srgb" else ""),
            "jitter_map": "jitter_yx = (-dy, +dx) of the file's (dy, dx)",
            "motion_map": "m = -(ch0, ch1), prev = cur + m",
        }


def _selftest() -> None:
    data = ArmSequence(max_frames=3)
    it = data[1]
    for k in ("lr", "hr", "depth", "motion", "motion_hr"):
        v = it[k]
        print(k, tuple(v.shape), float(v.min()), float(v.max()))
    print("jitter_raw", it["jitter_raw"].tolist(), "jitter_yx", it["jitter"].tolist())
    print(
        "depth_params",
        data.depth_params[:2].tolist(),
        "zNear",
        data.z_near[:2].tolist(),
    )
    print(data.provenance())


if __name__ == "__main__":
    _selftest()
