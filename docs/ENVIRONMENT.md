# Environment

The source archive records Python 3.12.13 and an environment inheriting system packages. It does not contain a complete original dependency lock. The supplied requirements therefore pin the directly observed release-validation stack, not an unverified historical environment.

| Component | Release version |
| --- | --- |
| Python | 3.12.13 used for validation; package requires ≥3.11 |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| NumPy | 2.4.2 |
| OpenCV | 4.13.0.92; use the headless distribution on servers |
| LMDB | 2.3.0 |
| safetensors | 0.7.0 |
| Pillow / SciPy | 12.1.1 / 1.17.1 |
| Triton | 3.6.0, optional Linux CUDA deployment |
| torchmetrics | 1.8.2, optional LPIPS |
| ColorVideoVDP / FLIP evaluator | 0.5.7 / 1.7, optional perceptual metrics |

Use `requirements.txt` for core execution, `requirements-metrics.txt` for paper metrics, `requirements-cuda.txt` for fused timing, and `requirements-dev.txt` for tests. Requirements list direct dependencies; retain `pip freeze` output with a reproduced experiment if a complete machine-specific lock is needed.

Install the correct PyTorch wheel first. CUDA runtime compatibility depends on its build and the GPU driver. The paper latency target is an NVIDIA H200. The eager PyTorch implementation also supports CPU execution, while fused kernels require CUDA and Triton.

Do not install both `opencv-python` and `opencv-python-headless` into one environment. `pack_qrisp.py` enables OpenEXR I/O before reading motion files. Older OpenCV builds compiled without OpenEXR cannot decode those inputs.

The STSSNet metric is loaded from an external upstream checkout for the exact published comparison. Its old TensorRT training environment is not required to load `utils/matlab_metric.py`; the release uses SciPy and scikit-image for that utility. No TensorRT or proprietary SDK is required for this project's core model.

LPIPS may download a pretrained AlexNet and learned metric parameters on first use. Configure the normal PyTorch cache location when using an offline machine. Those external metric parameters are not project training checkpoints.
