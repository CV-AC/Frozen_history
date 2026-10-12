# Upstream resources and licenses

The MIT license covers this project's code and documentation. It does not change the licenses of dependencies, datasets, pretrained metric networks, scene assets or rendered commercial baselines.

| Resource | Relationship to this release | Source and terms |
| --- | --- | --- |
| QRISP | Dataset adapter and independent QRISP-style network implementation; no dataset or official model is included | [Official dataset](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset); provider product agreement, research use |
| STSSNet | Dataset conventions and optional externally loaded MATLAB-style metric; no upstream source or checkpoints are vendored | [Official implementation](https://github.com/ryanhe312/STSSNet-AAAI2024), MIT for its code; obtain data under the separately supplied provider terms |
| Arm Neural Graphics Dataset | Zero-shot adapter; no safetensors or models included | [Official dataset card](https://huggingface.co/datasets/Arm/neural-graphics-dataset), Arm AI Model Community License |
| Arm Neural Graphics Model Gym | Reference for dataset conventions and public release organization; not a runtime dependency | [Official repository](https://github.com/arm/neural-graphics-model-gym), Apache-2.0 for code |
| PyTorch / torchvision | Model execution and metric networks | [PyTorch](https://github.com/pytorch/pytorch), BSD-style; [torchvision](https://github.com/pytorch/vision), BSD-3-Clause |
| Triton | Optional generated GPU kernels | [Official repository](https://github.com/triton-lang/triton), MIT |
| torchmetrics / LPIPS | Optional perceptual evaluation | [torchmetrics](https://github.com/Lightning-AI/torchmetrics), Apache-2.0; [LPIPS](https://github.com/richzhang/PerceptualSimilarity), BSD-2-Clause |
| FLIP | Optional perceptual evaluation | [Official repository](https://github.com/NVlabs/flip), BSD-3-Clause |
| ColorVideoVDP | Optional sequence/display evaluation | [Official repository](https://github.com/gfxdisp/ColorVideoVDP), see the installed version's license |

Additional dependencies retain their installed distribution notices. If external code or assets are later vendored, include their original copyright and license files. Public upstream attribution is intentionally retained independently of the removal of this project's personal metadata.

The README includes selected paper figures with QRISP scene renderings and supplied comparison captures. Their underlying dataset and scene-asset terms remain separate from the project code license. See [MEDIA.md](docs/MEDIA.md) for the figure inventory and evaluation notes. Raw datasets and third-party model weights are not included.

The release structure was checked against [STSSNet](https://github.com/ryanhe312/STSSNet-AAAI2024), [NeRF-SR](https://github.com/cwchenwang/NeRF-SR), and [Arm Model Gym](https://github.com/arm/neural-graphics-model-gym): separate installation, dataset acquisition, train/evaluate commands, configurations, citation and licensing.
