# Getting started

Run commands from the repository root. Start with the synthetic tests before downloading datasets or launching a long training run. The release contains no trained weights, so inference on real data requires a completed training run.

## 1. Choose an environment

Use Linux with Python 3.12 for training and evaluation. An NVIDIA GPU with a compatible driver is required for CUDA execution. Training uses bf16 autocast; check bf16 support before starting. The release does not establish a minimum GPU memory requirement or a training-time estimate. The paper's latency workload uses one H200 at batch 1 with 1080p output.

CPU execution is sufficient for the synthetic suite. The current review also runs that suite on macOS with Python 3.12; this does not validate macOS GPU execution. The engine selects CUDA when available and otherwise uses CPU.

Create a clean environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU tests on Linux:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
```

For CPU tests on macOS, use the PyPI wheels:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0
```

For CUDA on Linux, the following installs the CUDA 12.8 build:

```bash
nvidia-smi
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Choose a different supported wheel if your driver requires it. The [official version list](https://pytorch.org/get-started/previous-versions/) provides the PyTorch 2.10.0 installation commands. The CUDA version reported by `nvidia-smi` describes driver support; it does not identify the installed PyTorch build.

Install the project and test tools:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pip check
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)'
python -m pytest -q
python scripts/audit_release.py
```

Two perceptual-metric tests skip when their optional packages are absent. The integration test creates small synthetic LMDBs, performs four training updates, reloads the checkpoint, writes four predictions and runs sequence evaluation. All test artifacts live in temporary directories.

Install optional dependencies only for the tasks that use them:

```bash
python -m pip install -r requirements-metrics.txt
python -m pip install -r requirements-cuda.txt
python -m pip check
```

The second command is for Linux CUDA deployment. LPIPS downloads external metric weights on first use. Keep a record of the installed environment with each experiment:

```bash
mkdir -p results
python -m pip freeze > results/environment.txt
```

The requirements pin direct dependencies. Transitive dependencies and hardware-specific wheels remain dependent on the installation date and platform.

## 2. Set data paths

Obtain datasets through the providers linked in [DATASETS.md](DATASETS.md). Follow their download terms and retain the supplied notices with the data.

```bash
export FROZEN_HISTORY_DATA="$PWD/data"
export QRISP_RAW="$FROZEN_HISTORY_DATA/qrisp/raw"
export QRISP_LMDB="$FROZEN_HISTORY_DATA/qrisp/lmdb"
export STSS_PACKED="$FROZEN_HISTORY_DATA/stss/packed"
export ARM_TEST="$FROZEN_HISTORY_DATA/arm/nss_test_full_resolution_sample.safetensors"
mkdir -p "$QRISP_RAW" "$QRISP_LMDB" "$STSS_PACKED" results
```

Set these variables in every new shell before starting Python. Use a separate local data disk if needed; source edits are unnecessary. Data volume and packing time have not been measured in this review.

## 3. Prepare QRISP

Extract the provider files into the layout in [DATASETS.md](DATASETS.md). Keep `270p/CameraData` for every scale. Raw test scenes belong under `TestSet` for shipped-baseline evaluation.

```bash
python scripts/pack_qrisp.py --splits train --tile-grid 2x4
python scripts/pack_qrisp.py --splits test
python scripts/validate_qrisp.py --scene CBApocalypse --res 270p
```

The expected counts are 7,260 training frames and 1,500 test frames. Training uses tiled LMDBs; evaluation uses untiled LMDBs. Inspect the packing output for missing modalities and confirm that the split indexes and `meta_info.txt` files exist. Avoid `--allow-incomplete` for paper experiments. If conversion fails, repair the input and remove the incomplete database before repacking.

## 4. Check a training command

```bash
python scripts/train.py --config configs/train/qrisp_x2.json --dry-run
python scripts/train.py --config configs/train/qrisp_x4.json --dry-run
```

These commands print the resolved engine arguments without loading data or starting training. [CONFIGURATIONS.md](CONFIGURATIONS.md) lists model sizes, scales, loss settings and iteration budgets.

## 5. Run a short real-data check

After packing QRISP, use a separate run name for a four-update check:

```bash
python scripts/train.py --config configs/train/qrisp_x2.json --name smoke_x2 --dim 8 --blocks 1 --batch 1 --patch 6 --clip 4 --warmup 1 --iters 4 --workers 0 --eval-every 0 --ckpt-every 4 --log-every 1
python scripts/inference.py --run runs/smoke_x2 --max-frames 4 --out results/smoke_x2
```

Expect `runs/smoke_x2/config.json`, `runs/smoke_x2/last.pt` and four output PNGs. This small model checks data access and the training/inference path. Its settings and scores are unsuitable for comparison with the paper.

## 6. Train the paper models

```bash
python scripts/train.py --config configs/train/qrisp_x2.json
python scripts/train.py --config configs/train/qrisp_x3.json
python scripts/train.py --config configs/train/qrisp_x4.json
```

Run these sequentially on one GPU. The final models use 100,000 updates at ×2/×3 and 200,000 updates at ×4. Each run writes its resolved configuration and checkpoints under `runs/<name>`.

Resume an interrupted run with the same configuration and overrides:

```bash
test -f runs/qrisp_x4/last.pt
python scripts/train.py --config configs/train/qrisp_x4.json --resume
```

The engine restores model, optimizer, scheduler and iteration state. If `last.pt` is absent, `--resume` starts a new run. Check the file first. Do not change the model, schedule or loss while resuming. Data-loader random state is not fully restored, so resuming does not guarantee the same subsequent samples as uninterrupted training.

If the full configuration runs out of memory, use a separate name for reduced settings and record every override. A reduced batch or clip changes the experiment and requires separate reporting.

## 7. Infer and evaluate

```bash
python scripts/inference.py --run runs/qrisp_x4 --out results/qrisp_x4_predictions
python scripts/protocol_full.py qrisp_x4 --scale 4 --warmup 8 --lpips --out results/qrisp_x4.json
python scripts/static_segment.py qrisp_x4 --scale 4 --out results/qrisp_x4_static.json
python scripts/metrics_graphics.py --method qrisp_x4 --scale 4 --out results/qrisp_x4_graphics.json
```

Install metric dependencies before using `--lpips` or perceptual graphics metrics. Keep the full test split, eight-frame warm-up, reset rules and quantization settings in [PROTOCOL.md](PROTOCOL.md). Aggregate all five test sequences with equal scene weights. Clip validation during training serves a different purpose and cannot substitute for full-sequence scores.

## 8. Reproduce the temporal-loss experiment

```bash
python scripts/train.py --config configs/train/mechanism_r_l0.json
python scripts/train.py --config configs/train/mechanism_r_l025.json
python scripts/train.py --config configs/train/mechanism_j_l0.json
python scripts/train.py --config configs/train/mechanism_j_l025.json
python scripts/copy_fraction.py mechanism_r_l025 --scale 4 --out results/copy_fraction.json
python scripts/copy_cost.py --scales 4 3 2
```

The `r` variants separate output and sample lattices. The `j` variants provide the misregistration control. Compare matched settings and iteration budgets; additional loss weights and ablations are listed in the configuration catalogue.

## 9. Run transfer experiments

For STSS, pack the five archives listed in [DATASETS.md](DATASETS.md), validate them, and obtain the upstream metric utility if using the published STSSNet comparison:

```bash
python scripts/validate_stss.py
python scripts/train.py --config configs/train/stss_r_l0.json
python scripts/protocol_bench.py --dataset stss --method stss_r_l0 --fps-mode 30 --scenes Arena_test Lewis_test SunTemple_test Subway_test --out results/stss_comparison.json
```

Train only on `Lewis_train1`. The published comparison scores even original frame indices at 30 fps with composed motion. The default 60 fps evaluation is a separate protocol.

For Arm, provide the full-resolution test safetensors capture and transfer a QRISP-trained ×2 model:

```bash
python scripts/validate_arm.py --out results/arm_validation.json
python scripts/protocol_bench.py --dataset arm --method qrisp_x2 --exposure 2 --encode none --out results/arm_transfer.json
```

Use the 100-frame test capture without Arm training or tuning. The first eight frames warm up the state, leaving 92 scored frames.

## 10. Check deployment and latency

Train the deployment model before measuring trained latency:

```bash
python scripts/train.py --config configs/train/rep_64x4.json
python scripts/validate_deployment.py --run runs/rep_64x4 --precision fp32 --out results/deployment_parity.json
python scripts/benchmark.py --run runs/rep_64x4 --mode graph --precision bf16 --out results/trained_latency.json
```

Fused kernels support the registered ×4 normalized recurrence. Use eager execution for unsupported variants. A benchmark using `--config` builds random weights; use `--run` to load a trained checkpoint. Record the GPU, wheel build, precision, image dimensions and timing settings. H200 results require an H200 run; measurements on another GPU are separate results.

## 11. Prepare a public source archive

```bash
python -m pytest -q
python -m ruff check .
python -m pip check
python scripts/audit_release.py
python scripts/package_release.py
```

The packer updates `SHA256SUMS` and writes `frozen-history-source.zip`. Extract it into a new directory and upload that source tree. The archive excludes local datasets, weights, environments, outputs and manuscript files. It has normalized ZIP timestamps and permissions.

The release ZIP contains no `.git` directory. If you copy code into an existing Git repository, inspect its history, branches, tags, commit authors and remote settings separately. A source-file audit cannot remove information already stored in Git history. Account names and commit metadata also remain visible on GitHub when publishing under a personal account.

Create the remote repository only after reviewing the extracted tree. Follow [GitHub's repository guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories). Keep `LICENSE`, `THIRD_PARTY.md` and `CITATION.cff` with the code. Add the public paper URL and citation details when available.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: diffsr` | Activate `.venv`, run from the root, and install `-e . --no-deps`. |
| CUDA unavailable | Check `nvidia-smi`, the installed wheel and `torch.cuda.is_available()`. A CPU wheel cannot use CUDA. |
| CUDA out of memory | Confirm no competing process; test reduced settings under a separate run name. |
| Missing LMDB or index | Check environment variables and use tiled training / untiled test databases. |
| EXR decoding failure | Use the pinned headless OpenCV package and avoid installing both OpenCV distributions. |
| Missing camera metadata | Restore `270p/CameraData`; zero-jitter fallback is unsuitable for reproduction. |
| Missing checkpoint | Complete training first and check the run name and checkpoint path. |
| LPIPS fails offline | Prepare its external weights in the PyTorch cache before running offline. |
| Triton import or graph failure | Use Linux CUDA, install optional requirements, and test eager execution first. |
| STSS metric utility missing | Clone the upstream checkout and set `STSSNET_UTILS` as described in the dataset guide. |
