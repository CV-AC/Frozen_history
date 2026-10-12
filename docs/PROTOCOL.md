# Experimental protocol

These settings are taken from the manuscript and archived experiment configurations. Re-running the commands requires separately obtained data and trained checkpoints. Published scores are not asserted as release-validation results.

## Registration and recurrence

All model tensors are RGB in `[0,1]` with CHW layout. Backward motion is `(dy,dx)` in input-resolution pixels, with previous position equal to current position plus motion. It is enlarged with nearest-neighbor interpolation and multiplied by the scale for output-resolution warps. Image rows point down.

The stored jitter is y-up `(jy,jx)`; its image-coordinate sample offset is `o=(-jy,+jx)`. Registered R uses the nonjittered motion `m` for output-color history and unjittered depth. LR samples/features use `m + o_t - o_{t-1}`. The candidate skip and cold start use registered bicubic sampling, and the reconstruction-kernel phase uses image coordinates. The J control deliberately feeds jittered motion to the displayed-history warp, uses jittered depth and unregistered skip, and disables LR lattice correction. It remains a named experimental control.

The normalized accumulator carries weighted color `S` and weight `N`:

```text
S_t = rho_t * valid_t * warp(S_(t-1)) + w_t * candidate_t
N_t = rho_t * valid_t * warp(N_(t-1)) + w_t
output_t = S_t / N_t
alpha_t = w_t / N_t
```

For the final recipe the weight cap is 16. Rescale numerator and denominator together when capping, preserving their ratio. Learned reliability/retention can make alpha very small; a large `1/alpha` alone is not a demonstrated averaging window. Report accuracy beside temporal variation and copying. Direct-prediction models store their output as history without an accumulator; their reported accumulator alpha is intentionally NaN.

## Training

| Family | Scale | Trunk | Iterations | Clip | Temporal loss |
| --- | --- | --- | --- | --- | --- |
| Final QRISP | ×2 / ×3 | 192 channels, 8 blocks | 100k | 30 | Warped L1 derivative, λ=0.05 |
| Final QRISP | ×4 | 256 channels, 8 blocks | 200k | 30 | Warped L1 derivative, λ=0.05 |
| Alternative ×4 | ×4 | 192 channels, 8 blocks | 200k | 30 | Warped L1 derivative, λ=0.05 |
| Mechanism R/J | ×4 / ×2 | 192 channels, 8 blocks | 45k | 30 | λ=0, 0.05, 0.25, 1; see available configurations |
| QRISP-style R/J | ×4 | Jitter-conditioned comparison network | 15k | 30 | λ=0 / 0.25 |
| Direct prediction R/J | ×4 | 192 channels, 8 blocks | 45k | 30 | λ=0 / 0.25 |
| Registration component controls | ×4 | 192 channels, 8 blocks | 15k | 16 | λ=0 |
| STSS | ×2 | 192 channels, 8 blocks | 45k | 16 | λ=0, 0.05, 0.25 |

QRISP final training uses 66×66 LR crops, batch 8, AdamW with betas `(0.9,0.99)` and weight decay `1e-4`. OneCycleLR uses peak `6e-4`, a 5% increasing phase, and final division factor 200. Skip the first two frames of each training clip in the loss. Clips share crops and horizontal/vertical flips; motion and jitter signs transform with the images. Use each released configuration's actual learning rate for reparameterized and shift-MLP variants; shift variants use peak `2e-3`.

Reconstruction is RGB L2. The default temporal term matches the prediction's warped derivative to the reference's warped derivative, over in-bounds pixels, using the training motion convention. `--temporal-warp none` compares screen-space derivatives; `--temporal-objective l2` squares derivative error. Weights 0.25 and 2.35 are distinct squared-loss experiments; the latter matches the approximate logged scale of the L1 term and is not an equivalent-gradient normalization.

Original runtime checkpoint/config files are excluded. Public configuration names replace internal run identifiers. Model configurations preserve actual model kwargs. Supplementary cases retain original parameter values even when they represent exploratory or unsuccessful experiments; in particular the seed-0 jittered-depth leave-one-out run is a documented divergence, not a recommended recipe. No archived trained weights are included.

## QRISP full sequence scoring

Reset the state at every camera run and step every frame. Skip the first eight frames per run, leaving `5*(300-8)=1460` scored frames. Do not transfer state across scene changes or cuts. `protocol_full.py` resets by default. Neither TTA nor model ensembles are enabled for standard rows.

PSNR-Y uses MATLAB-style BT.601 limited-range luma, a maximum intensity of 255, and no border crop. SSIM-Y uses an 11×11 Gaussian window. Spatial scores are frame means within each scene and then an equal mean over the three scenes. LPIPS uses RGB with AlexNet and normalization to `[-1,1]`. The default protocol scores clamped floating predictions; `--quantize` is a separate round-to-8-bit condition.

Use the same **nonjittered motion** for temporal evaluation of every R/J/model/baseline arm. The default is that scale's field; `--tpsnr-field nonjit540` is the named cross-scale alternative. Temporal error is the difference between prediction and reference motion-compensated RGB derivatives. Reports distinguish mean dB capped at 60 per pair from pooled SSE/count PSNR. They are different aggregations and must not be interchanged.

Camera-static classification uses the median absolute nonjittered 270p motion magnitude per channel below 0.02 LR pixels; equivalent scaling is used at other resolutions. The manuscript's split has 97 static and 1,363 moving scored frames. Full-table metrics use scene means, whereas static/moving spatial subsets use frame means.

```bash
python scripts/protocol_full.py qrisp_x2 --scale 2 --warmup 8 --lpips --out results/qrisp_x2.json
python scripts/protocol_full.py qrisp_x3 --scale 3 --warmup 8 --lpips --out results/qrisp_x3.json
python scripts/protocol_full.py qrisp_x4 --scale 4 --warmup 8 --lpips --out results/qrisp_x4.json
python scripts/temporal_anatomy.py qrisp_x4 --scale 4 --out results/temporal_anatomy.json
```

`--max-frames`, nonunit stride and alternate resets are diagnostics. Their reports must not be used as complete paper-protocol rows.

## Static variation, copying and perceptual metrics

Static STD is the mean population standard deviation per RGB channel/pixel over AbandonedSchool/0001 frames 275–292 inclusive. Warm the model from that camera run's first frame. Report it in 8-bit gray levels. The main table uses rounded 8-bit outputs; mechanism analyses retain floating output STD. `static_segment.py` writes both `ours` and `ours_uint8`; the two are deliberately distinct.

`copy_fraction.py` measures whether each output differs from its own warped previous output by less than `0.5/255` in every channel, over valid pixels whose reference satisfies the same threshold. The reference therefore has copy fraction 1 on that mask. It also measures response to a seeded small perturbation of color history, reporting history gain and alpha-like response. Accuracy, copying, and temporal STD should be considered together. A frozen inaccurate output can have excellent STD.

FLIP uses rounded 8-bit images and 67 pixels/degree. ColorVideoVDP 0.5.7 uses `standard_fhd`: 24-inch 1080p sRGB, 200 cd/m², 0.6 m viewing distance and 60 fps for QRISP. JOD is measured per camera sequence and aggregated with equal scene weights. Static JOD separately scores School frames 274–292 and SeaPort frames 20–38 and 205–223.

```bash
python scripts/static_segment.py mechanism_r_l025 --scale 4 --out results/static.json
python scripts/copy_fraction.py mechanism_r_l025 --scale 4 --out results/copy.json
python scripts/metrics_graphics.py --method qrisp_x4 --scale 4 --out results/graphics.json
python scripts/metrics_graphics.py --method GT --scale 4 --sanity --out results/graphics_sanity.json
```

## Classical baselines and model-free copy cost

The TAAU and clamp baselines are implemented here without external SDK code. Their public selection configurations were chosen on QRISP training runs only, using three runs per training scene, eight warm-ups and PSNR-Y. Re-tune on training data using `taau_select.py` or `taa_clamp_select.py`, never the test set. The released ×2 selection is reused unchanged for transfer.

```bash
python scripts/protocol_taau.py --scale 4 --what both
python scripts/taau_select.py --scale 4 --out results/taau_selection.json
python scripts/taa_clamp_select.py --scale 4 --out results/clamp_selection.json
python scripts/copy_cost.py --scales 4 3 2
```

`copy_cost.py` evaluates the reference-warp residual with different field/lattice choices and separates static and moving content. It requires data but no trained model. `--runs` and `--max-frames` are truncated diagnostics.

## STSS

Train only on the first valid Lewis training file. Arena, SunTemple and Subway are unseen by these models. The final training configurations use ×2, no jitter and 16-frame clips. Reset at numbering gaps and recorded cuts.

```bash
python scripts/train.py --config configs/train/stss_r_l0.json
python scripts/train.py --config configs/train/stss_r_l005.json
python scripts/train.py --config configs/train/stss_r_l025.json
python scripts/protocol_stss.py stss_r_l0 --root data/stss/packed/Arena_test --lpips --out results/stss_arena_all.json
python scripts/protocol_bench.py --dataset stss --method stss_r_l0 --fps-mode 30 --scenes Arena_test Lewis_test SunTemple_test Subway_test --out results/stss_comparison.json
```

`protocol_stss.py` is the all-frame 60 fps mechanism protocol. The STSSNet published comparison uses **even original frame indices** (super-resolved frames), rounded 8-bit BT.601 luma, full images and the externally loaded `utils/matlab_metric.py`. Use `protocol_bench.py --fps-mode 30` for this comparison; motion over two source steps is composed before stepping the recurrent model. Selecting every other output after running a 60 fps model is a different experiment.

Published-comparison ranges are Arena 500–1500, Lewis/SunTemple 5–1000 and Subway 6–1000; the eight-step warm-up is applied within each selected run. `protocol_bench.py` defaults to 60 fps, so specify 30 explicitly for Table 3's comparison. The report includes both this repository's scores and the STSSNet-specific super-resolved-frame scores. Use the latter for the quoted STSSNet comparison.

## Arm zero-shot transfer

Transfer the QRISP-trained ×2 model without Arm training or tuning. Use the 100-frame full-resolution Bistro test sequence, 30 fps, state reset at start and eight-frame warm-up, leaving 92 scored frames. Both LR and GT use exposure 2 and the same Reinhard transform. Default motion/jitter signs come from the validated loader; sign-flipped variants are diagnostics. Arm has jittered depth only.

```bash
python scripts/protocol_bench.py --dataset arm --method qrisp_x2 --exposure 2 --encode none --out results/arm_transfer.json
python scripts/protocol_bench.py --dataset arm --method single_frame_x2 --exposure 2 --encode none --out results/arm_single_frame.json
python scripts/protocol_bench.py --dataset arm --method mechanism_x2_r_l0 --exposure 2 --encode none --out results/arm_recurrent_control.json
```

The single-frame and recurrent 45k controls are matched-budget comparisons, while the final ×2 model uses 100k iterations. Keep this distinction when interpreting transfer gains. Never use overlapping Arm crops for model selection.

## Deployment

The optional fused kernels accelerate the registered ×4 normalized recurrence. Convolution folding replaces reparameterized training branches with inference convolutions. CUDA graphs require fixed dimensions and stable buffers. Use the dataset-based `validate_deployment.py` to compare independently evolving eager and fused recurrent states; `benchmark.py` instead measures repeated fixed-input, fixed-state calls, corresponding to the timing workload.

```bash
python scripts/validate_deployment.py --run runs/rep_64x4 --precision fp32 --out results/deployment_parity.json
python scripts/benchmark.py --run runs/rep_64x4 --mode graph --precision bf16 --out results/timing.json
```

Record GPU, software build, precision, dimensions, warm-up, number of calls and kernel options. CUDA performance and full-data parity were not rerun during the CPU release audit; the validation report identifies this limit explicitly.
