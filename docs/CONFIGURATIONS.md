# Configuration catalogue

Each `configs/train/<name>.json` is a runnable training recipe. Its matching `configs/models/<name>.json` preserves model kwargs and input conventions for construction, checkpoint compatibility and synthetic timing. Source-configuration hashes are recorded in `config_manifest.json` without internal paths or run identifiers.

`r` means registered; `j` is the misregistration control. `l005`, `l025` and `l235` mean λ=0.05, 0.25 and 2.35. `short` is a 15k-iteration control. `single_frame` disables recurrent history. `registration_*` changes the named registration component. Supplementary configurations cover the manuscript tables' additional architecture, rate, seed and training-state comparisons; inspect the complete JSON when reproducing a specific variant.

The final ×4 configuration is `qrisp_x4` (256×8). `qrisp_x4_192` is the 192×8 alternative. A supplementary or failed leave-one-out run is not a recommended default.

| Configuration | Dataset | Architecture / trunk | Scale | Iterations | Clip | λ | Temporal warp/norm | Accumulation | Seed | Peak LR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ablation_blend](../configs/train/ablation_blend.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.05 | motion / l1 | blend | 0 | 0.0006 |
| [ablation_clip16](../configs/train/ablation_clip16.json) | qrisp | plain 192×8 | ×4 | 45000 | 16 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [direct_j_l0](../configs/train/direct_j_l0.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0 | motion / l1 | direct | 0 | 0.0006 |
| [direct_j_l025](../configs/train/direct_j_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l1 | direct | 0 | 0.0006 |
| [direct_r_l0](../configs/train/direct_r_l0.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0 | motion / l1 | direct | 0 | 0.0006 |
| [direct_r_l025](../configs/train/direct_r_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l1 | direct | 0 | 0.0006 |
| [mechanism_j_l0](../configs/train/mechanism_j_l0.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_j_l005](../configs/train/mechanism_j_l005.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_j_l025](../configs/train/mechanism_j_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_j_l1](../configs/train/mechanism_j_l1.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 1 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_r_l0](../configs/train/mechanism_r_l0.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_r_l005](../configs/train/mechanism_r_l005.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_r_l025](../configs/train/mechanism_r_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_r_l1](../configs/train/mechanism_r_l1.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 1 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_x2_j_l0](../configs/train/mechanism_x2_j_l0.json) | qrisp | plain 192×8 | ×2 | 45000 | 30 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_x2_j_l025](../configs/train/mechanism_x2_j_l025.json) | qrisp | plain 192×8 | ×2 | 45000 | 30 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_x2_r_l0](../configs/train/mechanism_x2_r_l0.json) | qrisp | plain 192×8 | ×2 | 45000 | 30 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [mechanism_x2_r_l025](../configs/train/mechanism_x2_r_l025.json) | qrisp | plain 192×8 | ×2 | 45000 | 30 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [qrisp_style_j_l0](../configs/train/qrisp_style_j_l0.json) | qrisp | QRISP-style f64 m5 | ×4 | 15000 | 30 | 0 | motion / l1 | direct blend | 0 | 0.0006 |
| [qrisp_style_j_l025](../configs/train/qrisp_style_j_l025.json) | qrisp | QRISP-style f64 m5 | ×4 | 15000 | 30 | 0.25 | motion / l1 | direct blend | 0 | 0.0006 |
| [qrisp_style_r_l0](../configs/train/qrisp_style_r_l0.json) | qrisp | QRISP-style f64 m5 | ×4 | 15000 | 30 | 0 | motion / l1 | direct blend | 0 | 0.0006 |
| [qrisp_style_r_l025](../configs/train/qrisp_style_r_l025.json) | qrisp | QRISP-style f64 m5 | ×4 | 15000 | 30 | 0.25 | motion / l1 | direct blend | 0 | 0.0006 |
| [qrisp_x2](../configs/train/qrisp_x2.json) | qrisp | plain 192×8 | ×2 | 100000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [qrisp_x3](../configs/train/qrisp_x3.json) | qrisp | plain 192×8 | ×3 | 100000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [qrisp_x4](../configs/train/qrisp_x4.json) | qrisp | plain 256×8 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [qrisp_x4_192](../configs/train/qrisp_x4_192.json) | qrisp | plain 192×8 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_j](../configs/train/registration_j.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_jittered_depth_seed0](../configs/train/registration_jittered_depth_seed0.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_jittered_depth_seed1](../configs/train/registration_jittered_depth_seed1.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 1 | 0.0006 |
| [registration_legacy_phase](../configs/train/registration_legacy_phase.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_motion_only](../configs/train/registration_motion_only.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_no_lr_rejitter](../configs/train/registration_no_lr_rejitter.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_no_skip](../configs/train/registration_no_skip.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_r_seed0](../configs/train/registration_r_seed0.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [registration_r_seed1](../configs/train/registration_r_seed1.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 1 | 0.0006 |
| [rep_64x4](../configs/train/rep_64x4.json) | qrisp | rep3 64×4 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [rep_96x4](../configs/train/rep_96x4.json) | qrisp | rep3 96×4 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [rep_96x8](../configs/train/rep_96x8.json) | qrisp | rep3 96×8 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [shift_64x6](../configs/train/shift_64x6.json) | qrisp | shiftmlp 64×6 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [shift_64x8](../configs/train/shift_64x8.json) | qrisp | shiftmlp 64×8 | ×4 | 200000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [single_frame_x2](../configs/train/single_frame_x2.json) | qrisp | plain 192×8 | ×2 | 45000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [single_frame_x4](../configs/train/single_frame_x4.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [squared_temporal_j_l025](../configs/train/squared_temporal_j_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l2 | normalized | 0 | 0.0006 |
| [squared_temporal_j_l235](../configs/train/squared_temporal_j_l235.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 2.35 | motion / l2 | normalized | 0 | 0.0006 |
| [squared_temporal_r_l025](../configs/train/squared_temporal_r_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | motion / l2 | normalized | 0 | 0.0006 |
| [squared_temporal_r_l235](../configs/train/squared_temporal_r_l235.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 2.35 | motion / l2 | normalized | 0 | 0.0006 |
| [stss_r_l0](../configs/train/stss_r_l0.json) | stss | plain 192×8 | ×2 | 45000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [stss_r_l005](../configs/train/stss_r_l005.json) | stss | plain 192×8 | ×2 | 45000 | 16 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [stss_r_l025](../configs/train/stss_r_l025.json) | stss | plain 192×8 | ×2 | 45000 | 16 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [stss_r_l025_short](../configs/train/stss_r_l025_short.json) | stss | plain 192×8 | ×2 | 15000 | 16 | 0.25 | motion / l1 | normalized | 0 | 0.0006 |
| [stss_r_l0_short](../configs/train/stss_r_l0_short.json) | stss | plain 192×8 | ×2 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_001](../configs/train/supplementary_001.json) | qrisp | plain 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_002](../configs/train/supplementary_002.json) | qrisp | mbv2 64×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_003](../configs/train/supplementary_003.json) | qrisp | pconv 128×6 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_004](../configs/train/supplementary_004.json) | qrisp | plain 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_005](../configs/train/supplementary_005.json) | qrisp | plain 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 1 | 0.0006 |
| [supplementary_006](../configs/train/supplementary_006.json) | qrisp | rep3 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_007](../configs/train/supplementary_007.json) | qrisp | rep3 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_008](../configs/train/supplementary_008.json) | qrisp | rep3 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 1 | 0.0006 |
| [supplementary_009](../configs/train/supplementary_009.json) | qrisp | star 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_010](../configs/train/supplementary_010.json) | qrisp | unet 96×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_011](../configs/train/supplementary_011.json) | qrisp | unet 192×9 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_012](../configs/train/supplementary_012.json) | qrisp | unet 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_013](../configs/train/supplementary_013.json) | qrisp | unet 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_014](../configs/train/supplementary_014.json) | qrisp | unet 96×9 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_015](../configs/train/supplementary_015.json) | qrisp | unet 96×9 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_016](../configs/train/supplementary_016.json) | qrisp | plain 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_017](../configs/train/supplementary_017.json) | qrisp | rep3 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_018](../configs/train/supplementary_018.json) | qrisp | dwmlp 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_019](../configs/train/supplementary_019.json) | qrisp | dwmlp 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_020](../configs/train/supplementary_020.json) | qrisp | linattn 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_021](../configs/train/supplementary_021.json) | qrisp | pconvglu 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_022](../configs/train/supplementary_022.json) | qrisp | plain 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_023](../configs/train/supplementary_023.json) | qrisp | plain 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_024](../configs/train/supplementary_024.json) | qrisp | poolmlp 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_025](../configs/train/supplementary_025.json) | qrisp | shiftmlp 48×3 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_026](../configs/train/supplementary_026.json) | qrisp | shiftmlp 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_027](../configs/train/supplementary_027.json) | qrisp | shiftmlp 48×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_028](../configs/train/supplementary_028.json) | qrisp | shiftmlp 48×6 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_029](../configs/train/supplementary_029.json) | qrisp | shiftmlp 48×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_030](../configs/train/supplementary_030.json) | qrisp | shiftmlp 48×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_031](../configs/train/supplementary_031.json) | qrisp | shiftmlp 64×3 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_032](../configs/train/supplementary_032.json) | qrisp | shiftmlp 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_033](../configs/train/supplementary_033.json) | qrisp | shiftmlp 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_034](../configs/train/supplementary_034.json) | qrisp | shiftmlp 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_035](../configs/train/supplementary_035.json) | qrisp | shiftmlp 64×4 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.004 |
| [supplementary_036](../configs/train/supplementary_036.json) | qrisp | shiftmlp 64×6 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_037](../configs/train/supplementary_037.json) | qrisp | shiftmlp 64×6 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.002 |
| [supplementary_038](../configs/train/supplementary_038.json) | qrisp | plain 192×8 | ×4 | 15000 | 16 | 0 | motion / l1 | normalized | 0 | 0.0006 |
| [supplementary_039](../configs/train/supplementary_039.json) | qrisp | plain 192×8 | ×4 | 15000 | 30 | 0.05 | motion / l1 | normalized | 0 | 0.0006 |
| [unwarped_j_l025](../configs/train/unwarped_j_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | none / l1 | normalized | 0 | 0.0006 |
| [unwarped_j_l1](../configs/train/unwarped_j_l1.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 1 | none / l1 | normalized | 0 | 0.0006 |
| [unwarped_r_l025](../configs/train/unwarped_r_l025.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 0.25 | none / l1 | normalized | 0 | 0.0006 |
| [unwarped_r_l1](../configs/train/unwarped_r_l1.json) | qrisp | plain 192×8 | ×4 | 45000 | 30 | 1 | none / l1 | normalized | 0 | 0.0006 |

## Overrides

```bash
python scripts/train.py --config configs/train/mechanism_r_l025.json --dry-run
python scripts/train.py --config configs/train/mechanism_r_l025.json --workers 4
```

Supply `--resume` only with the same architecture and training schedule. The launcher retains explicit booleans, so `--no-motion-jit` and `--no-depth-jit` are represented correctly. Change a model recipe by training from its training JSON; model JSON alone is for construction and loading, not a training command.

## Classical baselines

`configs/baselines/taau_x{2,3,4}.json` and `taa_clamp_x{2,3,4}.json` retain the chosen parameter set, train-only selection metric, candidate grid and training-run list. They contain configuration metadata, not dataset samples. Re-run selection using the scripts described in [PROTOCOL.md](PROTOCOL.md).
