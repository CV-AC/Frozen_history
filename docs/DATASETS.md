# Dataset acquisition, formats and terms

No images, tensors, LMDBs, rendered baselines or pretrained weights are distributed. Download data from the original providers, accept their terms, and keep their license files with the local data. The MIT code license does not license the datasets.

## Paths

| Variable | Default relative to repository root |
| --- | --- |
| `FROZEN_HISTORY_DATA` | `data` |
| `QRISP_RAW` | `data/qrisp/raw` |
| `QRISP_LMDB` | `data/qrisp/lmdb` |
| `STSS_PACKED` | `data/stss/packed` |
| `ARM_TEST` | `data/arm/nss_test_full_resolution_sample.safetensors` |
| `STSSNET_UTILS` | `external/STSSNet/utils` |

Environment variables are read when importing adapters. Set them before starting Python. Custom locations do not require editing source code.

## QRISP

Use the [official dataset page](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset) and its [downloads page](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset/downloads). The provider makes QRISP available for research use under its product license agreement; review and accept that agreement at download. Provider restrictions also apply to DLSS/FSR captures and underlying assets. This release does not mirror those files or replace the agreement with MIT.

Arrange the extracted scenes as:

```text
data/qrisp/raw/
  CBApocalypse/1080p/Enhanced/0000/0000.png
  CBApocalypse/270p/CameraData/0000/0000.json
  ...other training scenes and modalities...
  TestSet/AbandonedSchool/1080p/Enhanced/0000/0000.png
  TestSet/SeaPort/...
  TestSet/SpaceShipDemo/...
```

The parser also accepts each test scene directly under `QRISP_RAW` for packing, but shipped-baseline and graphics evaluation expect the canonical `TestSet` layout.

Training scenes: CBApocalypse, FloodedGrounds, FloodedGroundsBridges, ScifiBase, ScifiBaseNightStartStop, ScifiBaseStartStop, ScifiFacility, SunTemple, SunTempleBush and SunTempleLamps. Expected total: 7,260 frames. Test scenes: AbandonedSchool, SeaPort and SpaceShipDemo, totaling 1,500 frames in five 300-frame camera runs. Keep all modalities aligned by scene, sequence and frame.

GT is `1080p/Enhanced`. Input resolutions are 540p, 360p and 270p for ×2, ×3 and ×4. The paper uses jittered, mip-biased RGB, camera jitter, depth, and backward motion in both unjittered and jittered variants. Preserve `270p/CameraData` even when training at another scale; the packer reads the shared camera metadata there. Missing camera metadata must be repaired before paper reproduction rather than interpreting the fallback zero jitter as real metadata.

```bash
python scripts/pack_qrisp.py --splits train --tile-grid 2x4
python scripts/pack_qrisp.py --splits test
python scripts/validate_qrisp.py --scene CBApocalypse --res 270p
```

The packer writes `QRISP_<split>_<modality>[_t2x4].lmdb`, `meta_info.txt` and `QRISP_<split>_index.json`. Color remains lossless PNG bytes; geometry becomes float16 arrays, matching the original conversion. Each index records the frame key and `jitter_yx`. Training crops come from the non-overlapping 2×4 tile grid. Full-frame testing uses untiled databases. The default modality set includes all scales and both geometry variants; `--modalities` can select names listed in `scripts/pack_qrisp.py`. The packer refuses incorrect complete-split counts and missing modality frames.

For classical shipped baselines retain the raw test PNGs in their provider layout as well as LMDBs. DLSS2/FSR shipped captures use the provider's ×2 mode; selecting another motion resolution does not turn them into matched ×3/×4 baselines.

## STSS / STSSNet

Obtain archives through the [official STSSNet repository](https://github.com/ryanhe312/STSSNet-AAAI2024) and its linked [ModelScope dataset](https://www.modelscope.cn/datasets/ryanhe312/STSSNet-AAAI2024). The upstream code is MIT. Dataset permissions must be checked against the terms attached to the downloaded dataset; the code license alone is insufficient evidence of dataset permission. No dataset license text is invented or relicensed here.

The paper trains on **Lewis_train1 only**, using the 2,798 valid frames from that first training file. It does not train on Arena, SunTemple or Subway. Download `Lewis_train1.zip`, `Arena_test.zip`, `Lewis_test.zip`, `SunTemple_test.zip` and `Subway_test.zip` for that experiment.

```bash
python scripts/pack_stss.py --zip data/stss/raw/Lewis_train1.zip --dst data/stss/packed/Lewis_train1
python scripts/pack_stss.py --zip data/stss/raw/Arena_test.zip --dst data/stss/packed/Arena_test
python scripts/pack_stss.py --zip data/stss/raw/Lewis_test.zip --dst data/stss/packed/Lewis_test
python scripts/pack_stss.py --zip data/stss/raw/SunTemple_test.zip --dst data/stss/packed/SunTemple_test
python scripts/pack_stss.py --zip data/stss/raw/Subway_test.zip --dst data/stss/packed/Subway_test
python scripts/validate_stss.py
```

Alternatively `--src` accepts an extracted scene containing `NPY/` and `HR/`. The converter recognizes the 33- and 35-channel source formats, preserves raw LR color/depth/motion in a seven-channel float16 packed array, and verifies sampled conversions against source values. Stored color is BGR; the loader returns RGB. Depth is `clip(raw_depth/16, 0, 1)`. Motion is `(dy, dx)=(channel_5, -channel_6)` in the packed representation. Jitter is zero. Do not synthesize QRISP jitter or relabel this domain as jittered.

The converter records valid frame indices and camera cuts. Failed/corrupt input frames are omitted with the resulting gap recorded; sequences restart at gaps and recorded cuts. Reproduction expects the original valid-frame set, including the known gap at frame 1427 and cut at frame 2802, rather than silently replacing bad frames.

The exact STSSNet super-resolved-frame metric needs the upstream utility, loaded externally:

```bash
git clone https://github.com/ryanhe312/STSSNet-AAAI2024 external/STSSNet
export STSSNET_UTILS="$PWD/external/STSSNet/utils"
```

The external checkout is ignored by Git and omitted from the source archive; preserve its MIT notice. The command may also fetch upstream checkpoint files into the ignored directory; they are not distributed in this release.

## Arm Neural Graphics Dataset

Use the [official dataset card](https://huggingface.co/datasets/Arm/neural-graphics-dataset). It identifies the **Arm AI Model Community License**, separate from the Apache-2.0 license of Arm Model Gym code. Accept the dataset license and retain the original scene-asset notices.

Place the paper's full-resolution Bistro test capture `nss_test_full_resolution_sample.safetensors` at `ARM_TEST`. Provider layouts change across revisions; use the original NSS test capture with the required schema, and record the downloaded revision and file checksum. The paper evaluates its 100 test frames at ×2 and 30 fps. Cropped training/validation captures overlap this small demonstration scene; neither is used to train, tune or select the transferred model.

Required model inputs: `colour_linear`, `ground_truth_linear`, `depth`, `motion_lr`, `motion`, `jitter`; capture diagnostics also read `depth_params`, `zNear`, `exposure` and `viewProj`. The loader uses sequence-first CHW tensors. An incompatible file should be corrected at acquisition rather than guessed or replaced by fabricated geometry.

Both LR and reference HDR images use the same `Reinhard(exp(2) * RGB)` transform. Default evaluation leaves that result in the resulting linear display values (`--encode none`); `--encode srgb` applies the explicit sRGB OETF to both sides as a separately named variant. Negate the stored motion to obtain backward `(dy,dx)` displacement. Raw jitter `(dy,dx)` becomes stored y-up `(-dy,dx)`; depth maps to `min(1, 0.0013/(1-d))`. The capture supplies jittered depth only, a known mismatch from the QRISP registered depth convention.

```bash
python scripts/validate_arm.py --out results/arm_validation.json
```

The repository does not bundle NSS, FSR, DLSS or other external models. Published baseline scores, if quoted in the paper, are not rerun by downloading this source release.
