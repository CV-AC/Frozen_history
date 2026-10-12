# Paper figures and tables

The README uses five figures from the local arXiv manuscript of *Frozen History: When Temporal Losses Stop Accumulation in Learned Real-Time Supersampling*. The figures were checked visually before inclusion. Their captions and the selected table values were checked against the manuscript source.

## Figures

| Public asset | Paper figure | Contents |
| --- | --- | --- |
| [teaser.png](../assets/figures/teaser.png) | Figure 1 | Static reconstruction, temporal standard deviation and new-sample share under R/J registration and different temporal weights. |
| [pipeline.png](../assets/figures/pipeline.png) | Figure 2 | Network, normalized accumulator, registration conventions and model-free copy cost. |
| [temporal-loss.png](../assets/figures/temporal-loss.png) | Figure 3 | Accuracy, static perceptual quality and copy fraction as the temporal weight changes. |
| [reconstruction.png](../assets/figures/reconstruction.png) | Figure 4 | Selected crop comparisons at ×2, ×3 and ×4. |
| [temporal-stability.png](../assets/figures/temporal-stability.png) | Figure 5 | Static pixel variation and a moving space-time slice at ×2. |

PDF figures were rasterized at 2,400 pixels wide. The pipeline preserves the manuscript's 4,096-pixel image. Files use RGB PNG with no embedded text, EXIF, DPI or other image metadata. File names contain no submission identifiers or private paths. The source PDFs, author block and manuscript build files are excluded.

Each figure has an inline README image and a relative link to its full-resolution asset. The figures work when the repository is copied or renamed; they do not depend on private hosting or absolute filesystem paths.

## Tables

The temporal-loss table is selected from paper Table 1. It reports changes in PSNR-Y within the same model and pipeline when the warped L1 temporal weight rises from 0 to 0.25. Static and moving populations are 97 and 1,363 frames. The normalized and direct models use 45,000 training updates; the QRISP-style models use 15,000.

The reconstruction and latency tables are selected from paper Table 2. QRISP-L quality values are quoted published scores. Other reconstruction rows use the paper's harness. Our final models train for 100,000 updates at ×2/×3 and 200,000 at ×4. Missing metrics appear as `N/A`.

The ×4 latency table includes four of this project's models. Timings use one H200, 1080p output, batch 1, fused kernels, bf16 with fp32 accumulation and a CUDA graph. The measured recurrent step includes preparation, warps, network execution and accumulation, excluding rendering.

These values are paper results. Synthetic release checks do not reproduce them. Image-crop scores also differ from test-set means: the qualitative crops were selected for high gains within detailed regions. See [PROTOCOL.md](PROTOCOL.md) for reset rules, warm-up, quantization and aggregation.

## Attribution and data terms

The display figures contain QRISP scene renderings and, where labelled, supplied DLSS 2 comparison captures. Public method and dataset names remain visible. The upstream resources and their separate terms are listed in [THIRD_PARTY.md](../THIRD_PARTY.md) and [DATASETS.md](DATASETS.md). Inclusion in a paper figure does not change the underlying data or scene-asset terms.

The five figures are the only binary media files selected by the release packer. `.gitignore` permits these exact paths while continuing to ignore other image outputs. The source audit checks PNG integrity and rejects embedded image metadata. Visual inspection is still needed when replacing a figure, since a text scan cannot detect personal information drawn into pixels.
