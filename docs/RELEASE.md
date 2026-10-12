# Release preparation and validation

This is an anonymous source release for *Frozen History: When Temporal Losses Stop Accumulation in Learned Real-Time Supersampling*. The source repository is [CV-AC/Frozen_history](https://github.com/CV-AC/Frozen_history).

## Source selection

The archive inventory contains 75,402 regular files. Full text/code inspection covered 6,959 Python files, 3,193 shell files and the obsolete GLSL prototype, including many duplicate snapshots. Deduplication identified 1,569 distinct code contents; 1,140 distinct Python contents parsed successfully. Twelve excluded figure/profiler fragments are not valid standalone Python. See [source_review.json](source_review.json) for the aggregate review scope.

The selected implementation includes direct-prediction and unwarped-loss experiments as well as the normalized accumulator. Entry points use portable release commands. [source_manifest.json](source_manifest.json) records original source hashes and release hashes under public file names; [config_manifest.json](config_manifest.json) records original configuration hashes without internal run IDs.

The release includes 94 matching training/model configuration pairs and six classical-baseline selections. Every training configuration was checked through the actual training parser; intercepted training configuration output also matched its corresponding model-loading configuration in all 94 cases. The check used synthetic loader stubs and did not train 94 models. See [training_model_validation.json](training_model_validation.json).

## Changes for release

Python comments and module/class/function docstrings were removed using syntax-aware transformation. No personal authors, affiliations, submission numbers, hostnames, job metadata or private absolute paths are included in the selected payload. Public upstream URLs and licensing remain in documentation. Citation and MIT copyright use the project contributor group.

Dataset paths now come from `diffsr/paths.py` environment variables. Internal snapshot lookup was replaced by the release source. Portable training, inference, benchmark, deployment-validation, audit and packaging commands were added. The STSS packer previously used its module docstring as parser text; removing that docstring required replacing it with an explicit description. A missing synthetic-pattern dependency was also included after checking every package import.

The homepage includes five selected paper figures described in [MEDIA.md](MEDIA.md). No manuscript PDFs, internal notes, logs, environments, cached bytecode, raw dataset files, LMDBs, checkpoints, metric weights, compiled engine files or external upstream checkouts are included in the source package.

## Previously recorded checks

The pre-existing release record reports Python 3.12.13, PyTorch 2.10.0 and the direct dependencies listed in [ENVIRONMENT.md](ENVIRONMENT.md). CUDA was unavailable to that validation process. These are historical results. Current checks and publication gaps are documented in [OPEN_SOURCE_REVIEW.md](OPEN_SOURCE_REVIEW.md).

| Check | Result |
| --- | --- |
| Synthetic test suite with optional perceptual dependencies | **117 passed** |
| Core suite without optional metric packages | 115 tests available; two optional metric tests skip |
| Training parser and training/loading settings | 94/94 configurations consistent |
| Source-model parity | 12 four-frame R/J sequences across ×2/×3/×4 and normalized/direct prediction, bitwise equal |
| QRISP-style source parity | Four frames with a compact comparison network, bitwise equal |
| Source temporal-loss parity | Four warp/norm combinations, loss values and gradients bitwise equal |
| Synthetic end-to-end execution | Four training updates, checkpoint save, reload, inference and full-sequence scoring passed |
| Dataset conventions | Both STSS channel layouts and Arm signs/depth/tone mapping passed on generated inputs |
| Reparameterized convolution folding | Four-frame eager parity passed |
| Optional metrics | Identical-image FLIP = 0; identical-video ColorVideoVDP JOD = 10 |
| Command help / module imports | All 22 script help entries and 33 package module imports passed |
| Python static checks | Undefined-name checks and formatting passed |
| Package build | Source wheel built and imported from an independent installation |
| Citation metadata | Validated against the CFF schema |
| Payload audit | Python parses, no comments/docstrings, valid JSON, no detected private paths/emails/credentials/submission IDs |

Detailed source parity is recorded in [parity_validation.json](parity_validation.json). The source-code audit checks the explicitly selected release files; it does not claim to detect every conceivable secret through arbitrary obfuscation.

## Validation limits

Full dataset training and paper metrics were not rerun. No trained checkpoints are supplied. The published reconstruction values require the original data, configuration and completed training; synthetic checks do not reproduce those values.

CUDA/Triton kernels were parsed and imported but not executed on a GPU. H200 timings, bf16 execution, CUDA graph replay and real-data deployment parity remain to be measured in the appropriate CUDA environment. `scripts/validate_deployment.py` and `scripts/benchmark.py` are supplied for those checks. External LPIPS metric weights were not downloaded for the release audit.

The archive records an inherited system environment rather than a complete original lock. Requirements pin the observed validation stack and the perceptual-metric versions, with the CUDA wheel selected separately for the target machine.

## Make a source package

From the repository root:

```bash
python scripts/audit_release.py
python scripts/package_release.py
```

The packer uses an explicit allowlist, writes `SHA256SUMS`, and creates `frozen-history-source.zip`. ZIP timestamps and permissions are normalized and do not contain filesystem ownership metadata. Source, configurations, documentation, tests and the five approved homepage figures are included. `.gitignore` excludes local data, weights, results, PDFs and archives from a future Git checkout, with exact exceptions for the homepage figures.

Extract the ZIP to a fresh folder for upload. Keep the dataset providers' terms and upstream software notices separate from MIT. Publication identifiers and personal citation metadata can be added later when an identified release is desired.
