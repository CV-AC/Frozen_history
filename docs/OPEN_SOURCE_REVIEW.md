# Open-source review

Reviewed on 12 October 2026.

The repository meets the practical requirements for a public, source-only research release. It includes executable experiment code, configuration files, synthetic tests, installation instructions, dataset preparation, evaluation protocols, a license and citation metadata. Full paper reproduction remains unverified in this review because datasets, trained checkpoints and CUDA hardware were unavailable.

GitHub's [README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) describes the role of a README, license, citation file and contribution guidelines. The criteria below also cover experiment reproduction; they are a review of this repository, not a GitHub certification.

## Findings

| Area | Assessment | Evidence or remaining work |
| --- | --- | --- |
| Repository structure | Suitable | `diffsr/`, `scripts/`, `configs/`, `docs/` and `tests/` have separate roles. |
| Core experiment code | Present | Recurrent supersampling, registration, temporal losses, comparison models and evaluation entry points are included. |
| Configurations | Present | 94 training/model pairs and six classical-baseline configurations; parser coverage is included in the suite. |
| Installation | Documented | Python 3.12 instructions, pinned direct dependencies, CPU/CUDA wheel choices and optional dependency groups. |
| Step-by-step use | Documented | `GETTING_STARTED.md` covers synthetic checks, data packing, short runs, training, inference, transfer and deployment. |
| Data access | Documented, external | Provider links, expected layouts, counts and geometry conventions; data are not redistributed. |
| Tests | Included | Synthetic unit and integration tests exercise model behavior, adapters, training, checkpoint reload and scoring. |
| CI | Configured | CPU tests, dependency consistency, undefined-name checks, source audit and packaging run on push and pull requests. Hosted CI has not been observed in this review. |
| Licensing and attribution | Present | MIT project license and separate upstream/data notices. Upstream attribution remains in the release. |
| Citation | Anonymous metadata present | Contributor-group citation; public paper identifier and repository URL remain to be added when available. |
| Personal and infrastructure metadata | No matches in audited source | No detected personal email, private path, submission ID, job directive or common credential pattern. This is pattern-based inspection. |
| Comments and docstrings | Removed | Python token and AST checks enforce the requested convention. Explanations remain in documentation. |
| Packaging | Explicit source selection | Source, configurations, documentation, tests and five approved paper figures, with checksums and normalized ZIP metadata. |
| Prior Git history | Unavailable | The supplied directory had no `.git` during the review; earlier commit authors, revisions, tags and remotes could not be inspected. Publication starts with a new source-release history. |
| Full paper results | Unverified | Requires the original data, complete training and comparison against the paper's reported values. |
| Paper table and figure mapping | Partial | The homepage media inventory maps five main figures and selected Table 1/2 results. Numbered supplementary variants still lack a complete table/figure mapping and expected outputs. |
| GPU performance | Unverified | CUDA/Triton execution, bf16 behavior, graph replay and H200 latency require a GPU run. |
| Training resources | Incomplete | No measured minimum VRAM, dataset storage requirement or training duration is supplied. Measure these on the target hardware. |
| Complete environment lock | Incomplete | Direct packages are pinned; transitive packages and platform wheels need an experiment-specific environment record. |

## Changes made during this review

The release audit now checks macOS and Windows user paths, additional storage paths, scheduler options and more credential formats. It also rejects empty Python docstrings and scans the audit script itself. Regression tests cover those cases and a symlink escaping the source tree.

The English getting-started guide replaces the Chinese README. Remaining Chinese status messages in the static-segment script were translated. The guide explains the difference between a short setup check and the released paper configurations, states the resume behavior, and distinguishes random-weight timing from trained-model timing. CI now checks installed dependency consistency, runs the configured static checks and builds the source archive.

The previous validation report is labelled as a historical record. Its source-parity and optional-metric claims are preserved for reference, but they are not counted as checks performed in this review. The original research archive was not available for source-parity reassessment. A later homepage update used the local arXiv manuscript to check selected figures, captions and Table 1/2 values; this does not validate every paper experiment.

## Current validation

Current checks use a fresh Python 3.12.14 environment on macOS arm64, with PyTorch 2.10.0 and the direct packages from `requirements-dev.txt`. CUDA is unavailable.

| Check | Result |
| --- | --- |
| Full synthetic test suite | Homepage update: 131 passed, 2 skipped in 3.64 seconds. The initial cold run took 42.92 seconds before audit regressions were added. FLIP and ColorVideoVDP skipped because their optional packages were absent. |
| Audit regression tests | All 16 passed, including PNG integrity and metadata checks. |
| Synthetic integration | Four training updates, checkpoint reload, four inference outputs and sequence scoring passed within the full suite. |
| Dependency consistency | `pip check` passed. |
| Static checks | Configured Ruff undefined-name checks passed. |
| Formatting of changed audit files | Passed. |
| Editable package installation | Passed. |
| Wheel build | `frozen_history-1.0.0-py3-none-any.whl` built successfully. |
| Training command | The ×4 configuration dry run completed. |
| Static-segment command | Help entry completed after status-message translation. |
| Language scan | No Chinese text remains in the selected code and documentation. |
| Source payload audit | Passed with no detected issues. |
| Extracted archive | All 288 payload checksums matched, including the five paper figures; no extra files or broken document links; source-manifest hashes matched. |
| Archive metadata | ZIP timestamps and permissions normalized; no ZIP extra fields. |

The source archive and its checksums are regenerated after document changes. The rendered homepage was checked in a browser: five figures loaded and internal navigation resolved. Full dataset training, real-data inference, optional perceptual metrics, source parity against the original research archive and GPU execution were not run during this review.

## Before claiming paper reproduction

1. Record the downloaded dataset versions and checksums; complete the adapter validation commands.
2. Train the released configurations and retain their resolved settings, checkpoints and software environment records.
3. Run the full protocols with the documented sequence resets, warm-up, frame selection and aggregation.
4. Compare the generated metrics with the paper tables. Record discrepancies and any changed settings.
5. Validate eager/fused recurrence on real sequences and measure the timing workload on the stated GPU.
6. Report measured training time, peak GPU memory and local dataset storage for the reproduced runs.
7. Map each paper table and figure to its configuration, evaluation command and expected output, including the numbered supplementary variants.

Missing pretrained weights make first use slower, but do not prevent a source release. Providing checkpoints later would let readers evaluate the method before repeating training. Until full experiments have been run, describe the release as code and protocols for reproduction and state the validation limits.

## Publishing without private metadata

Use the newly generated archive and inspect its extracted contents before upload. The packer scans selected source files; it does not inspect arbitrary local files excluded from the payload. Local virtual environments and experiment records can contain installation paths and remain excluded.

Source anonymization does not anonymize a GitHub account or commit history. If the release must be anonymous, review the publishing account and commit metadata as part of publication. Preserve public upstream credits and required license notices. Add a paper URL and stable release tag when those are available.
