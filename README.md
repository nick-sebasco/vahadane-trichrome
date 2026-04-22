# vahadane-trichrome

Vahadane-style stain extraction and normalization for histology images, adapted for Mason's Trichrome workflows (3-stain setup).

## Why this project exists

Most open-source stain normalization implementations are adapted to H&E (2-stain assumptions).
Mason's Trichrome is a 3-stain problem, and those assumptions break down.

This project focuses on trichrome-first normalization so you can:

- map source slide color appearance to a target reference slide,
- preserve structure while reducing stain appearance drift,
- inspect tissue masks and stain swatches for debugging and QA.

## Reference

- Vahadane et al., *Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images* (IEEE TMI, 2016):
	https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7460968

## Features

- `VahadaneTrichromeExtractor`
- `VahadaneTrichromeNormalizer`
- Validation helpers for cohort Wasserstein distance and structure-preserving SSIM audits
- Source-to-target stain basis row alignment for 3 components
- Optional source tissue masking during transform
- Robust scientific tests for concentration recovery, alignment, sparsity, and structure-preservation sanity
- Debug outputs:
	- raw source/target/transformed exports
	- ROI mask images
	- stain swatches (including human-interpretable RGB mode)

## Installation

Package publishing is in progress. For now, use Poetry:

Requires Python 3.11 or 3.12.

```bash
poetry install
```

Then enter the environment:

```bash
poetry shell
```

Run tests:

```bash
pytest -q
```

## Testing

Core scientific suite:

```bash
poetry run pytest tests/test_vahdane_trichrome.py -q
```

Run with artifact paths printed:

```bash
poetry run pytest tests/test_vahdane_trichrome.py -q -s
```

Artifacts are written to `tests/test_outputs` when that folder exists, otherwise pytest `tmp_path` is used.

## Usage

### Example 1: Basic normalization

```python
import numpy as np
from vahadane_trichrome import VahadaneTrichromeNormalizer

# source_rgb and target_rgb must be uint8 arrays of shape (H, W, 3)
source_rgb: np.ndarray = ...
target_rgb: np.ndarray = ...

normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.65,
		regularizer=0.1,
		sort_mode="none",
)

normalizer.fit(target_rgb)
normalized_rgb = normalizer.transform(source_rgb)
```

### Example 2: Normalize with source tissue mask

```python
normalized_masked = normalizer.transform(
		source_rgb,
		apply_source_tissue_mask=True,
)
```

### Example 3: Save debug artifacts (ROI + swatches)

```python
roi_outputs = normalizer.save_roi_images(
		source_img=source_rgb,
		target_img=target_rgb,
		output_dir="outputs/vahadane_debug",
		prefix="run1",
)

swatch_outputs = normalizer.save_stain_vector_swatches(
		output_dir="outputs/vahadane_debug",
		prefix="run1",
		rgb=True,
)

print(roi_outputs)
print(swatch_outputs)
```

### Example 4: Validate batch-effect reduction across cohorts

```python
from vahadane_trichrome import cohort_wasserstein_matrix
from vahadane_trichrome import summarize_reference_cohort_improvement

# Run once on raw cohorts.
before = cohort_wasserstein_matrix(
    {
        "NW": nw_raw_paths,
        "BU": bu_raw_paths,
        "KD": kd_raw_paths,
    },
    feature_domain="od",
    luminosity_threshold=0.9,
    max_pixels_per_image=50_000,
)

# Normalize images separately with your chosen parameters, then evaluate the
# resulting outputs in a second pass.
after = cohort_wasserstein_matrix(
    {
        "NW": nw_raw_paths,
        "BU": bu_normalized_paths,
        "KD": kd_normalized_paths,
    },
    feature_domain="od",
    luminosity_threshold=0.9,
    max_pixels_per_image=50_000,
)

improvement = summarize_reference_cohort_improvement(
    before,
    after,
    reference_cohort="NW",
)

print(before.distance_matrix)
print(after.distance_matrix)
print(improvement.deltas)
```

### Example 5: Validate structure preservation with SSIM

```python
from vahadane_trichrome import paired_structural_similarity

result = paired_structural_similarity(
    sources=source_paths,
    transformed=normalized_paths,
    feature_domain="lab_l",
    luminosity_threshold=0.9,
)

print(result.mean_score)
print(result.scores)
```

`cohort_wasserstein_matrix` only computes cohort-to-cohort Wasserstein distances on the images you pass in.
It does not perform stain normalization internally, which keeps normalization settings under your control.
It is useful for batch-effect assessment because normalization should reduce the distance between an external cohort and the reference cohort.
`paired_structural_similarity` complements that by checking that normalization preserved tissue structure rather than simply forcing colors to match.

Important assumptions and limitations:
- Cohort Wasserstein distance here is based on pooled tissue-pixel feature distributions, not matched slide pairs or matched regions.
- It is informative for batch-effect assessment, but it is not a pure stain-only metric because biology, tissue composition, masking quality, scanner effects, and artifacts can also shift the distributions.
- It ignores spatial arrangement. Two cohorts can have similar color distributions but very different tissue structure.
- Saving distribution plots alongside the scalar distances is often helpful for interpretation, especially when comparing before-vs-after normalization runs.

## Input expectations

- RGB images as `np.uint8`
- shape `(H, W, 3)`
- fit first on a target image, then transform source images

## Current limitations

- Packaging to PyPI is not complete yet.
- API/module paths may change during packaging refactor.
- Legacy playground code in this repo may still reference comparison workflows not required for the core trichrome pipeline.

## Planned next algorithmic step

- Evaluate NNMF backend (HistomicsTK/Wu method) against current dictionary-learning behavior.
