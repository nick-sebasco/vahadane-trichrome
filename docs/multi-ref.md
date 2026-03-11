## Multi-target normalization

This project supports a multi-target reference workflow inspired by:

- Multi-target stain normalization for histology slides
	https://arxiv.org/html/2406.02077v2

For this codebase, the two relevant strategies are:

1. Concat
2. Median-post

We intentionally implement median-post, not arithmetic avg-post, because median
aggregation is more robust to outlier reference slides and unstable per-slide
dictionary-learning results.

## 1. Concat

Concat means combining multiple reference images into one large composite image,
then fitting a single target stain matrix on that combined image.

Conceptually:

```python
# Pseudocode
reference_images = [ref1, ref2, ref3, ...]
big_reference = concatenate_images(reference_images)

normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		dl_n_jobs=-1,
)
normalizer.fit(big_reference)
```

Notes:

- We do not provide an in-package concat image builder.
- You can create the composite externally by tiling images horizontally,
	vertically, or in a grid.
- For concat, using `dl_n_jobs=-1` is appropriate because there is one large
	dictionary-learning job and sklearn can use all available CPU cores.
- Concat can overweight visually dominant reference styles if one reference has
	much more tissue area than the others.

## 2. Median-post

Median-post computes one stain matrix per reference image, aligns the matrices,
then aggregates them with a row-wise median.

This is the implemented approach.

### Why alignment is required

Dictionary learning is permutation-invariant. Even if two reference images have
similar stain vectors, their learned rows can appear in different orders.

If matrices are aggregated without alignment first, the result is invalid.

This implementation therefore:

1. Computes one stain matrix per reference image.
2. Chooses an anchor matrix using a medoid-like similarity criterion.
3. Aligns every reference stain matrix to that anchor.
4. Aggregates aligned matrices with a median (default) or mean.
5. Aggregates per-channel target concentration scale statistics using the same
	 aligned row ordering.

### Parallelism strategy

Median-post parallelizes across reference images.

- Each reference image can compute its own stain matrix independently.
- The multi-target fit uses process-based parallelism across references.
- Inside each worker, sklearn dictionary learning is forced to `n_jobs=1` to
	avoid nested oversubscription.
- For non-parallel single-image extraction, `dl_n_jobs=-1` remains useful.

### Usage

```python
import numpy as np
from vahadane_trichrome import VahadaneTrichromeNormalizer

reference_images: list[np.ndarray] = [ref_a, ref_b, ref_c, ref_d]

normalizer = VahadaneTrichromeNormalizer(
		n_components=3,
		luminosity_threshold=0.83,
		sort_mode="none",
		dl_n_jobs=-1,
)

normalizer.fit_multi_target(
		reference_images,
		aggregation="median",
		max_workers=None,
)

normalized = normalizer.transform(source_rgb_uint8)
```

### Parameters

- `aggregation="median"`
	Recommended default.
- `aggregation="mean"`
	Available if you want paper-style avg-post behavior.
- `max_workers=None`
	Uses all available CPU cores for per-reference extraction.
- `anchor_index=None`
	Automatically selects the most representative reference matrix.

## Practical recommendation

If you want the most robust multi-reference behavior in this repository:

1. Use 5-20 diverse target references.
2. Use `fit_multi_target(..., aggregation="median")`.
3. Keep `sort_mode="none"`; alignment is handled explicitly during multi-target fitting.
4. Use a conservative luminosity threshold so background is not merged into one
	 huge connected component before stain extraction.

## Summary

- Concat is simple and can work well, but may overweight dominant references.
- Median-post is more robust and is the implemented multi-target workflow here.
- Correct stain-row alignment before aggregation is mandatory.