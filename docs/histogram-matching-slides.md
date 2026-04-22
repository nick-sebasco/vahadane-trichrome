# Histogram Matching For Cohort Stain Normalization

## Slide 1

### Why we need this

- Different cohorts can have substantial stain appearance drift.
- That drift can obscure downstream biological comparisons.
- We want cohort harmonization without destroying tissue structure.

Figure:

![](../outputs/histogram_matching_story_pack/02_cohort_pooling_schematic.png)

Speaker note:

This is really a cohort harmonization problem, not just a one-slide beautification problem.

---

## Slide 2

### Classic histogram matching

- Start with one source slide and one target slide.
- Learn how intensity values are distributed in each.
- Remap the source so its intensities follow the target distribution.

Key idea:

> Match percentile to percentile, not pixel to pixel.

---

## Slide 3

### The math in one picture

- For each channel, compute the source and target CDFs.
- For a source intensity value, find the target intensity at the closest CDF value.
- Store that mapping in a 256-entry lookup table.

Figure:

![](../outputs/histogram_matching_story_pack/01_blue_channel_hist_cdf_lut.png)

Speaker note:

If a source pixel is at the 80th percentile of blue intensity in the source slide, send it to about the 80th percentile of blue intensity in the target slide.

---

## Slide 4

### Why multi-target is better than one target

- One target slide may be atypical.
- Several target slides better represent the desired cohort appearance.
- Pooling targets reduces dependence on a single reference slide's quirks.

Message:

> Multi-target makes the destination more representative.

---

## Slide 5

### Why multi-source is the important extra step

- Fitting each source slide separately can overreact to slide-specific quirks.
- Pooling the source cohort defines a cohort-level starting distribution.
- Then we learn one stable cohort-to-cohort mapping and apply it consistently.

Message:

> Multi-source makes the departure point more representative.

Figure:

![](../outputs/histogram_matching_story_pack/02_cohort_pooling_schematic.png)

---

## Slide 6

### What our implementation does

- Detect tissue with a luminosity threshold.
- Refine the mask with connected components.
- Pool source tissue pixels and target tissue pixels by channel.
- Learn one LUT per RGB channel.
- Apply that fixed LUT to each source image.

Code:

- `HistogramMatchingNormalizer.fit(...)`
- `HistogramMatchingNormalizer.fit_multi_source_target(...)`
- `HistogramMatchingNormalizer.transform(...)`

---

## Slide 7

### Real example: single target vs cohort-level fit

- Same BU source slides
- Same chosen NW target for the single-target comparison
- Compare per-slide matching with the cohort-level mapping

Figure:

![](../outputs/histogram_matching_story_pack/03_single_vs_multi_examples.png)

Speaker note:

This is where we can talk through where the cohort-level result looks more stable or more representative of NW overall rather than one chosen NW slide.

---

## Slide 8

### Cohort-level distributions

- Histogram matching is fundamentally about distributions.
- CDF plots are especially natural here because the LUT is learned from CDF alignment.
- We should look at the cohort distribution story, not only one slide at a time.

Figure:

![](../outputs/histogram_matching_story_pack/04_distribution_dashboard.png)

---

## Slide 9

### Why this method is attractive

- Fast and deterministic
- Easy to visualize and explain
- Much cheaper than iterative stain-factorization methods
- Practical for cohort-wide fitting

Current level-4 PNG benchmark in this repo:

- Single-target fit: about `0.17 s`
- Single-target transform: about `0.23 s`
- Multi-source/target fit on 3 BU + 3 NW images: about `2.71 s`
- Multi-source/target transform: about `0.72 s` per image

---

## Slide 10

### Limits and honest framing

- This is channel-wise RGB matching, not stain-aware OD decomposition.
- Tissue composition still affects the learned cohort histogram.
- A fixed cohort LUT may under-handle unusual slides.
- Good masking still matters.

Closing line:

> This is a robust cohort harmonization method, not a full stain-physics model.
