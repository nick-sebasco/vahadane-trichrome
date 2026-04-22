# Histogram Matching Slide Outline

This is a slide-ready outline for explaining single-target and cohort-based histogram matching to a smart pathology audience.

## Slide 1: The Problem

Message:

Different cohorts can have real stain-distribution shifts even when the underlying tissue is comparable enough for downstream analysis to benefit from harmonization.

Say:

- BU and NW can differ in color appearance for reasons that are not the biology we care about.
- We want to reduce cohort-specific color drift without changing gross structure.

Figure:

- `02_cohort_pooling_schematic.png`

## Slide 2: What Classic Histogram Matching Does

Message:

Classic histogram matching uses one source slide and one target slide.

Say:

- It learns a mapping from source intensity values to target intensity values.
- The key idea is percentile matching, not pixel copying.

Figure:

- `01_blue_channel_hist_cdf_lut.png`

## Slide 3: The Core Intuition

Message:

A source pixel keeps its relative rank, but that rank is expressed in the target slide's color distribution.

Say:

- If a pixel is at the 80th percentile in the source, map it near the 80th percentile in the target.
- This is why the output looks target-like without simply becoming the target.

Figure:

- `01_blue_channel_hist_cdf_lut.png`

## Slide 4: Why Multi-Target Is Better Than One Target

Message:

One target slide can be idiosyncratic; several targets better represent the desired cohort appearance.

Say:

- A single reference can be too dark, too pale, or compositionally odd.
- Pooling target slides learns a more representative target distribution.

Figure:

- `02_cohort_pooling_schematic.png`

## Slide 5: Why Multi-Source Is The Important Extra Step

Message:

Multi-source changes the model from slide-specific normalization to cohort-specific normalization.

Say:

- If we fit one mapping per source slide, each slide can overfit its own quirks.
- Pooling the source cohort asks what the typical source distribution is.
- Then we learn one stable cohort-to-cohort mapping and apply it consistently.

Good phrase:

> "Multi-target makes the destination more representative. Multi-source makes the departure point more representative."

Figure:

- `02_cohort_pooling_schematic.png`

## Slide 6: What The Code Actually Does

Message:

The implementation is simple, deterministic, and inspectable.

Say:

- detect tissue
- ignore background
- pool source and target tissue pixels per channel
- build one 256-entry LUT per channel
- apply the same LUT to each source image

Code references:

- [src/vahadane_trichrome/methods/histogram_matching.py](/home/sebasn/vahadane-trichrome/src/vahadane_trichrome/methods/histogram_matching.py)

## Slide 7: Visual Comparison

Message:

Single-target and cohort-based matching can lead to visibly different outputs.

Say:

- Show one BU source against the chosen NW target.
- Compare single-target output with the cohort-trained output.
- Point out where the cohort-trained result looks more stable or more representative of the target cohort rather than one slide.

Figure:

- `03_single_vs_multi_examples.png`

## Slide 8: Distribution-Level Comparison

Message:

The visual change is not just anecdotal; we can inspect cohort-level histograms and CDFs.

Say:

- These overlays show whether normalized BU distributions move toward NW.
- CDF plots are especially good because histogram matching is fundamentally a CDF-matching procedure.

Figure:

- `04_distribution_dashboard.png`

## Slide 9: Why This Method Is Attractive

Message:

Histogram matching is fast, transparent, and easy to reason about.

Say:

- no iterative matrix factorization
- easy to visualize
- near-linear in pixel count
- practical for cohort-wide fitting

Use these numbers from the current repo example:

- single-target fit about `0.13 s`
- single-target transform about `0.20 s`
- cohort fit on 3 BU + 3 NW PNGs about `1.37 s`

## Slide 10: Limitations

Message:

This is useful harmonization, not a full stain-physics model.

Say:

- channel-wise RGB matching ignores joint color structure
- tissue composition still matters
- a fixed cohort LUT may under-handle unusual slides
- good masking remains important

Good closing line:

> "This is a robust cohort harmonization tool, not a perfect stain-separation model."

## Speaker Notes By Audience

### If the room is mostly pathologists

Emphasize:

- background exclusion
- cohort consistency
- preserving structure
- avoiding over-reliance on one reference slide

De-emphasize:

- formal CDF notation
- asymptotic complexity

### If the room is computational

Emphasize:

- percentile matching
- LUT construction
- cohort pooling
- complexity and failure modes

### If you only have 2 minutes

Use slides 1, 3, 5, 7, and 10.
