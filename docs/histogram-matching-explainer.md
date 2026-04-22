# Histogram Matching Explainer

This document is meant to help you explain histogram matching in conversation, in writing, and on slides.

The goal is not just to state the algorithm correctly. The goal is to build intuition at multiple levels:

1. what it is trying to do,
2. how the math works,
3. why multi-target helps,
4. why multi-source is a real conceptual improvement and not just "more data",
5. how this repository implements it.

## Naming

In this repository:

- `source` = the image or cohort we want to normalize
- `target` = the image or cohort we want the source to resemble

That means:

- our `target` is what many papers would call the `reference`
- our `source` is the image that starts out "unnormalized"

## The Main Idea In One Sentence

Histogram matching says:

> keep a pixel's relative rank within the source, but express that rank using the target's color distribution.

Another way to say it:

> if a pixel is "fairly dark blue" relative to the source slide, make it "fairly dark blue" relative to the target slide.

It is not copying the target image. It is matching relative position inside a distribution.

## The Mental Model That Usually Lands Best

Ignore images for a moment and imagine we take all blue-channel tissue pixels from a slide and sort them from darkest to brightest.

Now do the same for the target slide.

Histogram matching says:

- the darkest source pixels should map to the darkest target-like pixels
- the middle source pixels should map to the middle target-like pixels
- the brightest source pixels should map to the brightest target-like pixels

So the method preserves ordering, but changes the scale of values.

This is why it often feels like:

- "same structures"
- "same rough contrast ordering"
- "different color style"

That is the core intuition.

## A Tiny Concrete Example

Suppose the source blue values are:

$$
[20, 20, 40, 90, 200]
$$

and the target blue values are:

$$
[10, 30, 60, 140, 220]
$$

If we sort both lists, then:

- the 1st/2nd darkest source pixels with value `20` should map near the 1st/2nd darkest target values
- the middle source pixel with value `40` should map near the middle target value `60`
- the 4th source pixel with value `90` should map near `140`
- the brightest source pixel with value `200` should map near `220`

So a reasonable mapping would be roughly:

$$
20 \mapsto 10 \text{ or } 30,\qquad
40 \mapsto 60,\qquad
90 \mapsto 140,\qquad
200 \mapsto 220
$$

That is almost the whole method.

The histogram and CDF machinery is just the efficient, general way to do this for all 256 intensity values.

## Why Histograms Enter The Story

For an 8-bit image channel, possible intensity values are:

$$
0, 1, 2, \dots, 255
$$

A histogram answers:

> how many pixels had each possible intensity value?

So if we focus on one channel, like blue:

- the histogram tells us how blue values are distributed
- the cumulative histogram tells us what fraction of pixels are at or below each value

That cumulative view is exactly what we need for percentile matching.

## The Math, Built Slowly

### Step 1: Work with one channel

For now, pretend we are only matching one channel, for example the blue channel.

Let:

- $v$ = a possible intensity value, from $0$ to $255$
- $h_{\text{source}}(v)$ = number of source pixels with intensity exactly $v$
- $h_{\text{target}}(v)$ = number of target pixels with intensity exactly $v$

So if $h_{\text{source}}(120) = 500$, that means the source image has 500 pixels whose blue value is exactly 120.

### Step 2: Count how many pixels there are in total

Let:

$$
n_{\text{source}} = \sum_{v=0}^{255} h_{\text{source}}(v)
$$

$$
n_{\text{target}} = \sum_{v=0}^{255} h_{\text{target}}(v)
$$

These are just the total number of pixels being used for matching.

In this repository, these are usually not all pixels in the image. They are usually only the tissue pixels after masking.

### Step 3: Build the cumulative distribution function

First define the cumulative count:

$$
H_{\text{source}}(v) = \sum_{u=0}^{v} h_{\text{source}}(u)
$$

$$
H_{\text{target}}(v) = \sum_{u=0}^{v} h_{\text{target}}(u)
$$

Interpretation:

- $H_{\text{source}}(v)$ = how many source pixels have intensity less than or equal to $v$
- $H_{\text{target}}(v)$ = how many target pixels have intensity less than or equal to $v$

Now divide by the total number of pixels to get a fraction:

$$
F_{\text{source}}(v) = \frac{H_{\text{source}}(v)}{n_{\text{source}}}
$$

$$
F_{\text{target}}(v) = \frac{H_{\text{target}}(v)}{n_{\text{target}}}
$$

These are the empirical cumulative distribution functions, or CDFs.

Interpretation:

- $F_{\text{source}}(v)$ = fraction of source pixels with value $\leq v$
- $F_{\text{target}}(v)$ = fraction of target pixels with value $\leq v$

This is the key moment:

the CDF converts an intensity value into a percentile-like quantity.

If:

$$
F_{\text{source}}(120) = 0.80
$$

that means about 80% of source pixels are at or below intensity 120.

So intensity 120 sits around the 80th percentile of the source distribution.

### Step 4: Match equal percentile to equal percentile

Now suppose a source intensity value $v$ sits at source percentile:

$$
F_{\text{source}}(v)
$$

We want to find the target intensity $w$ that sits at the closest target percentile.

So the matching rule is:

$$
m(v) = \underset{w \in \{0,\dots,255\}}{\arg\min}
\left|F_{\text{target}}(w) - F_{\text{source}}(v)\right|
$$

In words:

- look up the percentile of source value $v$
- search over all target intensities $w$
- pick the one whose target CDF is closest

That gives a lookup table:

$$
m : \{0,\dots,255\} \to \{0,\dots,255\}
$$

meaning every possible source intensity gets assigned a target-like intensity.

### Step 5: Apply that lookup table to every pixel

If the source image is $I_{\text{source}}(x,y)$, then the matched image is:

$$
I_{\text{matched}}(x,y) = m\!\left(I_{\text{source}}(x,y)\right)
$$

So once the lookup table is learned, transformation is easy:

- read source pixel value
- look it up in the table
- write the mapped value

That is why histogram matching is fast.

## The One Equation To Remember

If you only want one idea to remember, it is this:

1. take a source intensity value
2. ask what percentile it sits at in the source
3. find the target intensity at about that same percentile
4. map the source value to that target value

That mapping is what the function $m$ means.

So:

$$
m(v)
$$

just means:

> "when the source intensity is $v$, what target intensity should it become?"

For example, if:

- source intensity $120$ sits near the 80th percentile in the source
- target intensity $95$ sits near the 80th percentile in the target

then:

$$
m(120) = 95
$$

That is the whole idea.

If you want the slightly more formal version, it is:

$$
m(v) = \text{the target intensity whose CDF value is closest to } F_{\text{source}}(v)
$$

In other words:

- compute the source percentile of $v$
- search through target intensities
- choose the one whose percentile is closest

The earlier equation with `argmin` says exactly the same thing, just in more compressed math notation.

## What The Code Actually Computes

The implementation is in [src/vahadane_trichrome/methods/histogram_matching.py](/home/sebasn/vahadane-trichrome/src/vahadane_trichrome/methods/histogram_matching.py).

The code does not store CDFs as numbers from 0 to 1.

Instead, in `_compute_discrete_cdf(...)`, it computes:

$$
\widetilde{F}(v) = \operatorname{round}\!\left(255 \cdot F(v)\right)
$$

So instead of cumulative probabilities in $[0,1]$, it uses integer-scaled cumulative values in $[0,255]$.

That is mathematically the same idea. It is just a discrete implementation detail that makes the LUT construction convenient.

## Why RGB Images Need Three LUTs

An RGB image has three channels:

- red
- green
- blue

This implementation matches each channel independently:

$$
m_R,\qquad m_G,\qquad m_B
$$

and applies:

$$
I_{\text{matched}}(x,y,c) = m_c\!\left(I_{\text{source}}(x,y,c)\right)
$$

for channel $c \in \{R,G,B\}$.

This is simple and fast, but it is also an approximation.

It matches:

- the red histogram,
- the green histogram,
- the blue histogram

separately.

It does **not** model the full joint color distribution or explicit stain physics.

That limitation matters and should be said clearly.

## Why Tissue Masking Matters So Much

If we learn histograms from the whole slide image patch, then a large amount of white background can dominate the distribution.

That is bad because the algorithm would spend much of its effort learning how to match whitespace rather than stained tissue.

So this repository usually does the following first:

1. detect tissue using a luminosity threshold
2. optionally clean the mask with connected components
3. use only masked tissue pixels to estimate the histograms

So in practice:

- $h_{\text{source}}(v)$ is really the histogram of source **tissue** pixels
- $h_{\text{target}}(v)$ is really the histogram of target **tissue** pixels

This is one of the most important implementation details.

## Single-Target Histogram Matching

Classic histogram matching is:

- one source image
- one target image

For each source image, learn a mapping from that source to that target.

Conceptually:

$$
\text{one source slide} \longrightarrow \text{one target slide}
$$

This is useful, but it can be unstable when either image is unusual.

## Why Multi-Target Is Intuitive

Multi-target means:

- do not define "the desired appearance" using just one target slide
- define it using a cohort of target slides

For one channel, instead of collecting target pixels from just one image, pool them from several:

$$
\text{target values}
=
\text{concatenate}\left(
\text{target}_1,\text{target}_2,\dots,\text{target}_M
\right)
$$

Then compute the target histogram and target CDF from that pooled set.

This asks:

> what does the target cohort look like overall?

instead of:

> what does this one chosen target slide look like?

That makes intuitive sense to most people right away.

## Why Multi-Source Is More Subtle, But Important

This repository goes one step further:

- it also pools the source cohort
- then learns one fixed cohort-to-cohort mapping

For one channel:

$$
\text{source values}
=
\text{concatenate}\left(
\text{source}_1,\text{source}_2,\dots,\text{source}_N
\right)
$$

and then:

$$
m_{\text{cohort}}(v)
=
\underset{w}{\arg\min}
\left|F_{\text{target cohort}}(w) - F_{\text{source cohort}}(v)\right|
$$

This is not just "more data."

It changes what the fitted model means.

### Per-slide fitting means

For each individual source slide:

- estimate that slide's own distribution
- build a mapping specifically for that slide

That is adaptive, but it can overreact to:

- unusual tissue composition
- scanner quirks
- section thickness differences
- masking errors
- one slide being especially dark or especially pale

### Multi-source fitting means

Estimate the source cohort distribution once, using many source slides, and treat that pooled distribution as the thing you are normalizing.

This means:

- the learned mapping represents a cohort, not a single slide
- every slide in that cohort is normalized using the same cohort-level rule
- the result is often more stable and more internally consistent

This is why I would describe multi-source as:

> learning a stain-normalization model for a cohort, not just for a slide

That is the important conceptual step.

## The Most Useful Verbal Contrast

This phrasing tends to work well in conversation:

- multi-target makes the destination more representative
- multi-source makes the starting point more representative

Or more formally:

- multi-target reduces over-reliance on one reference slide
- multi-source reduces over-reliance on one source slide

## How The Repository Implements It

### Core pieces

- `_compute_discrete_cdf(...)`
  - builds a 256-bin histogram with `np.bincount`
  - takes the cumulative sum
  - scales to the integer range `0..255`

- `build_histogram_specification_lut(source_channel, target_channel)`
  - computes source and target CDFs
  - for each possible source intensity, finds the target intensity with closest CDF value
  - returns a 256-entry LUT

- `histogram_specification(source_img, target_img, source_mask=..., target_mask=...)`
  - learns per-channel LUTs from masked source and target pixels
  - applies them channel by channel

- `build_cohort_histogram_specification_lut(source_images, target_images, ...)`
  - pools masked source pixels across a source cohort
  - pools masked target pixels across a target cohort
  - learns one fixed LUT per channel

### Normalizer behavior

`HistogramMatchingNormalizer` has two important modes:

- `fit(target_img)`
  - classic single-target mode
  - the mapping is computed with respect to one target image

- `fit_multi_source_target(source_imgs, target_imgs)`
  - cohort mode
  - the LUTs are learned once from pooled source and pooled target cohorts
  - those LUTs are then reused for every transformed source image

During `transform(...)`:

1. tissue is re-detected on the new source image
2. the learned LUT is applied
3. non-tissue can optionally be painted white

## Complexity And Speed

Let:

- $N$ = number of pixels in one image
- $C$ = number of channels, usually 3
- $L$ = number of gray levels per channel, here 256

### Single-image mode

The main costs are:

- tissue masking: roughly $O(N)$
- histogram counting: roughly $O(N)$
- LUT construction: roughly $O(L^2)$ in the current implementation
- LUT application: roughly $O(N)$ per channel

Since $L=256$ is tiny and fixed, runtime is dominated by scanning pixels, not by the LUT math.

### Cohort mode

If the total number of masked pixels pooled across the source and target cohorts is $P$, then fitting is approximately linear in $P$.

The transform step remains approximately linear in the size of the image being transformed.

### Practical benchmark from the current repo examples

From the level-4 BU/NW PNG examples already in this repository:

- single-target fit: about `0.17 s`
- single-target transform of one image: about `0.23 s`
- multi-source/target fit on 3 BU and 3 NW images: about `2.71 s`
- multi-source/target transform: about `0.72 s` per image on average

These are not WSI-level timings, but they support the main message:

- histogram matching is very fast
- the dominant cost is pixel scanning and masking
- it is much cheaper than iterative stain-factorization methods such as Vahadane

## Strengths Of This Approach

- Easy to explain.
- Easy to visualize.
- Deterministic.
- Fast.
- No iterative dictionary learning.
- Tissue masking helps focus the model on stained tissue rather than background.
- Cohort mode gives a stable, reusable cohort-level mapping.

## Honest Limitations

- It is channel-wise RGB matching, not stain-aware optical-density modeling.
- It ignores cross-channel dependence.
- It can still be influenced by tissue composition differences across cohorts.
- A fixed cohort LUT may be less adaptive for a truly unusual individual slide.
- Cohort fitting currently pools large pixel sets, so memory use grows with the number of pooled tissue pixels.
- Matching color distributions does not guarantee recovery of true biological stain concentrations.

## How To Explain It To A Pathology Audience

### One sentence

We remap source slide colors so their tissue-pixel distributions resemble the target cohort, while trying to preserve relative contrast ordering within the source.

### One paragraph

Histogram matching does not copy one slide onto another. Instead, it looks at how tissue-pixel intensities are distributed in the source and in the target, then remaps source values so that a pixel that was, say, unusually dark or unusually bright relative to its own slide stays similarly ranked after normalization, but now according to the target cohort's color distribution. Using multiple target slides makes the destination style more representative, and using multiple source slides makes the learned normalization model more representative of the cohort you are trying to harmonize.

### Phrases that help

- "percentile matching"
- "preserving relative ordering"
- "cohort-to-cohort mapping"
- "learn one stable mapping for the cohort"

### Phrases to avoid

- "recover the true stain concentrations"
- "preserve biology exactly"
- "multi-source is just more references"

That last one is especially important. Multi-source is not just more examples. It changes the model from slide-specific normalization to cohort-specific normalization.

## Recommended Figures

The story-pack script in this repository produces slide-ready figures:

- `01_blue_channel_hist_cdf_lut.png`
- `02_cohort_pooling_schematic.png`
- `03_single_vs_multi_examples.png`
- `04_distribution_dashboard.png`

These correspond naturally to:

- the single-channel math,
- the cohort intuition,
- the visual comparison,
- the cohort-level distribution story.
