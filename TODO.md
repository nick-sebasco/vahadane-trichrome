# Internal TODO (gitignored)

## Experiments

+ [x] try running at lower zarr levels 2
-- see: /home/sebasn/vahadane-trichrome/outputs/bu10_to_nw_single_level2_cd_omp & /home/sebasn/vahadane-trichrome/outputs/bu10_to_nw_multi_level2_lars_lasso_lars for level 2 runs.  the results don't look as dark as level 4 but same anomally of lighter target shifting the source darker somehow?  
-- Added new sanity test where we make a toy image with a dark target and light source and verify that the source gets darker in the normalized output: /home/sebasn/vahadane-trichrome/tests/test_vahadane_trichrome_image_sanity.py


+ [] swatch code to be post-sort

-- it seems like the purple in target swatch is being used innappropriately
-- optimization: take a levels 4 amount of random pixels from a level 0

+ [x] look into implementation of multi-reference
-- A. Matts idea of concatenating images is very good, I can do that.
-- B. Since it is mathematically valid to average W if we find W's separately for each image
    we can solve for each W in parallel, so that might be an advantage over A

-- paper uses 80:20 rule. we could take a random 20% sample of reference dataset

+ [] How long does this take to run?  do a break down by plotting run time at different zarr levels: 6, 4, 2, 0 and compare with Reinhardt
+ [] what is highest zarr level where I can still learn the black/ brown stain vector in trichrome
+ [] show effect of poor thresholding
+ [] show sparsity lambda effect
--what is standard practice for setting this.
+ [] how do we evaluate how well vahadane is working

+ [] Add cohort-based Wasserstein distance evaluation (paper-inspired)
Goal:
- Quantify stain-normalization consistency by comparing a reference cohort (ex: NW)
	to itself and to external cohorts (ex: BU, KD) using pairwise Wasserstein distances.

Implementation plan:
- Add utility function for cohort evaluation:
	- input: cohorts dict[str, list[path]], reference_cohort key
	- configurable feature domain: "od" (default), optional "lab_l"
	- optional tissue masking using luminosity threshold
	- optional per-image pixel subsampling cap for runtime control
- Implement 1D Wasserstein in pure NumPy (no new hard dependency):
	- validate against SciPy in tests when SciPy is available (optional test branch)
	- use scalar distributions consistently (do not pool unrelated channels silently)
- Compute and return:
	- reference internal pair distances: ref <-> ref
	- cross distances: ref <-> other cohort for each other cohort
	- summary statistics per comparison: n_pairs, mean, std, median, p05, p95
- Add LLN/convergence analysis helper:
	- randomize pair order
	- plot running mean Wasserstein vs number of pairs
	- verify convergence behavior for increasing sample size

Validation / experiments:
- Run NW as reference cohort:
	- NW <-> NW (internal baseline)
	- NW <-> BU
	- NW <-> KD
- Expectation: mean(NW<->NW) < mean(NW<->BU) and mean(NW<->NW) < mean(NW<->KD)
- Repeat with fixed random seeds to check stability.

Artifacts to save:
- CSV/JSON summary table of pairwise statistics
- LLN convergence plots per comparison
- optional histogram/KDE overlays for distance distributions

Definition of done:
- Utility callable from script/notebook with arbitrary number of cohorts
- Unit tests for 1D Wasserstein numerical sanity (symmetry, identity, shift behavior)
- End-to-end run on NW/BU/KD with saved outputs and brief interpretation note

+ [] show within slide normalization
+ [] show within cohort normalization
+ [] show NW-BU
+ [] show NW-KD
+ [] Show cluster image feature pc and show before after stain normaliztaion

## Engineering

+ [x] TiaToolbox has max_iter set to 3 which is too low, this implementation defaults to 100 and let's the user control.
+ [] add new threshold method and use connected components.
+ [x] Add load/ save model
+ [] Add NNMF backend (HistomicsTK/Wu-style) and benchmark against current method.
+ [] Add the smart patch based strategy from the paper

As explained in Sections III-A and III-B, a majority of computational time of SPCN is spent in the iterative optimization of SNMF, which slows its performance on WSI, especially when a computer RAM is limited with respect to the size of the WSI. Therefore, we propose a novel acceleration scheme for estimation of global color appearance matrix of a WSI based on smart patch sampling and patch-wise stain separation. The patches have the same resolution as the original WSI to preserve local structures, which could have been lost by using downsampling -- a trivial alternative. 

We sample patches centered at corner points on grid as shown by green dots in Fig. 4, and discard those that lie in whitespace by comparing their luminosity against a threshold (we use 0.9 for all of our testing). The luminosity is the $L$ value in the $L*a*b$ color space. Then we estimate the color basis matrices $W_i$ for each of the sampled patches indexed by $i$ using SNMF. Stain color columns in the $W_i$ are sorted by ranking the blue channel intensities such that first column corresponds to hematoxylin and second corresponds to eosin. 

Finally, we take element-wise median of these matrices to make color estimation more robust to artifacts such as folding, blurring and holes. We normalize this median matrix to have unit vector columns, and denote the final color matrix thus obtained as $W$. The stain separation for WSI is obtained through color deconvolution:

$$H = W^+V, \quad H \geq 0$$

where $W^+ = (W^T W)^{-1} W^T$ is the Moore-Penrose pseudo-inverse matrix of $W$. Note that this operation can also be done separately for sub-images of WSI by using a single color appearance matrix $W$ obtained for the entire image as described above i.e., obtaining $H$ through pseudo-inverse will hold for any sub-image and hence can be parallelized.

After stain separation of source and target WSIs, $V_s = W_s H_s$ and $V_t = W_t H_t$, respectively, we change the color appearance of the source WSI to that of the target WSI while preserving original source stain concentration to obtain normalized source WSI.

+ [] Are there gpu/ parallelization optimizations

## PyPI packaging

- Move implementation into `src/vahadane_trichrome/` package modules.
- Add `__init__.py` public API exports.
- Add optional extras for dev/testing.
