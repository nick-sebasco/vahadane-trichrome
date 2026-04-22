# Internal TODO (gitignored)


+ [x] make slides
+ [x] how do i describe in picture what its doing color histogram normalization
+ [x] how do we convey the math+ make intended audience ss

+ [] two very different images, transform toward chosen target individeual also do multi source transformation.  

+ [x] histogram matching should be done at level 0.

## 2 component vahadane
This section we disregard the 3rd stain component (dark nuclei)

+ [x] create NW multi-target
+ [] train vahadane stain normalization model
+ [] reuse stain normalization model from above and apply to all BU & kadmon images.
+ [] run feature extraction pipeline on new stain normalized features
+ [] Do before/ after PCA analysis

+ [x] Experiment with 2 component Vahdane
+ [x] multi-target histogram matching.  we dont want just a single reference hiostogram.  we want multiple targets and multiple sources then once we have learnd that mapping we apply to all indoividual source.
-- see slides: 
https://docs.google.com/presentation/d/1YiNX2aigXzY6Xy10B56pwwOZCwQf8xyQLs0ASHf77OQ/edit?usp=sharing

## Experiments

+ [x] try running at lower zarr levels 2
-- see: /home/sebasn/vahadane-trichrome/outputs/bu10_to_nw_single_level2_cd_omp & /home/sebasn/vahadane-trichrome/outputs/bu10_to_nw_multi_level2_lars_lasso_lars for level 2 runs.  the results don't look as dark as level 4 but same anomally of lighter target shifting the source darker somehow?  
-- Added new sanity test where we make a toy image with a dark target and light source and verify that the source gets darker in the normalized output: /home/sebasn/vahadane-trichrome/tests/test_vahadane_trichrome_image_sanity.py


+ [x] swatch code to be post-sort

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

+ [] show within slide normalization
+ [] show within cohort normalization
+ [] show NW-BU
+ [] show NW-KD
+ [] Show cluster image feature pc and show before after stain normaliztaion

## Engineering

+ [x] TiaToolbox has max_iter set to 3 which is too low, this implementation defaults to 100 and let's the user control.
+ [x] add new threshold method and use connected components.
+ [x] Add load/ save model
+ [x] Add NNMF backend (HistomicsTK/Wu-style) and benchmark against current method.
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
