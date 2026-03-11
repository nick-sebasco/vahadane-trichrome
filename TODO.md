# Internal TODO (gitignored)

## Experiments

+ [] look into implementation of multi-reference 
+ [] How long does this take to run?  do a break down by plotting run time at different zarr levels: 6, 4, 2, 0 and compare with Reinhardt
+ [] show effect of poor thresholding
+ [] show sparsity lambda effect
--what is standard practice for setting this.
+ [] how do we evaluate how well vahadane is working

+ [] show within slide normalization
+ [] show within cohort normalization
+ [] show NW-BU
+ [] show NW-KD
+ [] Show cluster image feature pc and show before after stain normaliztaion

## Engineering
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
