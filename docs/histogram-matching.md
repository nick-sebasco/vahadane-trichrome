Algorithm Documentation from [1]: Histogram SpecificationThe goal of histogram specification is to transform an input image with intensity levels $r$ into an output image with intensity levels $z$, such that the output image follows a specified probability density function (PDF), $p_z(z)$.I. The Continuous FormulationThe theoretical foundation relies on mapping both the input and the target to a uniform distribution via histogram equalization, and then linking them.Input Equalization: We define a continuous random variable $s$ representing the equalized input image. The transformation $T(r)$ uses the continuous PDF of the input image $p_r(r)$:$$s=T(r)=(L-1)\int_0^r p_r(w)dw$$Target Equalization: We define a similar transformation $G(z)$ using the specified/target PDF $p_z(z)$:$$G(z)=(L-1)\int_0^z p_z(t)dt=s$$Inverse Mapping: Because $G(z)=T(r)=s$, we can isolate the desired output intensities $z$ by applying the inverse transformation:$$z=G^{-1}[T(r)]=G^{-1}(s)$$II. The Discrete ImplementationIn digital images, we operate on discrete intensity levels (typically $L=256$, ranging from $0$ to $255$). We approximate the continuous integrals using cumulative sums (CDFs) of histograms.Step 1: Compute the histogram of the input image to find the discrete equalization transformation $T(r_k)$. Round the resulting values, $s_k$, to the integer range $[0, L-1]$.Step 2: Compute all values of the transformation function $G$ using the specified target histogram. Round these values to integers in the range $[0, L-1]$ and store them in a lookup table.Step 3: For every equalized input value $s_k$, find the corresponding target value $z_q$ such that $G(z_q)$ is closest to $s_k$. If the mapping is not unique (multiple $z_q$ yield the same minimum difference), choose the smallest $z_q$ by convention.Step 4: Form the output image by mapping every original pixel $r_k$ directly to $z_q$ using the composite lookup table $z_q = G^{-1}[T(r_k)]$.

Implementation:
```
import numpy as np

def compute_discrete_cdf(channel: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Computes the scaled and rounded CDF (Transformation function T or G) 
    for a single image channel as defined in Steps 1 & 2 of the textbook.
    """
    # np.bincount is significantly faster than np.histogram for integer arrays
    hist = np.bincount(channel.ravel(), minlength=L)
    
    # Compute discrete Cumulative Distribution Function
    cdf = hist.cumsum()
    
    # Normalize and scale to [0, L-1]
    cdf_normalized = cdf / cdf[-1]
    
    # "Round the resulting values to the integer range [0, L-1]"
    transformation_func = np.round(cdf_normalized * (L - 1)).astype(int)
    
    return transformation_func

def match_channel_gw(source_channel: np.ndarray, target_channel: np.ndarray, L: int = 256) -> np.ndarray:
    """
    Performs Histogram Specification on a single 2D channel.
    """
    # Step 1: Compute T(r) - The equalized source values (s_k)
    T = compute_discrete_cdf(source_channel, L)
    
    # Step 2: Compute G(z) - The equalized target values
    G = compute_discrete_cdf(target_channel, L)
    
    # Step 3: Create mapping from r -> z directly.
    # We build a Lookup Table (LUT) where LUT[r] = z
    lut = np.zeros(L, dtype=np.uint8)
    
    for r in range(L):
        s_k = T[r]
        
        # "find the corresponding value of z_q so that G(z_q) is closest to s_k"
        diff = np.abs(G - s_k)
        
        # np.argmin inherently returns the first index (smallest z_q) if there are ties, 
        # fulfilling the rule: "choose the smallest value by convention."
        z_q = np.argmin(diff)
        
        lut[r] = z_q
        
    # Step 4: Map the original pixels to the new values using the LUT
    # This is a highly optimized O(N) vectorized operation in NumPy
    matched_channel = lut[source_channel]
    
    return matched_channel

def histogram_specification(source_img: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """
    Main entry point. Applies textbook histogram specification to an RGB image.
    Expects source and target to be uint8 numpy arrays of shape (H, W, C).
    """
    if source_img.shape[-1] != target_img.shape[-1]:
         raise ValueError("Source and target images must have the same number of channels.")
         
    matched_img = np.zeros_like(source_img)
    
    # Apply algorithm channel by channel (e.g., R, G, B)
    for c in range(source_img.shape[-1]):
        matched_img[..., c] = match_channel_gw(source_img[..., c], target_img[..., c])
        
    return matched_img
```

References

1. Gonzalez, R. C. & Woods, R. E. Digital image processing, prentice hall. Up. Saddle River, NJ (2008).  Chapter 3 ■ Intensity Transformations and Spatial Filtering, 3.3 Histogram Processing