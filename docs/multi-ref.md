https://arxiv.org/html/2406.02077v2

Concatenation (Concat): This procedure involves stitching all reference images into a single, massive virtual image. The stain matrix $V_{ref}$ is then estimated from this aggregate dataset. While this captures the full diversity of the target domain, it is computationally expensive and memory-intensive

Average-Pre (Avg-pre): In this method, the principal components or SVD directions are computed for each reference image independently. These directions are then averaged to find the "robust extremes" that define the final stain vectors. This is particularly effective for methods like Macenko's, which relies on angular percentiles in the OD plane.

Average-Post (Avg-post): This is widely considered the most robust method for deep learning generalizability. A complete stain matrix $V_{ref}^t$ is computed for each individual reference image $t \in T$. The final reference matrix used for normalization is the arithmetic mean of these individual matrices: $V_{ref} = \mathbb{E}[V_{ref}^t]$.

Stochastic Normalization: During training, a random reference image is picked from $T$ for each mini-batch. This acts as a form of style augmentation, forcing the model to become invariant to minor color shifts within the target domain.