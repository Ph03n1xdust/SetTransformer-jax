# Set Transformer
This repository contains a jax-flax based implementation of the [Set Transformer](https://arxiv.org/pdf/1810.00825.pdf), which heavily borrows from the official [PyTorch version](https://github.com/juho-lee/set_transformer).

### Example
The ```max_regression.ipynb``` contains the jax-flax variant of the original max regression example, extended to also include a version using ISABs.

### Notes
1. The initial weights of the network (especially the Dense layer kernels) are very important, which by default differ between Pytorch and Flax. This has been mostly fixed by using a PyTorch like initialization, but can be tuned for even better performance.
2. The code does not contain a "Set Transformer" by itself, just the building blocks, since the architecture has to be tuned for the specific problem.
3. There is one known difference between the "official" and this implementation in the scaling of attention weights before applying the softmax function. This version scales with ```1/sqrt(dim_split)``` (I expect this to be "correct") while the official scales with ```1/sqrt(dim_split*num_heads)```. 