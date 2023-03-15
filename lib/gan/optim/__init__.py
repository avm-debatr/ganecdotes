"""
-------------------------------------------------------------------------------
The optim module consists of a CUDA-optimized implementations of two operations
in the GAN framework.

Fused Activation Functions:

The module contains fused implementations of activation functions for
performance optimization.
A convolution operation followed by activation is a performance bottleneck as
the GPU must wait for the convolution operation to be finished and written into
memory before applying the activation function. The read-write operation of
the convolutional layer is the source of the bottleneck.

If the activation is performed in-place with the convolution, it is a major
performance upgrade - this is referred to as operator fusion.
(For details, see
 https://docs.microsoft.com/en-us/windows/ai/directml/dml-fused-activations)

This is used in the ZtoWLatentMapper within the Generator
--------

Efficient 2D FIR Filter for Upsampling:

- stylegan uses a special implementation of a 2D FIR filter for upsampling
- this implementation is more than the bilinear upsampling operation using
  PyTorch ops
- the implementation consists of a fused kernel function for upsampling
  written in the same way as above.

Both the kernel codes are borrowed from the official StyleGAN2-pytorch-ada
implementation:

https://github.com/NVLabs/stylegan2-ada-pytorch.git

and rosinality's StyleGAN2 implementation:

https://github.com/rosinality/stylegan2-pytorch.git

-------------------------------------------------------------------------------
"""

from .fused_act import FusedLeakyReLU, fused_leaky_relu
from .upfirdn2d import upfirdn2d
