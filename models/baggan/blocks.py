from lib.util.util import *

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.gan.optim import fused_leaky_relu, upfirdn2d, conv2d_gradfix, \
                          FusedLeakyReLU

# =============================================================================
# Section for Z-to-W Mapper in StyleGAN2 Generator:
# =============================================================================

# PixelNormalizer -------------------------------------------------------------


class PixelNorm(nn.Module):
    def __init__(self):
        """
        -----------------------------------------------------------------------
        This block is used to normalize the latent vector input to the 
        Latent Mapping network as it translates z to w.
        -----------------------------------------------------------------------
        """
        super().__init__()
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param input:
        :return:
        -----------------------------------------------------------------------
        """
        out = x * torch.rsqrt(torch.mean(x.pow(2),
                                              dim=1,
                                              keepdim=True) + 1e-8)

        return out
    # -------------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        # return (f"{self.__class__.__name__}:"
        #         f"({self.weight.shape[1]}, {self.weight.shape[0]})"
        # )
        return (f"{self.__class__.__name__}:")
    # -------------------------------------------------------------------------


# EqualizedLinearBlock --------------------------------------------------------


class EqualizedLinearBlock(nn.Module):
    """
    ---------------------------------------------------------------------------
    Fully connected layer for latent mapping block - eight of these and you
    have an MLP to convert the uniformly sampled vector z to the disentangled
    vector w.
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 in_features,
                 out_features,
                 lr_multiplier=1,
                 bias=0,
                 activation='lrelu'):
        """
        -----------------------------------------------------------------------
        :param in_features:      no. of input channels
        :param out_features:     no. of output channels
        :param lr_multiplier:    learning rate multiplier
        :param bias:             add bias to MLP (if none - no bias is added
                                 else bias is initialized with given value)
        :param activation:       type of activation {linear | fused_lrelu}
        -----------------------------------------------------------------------
        """

        super().__init__()

        # only linear / leaky reLU activation allowed (only these kernels
        # are fused)
        assert activation in ['linear', 'lrelu']
        self.activation = activation

        # initialize weight, bias for the layer

        # print("|"*80, lr_multiplier)
        self.weight = nn.Parameter(torch.randn(out_features,
                                               in_features).div_(lr_multiplier))

        self.bias   = nn.Parameter(torch.zeros(out_features).fill_(bias)) \
                      if bias is not None else None

        # scale weight/bias
        self.weight_gain  = (1/math.sqrt(in_features))*lr_multiplier
        self.bias_gain    = lr_multiplier
        # ---------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param x:
        :return:
        -----------------------------------------------------------------------
        """

        if self.activation=='linear':
            return F.linear(x,
                            self.weight*self.weight_gain,
                            self.bias*self.bias_gain if self.bias is not None
                                                     else self.bias)

        if self.activation=='lrelu':

            out = F.linear(x,
                           self.weight * self.weight_gain)
            out = fused_leaky_relu(out,
                                    self.bias * self.bias_gain
                                    if self.bias is not None
                                    else self.bias)
            return out
        # ---------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        return (f"{self.__class__.__name__}, "
                f"W: ({self.weight.shape[1]}, {self.weight.shape[0]}),"
                f"B: {self.bias.shape},"
                f"A: {self.activation}")
    # -------------------------------------------------------------------------


# =============================================================================
# Section for Style-Based Synthesizer in StyleGAN2 Generator:
# Consists of:
# - Block for Constant Input (first block in Generator)
# - Style Blocks with Up/Down-sampling
# - Blocks for Noise Injection
# =============================================================================

# Gen Constant Input Block ----------------------------------------------------

class ConstantInputBlock(nn.Module):
    """
    ---------------------------------------------------------------------------
    This is the topmost block of the synthesizer that consists of a layer of 
    constant, learned weights.
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 num_chls=1,
                 blk_size=4):
        """
        -----------------------------------------------------------------------
        :param num_chls:
        :param size:
        -----------------------------------------------------------------------
        """
        super().__init__()

        self.const_block = nn.Parameter(torch.randn(1,
                                                    num_chls,
                                                    blk_size,
                                                    blk_size))
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        Since this is a constant input block, x is not realy used here. What
        is returned the input bloc repeated as per the batch size.

        :param   x:
        :return:
        -----------------------------------------------------------------------
        """

        return self.const_block.repeat(x.shape[0], 1, 1, 1)
    # -------------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        return (f"{self.__class__.__name__}, "
                f"W: ({self.const_block.shape})")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class UpDownSampler(nn.Module):
    """
    ---------------------------------------------------------------------------
    Layer for upsampling or downsampling using NVIDIA's FIR filter based
    resampler 
    (This resampler claims to be more efficient than the Pytorch's model 
    for up/downs)
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 kernel,
                 scale=2,
                 mode='up'):
        """
        -----------------------------------------------------------------------
        :param kernel:
        :param scale:
        :param mode:
        -----------------------------------------------------------------------
        """
        super().__init__()

        self.scale = scale
        self.mode = mode

        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1: kernel = kernel[None, :]*kernel[:, None]
        kernel *= 1/kernel.sum()

        if mode=='up': kernel *= (scale**2)
        self.register_buffer("kernel", kernel)

        plen = kernel.shape[0] - scale
        self.padding = [(plen + 1)//2, plen//2]

        if self.mode=='up': self.padding[0] += scale - 1
        self.padding = tuple(self.padding)
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param x:
        :return:
        -----------------------------------------------------------------------
        """
        if self.mode=='up':
            return upfirdn2d(x,
                             self.kernel,
                             up=self.scale,
                             down=1,
                             pad=self.padding)
        if self.mode=='down':
            return upfirdn2d(x,
                             self.kernel,
                             up=1,
                             down=self.scale,
                             pad=self.padding)
    # -------------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        return (f"{self.__class__.__name__}, "
                f"M: ({self.mode}),"
                f"K: {self.kernel},"
                f"S: {self.scale: .3f}")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class BlurFilter(nn.Module):
    """
    ---------------------------------------------------------------------------
    Filter for upsampling/downsampling operation using nvidia's upfir2d filter

    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 kernel,
                 pad,
                 scale=1):
        """
        -----------------------------------------------------------------------
        :param kernel:
        :param pad:
        :param scale:
        -----------------------------------------------------------------------
        """

        super().__init__()

        # Creates the kernel for upsampling
        kernel = torch.tensor(kernel, dtype=torch.float32)

        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]

        kernel /= kernel.sum()

        if scale > 1: kernel = kernel * (scale ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------

        :param input:
        :return:
        -----------------------------------------------------------------------
        """

        return upfirdn2d(x, self.kernel, pad=self.pad)
    # -------------------------------------------------------------------------

    # def __repr__(self):
    #     """
    #     -----------------------------------------------------------------------
    #
    #     :return:
    #     -----------------------------------------------------------------------
    #     """
    #     return (f"{self.__class__.__name__}, "
    #             f"M: ({self.kernel}),"
    #             f"K: {self.},"
    #             f"S: {self.scale}")
    # # -------------------------------------------------------------------------

# -----------------------------------------------------------------------------


class ConvLayerWithStyleMod(nn.Module):
    """
    ---------------------------------------------------------------------------
    Block within the synthesis network that performs the following operation:
    - Convolution with the generator weights
    - Style modulation of conv weights with w latent vector
    - Style demodulation fo conv weights
    - Up/downsampling of output
                                                     x
                ---------------------------------------------
                |                                    |      |
                |                                    v      |
                |   ---    -----    ------------    ----    |
        style ->|--|mod|->|demod|->|conv_weights|->|conv|   |
                |   ---    -----    ------------    ----    |
                |                                    |      |
                |                                    v      |
                |                                   ----    |
                |                                   |up|    |
                ---------------------------------------------
                                                     |
                                                    out
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 in_chls,
                 out_chls,
                 kernel_width,
                 style_dim,
                 demod=True,
                 mode='none',
                 fir_filter=[1,3,3,1],
                 fused=True):
        """
        -----------------------------------------------------------------------
        :param in_chls:
        :param out_chls:
        :param kernel_width:
        :param style_dim:
        :param demod:
        :param mode:
        :param fir_filter:
        :param fused:
        -----------------------------------------------------------------------
        """

        super().__init__()
        self.eps, self.kernel_width = 1e-8, kernel_width
        self.in_chls, self.out_chls = in_chls, out_chls
        self.mode = mode

        # Blur Filter for Up / Downsampling
        upscale = 2

        if self.mode=='up':
            plen = (len(fir_filter) - upscale) - (kernel_width - 1)
            self.blur = BlurFilter(fir_filter,
                                   pad=((plen + 1) // 2 + upscale - 1,
                                         plen // 2 + 1),
                                   scale=upscale)
        if self.mode=='down':
            plen = (len(fir_filter) - upscale) + (kernel_width - 1)
            self.blur = BlurFilter(fir_filter,
                                   pad=((plen + 1) // 2,
                                         plen // 2))

        # Value for scale comes from fan-in calculated in the paper
        self.scale, self.padding = 1/math.sqrt(in_chls*kernel_width**2), \
                                   kernel_width//2

        # convolution layer weights
        self.weight = nn.Parameter(torch.randn(1,
                                               out_chls,
                                               in_chls,
                                               kernel_width,
                                               kernel_width))

        # style modulation
        self.mod   = EqualizedLinearBlock(style_dim,
                                          in_chls,
                                          bias=1,
                                          activation='linear')
        self.demod = demod
        self.fused = fused
        # ---------------------------------------------------------------------

    def forward(self, x, style):
        """
        -----------------------------------------------------------------------
        Performs style modulation-demodulation with input style with a given x.
        Also performs up/downsampling with upfir2dnd filters

        :param x:       input feature fro mprevious synthesis block
        :param style:   w style from latent mapping block
        :return:
        -----------------------------------------------------------------------
        """

        b, c, h, w = x.shape

        # Modulation
        style  = self.mod(style).view(b, 1, c, 1, 1)
        weight = self.scale * self.weight * style

        # Demodulation
        if self.demod:
            demod = torch.rsqrt((weight**2).sum([2,3,4]) + self.eps)
            weight = weight*demod.view(b, self.out_chls, 1, 1, 1)

        weight = weight.view(b * self.out_chls,
                             c,
                             self.kernel_width,
                             self.kernel_width)

        # Up/Downsampling with Convolution
        if self.mode=='up':
            x = x.view(1, b*c, h, w)
            weight = weight.view(b,
                                 self.out_chls,
                                 c,
                                 self.kernel_width,
                                 self.kernel_width)
            weight = weight.transpose(1, 2).reshape(b*c,
                                                    self.out_chls,
                                                    self.kernel_width,
                                                    self.kernel_width)
            out = conv2d_gradfix.conv_transpose2d(x,
                                                  weight,
                                                  padding=0,
                                                  stride=2,
                                                  groups=b)
            _, _, h, w = out.shape
            out = out.view(b, self.out_chls, h, w)
            out = self.blur(out)

        elif self.mode=='down':
            x = self.blur(x)
            _, _, h, w = x.shape
            x = x.view(1, -1, h, w)
            out = conv2d_gradfix.conv2d(x,
                                        weight,
                                        padding=0,
                                        stride=2,
                                        groups=b)
            _, _, h, w = out.shape
            out = out.view(b, self.out_chls, h, w)

        else:
            x = x.view(1, -1, h, w)
            out = conv2d_gradfix.conv2d(x,
                                        weight,
                                        padding=self.padding,
                                        # stride=2,
                                        groups=b)
            _, _, h, w = out.shape
            out = out.view(b, self.out_chls, h, w)

        return out
    # -------------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        return (f"{self.__class__.__name__}, "
                f"M: ({self.mode}),"
                f"W: ({list(self.weight.shape)}),"
                f"S: {self.scale: .3f},"
                f"P: {self.padding}")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class NoiseAdditionBlock(nn.Module):
    """
    ---------------------------------------------------------------------------
    Block to add noise style to each style mod-demod block
    ---------------------------------------------------------------------------
    """
    def __init__(self):
        """
        -----------------------------------------------------------------------
        initialize the noise vector to image conversion weights
        -----------------------------------------------------------------------
        """
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))
    # -------------------------------------------------------------------------

    def forward(self,
                image,
                noise=None):
        """
        -----------------------------------------------------------------------
        Takes in image generated from synthesis block and adds noise to it

        :param image:   from style synthesis block
        :param noise:   if noise to be added externally
        :return:
        -----------------------------------------------------------------------
        """
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class SynthesisBlock(nn.Module):
    """
    ---------------------------------------------------------------------------
    The block combines the style mod-demod and noise injection operations
    Consider this block as one module for one img scale in the generator.
    
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 in_chls,
                 out_chls,
                 kernel_width,
                 style_dim,
                 mode='none',
                 fir_filter=[1,3,3,1],
                 demod=True
                 ):

        super().__init__()

        # Style Synthesis Block
        self.style_block = ConvLayerWithStyleMod(in_chls,
                                                 out_chls,
                                                 kernel_width,
                                                 style_dim,
                                                 mode=mode,
                                                 fir_filter=fir_filter,
                                                 demod=demod)

        self.noise_block = NoiseAdditionBlock()
        self.activation  = FusedLeakyReLU(out_chls)
    # -------------------------------------------------------------------------

    def forward(self,
                x,
                style,
                noise=None):
        """
        -----------------------------------------------------------------------
        :param x:
        :param style:
        :param noise:
        :return:
        -----------------------------------------------------------------------
        """
        out = self.style_block(x, style)
        out = self.noise_block(out, noise=noise)
        out = self.activation(out)

        return out
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class XToImageConverter(nn.Module):
    """
    ---------------------------------------------------------------------------
    This block converts the output of the synthesis block to an image form so 
    that it can be upsampled added as a skip connection at the output of next 
    block - this type of addition performs similar function as progressive 
    growing.
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 in_chls,
                 img_chls,
                 style_dim,
                 upsample=True,
                 fir_filter=[1, 3, 3, 1]):
        """
        -----------------------------------------------------------------------

        
        :param in_chls:
        :param img_chls:
        :param style_dim:
        :param upsample:
        :param fir_filter:
        -----------------------------------------------------------------------
        """
        super().__init__()

        if upsample:
            self.upsample = UpDownSampler(fir_filter)

        self.conv = ConvLayerWithStyleMod(in_chls,
                                          img_chls,
                                          1,
                                          style_dim,
                                          demod=False)
        self.bias = nn.Parameter(torch.zeros(1, img_chls, 1, 1))
    # -------------------------------------------------------------------------

    def forward(self,
                x,
                style,
                skip=None):
        """
        -----------------------------------------------------------------------
        combines output of synthesis block with the skip connection and noise

        out =  conv(x, style) + upsample(skip)

        :param x:
        :param style:
        :param skip:
        :return:
        -----------------------------------------------------------------------
        """

        out = self.conv(x, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# =============================================================================
# Section for StyleGAN Discriminator
# =============================================================================


class EqualizedConvBlock(nn.Module):
    def __init__(self,
                 in_chls,
                 out_chls,
                 kspb=(3, 1, 0, True)):
        """
        -----------------------------------------------------------------------
        :param in_chls:
        :param out_chls:
        :param kspb:
        -----------------------------------------------------------------------
        """

        super().__init__()
        k, s, p, b = kspb

        self.weight = nn.Parameter(torch.randn(out_chls,
                                               in_chls,
                                               k, k))

        self.scale = 1 / math.sqrt(in_chls * k ** 2)
        self.stride, self.padding = s, p
        self.bias = nn.Parameter(torch.zeros(out_chls)) if b else None
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param x:
        :return:
        -----------------------------------------------------------------------
        """

        out = conv2d_gradfix.conv2d(x,
                                    self.weight * self.scale,
                                    bias=self.bias,
                                    stride=self.stride,
                                    padding=self.padding)

        return out
    # -------------------------------------------------------------------------

    def __repr__(self):
        """
        -----------------------------------------------------------------------

        :return:
        -----------------------------------------------------------------------
        """
        return (f"{self.__class__.__name__}, "
                f"W: ({list(self.weight.shape)}),"
                f"S: {self.scale: .3f},"
                f"St: {self.stride},"
                f"P: {self.padding}")
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class DiscConvBlock(nn.Sequential):
    """
    ---------------------------------------------------------------------------
    Convolutional layers for discriminator with downsampling option
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 in_chls,
                 out_chls,
                 kernel_width,
                 downsample=False,
                 fir_filter=[1, 3, 3, 1],
                 bias=True,
                 act=True):
        """
        -----------------------------------------------------------------------
        :param in_chls:
        :param out_chls:
        :param kernel_width:
        :param downsample:
        :param fir_filter:
        :param bias:
        :param act:
        -----------------------------------------------------------------------
        """

        layers = []

        factor = 2
        plen = (len(fir_filter) - factor) + (kernel_width - 1)

        if downsample:

            layers.append(BlurFilter(fir_filter,
                                     pad=((plen + 1) // 2,
                                           plen      // 2)
                                     ))
            stride, self.padding = 2, 0

        else:
            stride, self.padding = 1, kernel_width // 2

        layers.append(EqualizedConvBlock(in_chls,
                                         out_chls,
                                         kspb=(kernel_width,
                                               stride,
                                               self.padding,
                                               bias and not act)))

        if act:
            layers.append(FusedLeakyReLU(out_chls, bias=bias))

        super().__init__(*layers)
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """
    ---------------------------------------------------------------------------
    Residual Block in discriminator to counter skip connection in generator
    - for multi-resolution learning similar in MSG-GAN
    ---------------------------------------------------------------------------
    """
    def __init__(self,
                 in_chls,
                 out_chls):
        """
        -----------------------------------------------------------------------
        :param in_chls:
        :param out_chls:
        -----------------------------------------------------------------------
        """
        super().__init__()

        self.conv1 = DiscConvBlock(in_chls, in_chls,  3)
        self.conv2 = DiscConvBlock(in_chls, out_chls, 3, downsample=True)
        self.skip  = DiscConvBlock(in_chls,
                                   out_chls,
                                   1,
                                   downsample=True,
                                   act=False,
                                   bias=False)
    # -------------------------------------------------------------------------

    def forward(self, x):
        """
        -----------------------------------------------------------------------
        :param input:
        :return:
        -----------------------------------------------------------------------
        """
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)

        return out
    # -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

