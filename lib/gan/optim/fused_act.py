"""
Module for fused Leaky ReLU activation used in ZtoW Mapper.
-   fused operation implemented in CUDA
-   this module contains Pytorch wrappers for the kernels

Following rosinality's steps, the fused activation is only used for
GPU implementation.
"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

# -----------------------------------------------------------------------------
# Load C++ implementations of the kernel function
module_path = os.path.dirname(__file__)

fused = load("fused",
              sources=[os.path.join(module_path, "fused_bias_act.cpp"),
                       os.path.join(module_path, "fused_bias_act_kernel.cu")],
)


class FusedLeakyReLUFunctionBackward(Function):
    """
    ---------------------------------------------------------------------------
    Customized differentiable torch Function for Backpropagation operation of
    fused activation
    ---------------------------------------------------------------------------
    """
    @staticmethod
    def forward(ctx,
                grad_output,
                out,
                bias,
                negative_slope,
                scale):
        """
        ----------------------------------------------------------------------
        :param ctx:             data to be saved for backward() operation
        :param grad_output:     gradient fed to node during backprop
        :param out:             output generated during forward prop
        :param bias:            for i/p
        :param negative_slope:  for leaky ReLU
        :param scale:           for i/p
        :return:
        ----------------------------------------------------------------------
        """

        # saving data for backward()
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        # apply fused bias activation to gradient
        grad_input = fused.fused_bias_act(grad_output.contiguous(),
                                          empty,
                                          out,
                                          3, # specifies to kernel that activation is lrelu
                                          1, # whether biase is present or not
                                          negative_slope,
                                          scale)

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias
    # -------------------------------------------------------------------------

    @staticmethod
    def backward(ctx,
                 gradgrad_input,
                 gradgrad_bias):
        """
        -----------------------------------------------------------------------
        :param ctx:
        :param gradgrad_input:
        :param gradgrad_bias:
        :return:
        -----------------------------------------------------------------------
        """
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input.contiguous(),
            gradgrad_bias,
            out,
            3,
            1,
            ctx.negative_slope,
            ctx.scale,
        )

        return gradgrad_out, None, None, None, None
    # -------------------------------------------------------------------------


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                bias,
                negative_slope,
                scale):
        """
        -----------------------------------------------------------------------
        :param ctx:
        :param input:
        :param bias:
        :param negative_slope:
        :param scale:
        :return:
        -----------------------------------------------------------------------
        """
        empty = input.new_empty(0)

        ctx.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input,
                                   bias,
                                   empty,
                                   3,
                                   0,
                                   negative_slope,
                                   scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out
    # -------------------------------------------------------------------------

    @staticmethod
    def backward(ctx,
                 grad_output):
        """
        -----------------------------------------------------------------------
        :param ctx:
        :param grad_output:
        :return:
        -----------------------------------------------------------------------
        """
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
        )

        if not ctx.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None
    # -------------------------------------------------------------------------


class FusedLeakyReLU(nn.Module):
    """
    ---------------------------------------------------------------------------
    Leaky ReLU layer based on fused activation
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 channel,
                 bias=True,
                 negative_slope=0.2,
                 scale=2 ** 0.5):
        """
        -----------------------------------------------------------------------

        :param channel:
        :param bias:
        :param negative_slope:
        :param scale:
        -----------------------------------------------------------------------
        """

        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale
    # -------------------------------------------------------------------------

    def forward(self, input):
        """
        -----------------------------------------------------------------------
        :param input:
        :return:
        -----------------------------------------------------------------------
        """
        return fused_leaky_relu(input,
                                self.bias,
                                self.negative_slope,
                                self.scale)
    # -------------------------------------------------------------------------


def fused_leaky_relu(input,
                     bias=None,
                     negative_slope=0.2,
                     scale=2 ** 0.5):
    """
    ---------------------------------------------------------------------------
    fused activation based on device for input.

    :param input:
    :param bias:
    :param negative_slope:
    :param scale:
    :return:
    ---------------------------------------------------------------------------
    """

    if input.device.type == "cpu":
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (
                F.leaky_relu(
                    input + bias.view(1,
                                      bias.shape[0],
                                      *rest_dim),
                    negative_slope=0.2
                )
                * scale
            )

        else:
            return F.leaky_relu(input, negative_slope=0.2) * scale

    else:
        return FusedLeakyReLUFunction.apply(
            input.contiguous(), bias, negative_slope, scale
        )
# -----------------------------------------------------------------------------
