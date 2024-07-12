import torch
import importlib

from torch.nn import functional as F

from .hook import HookManager
from typing import Optional


from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
from apex.normalization import  FusedLayerNormAffineFunction

from .layer_norm import LayerNorm
from apex._autocast_utils import _cast_if_autocast_enabled

class FusedLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, eps, memory_efficient=False):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward(input_, ctx.normalized_shape, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, None, invvar)
        else:
            ctx.save_for_backward(input_, mean, invvar)
        return output, mean, invvar

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, mean, invvar = ctx.saved_tensors
        grad_input = fused_layer_norm_cuda.backward(
            grad_output.contiguous(), mean, invvar, input_or_output,
            ctx.normalized_shape, ctx.eps, ctx.memory_efficient
        )
        return grad_input, None, None, None



class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps, memory_efficient=False):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps
        )
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, bias_, None, invvar)
        else:
            ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output, mean, invvar

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar, input_or_output,
            ctx.normalized_shape, weight_, bias_, ctx.eps, ctx.memory_efficient
        )
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_layer_norm_affine(input, weight, bias, normalized_shape, eps=1e-6, memory_efficient=False):
    args = _cast_if_autocast_enabled(input, weight, bias, normalized_shape, eps, memory_efficient)
    with torch.cuda.amp.autocast(enabled=False):
        return FusedLayerNormAffineFunction.apply(*args)


def fused_layer_norm(input, normalized_shape, eps=1e-6, memory_efficient=False):
    args = _cast_if_autocast_enabled(input, normalized_shape, eps, memory_efficient)
    with torch.cuda.amp.autocast(enabled=False):
        return FusedLayerNormFunction.apply(*args)



class FusedLayerNorm(ApexFusedLayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, hook: Optional[HookManager] = None):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.hook = hook or HookManager()

    def forward(self, input):
        if torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:

            return LayerNorm(input, hook = self.hook)

        if self.elementwise_affine:
            output, mean, invvar = fused_layer_norm_affine(
                input, self.weight, self.bias, self.normalized_shape, self.eps, self.memory_efficient,self.hook)
            self.hook("mean", ret=mean)
            sqrt_var = 1.0 / torch.sqrt(invvar + self.eps)
            self.hook("sqrt_var", ret=sqrt_var) 
            self.hook.finalize()
            return output
        else:
            output , mean, invvar = fused_layer_norm(input, self.normalized_shape, self.eps, self.memory_efficient,self.hook)
            self.hook("mean", ret=mean)
            sqrt_var = 1.0 / torch.sqrt(invvar + self.eps)
            self.hook("sqrt_var", ret=sqrt_var) 
            self.hook.finalize()
            return output

    


