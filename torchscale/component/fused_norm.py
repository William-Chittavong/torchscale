import torch
import importlib

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

from .hook import HookManager
from typing import Optional
from typing import Optional, Sequence
import numbers
#from apex.normalization import FusedLayerNorm as ApexFusedLayerNorm
#from apex.normalization import  FusedLayerNormAffineFunction

from .layer_norm import LayerNorm


__all__ = ["_cast_if_autocast_enabled"]


def _get_autocast_dtypes() -> Sequence[torch.dtype]:
    if torch.cuda.is_bf16_supported():
        return [torch.half, torch.bfloat16]
    return [torch.half]


def _get_current_dtype(dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if not torch.is_autocast_enabled():
        return torch.float or dtype
    else:
        return torch.get_autocast_gpu_dtype()


def _cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return torch.cuda.amp.autocast_mode._cast(args, torch.get_autocast_gpu_dtype())

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

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



class FusedLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, memory_efficient=False, hook: Optional[HookManager] = None):
        super().__init__()
        self.hook = hook or HookManager()
        
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

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

    


