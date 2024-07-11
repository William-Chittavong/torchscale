import torch
from torch import nn
from component.hook import HookManager
from typing import Optional
import numbers



class LayerNorm(nn.Module):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        hook: Optional[HookManager] = None,
    ):
        super().__init__()
        self.hook = hook or HookManager()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(
                torch.empty(
                    self.normalized_shape,
                )
            )
            self.bias = torch.nn.Parameter(
                torch.empty(
                    self.normalized_shape,
                )
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        assert self.normalized_shape == x.shape[-len(self.normalized_shape) :]
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]
        mean = self.hook("mean", ret=x.mean(dim=dims, keepdim=True))
        mean_x2 = (x**2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean**2
        x_norm = self.hook("mean_reduced", ret=(x - mean)) / self.hook(
            "sqrt_var", ret=torch.sqrt(var + self.eps)
        )
        if self.elementwise_affine:
            x_norm = self.hook("renorm.post", ret=self.weight * x_norm + self.bias)
        x_norm = self.hook("ln_post",ret = x_norm)
        self.hook.finalize()
        return x_norm.to(orig_type)
