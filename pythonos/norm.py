"""Normalisation layers.

Stage D (FuseNorm-DyT) plugs in here: add the module, register it in
NORM_REGISTRY, and select it from GPTConfig.norm_type — no changes needed
anywhere else in the stack.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalisation with an optional bias.

    torch.nn.LayerNorm has no way to drop the bias term, and bias-free norms
    train slightly better and faster, so this wraps the functional form
    directly.
    """

    def __init__(self, dim, bias=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        return nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape[0]}, bias={self.bias is not None}"


class RMSNorm(nn.Module):
    """Root-mean-square normalisation (Zhang & Sennrich, 2019).

    Drops the mean-centring step of LayerNorm and rescales by RMS only. Cheaper
    and generally equivalent in quality for decoder-only LMs. Not used by the
    Stage A baseline; available as a comparison point for Stage D.
    """

    def __init__(self, dim, bias=False, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        # compute in fp32 so the reciprocal sqrt is stable under autocast
        dtype = x.dtype
        xf = x.float()
        scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (xf * scale).to(dtype) * self.weight
        return out + self.bias if self.bias is not None else out


NORM_REGISTRY = {
    'layernorm': LayerNorm,
    'rmsnorm': RMSNorm,
}


def make_norm(config, dim):
    """Build the norm selected by config.norm_type."""
    try:
        cls = NORM_REGISTRY[config.norm_type]
    except KeyError:
        raise KeyError(f"unknown norm_type {config.norm_type!r}; "
                       f"available: {sorted(NORM_REGISTRY)}") from None
    return cls(dim, bias=config.bias)
