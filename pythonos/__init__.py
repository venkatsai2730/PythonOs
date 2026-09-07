"""PythonOS-1B MoE — nano-scale architecture validation.

Validates the locked v1.1 architecture one component at a time, at ~150M
params on a single GPU, before committing to full-scale pretraining. See
README.md for the stage plan and what each stage must prove.

Public surface:
    GPTConfig   every architecture flag, defaulting to the Stage A baseline
    GPT         the model
    build_layer_plan  per-layer attention settings derived from the config
"""

from .config import GPTConfig, build_layer_plan
from .model import GPT, Block
from .norm import LayerNorm, RMSNorm, make_norm, NORM_REGISTRY
from .attention import (SelfAttention, CausalSelfAttention,
                        MultiheadLatentAttention, precompute_rope_cache,
                        apply_rope, build_attn_mask)
from .ffn import FeedForward, MLP, Expert, MoE

__all__ = [
    'GPTConfig', 'build_layer_plan', 'GPT', 'Block',
    'LayerNorm', 'RMSNorm', 'make_norm', 'NORM_REGISTRY',
    'SelfAttention', 'CausalSelfAttention', 'MultiheadLatentAttention',
    'precompute_rope_cache', 'apply_rope', 'build_attn_mask',
    'FeedForward', 'MLP', 'Expert', 'MoE',
]
