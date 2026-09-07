"""Attention variants: dense multi-head (Stage A) and hybrid (Stage B).

SelfAttention is the Stage A baseline. MultiheadLatentAttention is Stage B's
MLA. Rotary embeddings and sliding-window masking are shared helpers, inert
unless the matching GPTConfig flags are set.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .norm import NORM_REGISTRY

# -----------------------------------------------------------------------------
# Rotary position embeddings (Su et al., RoFormer) and attention masking.


def precompute_rope_cache(head_dim, seq_len, theta=10000.0):
    """cos/sin tables of shape (seq_len, head_dim // 2)."""
    if head_dim % 2:
        raise ValueError(f"RoPE needs an even head dim, got {head_dim}")
    exponents = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = theta ** (-exponents)
    angles = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1) * inv_freq
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate (B, H, T, D) in adjacent dimension pairs.

    Pairs are (0,1), (2,3), ... — the interleaved convention. A cache built for
    the same convention must be used throughout; mixing conventions between q
    and k silently degrades attention rather than erroring.
    """
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    half = head_dim // 2
    c = cos[:seq_len].reshape(1, 1, seq_len, half).to(x.dtype)
    s = sin[:seq_len].reshape(1, 1, seq_len, half).to(x.dtype)
    lo, hi = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((lo * c - hi * s, lo * s + hi * c), dim=-1)
    return rotated.flatten(start_dim=-2)


def build_attn_mask(seq_len, window, device):
    """Boolean keep-mask of shape (1, 1, T, T), or None for plain causal.

    None means "use the SDPA is_causal fast path". A window of w restricts each
    query to itself and the w-1 keys before it.
    """
    if not window:
        return None
    positions = torch.arange(seq_len, device=device)
    lag = positions.unsqueeze(1) - positions.unsqueeze(0)
    keep = torch.logical_and(lag >= 0, lag < window)
    return keep.unsqueeze(0).unsqueeze(0)


def _causal_keep_mask(seq_len, device):
    positions = torch.arange(seq_len, device=device)
    return (positions.unsqueeze(1) >= positions.unsqueeze(0)).view(1, 1, seq_len, seq_len)


def split_heads(x, n_head, head_dim):
    """(B, T, n_head * head_dim) -> (B, n_head, T, head_dim)."""
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, n_head, head_dim).transpose(1, 2)


def merge_heads(x):
    """(B, n_head, T, head_dim) -> (B, T, n_head * head_dim)."""
    batch, n_head, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, seq_len, n_head * head_dim)


def dot_product_attention(q, k, v, keep_mask, dropout_p, scale=None):
    """Scaled dot-product attention, preferring the fused SDPA kernel.

    keep_mask=None selects the is_causal fast path. The manual branch exists
    only for PyTorch builds without SDPA and must stay numerically equivalent.
    """
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=keep_mask, dropout_p=dropout_p,
            is_causal=keep_mask is None, scale=scale)

    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    if keep_mask is None:
        keep_mask = _causal_keep_mask(q.shape[-2], q.device)
    scores = scores.masked_fill(keep_mask.logical_not(), float('-inf'))
    weights = F.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=dropout_p)
    return weights @ v


# -----------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Causal multi-head self-attention.

    Stage A uses this with every option off. Stage B optionally adds rotary
    embeddings, sliding-window masking, and cross-layer K/V reuse.

    Queries and keys/values are projected separately rather than as one fused
    matrix, because a layer that consumes K/V from an earlier layer needs no
    K/V projection at all and building one would waste the parameters.
    """

    def __init__(self, config, layer_idx=0, plan=None):
        super().__init__()
        plan = plan or {}
        if config.n_embd % config.n_head:
            raise ValueError(
                f"n_embd={config.n_embd} must divide by n_head={config.n_head}")

        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        self.use_rope = plan.get('rope', False)
        self.window = plan.get('window', 0)
        self.kv_role = plan.get('kv_role', 'own')  # 'own' | 'produce' | 'consume'

        inner = config.n_head * self.head_dim
        self.q_proj = nn.Linear(config.n_embd, inner, bias=config.bias)
        if self.kv_role != 'consume':
            self.kv_proj = nn.Linear(config.n_embd, 2 * inner, bias=config.bias)
        self.out_proj = nn.Linear(inner, config.n_embd, bias=config.bias)
        # marks a projection that writes into the residual stream, so the depth
        # dependent init scaling can find it without matching on parameter names
        self.out_proj.is_residual_out = True
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope=None, kv_cache=None):
        q = split_heads(self.q_proj(x), self.n_head, self.head_dim)

        if self.kv_role == 'consume':
            k, v = kv_cache['k'], kv_cache['v']
        else:
            kv = self.kv_proj(x)
            k, v = kv.split(kv.shape[-1] // 2, dim=-1)
            k = split_heads(k, self.n_head, self.head_dim)
            v = split_heads(v, self.n_head, self.head_dim)

        if self.use_rope:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            # a consumed key already carries its producer's rotation
            if self.kv_role != 'consume':
                k = apply_rope(k, cos, sin)

        if self.kv_role == 'produce':
            kv_cache['k'], kv_cache['v'] = k, v

        keep_mask = build_attn_mask(x.shape[1], self.window, x.device)
        attended = dot_product_attention(
            q, k, v, keep_mask, self.dropout_p if self.training else 0.0)
        return self.resid_dropout(self.out_proj(merge_heads(attended)))


class MultiheadLatentAttention(nn.Module):
    """MLA, following DeepSeek-V2/V3.

    Two mechanisms, both aimed at shrinking the inference KV cache:

    1. Low-rank K/V. The input is compressed to a single shared latent of
       width kv_lora_rank and up-projected to per-head keys and values inside
       the layer, so only the latent needs caching.
    2. Decoupled rotary keys. RoPE is position-dependent and cannot be folded
       into the up-projection, so the rotary part of the key is produced on a
       separate head-shared path and concatenated onto the compressed key.
       Hence the query/key head dim is qk_nope_head_dim + qk_rope_head_dim.

    Measurement caveat: the cache saving is an INFERENCE property and needs the
    absorption optimisation to be realised. Training memory here is not lower
    than plain attention and may be slightly higher, because the up-projected
    keys and values are still materialised in the forward pass. Compare cache
    bytes per token, not training peak memory.
    """

    def __init__(self, config, layer_idx=0, plan=None):
        super().__init__()
        plan = plan or {}
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.dropout_p = config.dropout
        self.use_rope = plan.get('rope', False)
        self.window = plan.get('window', 0)
        self.kv_role = plan.get('kv_role', 'own')

        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.qk_dim = self.nope_dim + self.rope_dim
        self.v_dim = config.v_head_dim
        self.kv_rank = config.kv_lora_rank
        self.q_rank = config.q_lora_rank

        # query path, optionally routed through a low-rank bottleneck
        if self.q_rank:
            self.q_down = nn.Linear(config.n_embd, self.q_rank, bias=False)
            self.q_latent_norm = _bias_free_norm(config, self.q_rank)
            self.q_up = nn.Linear(self.q_rank, self.n_head * self.qk_dim, bias=False)
        else:
            self.q_proj = nn.Linear(config.n_embd, self.n_head * self.qk_dim, bias=False)

        # key/value path, skipped entirely when inheriting from another layer
        if self.kv_role != 'consume':
            self.kv_down = nn.Linear(config.n_embd, self.kv_rank, bias=False)
            self.kv_latent_norm = _bias_free_norm(config, self.kv_rank)
            self.kv_up = nn.Linear(
                self.kv_rank, self.n_head * (self.nope_dim + self.v_dim), bias=False)
            if self.use_rope:
                self.k_rope_proj = nn.Linear(config.n_embd, self.rope_dim, bias=False)

        self.out_proj = nn.Linear(self.n_head * self.v_dim, config.n_embd, bias=config.bias)
        self.out_proj.is_residual_out = True
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, rope=None, kv_cache=None):
        batch, seq_len, _ = x.shape
        heads = self.n_head

        if self.q_rank:
            q = self.q_up(self.q_latent_norm(self.q_down(x)))
        else:
            q = self.q_proj(x)
        q = split_heads(q, heads, self.qk_dim)
        q_nope, q_rope = q.split([self.nope_dim, self.rope_dim], dim=-1)

        if self.kv_role == 'consume':
            k, v = kv_cache['k'], kv_cache['v']
        else:
            latent = self.kv_latent_norm(self.kv_down(x))
            kv = split_heads(self.kv_up(latent), heads, self.nope_dim + self.v_dim)
            k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
            if self.use_rope:
                cos, sin = rope
                # one rotary key head, broadcast across all query heads
                shared = self.k_rope_proj(x).view(batch, seq_len, 1, self.rope_dim)
                shared = apply_rope(shared.transpose(1, 2), cos, sin)
                k_rope = shared.expand(batch, heads, seq_len, self.rope_dim)
            else:
                k_rope = q_rope.new_zeros(batch, heads, seq_len, self.rope_dim)
            k = torch.cat([k_nope, k_rope], dim=-1)

        if self.use_rope:
            cos, sin = rope
            q_rope = apply_rope(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        if self.kv_role == 'produce':
            kv_cache['k'], kv_cache['v'] = k, v

        keep_mask = build_attn_mask(seq_len, self.window, x.device)
        attended = dot_product_attention(
            q, k, v, keep_mask, self.dropout_p if self.training else 0.0,
            scale=1.0 / math.sqrt(self.qk_dim))
        return self.resid_dropout(self.out_proj(merge_heads(attended)))


def _bias_free_norm(config, dim):
    """Norm over an MLA latent. Always bias-free, whatever config.bias says."""
    return NORM_REGISTRY[config.norm_type](dim, bias=False)


# backwards-compatible alias: the Stage A baseline attention
CausalSelfAttention = SelfAttention
