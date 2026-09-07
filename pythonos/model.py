"""The decoder-only transformer: one block, and the full stack.

Stage E restructures Block and the stack's layer loop to carry several
residual streams instead of a single tensor. That is a signature change
through the whole stack, not a config flag.
"""

import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention import (MultiheadLatentAttention, SelfAttention,
                        precompute_rope_cache)
from .config import GPTConfig, build_layer_plan
from .ffn import FeedForward, MoE
from .hyper import HyperConnection, stream_cosine_similarity, stream_norms
from .norm import make_norm


class Block(nn.Module):
    """One pre-norm transformer layer: attention sublayer, then feed-forward."""

    def __init__(self, config, layer_idx=0, plan=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.plan = plan or {}

        self.attn_norm = make_norm(config, config.n_embd)
        attn_cls = MultiheadLatentAttention if config.use_mla else SelfAttention
        self.attn = attn_cls(config, layer_idx, self.plan)

        self.ffn_norm = make_norm(config, config.n_embd)
        # leading layers stay dense: routing is unreliable before the
        # representations have settled
        self.is_moe = config.use_moe and layer_idx >= config.moe_first_k_dense
        self.ffn = MoE(config, layer_idx) if self.is_moe else FeedForward(config)

        # Stage E: one hyper-connection per residual add. In 'single' mode none
        # are built and the plain residual path below is used unchanged.
        self.n_streams = config.mhc_streams if config.mhc_mode != 'single' else 1
        if config.mhc_mode != 'single':
            self.hc_attn = self._make_connection(config)
            self.hc_ffn = self._make_connection(config)

    @staticmethod
    def _make_connection(config):
        return HyperConnection(config.mhc_streams, config.mhc_mode,
                               config.mhc_sinkhorn_iters, config.mhc_init_logit,
                               config.mhc_init_scheme)

    def forward(self, x, rope=None, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), rope=rope, kv_cache=kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_streams(self, streams, rope=None, kv_cache=None):
        """Stage E path. `streams` is (B, T, n_streams, d)."""
        pooled = self.hc_attn.aggregate(streams)
        delta = self.attn(self.attn_norm(pooled), rope=rope, kv_cache=kv_cache)
        streams = self.hc_attn.combine(streams, delta)

        pooled = self.hc_ffn.aggregate(streams)
        delta = self.ffn(self.ffn_norm(pooled))
        return self.hc_ffn.combine(streams, delta)

    # kept so external code and tests can reach the sublayers by their
    # historical names
    @property
    def ln_1(self):
        return self.attn_norm

    @property
    def ln_2(self):
        return self.ffn_norm

    @property
    def mlp(self):
        return self.ffn

    @property
    def hc_mlp(self):
        return self.hc_ffn


class GPT(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, config):
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("vocab_size and block_size must both be set")
        self.config = config
        self.layer_plan = build_layer_plan(config)

        parts = {'wte': nn.Embedding(config.vocab_size, config.n_embd)}
        # Stage A learns absolute positions. With rotary or pure NoPE there is
        # no position table at all; position enters inside the attention layers.
        if config.learned_pos_emb:
            parts['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        parts['drop'] = nn.Dropout(config.dropout)
        parts['h'] = nn.ModuleList(
            Block(config, idx, plan) for idx, plan in enumerate(self.layer_plan))
        parts['ln_f'] = make_norm(config, config.n_embd)
        self.transformer = nn.ModuleDict(parts)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # tie input and output embeddings (Press & Wolf, 2017): fewer
        # parameters and generally better perplexity at this scale
        self.transformer.wte.weight = self.lm_head.weight

        # Stage E: collapse the streams back to one vector before the final
        # norm. The e_0 initialisation means only stream 0 is read at step 0,
        # and stream 0 carries the ordinary residual computation.
        self.mhc_enabled = config.mhc_mode != 'single'
        if self.mhc_enabled:
            readout = torch.zeros(config.mhc_streams)
            readout[0] = 1.0
            self.mhc_readout = nn.Parameter(readout)
        self.mhc_capture = True
        self._mhc_stats = {}
        self._mhc_grad_norms = {}
        self.last_lm_loss = None

        if config.use_rope:
            rotary_width = (config.qk_rope_head_dim if config.use_mla
                            else config.n_embd // config.n_head)
            cos, sin = precompute_rope_cache(
                rotary_width, config.block_size, config.rope_theta)
            # buffers, not parameters: no gradient, not saved as weights, and
            # shared by every layer that uses them
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_module)
        self._scale_residual_projections()
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    # ---------------------------------------------------------------- init

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self):
        """Damp projections that write into the residual stream by 1/sqrt(2L).

        Without this the residual variance grows with depth. Layers tag their
        output projection with `is_residual_out` rather than relying on a
        parameter-name convention, so renaming a submodule cannot silently
        disable the scaling.
        """
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, 'is_residual_out', False):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    # ---------------------------------------------------------------- counts

    def get_num_params(self, non_embedding=True):
        """Total parameters.

        With non_embedding=True the position table is excluded. The token
        embedding is not, because weight tying makes it the output layer too.
        """
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.config.learned_pos_emb:
            total -= self.transformer.wpe.weight.numel()
        return total

    def param_report(self):
        """Total against active parameters.

        These diverge sharply for MoE, and quoting one of them alone is how
        parameter budgets end up wrong. Active is what a token actually flows
        through; total is what occupies memory and optimiser state.
        """
        total = sum(p.numel() for p in self.parameters())
        idle = 0
        expert_hidden = None
        for block in self.transformer.h:
            if not block.is_moe:
                continue
            moe = block.ffn
            expert_hidden = moe.expert_hidden
            per_expert = sum(p.numel() for p in moe.routed_experts[0].parameters())
            idle += (moe.n_routed - moe.top_k) * per_expert
        return {'total': total, 'active': total - idle, 'inactive': idle,
                'expert_hidden': expert_hidden}

    # ---------------------------------------------------------------- forward

    def forward(self, idx, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size "
                             f"{self.config.block_size}")

        hidden = self.transformer.wte(idx)
        if self.config.learned_pos_emb:
            positions = torch.arange(seq_len, dtype=torch.long, device=idx.device)
            hidden = hidden + self.transformer.wpe(positions)
        hidden = self.transformer.drop(hidden)

        rope = (self.rope_cos, self.rope_sin) if self.config.use_rope else None
        group = max(1, self.config.kv_share_group)
        # one cache per sharing group, filled in by the group's first layer
        caches = [{} for _ in range(math.ceil(self.config.n_layer / group))]

        if self.mhc_enabled:
            streams = hidden.unsqueeze(2).expand(-1, -1, self.config.mhc_streams, -1)
            for idx_layer, block in enumerate(self.transformer.h):
                streams = block.forward_streams(
                    streams, rope=rope, kv_cache=caches[idx_layer // group])
                if self.config.mhc_instrument and self.mhc_capture:
                    self._record_stream_stats(idx_layer, streams)
            hidden = torch.einsum('n,btnd->btd',
                                  self.mhc_readout.to(streams.dtype), streams)
        else:
            for idx_layer, block in enumerate(self.transformer.h):
                hidden = block(hidden, rope=rope,
                               kv_cache=caches[idx_layer // group])

        hidden = self.transformer.ln_f(hidden)

        if targets is None:
            # inference: only the last position's logits are needed
            return self.lm_head(hidden[:, -1:, :]), None

        logits = self.lm_head(hidden)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                               ignore_index=-1)
        # Auxiliary losses are added only while training. Including them at
        # evaluation would stop validation loss being pure cross-entropy and
        # silently break every cross-stage and cross-weight comparison.
        # last_lm_loss always holds the clean value.
        self.last_lm_loss = loss.detach()
        if self.training:
            aux = [b.ffn.aux_loss for b in self.transformer.h if b.is_moe]
            if aux:
                loss = loss + torch.stack(aux).sum()
        return logits, loss

    # ------------------------------------------------------- instrumentation

    def _record_stream_stats(self, layer_idx, streams):
        """Snapshot per-layer stream statistics for the current forward pass.

        Similarity and norms come straight off the values. Gradient norms
        cannot — they do not exist until backward — so a hook is attached here
        and fills in the entry that mhc_report reads afterwards.
        """
        self._mhc_stats[layer_idx] = {
            'cos_sim': stream_cosine_similarity(streams),
            'stream_norms': stream_norms(streams),
        }
        if streams.requires_grad:
            def capture(grad, index=layer_idx):
                # grad is (B, T, n, d): reduce to one norm per stream
                self._mhc_grad_norms[index] = grad.float().pow(2).sum(
                    dim=(0, 1, 3)).sqrt().tolist()
            streams.register_hook(capture)

    def mhc_report(self):
        """Stage E instrumentation, or None when running a single stream.

        This is the point of Stage E: measure the stream-collapse risk on our
        own checkpoints rather than taking either narrative on trust.

        cos_sim            mean pairwise cosine similarity between streams.
                           Near 1.0 means the streams have become copies and
                           the "n streams carry n signals" reading has failed.
        dev_from_identity  Frobenius norm of A - I. Near zero means mixing has
                           collapsed to independent per-stream residuals.
        grad_norms         per-stream gradient norm. A stream near zero is
                           receiving no learning signal.
        stream_norms       per-stream RMS magnitude. One stream dominating here
                           is the single-dominant-stream failure mode.

        cos_sim, grad_norms and stream_norms need mhc_instrument=True; the
        mixing diagnostics read parameters and are always available.
        """
        if not self.mhc_enabled:
            return None

        per_layer = []
        for index, block in enumerate(self.transformer.h):
            entry = {'layer': index,
                     'attn_mixing': block.hc_attn.diagnostics(),
                     'mlp_mixing': block.hc_ffn.diagnostics()}
            entry.update(self._mhc_stats.get(index, {}))
            if index in self._mhc_grad_norms:
                entry['grad_norms'] = self._mhc_grad_norms[index]
            per_layer.append(entry)

        deviations = [e[key]['dev_from_identity']
                      for e in per_layer for key in ('attn_mixing', 'mlp_mixing')]
        report = {
            'mode': self.config.mhc_mode,
            'n_streams': self.config.mhc_streams,
            'readout': self.mhc_readout.detach().float().tolist(),
            'per_layer': per_layer,
            'mean_dev_from_identity': sum(deviations) / len(deviations),
            'max_dev_from_identity': max(deviations),
        }
        similarities = [e['cos_sim'] for e in per_layer if 'cos_sim' in e]
        if similarities:
            report['mean_cos_sim'] = sum(similarities) / len(similarities)
            report['max_cos_sim'] = max(similarities)
        gradients = [g for e in per_layer for g in e.get('grad_norms', [])]
        if gradients:
            report['min_stream_grad_norm'] = min(gradients)
            report['max_stream_grad_norm'] = max(gradients)
        return report

    def moe_report(self):
        """Routing instrumentation for the most recent forward pass.

        Stage C exists to detect routing collapse: a few experts absorbing
        nearly all traffic while the rest go idle. Read `dead` and
        `max_over_mean` first — the latter is 1.0 under uniform load and rises
        toward n_routed_experts as traffic concentrates. Per-layer counts show
        where it collapses, which is usually not uniform with depth.

        Returns None for a model without MoE layers.
        """
        layers = [(i, b.ffn) for i, b in enumerate(self.transformer.h) if b.is_moe]
        if not layers:
            return None

        per_layer = []
        for index, moe in layers:
            stats = moe.stats
            per_layer.append({
                'layer': index,
                'counts': stats['counts'].tolist(),
                'dead': int(stats['dead_experts']),
                'max_over_mean': float(stats['max_over_mean']),
                'load_balance': float(stats['load_balance']),
                'ortho': float(stats['ortho']),
                'variance': float(stats['variance']),
            })
        return {
            'n_routed_experts': layers[0][1].n_routed,
            'per_layer': per_layer,
            'total_dead': sum(e['dead'] for e in per_layer),
            'worst_max_over_mean': max(e['max_over_mean'] for e in per_layer),
            'mean_load_balance': sum(e['load_balance'] for e in per_layer) / len(per_layer),
        }

    # ---------------------------------------------------------------- optim

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """AdamW with decay applied only to matrix-shaped parameters.

        Biases, norm gains and the hyper-connection vectors are 1-D and are
        left undecayed: shrinking them toward zero fights what they are for.
        """
        trainable = [p for p in self.parameters() if p.requires_grad]
        matrices = [p for p in trainable if p.dim() >= 2]
        vectors = [p for p in trainable if p.dim() < 2]
        groups = [
            {'params': matrices, 'weight_decay': weight_decay},
            {'params': vectors, 'weight_decay': 0.0},
        ]
        print(f"decayed tensors: {len(matrices)} "
              f"({sum(p.numel() for p in matrices):,} params); "
              f"undecayed: {len(vectors)} ({sum(p.numel() for p in vectors):,})")

        supports_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = supports_fused and device_type == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 fused=True) if use_fused else torch.optim.AdamW(
                                     groups, lr=learning_rate, betas=betas)

    def estimate_mfu(self, fwdbwd_per_iter, dt, peak_flops=312e12):
        """Model FLOPs utilisation, as a fraction of peak device throughput.

        Uses the 6N + attention estimate from the PaLM paper's appendix.
        Defaults to A100 bf16 peak; pass peak_flops for other hardware or the
        number is meaningless.

        For MoE the count must be ACTIVE parameters, not total — using total
        overstates utilisation by total/active (2.14x at the Stage C1 config),
        which would make a healthy run look impossible.
        """
        cfg = self.config
        n_params = (self.param_report()['active'] if cfg.use_moe
                    else self.get_num_params())
        head_dim = cfg.n_embd // cfg.n_head
        per_token = 6 * n_params + 12 * cfg.n_layer * cfg.n_head * head_dim * cfg.block_size
        per_iter = per_token * cfg.block_size * fwdbwd_per_iter
        return (per_iter / dt) / peak_flops

    # ---------------------------------------------------------------- sample

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend `idx` by `max_new_tokens` tokens.

        No incremental cache: each step re-runs the whole (cropped) prefix, so
        cost is quadratic in the number of tokens generated. Adequate for
        inspecting a checkpoint, not for serving. Put the model in eval() mode
        first or dropout will be active.
        """
        for _ in range(max_new_tokens):
            window = idx[:, -self.config.block_size:]
            logits, _ = self(window)
            logits = logits[:, -1, :].float() / max(temperature, 1e-8)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float('-inf'))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        return idx
