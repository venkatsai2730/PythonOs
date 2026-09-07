"""Feed-forward blocks: dense (Stage A) and fine-grained MoE (Stage C)."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """Position-wise feed-forward network: expand, activate, project back.

    The Stage A baseline uses a 4x expansion with GELU.
    """

    def __init__(self, config, hidden=None):
        super().__init__()
        hidden = hidden or config.ffn_mult * config.n_embd
        self.hidden = hidden
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.down.is_residual_out = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.gelu(self.up(x))))


class Expert(nn.Module):
    """One fine-grained MoE expert.

    Same shape as FeedForward but with an independently chosen hidden width, so
    N experts can each be narrower than one dense feed-forward. No dropout: the
    parent MoE applies it once to the combined output.
    """

    def __init__(self, config, hidden):
        super().__init__()
        self.up = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.down.is_residual_out = True

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoE(nn.Module):
    """Fine-grained mixture of experts with shared and routed experts.

    Shared experts process every token unconditionally. The router selects
    moe_top_k of the routed experts per token. Active experts per token is
    therefore n_shared_experts + moe_top_k.

    Three auxiliary losses, weighted independently so each can be isolated:

    load-balance   Switch-Transformer form, E * sum_i f_i * P_i, where f_i is
                   the share of routing slots taken by expert i and P_i its
                   mean router probability. Minimised at 1.0 under uniform
                   load. The one well-understood term of the three.

    orthogonality  Mean squared off-diagonal of the Gram matrix of L2
                   normalised router rows. Pushes expert query directions
                   apart, the intent being specialisation.

    variance       Mean per-token variance of the routing distribution,
                   negated, so minimising it produces confident peaked routing.

    IMPORTANT, and absent from the architecture doc: the variance term and the
    load-balancing term pull in opposite directions. Load balancing wants every
    expert to see equal traffic; variance wants each token to commit hard to
    one expert. Both can hold at once, with different tokens committing to
    different experts, but the weight ratio decides which wins and a poor ratio
    either collapses routing or flattens it. Both default to zero weight so
    load balancing alone is validated first.

    The orthogonality and variance terms are not standard named losses with a
    single canonical definition. The forms above are a reading of the doc's
    intent, not a reproduction of published formulae — treat their exact shape
    as a design choice under test.
    """

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_shared = config.n_shared_experts
        self.n_routed = config.n_routed_experts
        self.top_k = config.moe_top_k
        if self.top_k > self.n_routed:
            raise ValueError(
                f"moe_top_k={self.top_k} exceeds n_routed_experts={self.n_routed}")
        self.weights = {
            'load_balance': config.moe_aux_loss_weight,
            'ortho': config.moe_ortho_loss_weight,
            'variance': config.moe_var_loss_weight,
        }

        # width 0 means "match one dense feed-forward's ACTIVE parameter
        # count", so a loss comparison against the dense baseline reflects
        # conditional computation rather than a change in capacity
        hidden = config.moe_expert_hidden
        active = self.n_shared + self.top_k
        if hidden == 0:
            hidden = (config.ffn_mult * config.n_embd) // active
        self.expert_hidden = hidden

        self.shared_experts = nn.ModuleList(
            Expert(config, hidden) for _ in range(self.n_shared))
        self.routed_experts = nn.ModuleList(
            Expert(config, hidden) for _ in range(self.n_routed))
        self.router = nn.Linear(config.n_embd, self.n_routed, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # refreshed each forward; read by GPT for the loss and by moe_report
        self.aux_loss = None
        self.stats = {}

    def _auxiliary_losses(self, probs, counts, n_slots):
        share = counts.to(probs.dtype) / max(n_slots, 1)
        mean_prob = probs.mean(dim=0)
        load_balance = self.n_routed * torch.sum(share * mean_prob)

        directions = F.normalize(self.router.weight, dim=-1)
        gram = directions @ directions.t()
        gram_off = gram - torch.diag_embed(torch.diagonal(gram))
        pairs = max(self.n_routed * (self.n_routed - 1), 1)
        ortho = torch.sum(gram_off.pow(2)) / pairs

        variance = -probs.var(dim=-1).mean()
        return {'load_balance': load_balance, 'ortho': ortho, 'variance': variance}

    def forward(self, x):
        batch, seq_len, width = x.shape
        tokens = x.reshape(-1, width)
        n_tokens = tokens.shape[0]

        probs = F.softmax(self.router(tokens), dim=-1)
        top_probs, top_idx = torch.topk(probs, self.top_k, dim=-1)
        # renormalise across the chosen experts so the gates sum to one
        gates = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        combined = torch.zeros_like(tokens)
        counts = torch.zeros(self.n_routed, dtype=torch.long, device=x.device)
        for index, expert in enumerate(self.routed_experts):
            token_pos, slot = torch.nonzero(top_idx == index, as_tuple=True)
            counts[index] = token_pos.numel()
            if token_pos.numel() == 0:
                continue
            weighted = expert(tokens[token_pos]) * gates[token_pos, slot].unsqueeze(-1)
            combined = combined.index_add(0, token_pos, weighted.to(combined.dtype))

        for expert in self.shared_experts:
            combined = combined + expert(tokens)

        terms = self._auxiliary_losses(probs, counts, n_tokens * self.top_k)
        self.aux_loss = sum(self.weights[name] * value for name, value in terms.items())

        counts_f = counts.to(probs.dtype)
        self.stats = {
            'counts': counts.detach(),
            'dead_experts': torch.sum(counts == 0).detach(),
            # 1.0 under uniform load, rising toward n_routed as traffic
            # concentrates on a few experts
            'max_over_mean': (counts.max().to(probs.dtype)
                              / counts_f.mean().clamp_min(1e-9)).detach(),
            **{name: value.detach() for name, value in terms.items()},
        }
        return self.dropout(combined.view(batch, seq_len, width))


# the Stage A baseline feed-forward, under its historical name
MLP = FeedForward
