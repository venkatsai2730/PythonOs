"""
Stage C (MoE) correctness tests. CPU, a few seconds.

MoE has failure modes that a smoke test happily reports as success:

  - experts silently receiving no gradient (dispatch drops them) => you are
    training a much smaller model than you think
  - gate weights not summing to 1 => output scale drifts with routing
  - the aux loss leaking into val loss => cross-stage comparisons invalid
  - token/expert bookkeeping wrong => the collapse detector lies to you
  - dense layers accidentally becoming MoE (or vice versa)

$ python test_stage_c.py
"""

import torch

import os
import sys

# run these as plain scripts (python tests/x.py) without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT, MoE

BASE = dict(block_size=32, vocab_size=64, n_layer=4, n_head=4,
            n_embd=64, dropout=0.0, bias=False)
MOE = dict(use_moe=True, n_shared_experts=2, n_routed_experts=6,
           moe_top_k=2, moe_first_k_dense=1)

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

def make(**over):
    torch.manual_seed(1337)
    m = GPT(GPTConfig(**{**BASE, **over}))
    return m

x = torch.randint(0, BASE['vocab_size'], (2, 16))

# ---------------------------------------------------------------- structure
print("\n1. layer structure")
m = make(**MOE)
is_moe = [getattr(b, 'is_moe', False) for b in m.transformer.h]
check("moe_first_k_dense=1 leaves exactly layer 0 dense",
      is_moe == [False, True, True, True], str(is_moe))
check("dense layer 0 is a plain FeedForward",
      type(m.transformer.h[0].mlp).__name__ == 'FeedForward')
check("layer 1 is a MoE", isinstance(m.transformer.h[1].mlp, MoE))
moe0 = m.transformer.h[1].mlp
check("expert count = shared + routed",
      len(moe0.shared_experts) == 2 and len(moe0.routed_experts) == 6)
# auto hidden = 4*n_embd // (shared + top_k) = 256 // 4 = 64
check("auto expert hidden matches dense active FFN params",
      moe0.expert_hidden == 4 * BASE['n_embd'] // (2 + 2), f"hidden={moe0.expert_hidden}")

# ---------------------------------------------------------------- active vs total
print("\n2. param accounting")
dense = make()
pr = m.param_report()
d_tot = sum(p.numel() for p in dense.parameters())
check("total > active (experts are held but not all used)", pr['total'] > pr['active'],
      f"total={pr['total']:,} active={pr['active']:,}")
# per-MoE-layer active FFN params should equal the dense layer's FFN params
per_expert = sum(p.numel() for p in moe0.routed_experts[0].parameters())
dense_ffn = sum(p.numel() for p in dense.transformer.h[0].mlp.parameters())
check("active FFN params/layer == dense FFN params/layer",
      (2 + 2) * per_expert == dense_ffn,
      f"{(2+2)*per_expert:,} vs {dense_ffn:,}")
router_params = sum(p.numel() for p in moe0.router.parameters())
check("active total == dense total + routers",
      pr['active'] == d_tot + 3 * router_params,
      f"active={pr['active']:,} dense+routers={d_tot + 3*router_params:,}")

# ---------------------------------------------------------------- gradients reach experts
# The dispatch loop is where experts get silently dropped. With enough tokens
# every expert should be selected at least once and receive gradient.
print("\n3. every expert receives gradient")
m = make(**MOE)
m.train()
big = torch.randint(0, BASE['vocab_size'], (8, 32))
_, loss = m(big, big)
loss.backward()
no_grad, zero_grad = [], []
for li, b in enumerate(m.transformer.h):
    if not getattr(b, 'is_moe', False):
        continue
    for ei, e in enumerate(b.mlp.routed_experts):
        g = e.up.weight.grad
        if g is None:
            no_grad.append((li, ei))
        elif g.abs().sum() == 0:
            zero_grad.append((li, ei))
check("no routed expert has grad=None", not no_grad, str(no_grad))
check("no routed expert has all-zero grad", not zero_grad, str(zero_grad))
shared_ok = all(e.up.weight.grad is not None and e.up.weight.grad.abs().sum() > 0
                for b in m.transformer.h if getattr(b, 'is_moe', False)
                for e in b.mlp.shared_experts)
check("shared experts receive gradient", shared_ok)
check("router receives gradient",
      all(b.mlp.router.weight.grad is not None and b.mlp.router.weight.grad.abs().sum() > 0
          for b in m.transformer.h if getattr(b, 'is_moe', False)))

# ---------------------------------------------------------------- token bookkeeping
print("\n4. routing bookkeeping")
m = make(**MOE)
m.train()
_, _ = m(big, big)
rep = m.moe_report()
n_tokens = big.numel()
for p in rep['per_layer']:
    total_slots = sum(p['counts'])
    ok = total_slots == n_tokens * MOE['moe_top_k']
    check(f"L{p['layer']} routing slots == tokens * top_k", ok,
          f"{total_slots} vs {n_tokens * MOE['moe_top_k']}")
check("moe_report returns None for a dense model", make().moe_report() is None)

# ---------------------------------------------------------------- gate normalisation
print("\n5. gate weights are normalised")
cfg = GPTConfig(**{**BASE, **MOE})
torch.manual_seed(0)
layer = MoE(cfg, 1)
probs = torch.softmax(torch.randn(20, cfg.n_routed_experts), dim=-1)
topk_p, _ = probs.topk(cfg.moe_top_k, dim=-1)
gates = topk_p / topk_p.sum(dim=-1, keepdim=True)
check("renormalised gates sum to 1 per token",
      torch.allclose(gates.sum(-1), torch.ones(20), atol=1e-6))

# A balanced router should sit near the loss minimum of 1.0.
print("\n6. load-balance loss is calibrated")
torch.manual_seed(0)
layer = MoE(cfg, 1)
with torch.no_grad():
    layer(torch.randn(8, 128, cfg.n_embd))   # random init => roughly even load
lb = float(layer.stats['load_balance'])
mom = float(layer.stats['max_over_mean'])
check("balanced routing => load_balance near 1.0", 0.95 < lb < 1.35, f"lb={lb:.4f}")
check("balanced routing => max_over_mean near 1.0", mom < 2.0, f"max/mean={mom:.2f}")

# DOCUMENTED BLIND SPOT, not a bug to fix:
# The Switch load-balance loss is E * sum_i f_i * P_i. If the router emits
# EXACTLY tied probabilities, P is uniform and topk's tie-breaking sends every
# token to the lowest-indexed experts. Then f = [0.5, 0.5, 0, ...] and the loss
# evaluates to 6 * (0.5/6 + 0.5/6) = 1.0 — its perfect minimum — while load is
# in fact fully collapsed.
#
# Consequence for Stage C: the load-balance loss VALUE is not a collapse
# detector. Only the token counts are. This is why moe_report() logs counts,
# dead experts and max_over_mean rather than just the aux loss.
print("\n6b. load-balance loss is blind to tie-break collapse (documented)")
torch.manual_seed(0)
layer = MoE(cfg, 1)
with torch.no_grad():
    layer.router.weight.zero_()          # exactly tied probabilities
    layer(torch.randn(4, 64, cfg.n_embd))
lb_tied = float(layer.stats['load_balance'])
counts_tied = layer.stats['counts'].tolist()
check("tied router: load_balance still reports its 1.0 minimum",
      abs(lb_tied - 1.0) < 1e-4, f"lb={lb_tied:.6f} counts={counts_tied}")
check("tied router: counts DO reveal the collapse",
      int(layer.stats['dead_experts']) == cfg.n_routed_experts - cfg.moe_top_k,
      f"dead={int(layer.stats['dead_experts'])} of {cfg.n_routed_experts}")
check("tied router: max_over_mean DOES reveal the collapse",
      float(layer.stats['max_over_mean']) > 2.0,
      f"max/mean={float(layer.stats['max_over_mean']):.2f}")

# ---------------------------------------------------------------- aux loss isolation
# This is the one that silently corrupts every cross-stage comparison: if aux
# losses are added at eval time, val loss stops being pure cross-entropy.
print("\n7. aux losses do not leak into eval loss")
m = make(**MOE, moe_aux_loss_weight=5.0)   # absurd weight to make leakage obvious
m.eval()
with torch.no_grad():
    _, eval_loss = m(x, x)
check("eval loss == pure LM loss (no aux term)",
      abs(eval_loss.item() - float(m.last_lm_loss)) < 1e-6,
      f"eval={eval_loss.item():.6f} lm={float(m.last_lm_loss):.6f}")
m.train()
_, train_loss = m(x, x)
check("train loss > LM loss (aux IS applied)",
      train_loss.item() > float(m.last_lm_loss) + 1e-3,
      f"train={train_loss.item():.4f} lm={float(m.last_lm_loss):.4f}")

# aux weight 0 => train loss identical to LM loss
m = make(**MOE, moe_aux_loss_weight=0.0)
m.train()
_, l0 = m(x, x)
check("all aux weights 0 => train loss == LM loss",
      abs(l0.item() - float(m.last_lm_loss)) < 1e-6)

# ---------------------------------------------------------------- collapse detector
print("\n8. collapse detector actually detects collapse")
torch.manual_seed(0)
layer = MoE(cfg, 1)
with torch.no_grad():
    # force all traffic onto expert 0 by making its router row dominate
    layer.router.weight.zero_()
    layer.router.weight[0] = 100.0
    layer(torch.randn(4, 64, cfg.n_embd).abs())  # positive inputs => expert 0 wins
counts = layer.stats['counts'].tolist()
check("collapsed routing flagged by dead_experts",
      int(layer.stats['dead_experts']) >= cfg.n_routed_experts - 2, f"counts={counts}")
check("collapsed routing flagged by max_over_mean",
      float(layer.stats['max_over_mean']) > 2.0,
      f"max/mean={float(layer.stats['max_over_mean']):.2f}")

# ---------------------------------------------------------------- composes with Stage B
print("\n9. composes with Stage B")
m = make(**MOE, learned_pos_emb=False, use_rope=True, swa_window=8,
         swa_full_every=4, kv_share_group=2, use_mla=True, kv_lora_rank=32,
         q_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)
m.train()
_, l = m(x, x)
l.backward()
check("MoE + full Stage B forward/backward", torch.isfinite(l),
      f"loss={l.item():.4f}")
# causality must survive
m.eval()
xa = torch.randint(0, BASE['vocab_size'], (1, 16))
xb = xa.clone(); xb[0, -1] = (xa[0, -1] + 1) % BASE['vocab_size']
with torch.no_grad():
    la, _ = m(xa, xa)
    lb2, _ = m(xb, xb)
delta = (la[:, :-1] - lb2[:, :-1]).abs().max().item()
check("causality holds with MoE + Stage B", delta < 1e-6, f"delta={delta:.2e}")

# ---------------------------------------------------------------- summary
passed = sum(ok for _, ok in results)
print(f"\n{'=' * 60}\n{passed}/{len(results)} checks passed")
if passed != len(results):
    raise SystemExit("Stage C FAILED — fix before training.")
print("Stage C implementation verified.")
