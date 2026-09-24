"""
Parameter accounting for every stage, at nano scale and at the full spec.

For MoE, total and active params diverge sharply, and quoting only one is how
training budgets end up wrong. This script prints both, and checks whether the
architecture doc's full-scale figures are internally consistent.

$ python scripts/param_report.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT

NANO = dict(block_size=1024, vocab_size=49152, n_layer=8, n_head=16,
            n_embd=1024, dropout=0.0, bias=False)
MLA = dict(use_mla=True, kv_lora_rank=256, q_lora_rank=512,
           qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64)
MOE = dict(use_moe=True, n_shared_experts=2, n_routed_experts=14,
           moe_top_k=2, moe_first_k_dense=1)

STAGES = {
    'A  dense baseline': dict(),
    'B1 rope+nope': dict(learned_pos_emb=False, use_rope=True, nope_layers=(3, 7)),
    'B2 swa 5:1': dict(swa_window=256, swa_full_every=6),
    'B3 mla (+rope)': dict(learned_pos_emb=False, use_rope=True, **MLA),
    'B4 kv sharing': dict(kv_share_group=2),
    'B  full hybrid': dict(learned_pos_emb=False, use_rope=True, nope_layers=(6, 7),
                           swa_window=256, swa_full_every=6, kv_share_group=2, **MLA),
    'C1 moe': dict(**MOE),
    'C  moe + full B': dict(learned_pos_emb=False, use_rope=True, nope_layers=(6, 7),
                            swa_window=256, swa_full_every=6, kv_share_group=2,
                            **MLA, **MOE),
}

print("NANO SCALE (d_model 1024, 8 layers, vocab 49152 -- StarCoder2 BPE)")
print(f"  {'stage':20s} {'total':>10s} {'active':>10s} {'active vs A':>13s}  note")
baseline = None
for name, over in STAGES.items():
    model = GPT(GPTConfig(**NANO, **over))
    pr = model.param_report()
    if baseline is None:
        baseline = pr['active']
    delta = 100 * (pr['active'] - baseline) / baseline
    note = ''
    if pr['inactive']:
        note = (f"expert h={pr['expert_hidden']}, "
                f"{pr['inactive'] / 1e6:.0f}M held but inactive; "
                f"~{pr['total'] * 12 / 1e9:.1f}GB fp32 AdamW state")
    print(f"  {name:20s} {pr['total'] / 1e6:9.2f}M {pr['active'] / 1e6:9.2f}M "
          f"{delta:+12.2f}%  {note}")

print("\nFULL SPEC (d_model 2048, 32 layers, 16 experts = 2 shared + 14 routed,")
print("           top-2, 31 MoE layers, vocab 49152 -- StarCoder2 BPE)")
d, L, vocab, moe_layers = 2048, 32, 49152, 31
E_total, E_active = 16, 4
base = L * 4 * d * d + vocab * d          # attention + embeddings
dense_ff = (L - moe_layers) * 2 * d * 4 * d


def totals(h):
    per_expert = 2 * d * h
    total = base + dense_ff + moe_layers * E_total * per_expert
    active = base + dense_ff + moe_layers * E_active * per_expert
    return total, active


print(f"  non-FFN base (attn + embeddings): {base / 1e9:.2f}B")
print(f"  {'expert hidden':18s} {'total':>8s} {'active':>8s}")
for mult, label in [(1, 'h = d'), (2, 'h = 2d'), (4, 'h = 4d (dense-equiv)')]:
    t, a = totals(d * mult)
    print(f"  {label:18s} {t / 1e9:7.2f}B {a / 1e9:7.2f}B")

# Solve each doc constraint separately to show they disagree
per_h_total = moe_layers * E_total * 2 * d
per_h_active = moe_layers * E_active * 2 * d
h_for_active_2b = (2.0e9 - base - dense_ff) / per_h_active
h_for_total_95b = (9.5e9 - base - dense_ff) / per_h_total
t1, a1 = totals(h_for_active_2b)
t2, a2 = totals(h_for_total_95b)

print("\n  Doc claims ~9-10B total with ~2B active. Solving each separately:")
print(f"    pin active = 2.00B -> h ~= {h_for_active_2b:.0f} -> total = {t1 / 1e9:.2f}B")
print(f"    pin total  = 9.50B -> h ~= {h_for_total_95b:.0f} -> active = {a2 / 1e9:.2f}B")
print("\n  These do not reconcile. With 16 experts and 4 active, total expert")
print("  params are exactly 4x active expert params, so the ratio is fixed by")
print("  the architecture and cannot be tuned. Whichever figure the training")
print("  budget is built on must be the one that is fixed; correct the other.")
