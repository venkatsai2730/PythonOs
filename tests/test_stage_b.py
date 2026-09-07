"""
Stage B correctness tests. CPU, a few seconds.

These target the failure modes that do NOT show up as a crash and do NOT show
up as an obviously bad loss curve — they show up as a model that quietly
cheats or quietly ignores part of its input:

  - broken causality (attending to the future) => loss looks great, model is worthless
  - an off-by-one sliding window => silently wrong receptive field
  - "KV sharing" that doesn't actually share => you measure nothing
  - RoPE that isn't relative => you built absolute positions with extra steps

$ python test_stage_b.py
"""

import torch

import os
import sys

# run these as plain scripts (python tests/x.py) without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT, build_layer_plan

BASE = dict(block_size=64, vocab_size=128, n_layer=4, n_head=4,
            n_embd=64, dropout=0.0, bias=False)
MLA_DIMS = dict(use_mla=True, kv_lora_rank=32, q_lora_rank=32,
                qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)

results = []

def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

def make(**over):
    torch.manual_seed(1337)
    m = GPT(GPTConfig(**{**BASE, **over}))
    m.eval()
    return m

VARIANTS = {
    'stage_a (dense)':   dict(),
    'b1 rope+nope':      dict(learned_pos_emb=False, use_rope=True, nope_layers=(1, 3)),
    'b2 swa':            dict(swa_window=8, swa_full_every=4),
    'b3 mla':            dict(learned_pos_emb=False, use_rope=True, **MLA_DIMS),
    'b4 kvshare':        dict(kv_share_group=2),
    'b_full':            dict(learned_pos_emb=False, use_rope=True, nope_layers=(2, 3),
                              swa_window=8, swa_full_every=4, kv_share_group=2, **MLA_DIMS),
}

# ---------------------------------------------------------------- forward/backward
print("\n1. forward + backward, finite loss")
for name, over in VARIANTS.items():
    m = make(**over)
    m.train()
    x = torch.randint(0, BASE['vocab_size'], (2, 32))
    _, loss = m(x, x)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    ok = (torch.isfinite(loss) and all(g is not None for g in grads)
          and all(torch.isfinite(g).all() for g in grads))
    n = sum(p.numel() for p in m.parameters())
    check(name, ok, f"loss={loss.item():.4f} params={n:,}")

# ---------------------------------------------------------------- causality
# Perturb the LAST token; logits at every earlier position must be unchanged.
# This is the single most important test here: a mask bug that leaks the future
# makes the loss curve look *better*, so nothing downstream would catch it.
print("\n2. causality (future tokens cannot affect earlier logits)")
for name, over in VARIANTS.items():
    m = make(**over)
    x = torch.randint(0, BASE['vocab_size'], (1, 32))
    x2 = x.clone()
    x2[0, -1] = (x[0, -1] + 1) % BASE['vocab_size']
    with torch.no_grad():
        a, _ = m(x, x)     # targets given => full logits, not just last position
        b, _ = m(x2, x2)
    delta = (a[:, :-1] - b[:, :-1]).abs().max().item()
    check(name, delta < 1e-6, f"max delta at earlier positions = {delta:.2e}")

# ---------------------------------------------------------------- window bounds
# In a 1-layer windowed model, a token more than `window` steps back must have
# exactly zero influence; a token just inside the window must have some.
print("\n3. sliding window receptive field (single windowed layer, window=8)")
W = 8
m = make(n_layer=1, swa_window=W, swa_full_every=99)  # never full
x = torch.randint(0, BASE['vocab_size'], (1, 32))
QUERY = 20
with torch.no_grad():
    ref, _ = m(x, x)
def influence(pos):
    x2 = x.clone()
    x2[0, pos] = (x[0, pos] + 1) % BASE['vocab_size']
    with torch.no_grad():
        alt, _ = m(x2, x2)
    return (ref[0, QUERY] - alt[0, QUERY]).abs().max().item()

inside = influence(QUERY - W + 1)   # oldest key still inside the window
outside = influence(QUERY - W)      # exactly one step too far back
check("token inside window influences query", inside > 1e-6, f"delta={inside:.2e}")
check("token outside window has zero influence", outside < 1e-9, f"delta={outside:.2e}")

# ---------------------------------------------------------------- KV sharing is real
print("\n4. cross-layer KV sharing actually shares")
plan = build_layer_plan(GPTConfig(**BASE, kv_share_group=2))
roles = [p['kv_role'] for p in plan]
check("layer plan alternates produce/consume", roles == ['produce', 'consume'] * 2, str(roles))

shared = make(kv_share_group=2)
own = make()
# consumers must have no k/v projection at all
consumer = shared.transformer.h[1].attn
check("consumer layer has no K/V projection", not hasattr(consumer, 'kv_proj'))
check("consumer layer still has a query projection", hasattr(consumer, 'q_proj'))
saved = sum(p.numel() for p in own.parameters()) - sum(p.numel() for p in shared.parameters())
expected = 2 * 2 * BASE['n_embd'] ** 2   # 2 consumers x (k,v) x n_embd^2
check("param saving matches k+v projections removed", saved == expected,
      f"saved={saved:,} expected={expected:,}")

# ---------------------------------------------------------------- RoPE is relative
# With no absolute position embedding, a full-attention RoPE layer should give
# the same answer for a subsequence regardless of where the window starts,
# once the prefix is long enough to be outside the causal dependency. We test
# the weaker, exact property: RoPE with theta -> identity at position 0.
print("\n5. RoPE sanity")
cos, sin = __import__('pythonos').precompute_rope_cache(8, 16, 10000.0)
check("cos[0]==1, sin[0]==0 (no rotation at position 0)",
      torch.allclose(cos[0], torch.ones(4)) and torch.allclose(sin[0], torch.zeros(4)))
from pythonos import apply_rope
v = torch.randn(1, 1, 1, 8)
check("apply_rope is identity at position 0", torch.allclose(apply_rope(v, cos, sin), v, atol=1e-6))
# rotation must preserve norm
v16 = torch.randn(1, 1, 16, 8)
rot = apply_rope(v16, cos, sin)
check("apply_rope preserves per-head norm",
      torch.allclose(v16.norm(dim=-1), rot.norm(dim=-1), atol=1e-5))

# ---------------------------------------------------------------- no positions => permutation
# A pure-NoPE model with no wpe and no RoPE has no way to know absolute position.
# Causal masking still makes order matter, but a single-token input must give
# the same logits wherever we claim it sits. Cheap guard against a stray
# absolute-position leak.
print("\n6. pure NoPE has no absolute position signal")
m = make(learned_pos_emb=False, use_rope=False)
tok = torch.tensor([[7]])
with torch.no_grad():
    l1, _ = m(tok, tok)
    seq = torch.tensor([[7, 7, 7, 7]])
    l4, _ = m(seq, seq)
# first position of the length-4 run must match the length-1 run
check("logits at position 0 independent of sequence length",
      torch.allclose(l1[0, 0], l4[0, 0], atol=1e-6),
      f"delta={(l1[0,0]-l4[0,0]).abs().max().item():.2e}")

# ---------------------------------------------------------------- guard rail
print("\n7. incoherent config is rejected")
try:
    build_layer_plan(GPTConfig(**BASE, use_rope=True, nope_layers=(1,), kv_share_group=2))
    check("nope_layers misaligned with kv_share_group raises", False)
except AssertionError:
    check("nope_layers misaligned with kv_share_group raises", True)

# ---------------------------------------------------------------- summary
passed = sum(ok for _, ok in results)
print(f"\n{'='*60}\n{passed}/{len(results)} checks passed")
if passed != len(results):
    raise SystemExit("Stage B FAILED — fix before training.")
print("Stage B implementation verified.")
