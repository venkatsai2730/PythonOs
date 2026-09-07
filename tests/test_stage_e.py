"""
Stage E (mHC / Hyper-Connections) correctness tests. CPU, a few seconds.

Stage E replaces the single residual stream with n streams, which is a
structural change to the whole layer stack. The failure modes that do not
announce themselves:

  - the reparameterisation not actually reducing to the baseline at init, so
    every measured difference is confounded with an init change
  - Sinkhorn silently producing a non-doubly-stochastic matrix (or NaNs, if
    done outside log space) — mHC then isn't manifold-constrained at all
  - streams that receive no gradient, so "4 streams" is really 1 plus 3 dead
  - causality quietly broken by the stream plumbing
  - instrumentation that reports healthy numbers because it is reading the
    wrong tensor

$ python tests/test_stage_e.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT
from pythonos.hyper import (HyperConnection, sinkhorn_log,
                            stream_cosine_similarity, stream_norms)

BASE = dict(block_size=64, vocab_size=128, n_layer=4, n_head=4,
            n_embd=64, dropout=0.0, bias=False)
MLA = dict(use_mla=True, kv_lora_rank=32, q_lora_rank=32,
           qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)
MOE = dict(use_moe=True, n_shared_experts=2, n_routed_experts=6,
           moe_top_k=2, moe_first_k_dense=1)

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def make(**over):
    torch.manual_seed(1337)
    return GPT(GPTConfig(**{**BASE, **over}))


x = torch.randint(0, BASE['vocab_size'], (2, 32),
                  generator=torch.Generator().manual_seed(5))


def logits_of(**over):
    m = make(**over)
    m.eval()
    with torch.no_grad():
        lg, ls = m(x, x)
    return lg, ls.item()


# ------------------------------------------------------- baseline equivalence
# The single most important property in this stage. HC is built so that
# alpha = beta = readout = e_0 and A = I, meaning stream 0 carries exactly the
# standard residual computation and the rest sit inert. If this does not hold
# EXACTLY, then every later HC-vs-baseline loss difference is contaminated by
# an initialisation change and Stage E measures nothing.
print("\n1. 'hc' is bit-identical to the dense baseline at init")
base_logits, base_loss = logits_of()
for n in (2, 4, 8):
    lg, ls = logits_of(mhc_mode='hc', mhc_streams=n)
    delta = (lg - base_logits).abs().max().item()
    check(f"hc n={n} reproduces baseline exactly", delta == 0.0,
          f"max|dlogit|={delta:.3e} loss={ls:.10f}")

print("\n2. 'single' mode carries no Stage E machinery")
m = make()
check("no hyper-connection modules", not hasattr(m.transformer.h[0], 'hc_attn'))
check("no readout parameter", not hasattr(m, 'mhc_readout'))
check("mhc_report() returns None", m.mhc_report() is None)
hc_model = make(mhc_mode='hc', mhc_streams=4)
overhead = sum(p.numel() for p in hc_model.parameters()) - sum(p.numel() for p in m.parameters())
# per block: 2 connections x (alpha n + beta n + A n^2), plus one readout
expected = BASE['n_layer'] * 2 * (4 + 4 + 16) + 4
check("param overhead is exactly the connection weights", overhead == expected,
      f"{overhead} vs {expected}")

# ------------------------------------------------------- mHC init deviation
# mHC reaches A through a Sinkhorn projection, and identity is only a limit
# point of that projection. So mHC starts NEAR the baseline, not at it. We
# assert the deviation is small AND that it shrinks as the init logit grows —
# that is what shows the deviation is the projection and nothing else.
print("\n3. 'mhc' starts near (not at) the baseline, controllably")
prev = None
for logit in (5.0, 10.0, 20.0):
    lg, ls = logits_of(mhc_mode='mhc', mhc_streams=4, mhc_init_logit=logit)
    delta = (lg - base_logits).abs().max().item()
    print(f"      init_logit={logit:<5} max|dlogit|={delta:.3e}")
    if prev is not None:
        check(f"deviation shrinks as init_logit rises to {logit}", delta < prev,
              f"{delta:.3e} < {prev:.3e}")
    prev = delta
check("mhc deviation at default logit is small", prev is not None and prev < 1e-2)

# ------------------------------------------------------- Sinkhorn correctness
print("\n4. Sinkhorn-Knopp projects onto the doubly-stochastic manifold")
# Columns are normalised last, so they are exact regardless of conditioning.
torch.manual_seed(0)
for n in (2, 4, 8):
    A = sinkhorn_log(torch.randn(n, n) * 2.0, 30)
    col = (A.sum(dim=0) - 1).abs().max().item()
    check(f"n={n}: columns sum to 1 exactly", col < 1e-5, f"col_dev={col:.2e}")
    check(f"n={n}: all entries non-negative", (A >= 0).all().item())
# Given enough iterations, rows converge too.
A = sinkhorn_log(torch.randn(4, 4, generator=torch.Generator().manual_seed(0)) * 2.0, 2000)
check("rows converge given enough iterations",
      (A.sum(dim=1) - 1).abs().max().item() < 1e-5,
      f"row_dev@2000={(A.sum(dim=1) - 1).abs().max().item():.2e}")

# DOCUMENTED LIMITATION, not a bug to paper over.
# Sinkhorn converges only linearly, so at any practical iteration count a
# sufficiently skewed A is NOT actually doubly stochastic — mHC would then be
# unconstrained while claiming otherwise. Encoded here so the limitation
# cannot quietly regress, and monitored via row_sum_dev during training.
print("\n4b. fixed-iteration Sinkhorn does NOT converge for skewed A (documented)")
worst = 0.0
for n in (2, 4, 8):
    for s in range(300):
        g = torch.Generator().manual_seed(s)
        A = sinkhorn_log(torch.randn(n, n, generator=g) * 2.0, 20)
        worst = max(worst, (A.sum(dim=1) - 1).abs().max().item())
check("skewed A at default iters leaves a real residual (>1e-3)", worst > 1e-3,
      f"worst row_dev={worst:.2e} — this is why row_sum_dev is logged")
# ...but the regime mHC actually trains in converges immediately.
near = sinkhorn_log(10.0 * torch.eye(4), 8)
check("near-identity A converges at just 8 iterations",
      (near.sum(dim=1) - 1).abs().max().item() < 1e-6,
      f"row_dev={(near.sum(dim=1) - 1).abs().max().item():.2e}")

# Log space is not a stylistic choice: the strongly-diagonal init lives at
# exp(-20) and the naive exp/divide formulation underflows to NaN there.
A = sinkhorn_log(20.0 * torch.eye(6), 8)
check("no NaN/Inf at the strongly-diagonal init", torch.isfinite(A).all().item())
check("strongly-diagonal logits give near-identity A",
      (A - torch.eye(6)).norm().item() < 1e-3,
      f"||A-I||={(A - torch.eye(6)).norm().item():.2e}")
# identity IS doubly stochastic, so the projection must leave it reachable
check("identity is a fixed point of the projection",
      (sinkhorn_log(torch.eye(4) * 50, 8) - torch.eye(4)).norm().item() < 1e-6)

# ------------------------------------------------------- forward/backward
print("\n5. forward + backward across all three modes")
VARIANTS = {
    'single': dict(),
    'hc': dict(mhc_mode='hc', mhc_streams=4),
    'mhc': dict(mhc_mode='mhc', mhc_streams=4),
}
for name, over in VARIANTS.items():
    m = make(**over, mhc_instrument=True)
    m.train()
    _, loss = m(x, x)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    ok = (torch.isfinite(loss)
          and all(g is not None and torch.isfinite(g).all() for g in grads))
    check(f"{name}: finite loss and gradients", ok, f"loss={loss.item():.4f}")

# mixing matrices must receive gradient, or the mode is decorative
for name in ('hc', 'mhc'):
    m = make(**VARIANTS[name])
    m.train()
    _, loss = m(x, x)
    loss.backward()
    b = m.transformer.h[0]
    got = all(p.grad is not None and p.grad.abs().sum() > 0
              for p in (b.hc_attn.A_param, b.hc_attn.alpha, b.hc_attn.beta))
    check(f"{name}: mixing matrix, alpha and beta all receive gradient", got)
check("readout receives gradient",
      (lambda: (lambda mm: (mm[1].backward(), mm[0].mhc_readout.grad is not None
                            and mm[0].mhc_readout.grad.abs().sum() > 0)[1])(
          (lambda m: (m, m(x, x)[1]))(make(mhc_mode='hc', mhc_streams=4).train())))())

# ------------------------------------------------------- causality
print("\n6. causality survives the stream plumbing")
for name, over in VARIANTS.items():
    m = make(**over)
    m.eval()
    xa = torch.randint(0, BASE['vocab_size'], (1, 24))
    xb = xa.clone()
    xb[0, -1] = (xa[0, -1] + 1) % BASE['vocab_size']
    with torch.no_grad():
        la, _ = m(xa, xa)
        lb, _ = m(xb, xb)
    delta = (la[:, :-1] - lb[:, :-1]).abs().max().item()
    check(f"{name}: future token cannot affect earlier logits", delta < 1e-6,
          f"delta={delta:.2e}")

# ------------------------------------------------------- instrumentation
print("\n7. instrumentation reports real numbers")
m = make(mhc_mode='mhc', mhc_streams=4, mhc_instrument=True)
m.train()
_, loss = m(x, x)
loss.backward()          # grad-norm hooks fire here, not during forward
rep = m.mhc_report()
check("report covers every layer", len(rep['per_layer']) == BASE['n_layer'])
check("per-stream gradient norms recorded (needs backward)",
      all('grad_norms' in p and len(p['grad_norms']) == 4 for p in rep['per_layer']))
check("cosine similarity recorded", all('cos_sim' in p for p in rep['per_layer']))
check("stream norms recorded",
      all(len(p['stream_norms']) == 4 for p in rep['per_layer']))
check("mixing diagnostics present for both sublayers",
      all('attn_mixing' in p and 'mlp_mixing' in p for p in rep['per_layer']))
check("mhc mixing is doubly stochastic in practice",
      all(p['attn_mixing']['row_sum_dev'] < 1e-4
          and p['attn_mixing']['col_sum_dev'] < 1e-4 for p in rep['per_layer']),
      f"worst row dev={max(p['attn_mixing']['row_sum_dev'] for p in rep['per_layer']):.2e}")
print(f"      at init: mean cos_sim={rep['mean_cos_sim']:.4f}  "
      f"mean ||A-I||={rep['mean_dev_from_identity']:.2e}  "
      f"stream grad norms L0={[round(g, 4) for g in rep['per_layer'][0]['grad_norms']]}")

# Without instrumentation the mixing diagnostics must still work (they are
# read from parameters, not activations), but the activation stats must not
# be silently fabricated.
m = make(mhc_mode='mhc', mhc_streams=4, mhc_instrument=False)
m.eval()
with torch.no_grad():
    m(x, x)
rep_off = m.mhc_report()
check("mixing diagnostics available without instrumentation",
      'mean_dev_from_identity' in rep_off)
check("activation stats absent without instrumentation",
      'mean_cos_sim' not in rep_off
      and not any('cos_sim' in p for p in rep_off['per_layer']))

# ------------------------------------------------------- collapse detectors
# The detectors must actually fire. Force each failure mode by hand.
print("\n8. collapse detectors fire on forced collapse")
H_same = torch.randn(2, 8, 1, 16).expand(2, 8, 4, 16)
check("identical streams => cos_sim ~ 1.0",
      abs(stream_cosine_similarity(H_same) - 1.0) < 1e-4,
      f"cos={stream_cosine_similarity(H_same):.6f}")
H_diff = torch.randn(2, 8, 4, 16)
check("independent streams => cos_sim ~ 0.0",
      abs(stream_cosine_similarity(H_diff)) < 0.2,
      f"cos={stream_cosine_similarity(H_diff):.4f}")
H_dom = torch.randn(2, 8, 4, 16)
H_dom[:, :, 0, :] *= 50.0
norms = stream_norms(H_dom)
check("one dominant stream shows up in stream_norms",
      norms[0] > 10 * max(norms[1:]), f"norms={[round(n, 2) for n in norms]}")

hc = HyperConnection(4, 'hc')
check("A = I at init => dev_from_identity == 0",
      hc.diagnostics()['dev_from_identity'] == 0.0)
with torch.no_grad():
    hc.A_param.copy_(torch.full((4, 4), 0.25))
check("uniform mixing => dev_from_identity large",
      hc.diagnostics()['dev_from_identity'] > 1.0,
      f"dev={hc.diagnostics()['dev_from_identity']:.3f}")

# ------------------------------------------------------- composition
print("\n9. composes with Stages B and C")
m = make(mhc_mode='mhc', mhc_streams=4, mhc_instrument=True,
         learned_pos_emb=False, use_rope=True, nope_layers=(2, 3),
         swa_window=8, swa_full_every=4, kv_share_group=2, **MLA, **MOE)
m.train()
_, loss = m(x, x)
loss.backward()
check("mHC + full Stage B + Stage C forward/backward", torch.isfinite(loss),
      f"loss={loss.item():.4f}")
rep = m.mhc_report()
check("instrumentation still works in the full stack",
      rep is not None and 'mean_cos_sim' in rep,
      f"cos_sim={rep['mean_cos_sim']:.4f}")
check("MoE routing still reported in the full stack",
      m.moe_report() is not None)
m.eval()
xa = torch.randint(0, BASE['vocab_size'], (1, 24))
xb = xa.clone()
xb[0, -1] = (xa[0, -1] + 1) % BASE['vocab_size']
with torch.no_grad():
    la, _ = m(xa, xa)
    lb, _ = m(xb, xb)
check("causality holds in the full stack",
      (la[:, :-1] - lb[:, :-1]).abs().max().item() < 1e-6)

# ------------------------------------------------------- summary
passed = sum(ok for _, ok in results)
print(f"\n{'=' * 60}\n{passed}/{len(results)} checks passed")
if passed != len(results):
    raise SystemExit("Stage E FAILED — fix before training.")
print("Stage E implementation verified.")
