"""
Stage A invariance — the guard that protects every cross-stage comparison.

Stage A's loss curve is the reference point for Stages B through G. If a later
stage's code changes perturb the dense baseline's forward pass even slightly,
every comparison silently shifts and nothing downstream notices.

So the Stage A path is pinned here to exact values: parameter count, loss to
10 decimal places, and a hash of the output logits, all under a fixed seed.
Any change to these means the baseline moved. Run this after touching
anything in pythonos/.

If a change to the baseline is INTENTIONAL, update the constants below AND
retrain Stage A — do not update the constants alone, because the existing
Stage A checkpoint would then no longer correspond to the code.

$ python tests/test_stage_a_invariance.py
"""

import hashlib

import torch

import os
import sys

# run these as plain scripts (python tests/x.py) without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT

# Reference captured before any stage work, then re-verified after the Stage B,
# Stage C and Stage E work, the package split, and the full rewrite of the
# model and training code.
#
# The rewrite preserved these numbers exactly, which was not guaranteed: the
# projections were restructured (a fused QKV matrix became separate query and
# key/value projections) but module registration order and the cumulative
# initialisation draw order stayed the same, so the RNG consumes identically
# and the resulting weights are the same values in the same places. Every
# figure recorded in STAGES.md therefore still refers to this architecture.
REFERENCE = dict(
    params=836736,
    loss=4.5623674393,
    logits_sha='a255812729078c1511b61e331b52316e',
)
CONFIG = dict(block_size=128, vocab_size=256, n_layer=4, n_head=4,
              n_embd=128, dropout=0.0, bias=False)

torch.manual_seed(1337)
model = GPT(GPTConfig(**CONFIG))
model.eval()
x = torch.randint(0, 256, (2, 64), generator=torch.Generator().manual_seed(99))
with torch.no_grad():
    logits, loss = model(x, x)

actual = dict(
    params=sum(p.numel() for p in model.parameters()),
    loss=round(loss.item(), 10),
    logits_sha=hashlib.sha256(logits.numpy().tobytes()).hexdigest()[:32],
)

print("\nStage A invariance")
ok = True
for key, expected in REFERENCE.items():
    got = actual[key]
    match = (abs(got - expected) < 1e-9) if isinstance(expected, float) else (got == expected)
    ok &= match
    print(f"  [{'PASS' if match else 'FAIL'}] {key}: {got}"
          + ("" if match else f"  (expected {expected})"))

# The dense baseline must not carry any stage B/C machinery
cfg = GPTConfig(**CONFIG)
defaults_off = {
    'use_rope': False, 'use_mla': False, 'use_moe': False,
    'learned_pos_emb': True, 'swa_window': 0, 'kv_share_group': 1,
    'nope_layers': (),
}
for flag, expected in defaults_off.items():
    got = getattr(cfg, flag)
    match = got == expected
    ok &= match
    print(f"  [{'PASS' if match else 'FAIL'}] default {flag} == {expected}"
          + ("" if match else f"  (got {got})"))

has_moe = any(getattr(b, 'is_moe', False) for b in model.transformer.h)
ok &= not has_moe
print(f"  [{'PASS' if not has_moe else 'FAIL'}] no MoE layers in the dense baseline")

if not ok:
    raise SystemExit("\nStage A baseline has MOVED — every cross-stage "
                     "comparison is invalid until this is resolved.")
print("\nStage A baseline unchanged.")
