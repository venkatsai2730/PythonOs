"""
Structural sanity check — the standing gate for every stage.

Overfit ONE fixed batch to ~zero loss. This is not training; it is a proof that
gradients flow, the optimizer steps, the loss masking is correct, and save/
resume round-trips. Runs on CPU in well under a minute per variant.

A model that cannot memorise a single batch has a structural bug. Run this
before spending any GPU hours, and re-run it after every architecture change.
Stage B variants are included so an attention change that breaks learnability
is caught here rather than 3000 iterations into a GPU run.

  $ python sanity_overfit.py                     # all variants
  $ python sanity_overfit.py --variant b3_mla    # just one
"""

import argparse
import os
import tempfile

import torch

import os
import sys

# run these as plain scripts (python tests/x.py) without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPTConfig, GPT

# deliberately tiny — we are testing the machinery, not the model
BASE = dict(
    block_size=64,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_embd=64,
    dropout=0.0,  # dropout would make exact memorisation impossible
    bias=False,
)
# scaled-down MLA dims to suit the tiny base model
MLA = dict(use_mla=True, kv_lora_rank=32, q_lora_rank=32,
           qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16)

VARIANTS = {
    "stage_a": dict(),
    "b1_rope_nope": dict(learned_pos_emb=False, use_rope=True, nope_layers=(1, 3)),
    "b2_swa": dict(swa_window=8, swa_full_every=4),
    "b3_mla": dict(learned_pos_emb=False, use_rope=True, **MLA),
    "b4_kvshare": dict(kv_share_group=2),
    "b_full": dict(learned_pos_emb=False, use_rope=True, nope_layers=(2, 3),
                   swa_window=8, swa_full_every=4, kv_share_group=2, **MLA),
    # Stage C. Aux weights are set to 0 for the overfit test: load balancing
    # actively fights memorisation of a single batch (it wants traffic spread
    # across experts, memorisation wants whatever routing minimises LM loss),
    # so a nonzero weight would make this gate measure the wrong thing.
    "c_moe": dict(use_moe=True, n_shared_experts=2, n_routed_experts=6,
                  moe_top_k=2, moe_first_k_dense=1, moe_aux_loss_weight=0.0),
    "c_moe_plus_b": dict(use_moe=True, n_shared_experts=2, n_routed_experts=6,
                         moe_top_k=2, moe_first_k_dense=1, moe_aux_loss_weight=0.0,
                         learned_pos_emb=False, use_rope=True, nope_layers=(2, 3),
                         swa_window=8, swa_full_every=4, kv_share_group=2, **MLA),
    # Stage E. 'uniform' is deliberately included even though it is the
    # known-collapsing control: it must still be able to LEARN (overfit one
    # batch), because collapsed streams and a broken model are different
    # failures and this gate distinguishes them.
    "e2_hc": dict(mhc_mode='hc', mhc_streams=4),
    "e3_mhc": dict(mhc_mode='mhc', mhc_streams=4),
    "e4_hc_random": dict(mhc_mode='hc', mhc_streams=4, mhc_init_scheme='random'),
    "e_uniform_control": dict(mhc_mode='hc', mhc_streams=4, mhc_init_scheme='uniform'),
    "e_mhc_full_stack": dict(mhc_mode='mhc', mhc_streams=4,
                             use_moe=True, n_shared_experts=2, n_routed_experts=6,
                             moe_top_k=2, moe_first_k_dense=1, moe_aux_loss_weight=0.0,
                             learned_pos_emb=False, use_rope=True, nope_layers=(2, 3),
                             swa_window=8, swa_full_every=4, kv_share_group=2, **MLA),
}

BATCH_SIZE = 4
MAX_STEPS = 600
TARGET_LOSS = 0.05
DEVICE = "cpu"


def run_variant(config, name):
    torch.manual_seed(1337)
    model = GPT(config).to(DEVICE)
    optimizer = model.configure_optimizers(
        weight_decay=0.0, learning_rate=1e-3, betas=(0.9, 0.95), device_type=DEVICE
    )

    # one fixed batch, reused every step
    gen = torch.Generator().manual_seed(7)
    x = torch.randint(0, config.vocab_size, (BATCH_SIZE, config.block_size), generator=gen)
    y = torch.randint(0, config.vocab_size, (BATCH_SIZE, config.block_size), generator=gen)
    x, y = x.to(DEVICE), y.to(DEVICE)

    with torch.no_grad():
        _, initial = model(x, y)
    expected = torch.log(torch.tensor(float(config.vocab_size)))

    steps = MAX_STEPS
    final_loss = float("nan")
    for step in range(MAX_STEPS):
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        if final_loss < TARGET_LOSS:
            steps = step + 1
            break

    # `final_loss` was measured BEFORE the last optimizer step, so re-measure
    # to get a value the saved weights actually correspond to
    model.eval()
    with torch.no_grad():
        _, post = model(x, y)
    post_loss = post.item()

    # checkpoint round-trip: save, reload into a fresh model, confirm identical
    ckpt_path = os.path.join(tempfile.mkdtemp(), "sanity_ckpt.pt")
    torch.save({"model": model.state_dict(), "model_args": vars(config)}, ckpt_path)
    blob = torch.load(ckpt_path, weights_only=False)
    reloaded = GPT(GPTConfig(**blob["model_args"])).to(DEVICE)
    reloaded.load_state_dict(blob["model"])
    reloaded.eval()
    with torch.no_grad():
        _, reloaded_loss = reloaded(x, y)

    checks = {
        "initial loss ~= ln(vocab_size)": abs(initial.item() - expected.item()) < 0.5,
        f"overfits one batch to < {TARGET_LOSS}": final_loss < TARGET_LOSS,
        "checkpoint round-trip preserves loss": abs(reloaded_loss.item() - post_loss) < 1e-5,
    }
    print(f"  init {initial.item():.4f} (ln(vocab)={expected:.4f}) -> "
          f"{final_loss:.5f} in {steps} steps, {model.get_num_params():,} params")
    for label, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    args = parser.parse_args()
    to_run = list(VARIANTS) if args.variant == "all" else [args.variant]

    outcomes = {}
    for variant in to_run:
        print(f"\n=== {variant} ===")
        config = GPTConfig(**{**BASE, **VARIANTS[variant]})
        outcomes[variant] = run_variant(config, variant)

    print(f"\n{'=' * 60}")
    for variant, ok in outcomes.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {variant}")
    if not all(outcomes.values()):
        raise SystemExit("\nSanity check FAILED — fix this before training.")
    print(f"\nAll {len(outcomes)} variant(s) passed — machinery is sound.")


if __name__ == "__main__":
    main()
