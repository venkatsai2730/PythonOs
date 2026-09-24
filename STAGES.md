# PythonOS-1B MoE — nano validation stages, mapped to this codebase

Stage A is the dense baseline; every later stage is compared against it on
**byte-identical data, same seed, same data order**. See NOTICE.md for
provenance and the attribution table.

> **Tokenizer changed after these figures were recorded.** Everything below
> was measured with GPT-2 BPE (vocab 50,304, padded). The corpus builders and
> `GPTConfig`'s default now use StarCoder2 BPE (vocab 49,152 — see
> `pythonos/tokenizer.py` for why) after measuring it 27% more token-efficient
> on this repo's own code-heavy corpus. Params shift slightly with the smaller
> vocab (e.g. Stage A: ~153.24M total under GPT-2 -> ~152.06M under
> StarCoder2) but the *shape* of every finding below — B1 beating dense, mHC's
> near-identity collapse, the load-balance blind spot, and so on — does not
> depend on which BPE vocabulary produced the token ids. A corpus built with
> the old tokenizer is not byte-comparable to one built with the new one:
> re-verify `manifest.json`'s `tokenizer` field, not just its sha256, before
> treating any two runs as the same frozen slice.

## Stage A — dense baseline (current)

No MoE, no mHC, no BLT. Standard pre-norm transformer, StarCoder2 BPE.

| Step | Command | Gate | Status |
|---|---|---|---|
| A0. Environment smoke test | `python data/tiny_smoke/prepare.py` then `python train.py configs/smoke_tiny.py --device=cpu --compile=False --max_iters=20 --eval_interval=10 --eval_iters=5` | runs end to end, loss decreases, checkpoint written | **re-run needed** — the original figures used a corpus replaced during the rewrite |
| A1. Structural sanity check | `python tests/sanity_overfit.py` | loss → <0.05 on a single fixed batch | **PASS** (CPU) — init loss 5.556 vs ln(256)=5.545, overfits to 0.049 in 156 steps, ckpt round-trip exact |
| A2. Corpus prep | `python data/pythonos_code/prepare.py` | `train.bin` + `val.bin` + `meta.pkl` + `manifest.json` written, token count and sha256 recorded | not started — needs HF auth for the-stack-dedup |
| A3. Baseline run | `python train.py configs/stage_a_nano.py` | sane decreasing loss curve on real data; record final val loss | not started — **needs a GPU** |

A0/A1 were run on CPU only, which is all they need. A3 is ~400M tokens through
a 153M-param model and is not CPU-feasible (see compute plan — Kaggle T4 or a
paid hourly instance).

Local CPU env: `.venv` one level up (`../.venv/Scripts/python.exe` on Windows),
torch 2.14.0+cpu, numpy 2.5.3, tiktoken, datasets 5.0.1.

Note: A0's recorded loss figures came from the tiny Shakespeare corpus used
before the rewrite. That dataset was replaced by `data/tiny_smoke`, so re-run
A0 to get comparable smoke numbers. It is a smoke test, not a stage baseline.

A1 is the important one: it proves gradients flow, the optimizer steps, the loss
masking is right, and save/resume works — *before* any GPU hours are spent. A
model that cannot memorise one batch has a bug, not a data problem.

**Stage A deliverable**: `out-stage-a/ckpt.pt` + the logged val-loss curve. This
is the reference point for B–G. Do not regenerate it later with different data.

## Stage B — hybrid attention (implemented, untrained)

Stage B bundles **four** independently novel techniques. Training them as one
change would make any loss delta unattributable — the exact failure mode this
project exists to avoid. So each is behind its own config flag with its own
config file, and `stage_b_full.py` only runs after the four have run alone.

Every flag defaults to OFF. **Verified: the Stage A forward pass is bit-identical
after the refactor** (same param count, same loss to 10 d.p., same logits SHA).

| Config | Component | Params vs Stage A | Compare against |
|---|---|---|---|
| `stage_b1_rope_nope.py` | RoPE, with NoPE on layers 3 and 7 | 152.19M (−0.68%, wpe removed) | Stage A |
| `stage_b2_swa.py` | 5:1 sliding window / full split, window 256 | 153.24M (identical) | Stage A |
| `stage_b3_mla.py` | MLA (needs RoPE, so it is MLA+RoPE) | 144.07M (−5.98%) | **B1, not Stage A** |
| `stage_b4_kvshare.py` | Cross-layer KV sharing, groups of 2 | 144.85M (−5.47%) | Stage A, with the confound noted |
| `stage_b_full.py` | all four | 140.76M (−8.14%) | B1–B4 |

**Read the param column before reading any loss curve.** B3 and B4 each remove
~6% of parameters, so a raw loss comparison against Stage A conflates the
architecture change with a capacity reduction. B2 is the only clean one.
B3 must be compared to B1 rather than Stage A, because MLA's decoupled-rotary
design presumes RoPE — comparing it to Stage A would confound MLA with the
position-encoding change.

**MLA caveat**: the ~7x KV-cache reduction (288 vs 2048 floats/token/layer) is
an *inference* saving and needs the absorption optimisation. Training memory
will not drop and may rise slightly. Measure cache size per token, not
training peak memory, or this stage will look like a failure when it isn't.

### Verification

```
python tests/test_stage_b.py          # 23 structural checks
python tests/sanity_overfit.py        # all 6 variants overfit one batch
```

`test_stage_b.py` targets the bugs that do not announce themselves:
- **causality** — a mask that leaks the future makes the loss curve look
  *better*, so nothing downstream would catch it. Checked per variant.
- **window bounds** — a token exactly one step outside the window must have
  provably zero influence (verified 0.00e+00), one just inside must have some.
- **sharing is real** — consumer layers must have no K/V projection at all,
  and the param saving must equal exactly what was removed.
- **RoPE is a rotation** — identity at position 0, norm-preserving.
- incoherent configs rejected (NoPE misaligned with KV-share group boundaries,
  out-of-range `nope_layers`, `n_layer` not divisible by `kv_share_group`).

All 23 pass; all 6 variants overfit a single batch in 159–177 steps
(Stage A: 161), so no variant has a learnability regression.

### Gotcha fixed while implementing this

`train.py` built `model_args` from a hardcoded key list, so **every Stage B flag
would have been silently dropped** — the config files would have set them,
`GPTConfig` would never have seen them, and all four sub-stages would have
trained as Stage A while appearing to work. Flags are now threaded through
`model_args` and `ARCH_KEYS` (which also gates checkpoint-resume matching).
Worth remembering for Stages C–H: adding a config flag requires touching
`train.py`, not just `model.py`.

Megatron-Core note: MLA (`MLATransformerConfig`) and NoPE (`no_rope`) are native
config there, not custom code — relevant only after nano validation.

## Stage C — fine-grained MoE (implemented, untrained)

16 experts: 2 shared (always active) + 14 routed (top-2) = 4 active/token.
Touches only `MLP` (replaced by `MoE` per layer) and adds aux losses inside
`GPT.forward`. Layer 0 stays dense (`moe_first_k_dense=1`).

Split into three configs so each aux loss is attributable:

| Config | Aux losses | Compare against |
|---|---|---|
| `stage_c1_moe_loadbal.py` | load balancing only (0.01) | Stage A |
| `stage_c2_moe_ortho.py` | + router orthogonality (0.001) | C1 |
| `stage_c3_moe_all_losses.py` | + routing variance (0.001) | C2, **as a sweep** |

### Param accounting (measured, not estimated)

| | total | active | expert hidden |
|---|---|---|---|
| Stage A dense | 153.24M | 153.24M | — |
| C1, 14 routed | **329.50M** | **153.34M** (+0.07% vs A) | 1024 |
| C1, 6 routed (T4 fallback) | 212.00M | 153.28M (+0.03%) | 1024 |

`moe_expert_hidden=0` auto-sizes each expert to `4*n_embd // active_experts`,
which makes **active** FFN params exactly equal to the dense FFN. So a val-loss
comparison against Stage A is clean — the only difference is conditional
computation, not capacity. **Total** params are 2.15x, i.e. ~3.95GB of fp32
AdamW state before activations. Viable on a 16GB T4; use `--n_routed_experts=6`
if it OOMs.

### Two findings from implementing this

**1. The load-balance loss is not a collapse detector.** Its value can read
exactly 1.0 — its perfect minimum — while load is fully collapsed. If the
router emits tied probabilities, `P` is uniform and `topk`'s tie-breaking
sends every token to the lowest-indexed experts, giving `f = [0.5, 0.5, 0, …]`
and `E * Σ f_i P_i = 6 × (0.5/6 + 0.5/6) = 1.0`. Encoded as test 6b, which
asserts the loss stays blind *and* that the counts catch it. **Consequence:
never judge routing health from the aux loss value — only from the token
counts.** This is why `moe_report()` logs counts, dead experts and
max/mean rather than just the loss.

**2. The variance and load-balancing losses are in direct opposition**, which
the architecture doc does not address. Load balancing wants uniform traffic;
the variance (confidence) loss wants each token to commit hard to one expert.
Both can hold simultaneously — different tokens committing to different
experts — but the weight ratio decides which wins, and a bad ratio either
collapses routing or flattens it. Hence C3 is specified as a *sweep* over
`moe_var_loss_weight` (1e-4, 1e-3, 1e-2) plotted against `max_over_mean`, not
a single run. If confident routing and balanced load prove unachievable
together at this scale, that is a finding about the spec, not a failed run.

Also note: the orthogonality and variance losses are **not** standard named
losses with canonical formulas. The implemented forms (mean squared
off-diagonal of the normalised router Gram matrix; negated mean per-token
routing variance) are a reasonable reading of the doc's intent, not a
reproduction of a published result. Treat their exact shape as under test.

### Instrumentation

`train.py` prints every `moe_log_interval` steps (default 250):

```
[moe] iter 250: dead=0 worst max/mean=1.66 lb=1.0108
  L1  dead=0  max/mean=1.43 counts=[52, 36, 61, 24, 37, 46]
  L2  dead=0  max/mean=1.66 counts=[71, 28, 44, 43, 35, 35]
```

`max_over_mean` is 1.0 at uniform load and rises toward `n_routed_experts` as
traffic concentrates. Collapse is usually **not** uniform across depth, so the
per-layer counts matter more than the aggregate. Same fields go to wandb.

### Aux-loss isolation

Aux losses are added **only when `model.training`**. Including them at eval
would make val loss stop being pure cross-entropy and silently break every
cross-stage comparison (and every comparison between aux-weight settings).
`model.last_lm_loss` always holds the clean value. Test 7 asserts this with a
deliberately absurd aux weight of 5.0.

### What this stage cannot answer

Whether experts map to the four domains (DSA / ML / Agentic / Theory). Routing
is learned; labelling expert 1 "DSA" does nothing. On a pure-Python corpus with
no domain labels, C1–C3 can only show whether experts differentiate *at all*.
Testing the doc's Section 9 claim needs a labelled multi-domain corpus, which
is a full-pretraining concern.

### Verification

```
python tests/test_stage_c.py          # 29 checks
python tests/sanity_overfit.py        # 8 variants incl. c_moe, c_moe_plus_b
```

`test_stage_c.py` covers: layer structure; active-vs-total param accounting
(asserted against the dense model exactly); **every routed expert receives
nonzero gradient** (the dispatch loop is where experts get silently dropped —
that failure trains a much smaller model than you think while looking fine);
routing slot bookkeeping (`Σ counts == tokens × top_k`); gate renormalisation;
load-balance calibration and its tie-break blind spot; aux-loss isolation from
eval; that the collapse detector actually fires on forced collapse; and that
MoE composes with full Stage B including **causality still holding**.

All 29 pass. Stage A remains bit-identical (same logits SHA) and Stage B's 23
checks still pass.

## Stage E — mHC / Hyper-Connections (implemented, untrained)

Replaces the single residual stream with `n` streams. Per residual connection:

    x_in  = alpha . H              aggregate   (alpha: n)
    y     = sublayer(norm(x_in))
    H_out = A @ H + beta (x) y     combine     (A: n x n, beta: n)

Three modes, run side by side as the plan requires:

| Config | Mode | Compare against |
|---|---|---|
| `stage_e1_single.py` | single stream — the **control** | — |
| `stage_e2_hc.py` | unconstrained `A` | E1 |
| `stage_e3_mhc.py` | `A` projected onto the doubly-stochastic manifold (Sinkhorn-Knopp) | E2, **as a sweep over `mhc_init_logit`** |
| `stage_e4_random_init.py` | HC with symmetry-breaking init | E2 |

This is a genuine structural change: `Block.forward_streams` consumes and
returns `(B, T, n, d)` instead of `(B, T, d)`, and `GPT.forward` has a separate
stream loop. `mhc_mode='single'` keeps the original one-tensor path, so the
Stage A baseline is bit-for-bit untouched (asserted).

Param overhead is negligible — 388 params at 8 layers, n=4. This stage is not
a capacity comparison.

### Verified: HC is bit-identical to the baseline at init

With `alpha = beta = readout = e_0` and `A = I`, stream 0 carries exactly the
standard residual computation and the rest sit inert. Measured
`max|dlogit| = 0.000e+00` against the dense baseline for n = 2, 4 and 8. This
matters because it means any HC-vs-baseline difference later is **learned**,
not an artifact of reparameterisation.

mHC cannot be exactly identity-initialised — identity is only a limit point of
the Sinkhorn projection — so it starts *near* the baseline, controllably:
`init_logit` 5 / 10 / 20 gives `max|dlogit|` of 3.4e-2 / 2.4e-4 / 2.4e-7.

### Finding 1: mHC's mixing matrix barely learns

Measured over 300 steps at lr 1e-3, nano scale:

| `mhc_init_logit` | `\|\|A−I\|\|` start → end | final `diag_mean` |
|---|---|---|
| 1 | 1.21 → 1.17 | 0.475 |
| 3 | 0.300 → 0.275 | 0.873 |
| 5 | 0.0457 → 0.0438 | 0.980 |
| 10 (default) | 3.14e-4 → 3.38e-4 | 0.9999 |

`A` barely moves from wherever the init puts it — and in every case moves
slightly *toward* identity, not away. The Sinkhorn parameterisation is stiff:
**the mixing regime is effectively set by the init hyperparameter rather than
learned.** At the default `init_logit=10`, mHC is numerically indistinguishable
from independent per-stream residuals — the exact near-identity collapse the
design review flagged, reproduced on our own setup.

So `mhc_init_logit` is not an init detail, it is the most important knob in
this stage, and E3 must be run as a sweep (1, 3, 5, 10). Unconstrained HC by
contrast does move: `||A−I||` grows 0 → 7.8e-2.

**Caveat: 300 steps at nano scale on a memorisation task is short.** Confirm at
the full 6100-step length before concluding the matrix genuinely cannot learn.

### Finding 2: the initialisation trilemma

You cannot have all three of exact baseline equivalence at init, equal gradient
across streams, and `n` distinguishable streams.

| scheme | equivalence | gradient | distinct roles |
|---|---|---|---|
| `e0` | exact | [0.78, 0.093, 0.093, 0.093] | **2, not n** |
| `uniform` | exact | equal | **1** — provably collapses |
| `random` | approximate | unequal | n |

- **`e0`**: streams 1..n-1 stay mutually interchangeable, so 4 streams give
  only *two* distinct roles. Their measured gradient norms are identical to 5
  significant figures. Non-primary streams also get ~5000x less signal at init.
- **`uniform`**: perfectly symmetric, so all streams receive identical
  gradients and stay identical forever. Measured cosine similarity after 300
  steps: 0.976 (hc) / **0.999** (mhc). Kept as a control that *demonstrates*
  the collapse rather than risking it.
- **`random`** (E4): gives up exact init equivalence to break all symmetry.

This is structural, not a bug: inert identical streams are interchangeable by
construction, so any exactly-equivalent init inherits the cap. **If the doc's
"4 streams = 4 reasoning signals" reading is to be reachable at all, it needs
an asymmetric init** — that is what E4 tests.

### Finding 3: Sinkhorn does not converge at a fixed iteration count

Worst-case row-sum residual over 900 random skewed logit matrices:

| iters | 8 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| residual | 1.1e-1 | 2.7e-2 | 8.9e-3 | 4.1e-3 | 1.4e-3 | 1.9e-4 |

Convergence is linear, so brute force is no fix — 100 iterations still leaves
4e-3 and costs ~3.3ms per call, which at `2 x n_layer` calls per forward is
prohibitive. Columns are normalised last so they are always exact; rows are
the ones that lag.

**Near-identity matrices converge immediately** (6e-8 at 8 iterations), which
is the regime training starts in, so the default `mhc_sinkhorn_iters=20` is
cheap and exact there. The risk is `A` drifting skewed mid-training, at which
point **A is silently not doubly stochastic and mHC is not manifold-constrained
at all**. Hence `row_sum_dev` is logged every `mhc_log_interval` and
`train.py` prints a warning above 1e-3. Encoded as test 4b so the limitation
cannot quietly regress.

Also note the projection is done in log space deliberately: the naive
exp/divide form underflows to NaN at the strongly-diagonal init (exp(-20)).

### Instrumentation

Per the plan — per-stream gradient norm, inter-stream cosine similarity, and
mixing-matrix deviation from identity — logged every `mhc_log_interval`:

```
[mhc] iter 250: mode=mhc ||A-I|| mean=3.146e-04 max=3.146e-04       cos mean=0.9452 max=0.9750 grad=[3.65e-06,4.96e-01] rowdev=6.0e-08
  readout=[1.0, -0.0, -0.0, -0.0]
  L0  ||A-I||=3.146e-04 diag=0.9999 cos=0.9745 grad=[0.50661, 0.00012, 0.00012, 0.00012]
  L3  ||A-I||=3.146e-04 diag=0.9999 cos=0.9180 grad=[0.42546, 0.0, 0.0, 0.0]
```

Gradient norms require a backward pass, so they are captured by a hook on the
stream tensor rather than read during forward. Requires `mhc_instrument=True`;
mixing diagnostics are always available since they read parameters, not
activations. Note L3's non-primary streams show gradient exactly 0 — the
`e_0` readout means final-layer streams 1..3 feed nothing.

### Verification

```
python tests/test_stage_e.py    # 48 checks
```

Covers: bit-exact baseline equivalence for HC at n = 2/4/8; controlled mHC init
deviation; Sinkhorn double-stochasticity, its non-convergence limitation, log-space
stability and identity as a fixed point; forward/backward across all three
modes; that `A`, alpha, beta and readout all receive gradient (else the mode is
decorative); **causality surviving the stream plumbing** in all modes;
instrumentation reporting real numbers and *not* fabricating activation stats
when disabled; every collapse detector firing on forced collapse; and
composition with full Stage B + Stage C with causality intact.

## Stages D, F, G, H

- **D — FuseNorm-DyT**: swap `LayerNorm` in `pythonos/norm.py`, select via config.
- **F — BLT**: separate data pipeline (raw bytes -> entropy patcher -> local
  encoder -> global transformer -> local decoder). Does not fit this structure;
  expect separate modules, not a flag.
- **G — Muon+ vs AdamW**: touches `GPT.configure_optimizers`.
- **H — GRPO/S-GRPO**: problem-count driven, not token-count driven.

## Full-scale param math — the doc's numbers do not reconcile

The design review flagged that the doc's ~8-10B total may undercount FFN
expert params. Checked at the full spec (d_model 2048, 32 layers, 16 experts =
2 shared + 14 routed, top-2, 31 MoE layers, vocab 50304). Non-FFN base
(attention + embeddings) is ~0.64B.

| expert hidden | total | active |
|---|---|---|
| h = d (2048) | 4.83B | 1.71B |
| h = 2d | 8.99B | 2.75B |
| h = 4d (dense-equivalent) | 17.32B | 4.83B |

**The doc's claimed pair — ~9-10B total with ~2B active — is not achievable at
any expansion ratio.** Solving each constraint separately:

- pin **active = 2B** → h ≈ 2678 → total lands at **~6.1B**, not 9-10B
- pin **total = 9.5B** → h ≈ 4361 → active lands at **~2.86B**, not 2B

The two figures are inconsistent by 40-50%. Total and active are coupled
through the same `h`: with 16 experts and 4 active, `total_expert_params =
4 x active_expert_params` exactly, so the ratio is fixed by the architecture,
not tunable. Whichever number the training budget is actually built on needs
to be the one that gets fixed, and the other corrected to match.

Reproduce with `model.param_report()` at nano scale, or the arithmetic above
at full scale.

## Invariants across all stages

- Same corpus slice, same seed (`1337`), same data order. Changing the corpus
  between stages invalidates every comparison.
- One component per stage. Never combine untested.
- Record: final val loss, loss curve, peak memory, tokens/sec, wall clock.
