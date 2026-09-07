# pythonos-nano

Nano-scale validation of the **PythonOS-1B MoE** architecture (locked v1.1 spec)
before committing to full-scale pretraining.

The full spec stacks ~13 independently novel techniques (BLT, PLE, mHC,
FuseNorm-DyT, MLA, hybrid SWA, RoPE+NoPE, cross-layer KV sharing, fine-grained
MoE, Muon+/AdamW hybrid, FP8, GRPO/S-GRPO, PRM). Combining them untested makes
any training failure unattributable to a specific cause. So this repo builds
and validates **one component at a time** at ~150M params on a single GPU,
against a dense baseline that never moves.

Independent implementation, MIT licensed. The techniques are published
research (Vaswani, DeepSeek, Su, Fedus, Loshchilov et al.) — see `NOTICE.md`
for the full attribution table and the repository's provenance.

## The core discipline

1. **One component per stage.** Never combine untested.
2. **The baseline never moves.** `tests/test_stage_a_invariance.py` pins the
   Stage A forward pass to an exact parameter count, loss, and logits hash. If
   a later stage's code perturbs it, every comparison silently shifts.
3. **Identical data, identical order, every stage.** The corpus is frozen and
   hash-verified at train time (`pythonos/data.py`); a mismatch refuses to train.
4. **Read the param column before the loss column.** Several stages change
   parameter count as a side effect. A loss delta against Stage A that ignores
   this is measuring capacity, not architecture.

## Quickstart

```bash
pip install -e .                 # or: pip install torch numpy
pip install -e '.[data]'         # only needed to build the corpus

python tests/run_all.py          # every gate, CPU, ~2 min
python scripts/param_report.py   # total vs active params, all stages
```

Then, in order:

```bash
# A0. environment smoke test — unmodified-style run on a tiny dataset
python data/tiny_smoke/prepare.py
python train.py configs/smoke_tiny.py --device=cpu --compile=False \
    --max_iters=20 --eval_interval=10 --eval_iters=5

# A2. freeze the corpus (needs HF auth for the-stack-dedup;
#     use --dataset=codeparrot/codeparrot-clean for an ungated source)
python data/pythonos_code/prepare.py --target 400e6

# A3. the baseline run — NEEDS A GPU
python train.py configs/stage_a_nano.py
```

On a 16GB T4 add `--dtype=float16 --compile=False`.

## Layout

```
pythonos/            the model, as importable modules
  config.py          GPTConfig — every stage flag, all defaulting to Stage A
  norm.py            LayerNorm            <- Stage D swaps in here
  attention.py       dense MHA, MLA, RoPE, sliding-window masks
  ffn.py             dense MLP, fine-grained MoE
  model.py           Block, GPT           <- Stage E restructures these
  data.py            batch loader + corpus freeze verification
  settings.py        RunSettings dataclass, config-file and CLI loading
  hyper.py           hyper-connections + Sinkhorn (Stage E)
configs/             one file per stage/sub-stage
tests/               run_all.py plus per-stage gates
pythonos/settings.py RunSettings + config-file/CLI loading
scripts/             param_report.py
data/<dataset>/      prepare.py per dataset, writes train.bin/val.bin/manifest.json
train.py             training loop (single GPU or DDP)
STAGES.md            the detailed stage record — findings, gates, caveats
```

**`STAGES.md` is the substantive document.** It records what each stage must
prove, what has been verified, the param accounting, and the findings so far.
Read it before starting a stage.

## Status

| Stage | What | State |
|---|---|---|
| A | dense baseline | code verified; **untrained** (needs corpus + GPU) |
| B | MLA, 5:1 SWA, RoPE+NoPE, cross-layer KV sharing | implemented, 23 checks pass, untrained |
| C | fine-grained MoE + aux losses | implemented, 29 checks pass, untrained |
| D | FuseNorm-DyT | not started (seam: `pythonos/norm.py`) |
| E | mHC / Hyper-Connections | implemented, 48 checks pass, untrained |
| F | BLT | not started — separate data pipeline and topology, not a flag |
| G | Muon+ vs AdamW | not started (seam: `GPT.configure_optimizers`) |
| H | GRPO/S-GRPO | not started |

**Nothing has been trained yet.** There are no weights in this repo and no
base model — it is a verified architecture implementation plus a test harness.

## Findings so far

Details, measured tables and reproduction steps in `STAGES.md`.

1. **mHC's mixing matrix barely learns.** `||A-I||` stays within ~10% of
   wherever `mhc_init_logit` puts it over 300 steps, at every init tried, and
   drifts *toward* identity rather than away. At the default it is
   numerically indistinguishable from independent per-stream residuals — the
   near-identity collapse the design review flagged, on our own setup. The
   Sinkhorn parameterisation is stiff, so the mixing regime is set by an init
   hyperparameter rather than learned. Needs confirming at full run length.
2. **Hyper-Connections face an initialisation trilemma.** Exact baseline
   equivalence at init, equal gradient across streams, and n distinguishable
   streams cannot all hold. The natural `e_0` init gives only **two** distinct
   stream roles regardless of n; the symmetric alternative provably collapses
   (cosine similarity 0.999 after 300 steps). Reaching "4 streams = 4 signals"
   requires giving up exact init equivalence.
3. **Sinkhorn-Knopp does not converge at a fixed iteration count.** Worst-case
   row residual is 1.1e-1 at 8 iterations and still 1.9e-4 at 400;
   convergence is linear so brute force is not a fix. Near-identity converges
   instantly, which is where training starts — but if `A` drifts skewed, mHC
   is silently no longer manifold-constrained. `row_sum_dev` is logged and
   warned on.
4. **The MoE load-balance loss is not a collapse detector.** It can read
   exactly 1.0 — its perfect minimum — while load is fully collapsed, because
   tied router probabilities make `topk`'s tie-breaking send every token to the
   lowest-indexed experts. Judge routing health from token counts only.
5. **The doc's full-scale param figures do not reconcile.** ~9-10B total with
   ~2B active is unachievable at any expansion ratio: pinning active at 2B
   gives ~6.1B total, pinning total at 9.5B gives ~2.86B active. With 16
   experts and 4 active, total expert params are exactly 4x active, so the
   ratio is fixed by the architecture. Run `scripts/param_report.py`.

## Known limitations

- **No *efficient* inference path.** `generate()` exists and is verified working
  across Stage A, full Stage B, Stage C and Stage E — but it has no incremental
  KV cache: it re-runs the whole (cropped) prefix for every token, so cost is
  quadratic in tokens generated.
  So the cache savings from MLA and cross-layer KV sharing are architecturally
  present but not *realised* anywhere. Demonstrating them is a separate build.
  (`sample.py` was deleted as a CLI, not the capability.)
- **MLA's KV-cache win is inference-only** and needs the absorption
  optimisation. Training memory will not drop and may rise slightly.
- Stage F (BLT) will not fit this structure; expect separate modules.
- **`torch.compile` is untested against Stages C and E.** The MoE expert loop,
  the Sinkhorn iteration and the backward hook that captures per-stream
  gradient norms are all things Dynamo may graph-break on or drop; it could
  not be verified on the dev machine (no MSVC toolchain). `train.py` defaults
  to `compile=True`. **Run the first Stage C/E job with `--compile=False`**,
  then re-enable deliberately and confirm the `[moe]`/`[mhc]` logs still
  populate.
- **Stage E instrumentation is expensive and sampled, not continuous.** ~404ms
  per forward at nano shapes, so it runs only on `mhc_log_interval` steps. The
  reported stats are therefore a snapshot of one micro-batch, not a running
  average.
