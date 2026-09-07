# Stage A — dense baseline for PythonOS-1B MoE nano validation.
#
# Standard pre-norm transformer, GPT-2 BPE, no MoE / mHC / BLT / MLA.
# This run produces the reference loss curve that Stages B-G are compared to.
#
#   $ python train.py config/stage_a_nano.py
#
# On a 16GB T4, add: --dtype=float16 --compile=False
# (bfloat16 needs Ampere+; torch.compile on T4 is often slower than it saves)

out_dir = 'out-stage-a'

# eval often enough to get a usable curve, not so often it dominates runtime
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True  # we want the final checkpoint, not just the best

wandb_log = False  # flip on via --wandb_log=True once you have a project
wandb_project = 'pythonos-nano'
wandb_run_name = 'stage-a-dense-baseline'

# --- data -------------------------------------------------------------------
# FROZEN slice. Stages B-G must reuse this byte-for-byte. Do not regenerate.
dataset = 'pythonos_code'

block_size = 1024
batch_size = 8                    # micro-batch; lower this first if you OOM
gradient_accumulation_steps = 8   # raise this to keep tokens/iter constant
# tokens per iter = 8 * 8 * 1024 = 65,536

# --- model ------------------------------------------------------------------
# Verified: 153.2M total params, of which ~100.7M is transformer body
# (8 x 12.6M) and ~51.5M is the tied token embedding / lm_head. Note that
# get_num_params() reports 152.2M "non-embedding" because it only subtracts
# wpe — the tied wte counts, since it doubles as the output head.
#
# 8 layers keeps us in the "6-8 layers" nano envelope, marginally over the
# 150M param target. d_model 1024 is half the full spec's 2048, so per-layer
# shapes scale cleanly when we move up. Drop to n_layer=7 for ~140M if the
# param budget needs to be strict.
n_layer = 8
n_head = 16
n_embd = 1024
dropout = 0.0   # pretraining regime: no dropout
bias = False    # slightly better and faster than GPT-2's bias=True

# --- optimizer --------------------------------------------------------------
# AdamW only. Muon+ is Stage G — introducing it here would confound A vs G.
learning_rate = 6e-4
min_lr = 6e-5
max_iters = 6100          # 6100 * 65,536 = ~400M tokens
lr_decay_iters = 6100     # == max_iters, per Chinchilla
warmup_iters = 200
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# --- reproducibility --------------------------------------------------------
# train.py seeds torch with 1337. Same seed + same frozen .bin files =
# same data order across every stage. Do not change either.
