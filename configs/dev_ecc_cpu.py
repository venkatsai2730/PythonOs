# Pipeline validation run — real corpus, CPU, small model.
#
# Purpose: prove the whole path works on real text (corpus -> loader -> model
# -> loop -> checkpoint -> eval) and produce a real loss curve. This is NOT
# Stage A and its numbers are not a baseline:
#   - the corpus is one project's repo (~10.5M tokens, 57% markdown)
#   - the model is 16M params, not 153M
#   - it sees ~0.8M tokens, not 400M
#
# Sized from a measured CPU throughput of ~1660 tok/s at this shape, so ~400
# steps is roughly 8 minutes.
#
#   python data/ecc_repo/prepare.py --source /c/Temp/ecc
#   python train.py configs/dev_ecc_cpu.py
#   python scripts/evaluate.py --ckpt out-dev-ecc/ckpt.pt --sample

out_dir = 'out-dev-ecc'
wandb_project = 'pythonos-nano-dev'
wandb_run_name = 'dev-ecc-cpu'

dataset = 'ecc_repo'

# ~16M params. Dominated by the tied embedding (49,152 rows -- StarCoder2 BPE,
# see pythonos/tokenizer.py), which is also why CPU throughput is what it is:
# the output matmul is the bulk of the work.
n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
dropout = 0.0
bias = False

batch_size = 8
gradient_accumulation_steps = 1     # 2,048 tokens per step

max_iters = 400
lr_decay_iters = 400
warmup_iters = 40
learning_rate = 1e-3                # small model, so a higher rate is fine
min_lr = 1e-4
beta2 = 0.95
grad_clip = 1.0
weight_decay = 0.1

eval_interval = 50
eval_iters = 20
log_interval = 25
always_save_checkpoint = True

device = 'cpu'
dtype = 'float32'
compile = False                     # no MSVC toolchain here; also see README
