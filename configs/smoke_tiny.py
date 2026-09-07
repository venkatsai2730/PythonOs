# Environment smoke test — a miniature model on the tiny generated corpus.
#
# Purpose is to prove the loop runs end to end (CUDA, checkpointing, resume),
# not to train anything useful. This dataset is NOT frozen and must never be
# used for a stage comparison.
#
#   python data/tiny_smoke/prepare.py
#   python train.py configs/smoke_tiny.py --device=cpu --compile=False

out_dir = 'out-smoke'
wandb_project = 'pythonos-nano-smoke'
wandb_run_name = 'smoke-tiny'

dataset = 'tiny_smoke'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False   # tiny corpus: only keep improvements

gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2                    # small corpus, so some regularisation

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99                     # few tokens per step, so a longer average
warmup_iters = 100
