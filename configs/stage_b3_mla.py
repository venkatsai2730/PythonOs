# Stage B3 — MLA, in isolation.
#
# Replaces MHA with Multihead Latent Attention. RoPE is enabled here because
# MLA's decoupled-rotary design assumes it; running MLA with no positional
# signal at all would not test the intended architecture. That means B3 is
# strictly "MLA + RoPE" — compare it against B1 (RoPE alone), NOT against
# Stage A, or the MLA contribution is confounded with the position change.
#
# Head dims for d_model=1024, n_head=16:
#   qk head dim = 64 nope + 32 rope = 96   (vs 64 for MHA)
#   v head dim  = 64
#   KV cache per token = kv_lora_rank + qk_rope_head_dim = 256 + 32 = 288
#   vs MHA: 2 * n_embd = 2048  ->  ~7.1x smaller cache
#
# Reminder: that 7.1x is an INFERENCE-time saving and requires the absorption
# optimisation. Training memory here will not drop and may rise slightly.
#
#   $ python train.py config/stage_b3_mla.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-b3'
wandb_run_name = 'stage-b3-mla'

learned_pos_emb = False
use_rope = True

use_mla = True
kv_lora_rank = 256
q_lora_rank = 512
qk_nope_head_dim = 64
qk_rope_head_dim = 32
v_head_dim = 64
