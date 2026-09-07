# Stage B (full) — all four hybrid-attention components together.
#
# Run this ONLY after B1-B4 have each been validated alone. Its purpose is to
# check the components compose without interacting badly, not to attribute
# anything: if this run misbehaves, the sub-stage runs are what tell you why.
#
# nope_layers must cover whole KV-sharing groups (build_layer_plan asserts
# this). With kv_share_group=2 the groups are (0,1)(2,3)(4,5)(6,7), so (6,7)
# is legal and (3,7) is not.
#
#   $ python train.py config/stage_b_full.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-b-full'
wandb_run_name = 'stage-b-full-hybrid-attn'

# B1: RoPE + NoPE
learned_pos_emb = False
use_rope = True
nope_layers = (6, 7)

# B2: 5:1 sliding window
swa_window = 256
swa_full_every = 6

# B3: MLA
use_mla = True
kv_lora_rank = 256
q_lora_rank = 512
qk_nope_head_dim = 64
qk_rope_head_dim = 32
v_head_dim = 64

# B4: cross-layer KV sharing
kv_share_group = 2
