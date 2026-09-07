# Stage B1 — RoPE + NoPE, in isolation.
#
# Replaces Stage A's learned absolute position embedding (wpe) with rotary
# embeddings, and gives layers 3 and 7 no positional signal at all.
#
# Removing wpe drops 1.05M params vs Stage A (1024 x 1024). That is a 0.7%
# param difference, small but not nothing — note it when reading the loss delta.
#
#   $ python train.py config/stage_b1_rope_nope.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-b1'
wandb_run_name = 'stage-b1-rope-nope'

learned_pos_emb = False   # drop wpe
use_rope = True
rope_theta = 10000.0
nope_layers = (3, 7)      # 2 of 8 layers carry no positional information
