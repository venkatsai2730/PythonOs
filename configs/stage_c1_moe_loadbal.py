# Stage C1 — MoE with load-balancing loss only.
#
# 16 experts: 2 shared (always active) + 14 routed (top-2) = 4 active/token.
# Orthogonality and variance losses are OFF here. Load balancing is the one
# well-understood aux loss; validate the routing machinery with it alone
# before adding losses whose formulation is itself under test.
#
# PARAM ACCOUNTING (d_model=1024, 8 layers, layer 0 dense):
#   expert hidden = 4*1024 // 4 active = 1024
#   total  ~328M   <- what sits in memory and optimiser state
#   active ~152M   <- what each token flows through (Stage A: 153.2M)
# Active params are matched to Stage A by construction, so a val-loss
# comparison against Stage A is meaningful. Total params are 2.1x, which is
# the honest cost: ~5.2GB of AdamW state in fp32. Tight but viable on a 16GB
# T4; drop n_routed_experts to 6 if it OOMs.
#
# WHAT THIS STAGE MUST ANSWER (from the design review):
#   1. Does routing collapse to 2-3 dominant experts? Watch dead / max_over_mean.
#   2. Does expert<->domain correspondence emerge at all? Routing is LEARNED;
#      labelling expert 1 "DSA" does nothing on its own. This run cannot fully
#      answer that on a pure-Python corpus with no domain labels - it can only
#      show whether experts differentiate at all.
#
#   $ python train.py config/stage_c1_moe_loadbal.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-c1'
wandb_run_name = 'stage-c1-moe-loadbal'

use_moe = True
n_shared_experts = 2
n_routed_experts = 14
moe_top_k = 2
moe_expert_hidden = 0    # auto -> 1024, matches dense active FFN params
moe_first_k_dense = 1    # layer 0 stays dense

moe_aux_loss_weight = 0.01
moe_ortho_loss_weight = 0.0
moe_var_loss_weight = 0.0

moe_log_interval = 250

# MoE roughly doubles resident params; halve the micro-batch and double
# accumulation to keep tokens/iter identical to Stage A (65,536).
batch_size = 4
gradient_accumulation_steps = 16
