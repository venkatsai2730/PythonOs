# Stage C2 — C1 plus the router orthogonality loss.
#
# Adds one term to C1 so its effect is attributable. See the MoE docstring in
# model.py: this loss is the mean squared off-diagonal of the Gram matrix of
# L2-normalised router rows. That is an interpretation of the doc's intent,
# not a published formula — the weight below is a starting guess, and this
# stage is partly a weight-sensitivity probe.
#
# Compare against C1, not Stage A.
#
#   $ python train.py config/stage_c2_moe_ortho.py

include('configs/stage_c1_moe_loadbal.py')

out_dir = 'out-stage-c2'
wandb_run_name = 'stage-c2-moe-ortho'

moe_ortho_loss_weight = 0.001
