# Stage C3 — C2 plus the routing-variance (confidence) loss.
#
# READ THIS BEFORE RUNNING: the variance loss and the load-balancing loss are
# in direct tension. Load balancing wants uniform traffic across experts;
# variance wants each token to commit hard to one expert. Both can hold at
# once (different tokens committing to different experts), but the weight
# ratio decides which wins, and a bad ratio either collapses routing or
# flattens it into 14-way mush.
#
# So the useful output of this stage is not one loss curve — it is a small
# sweep. Run at least:
#   --moe_var_loss_weight=0.0001
#   --moe_var_loss_weight=0.001
#   --moe_var_loss_weight=0.01
# and plot max_over_mean against val loss. If confident routing and balanced
# load turn out to be unachievable together at this scale, that is a real
# finding about the spec, not a failed run.
#
#   $ python train.py config/stage_c3_moe_all_losses.py

include('configs/stage_c2_moe_ortho.py')

out_dir = 'out-stage-c3'
wandb_run_name = 'stage-c3-moe-all-losses'

moe_var_loss_weight = 0.001
moe_log_interval = 100   # tighter logging: this is the unstable configuration
