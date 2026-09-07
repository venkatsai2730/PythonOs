# Stage E3 — manifold-constrained Hyper-Connections, 4 streams.
#
# A is projected onto the doubly-stochastic manifold (Birkhoff polytope) by
# Sinkhorn-Knopp each forward pass.
#
# READ THIS FIRST — measured at nano scale over 300 steps at lr 1e-3:
#   ||A - I|| barely moves from wherever mhc_init_logit puts it.
#     init_logit=1  -> 1.21 -> 1.17
#     init_logit=3  -> 0.300 -> 0.275
#     init_logit=5  -> 0.0457 -> 0.0438
#     init_logit=10 -> 3.1e-4 -> 3.4e-4
#   The Sinkhorn parameterisation is stiff: the mixing regime is effectively
#   SET BY THE INIT rather than learned. So mhc_init_logit is not an init
#   detail, it is the most important knob in this stage, and E3 should be run
#   as a sweep over it (1, 3, 5, 10) rather than once at the default.
#
#   Caveat: 300 steps is short. Confirm at full 6100-step length before
#   concluding the matrix genuinely cannot learn.
#
# Also watch row_sum_dev in the [mhc] log. Sinkhorn converges only linearly,
# so if A drifts to a skewed regime the projection stops holding and mHC is
# no longer manifold-constrained. Raise mhc_sinkhorn_iters if it climbs.
#
#   $ python train.py configs/stage_e3_mhc.py
#   $ python train.py configs/stage_e3_mhc.py --mhc_init_logit=3.0 --out_dir=out-stage-e3-l3

include('configs/stage_a_nano.py')

out_dir = 'out-stage-e3'
wandb_run_name = 'stage-e3-mhc'

mhc_mode = 'mhc'
mhc_streams = 4
mhc_init_scheme = 'e0'
mhc_init_logit = 10.0
mhc_sinkhorn_iters = 20
mhc_instrument = True
mhc_log_interval = 250
