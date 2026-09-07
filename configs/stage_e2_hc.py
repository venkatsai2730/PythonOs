# Stage E2 — plain (unconstrained) Hyper-Connections, 4 streams.
#
# A is a free 4x4 matrix. At init this is BIT-IDENTICAL to Stage A / E1
# (alpha = beta = readout = e_0, A = I), so any divergence is learned rather
# than an artifact of reparameterisation.
#
# Param overhead: 2 connections/block x (4 + 4 + 16) + 4 readout = 388 params
# for 8 layers. Negligible — this stage is not a capacity comparison.
#
#   $ python train.py configs/stage_e2_hc.py

include('configs/stage_a_nano.py')

out_dir = 'out-stage-e2'
wandb_run_name = 'stage-e2-hyperconnections'

mhc_mode = 'hc'
mhc_streams = 4
mhc_init_scheme = 'e0'
mhc_instrument = True
mhc_log_interval = 250
