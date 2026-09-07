# Stage E1 — single stream. The CONTROL.
#
# Identical architecture to Stage A; exists so the three Stage E variants are
# run through the same harness with the same instrumentation switched on, and
# compared to each other rather than to a differently-configured run.
#
#   $ python train.py configs/stage_e1_single.py

include('configs/stage_a_nano.py')

out_dir = 'out-stage-e1'
wandb_run_name = 'stage-e1-single-stream'

mhc_mode = 'single'
mhc_instrument = True   # no-op for single, kept for harness symmetry
