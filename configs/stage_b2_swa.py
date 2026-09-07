# Stage B2 — sliding-window attention, in isolation.
#
# 5:1 split: layers 0-4 and 6-7 attend to a 256-token window, layer 5 keeps
# full 1024-token attention. Param count is IDENTICAL to Stage A — this changes
# only the mask, so any loss delta is purely the cost of restricted context.
#
# Expect a small loss penalty and a throughput/memory win. On a 1024-token
# context the win is modest; SWA pays off at long context. A negative result
# here is weak evidence, same caveat as BLT at nano scale.
#
#   $ python train.py config/stage_b2_swa.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-b2'
wandb_run_name = 'stage-b2-swa'

swa_window = 256
swa_full_every = 6   # every 6th layer is full attention -> 5:1
