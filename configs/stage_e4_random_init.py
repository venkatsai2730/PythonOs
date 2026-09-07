# Stage E4 — HC with symmetry-breaking random init.
#
# Addresses a structural limit found in E2: with the 'e0' init, streams
# 1..n-1 are mutually interchangeable, so 4 streams give only TWO distinct
# roles. Measured per-stream gradient norms after 300 steps were
# [0.78, 0.093, 0.093, 0.093] — the last three identical to 5 s.f.
#
# This config perturbs alpha/beta so all four streams are distinguishable from
# step 0. The cost is that it is no longer exactly equal to the baseline at
# init, so compare it to E2 (same mode, different init) and not to E1.
#
# If E4 differentiates streams (cos_sim well below E2's) while E2 does not,
# the conclusion is that the doc's "4 streams = 4 reasoning signals" story
# needs an asymmetric init to be reachable at all.
#
#   $ python train.py configs/stage_e4_random_init.py

include('configs/stage_e2_hc.py')

out_dir = 'out-stage-e4'
wandb_run_name = 'stage-e4-hc-random-init'

mhc_init_scheme = 'random'
