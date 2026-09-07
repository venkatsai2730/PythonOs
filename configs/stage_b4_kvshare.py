# Stage B4 — cross-layer KV sharing, in isolation.
#
# Pairs of layers (0,1), (2,3), (4,5), (6,7) compute K/V once in the first
# layer of the pair and reuse it in the second. Halves the KV cache.
#
# This REMOVES PARAMETERS: each consuming layer drops its k and v projections
# (2 * 1024^2 = 2.1M each), so 4 consumers = 8.4M fewer params, ~5.5% below
# Stage A. That is a large enough gap that a raw loss comparison against
# Stage A is not apples-to-apples. To read this stage honestly, either accept
# the confound explicitly or run a param-matched control.
#
#   $ python train.py config/stage_b4_kvshare.py

include('configs/stage_a_nano.py')  # inherit the Stage A baseline

out_dir = 'out-stage-b4'
wandb_run_name = 'stage-b4-kvshare'

kv_share_group = 2   # 1 = off; 2 = each adjacent pair shares K/V
