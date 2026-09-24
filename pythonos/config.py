"""Model configuration and the per-layer architecture plan.

Every stage flag defaults to Stage A behaviour, so the dense baseline stays
reproducible no matter what later stages add here."""

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 49152 # StarCoder2 BPE (see pythonos/tokenizer.py); already a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True   # bias terms in Linear and norm layers
    ffn_mult: int = 4   # feed-forward hidden width as a multiple of n_embd
    norm_type: str = 'layernorm'   # key into pythonos.norm.NORM_REGISTRY

    # --- Stage B: hybrid attention. Every flag below defaults to OFF, so the
    # --- Stage A dense baseline is bit-for-bit unchanged.

    # B1: positional encoding. learned_pos_emb is Stage A's wpe. Setting
    # use_rope replaces it with rotary; nope_layers then names the layers that
    # get NO positional signal at all (RoPE+NoPE hybrid).
    learned_pos_emb: bool = True
    use_rope: bool = False
    rope_theta: float = 10000.0
    nope_layers: tuple = ()

    # B2: sliding-window attention. swa_window is the window size in tokens;
    # every swa_full_every-th layer keeps full attention. swa_full_every=6
    # gives the spec's 5:1 windowed:full split.
    swa_window: int = 0
    swa_full_every: int = 6

    # B3: MLA. Head dims are independent of n_embd/n_head once this is on.
    use_mla: bool = False
    kv_lora_rank: int = 256
    q_lora_rank: int = 512      # 0 disables query compression
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # B4: cross-layer KV sharing. N>1 means each group of N consecutive layers
    # computes K/V once, in the group's first layer, and reuses it.
    kv_share_group: int = 1

    # --- Stage C: fine-grained MoE. Off by default.
    use_moe: bool = False
    n_shared_experts: int = 2      # always active
    n_routed_experts: int = 14     # router picks moe_top_k of these
    moe_top_k: int = 2             # => 2 + 2 = 4 active experts per token
    moe_expert_hidden: int = 0     # 0 = auto: 4*n_embd // active_experts
    moe_first_k_dense: int = 1     # leading layers that stay dense MLP
    moe_aux_loss_weight: float = 0.01   # load balancing
    moe_ortho_loss_weight: float = 0.0  # router orthogonality (see MoE docstring)
    moe_var_loss_weight: float = 0.0    # routing confidence (fights load balancing)

    # --- Stage E: (manifold-constrained) Hyper-Connections.
    # 'single' is the Stage A baseline and keeps the plain residual code path.
    # 'hc' is unconstrained mixing; 'mhc' constrains the mixing matrix to the
    # doubly-stochastic manifold via Sinkhorn-Knopp. Run all three side by side.
    mhc_mode: str = 'single'        # 'single' | 'hc' | 'mhc'
    mhc_streams: int = 4            # residual streams (ignored when 'single')
    # Sinkhorn-Knopp iterations for 'mhc'. Exact for the near-identity regime
    # training starts in; convergence degrades if A drifts to a skewed regime.
    # Watch row_sum_dev in the mhc log and raise this if it climbs.
    mhc_sinkhorn_iters: int = 20
    mhc_init_logit: float = 10.0    # diagonal dominance of A at init for 'mhc'
    mhc_init_scheme: str = 'e0'     # 'e0' | 'uniform' — see hyper.py; both are
                                    # residual-equivalent at init but distribute
                                    # gradient across streams very differently
    mhc_instrument: bool = False    # collect per-stream stats (costs a little)
def build_layer_plan(config):
    """Per-layer attention settings. Stage A yields all-default plans."""
    full_every = config.swa_full_every
    group = max(1, config.kv_share_group)

    # A typo'd or stale nope_layers would otherwise silently do nothing — the
    # run would look like it tested RoPE+NoPE while actually testing RoPE.
    bad = [i for i in tuple(config.nope_layers) if not 0 <= i < config.n_layer]
    assert not bad, (f"nope_layers {bad} out of range for n_layer="
                     f"{config.n_layer} (valid: 0..{config.n_layer - 1})")
    assert config.n_layer % group == 0, (
        f"n_layer={config.n_layer} must be divisible by kv_share_group={group}, "
        f"otherwise the final group is short and the KV saving is misreported")

    # MLA's qk_rope_head_dim slice exists to carry the decoupled rotary key.
    # With RoPE off entirely, those dims are all-zero on both q and k: they
    # cost compute and contribute nothing, and nothing would report it.
    if config.use_mla and not config.use_rope and config.qk_rope_head_dim:
        raise AssertionError(
            f"use_mla=True with use_rope=False leaves qk_rope_head_dim="
            f"{config.qk_rope_head_dim} dims permanently zero (wasted compute, "
            f"no positional signal). Either set use_rope=True — MLA's decoupled "
            f"rotary design assumes it — or set qk_rope_head_dim=0 to opt out "
            f"explicitly.")

    plans = []
    for i in range(config.n_layer):
        is_full = (not config.swa_window) or ((i + 1) % full_every == 0)
        if group == 1:
            kv_role = 'own'
        else:
            kv_role = 'produce' if i % group == 0 else 'consume'
        plans.append({
            'rope': config.use_rope and i not in tuple(config.nope_layers),
            'window': 0 if is_full else config.swa_window,
            'kv_role': kv_role,
        })

    # Sharing K/V between layers whose rotary treatment differs is incoherent:
    # a consumed key already has the producer's RoPE baked in.
    for i, p in enumerate(plans):
        if p['kv_role'] == 'consume':
            producer = plans[(i // group) * group]
            assert p['rope'] == producer['rope'], (
                f"layer {i} shares KV with layer {(i // group) * group} but their "
                f"RoPE settings differ; align nope_layers with kv_share_group "
                f"boundaries (nope_layers must cover whole groups)")
    return plans
