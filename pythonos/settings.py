"""Run settings: every knob for a training run, plus config-file/CLI loading.

Replaces the exec-a-config-file pattern with an explicit dataclass. Two
practical wins beyond tidiness:

  - an unknown key in a config file is an error rather than a silently
    ignored global, so a typo'd flag cannot make a run quietly train the
    wrong architecture
  - int values are accepted for float fields (a config file saying
    `warmup = 200` no longer needs to say `200.0`)

Config files are still plain Python, evaluated in a namespace containing the
current settings, so `batch_size = batch_size // 2` and
`include('configs/base.py')` both work.
"""

import ast
import os
from dataclasses import dataclass, fields, replace

# architecture keys, which must match exactly when resuming a checkpoint.
# Instrumentation and schedule keys are deliberately absent: forcing those to
# a checkpoint's values would ignore the flags given on the resume command.
ARCH_KEYS = (
    'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
    'dropout', 'ffn_mult', 'norm_type',
    'learned_pos_emb', 'use_rope', 'rope_theta', 'nope_layers',
    'swa_window', 'swa_full_every',
    'use_mla', 'kv_lora_rank', 'q_lora_rank', 'qk_nope_head_dim',
    'qk_rope_head_dim', 'v_head_dim', 'kv_share_group',
    'use_moe', 'n_shared_experts', 'n_routed_experts', 'moe_top_k',
    'moe_expert_hidden', 'moe_first_k_dense', 'moe_aux_loss_weight',
    'moe_ortho_loss_weight', 'moe_var_loss_weight',
    'mhc_mode', 'mhc_streams', 'mhc_sinkhorn_iters', 'mhc_init_logit',
    'mhc_init_scheme',
)


@dataclass
class RunSettings:
    # --- output and checkpointing
    out_dir: str = 'out'
    init_from: str = 'scratch'          # 'scratch' | 'resume'
    always_save_checkpoint: bool = True
    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False
    log_interval: int = 1

    # --- run logging
    wandb_log: bool = False
    wandb_project: str = 'pythonos-nano'
    wandb_run_name: str = 'run'

    # --- data
    dataset: str = 'pythonos_code'
    verify_corpus_strict: bool = True
    batch_size: int = 12               # micro-batch
    gradient_accumulation_steps: int = 40
    block_size: int = 1024

    # --- model, Stage A
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    ffn_mult: int = 4
    norm_type: str = 'layernorm'
    vocab_size: int = None             # filled from the dataset's meta.pkl

    # --- Stage B: hybrid attention (all off => Stage A)
    learned_pos_emb: bool = True
    use_rope: bool = False
    rope_theta: float = 10000.0
    nope_layers: tuple = ()
    swa_window: int = 0
    swa_full_every: int = 6
    use_mla: bool = False
    kv_lora_rank: int = 256
    q_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    kv_share_group: int = 1

    # --- Stage C: fine-grained MoE (off => Stage A)
    use_moe: bool = False
    n_shared_experts: int = 2
    n_routed_experts: int = 14
    moe_top_k: int = 2
    moe_expert_hidden: int = 0
    moe_first_k_dense: int = 1
    moe_aux_loss_weight: float = 0.01
    moe_ortho_loss_weight: float = 0.0
    moe_var_loss_weight: float = 0.0
    moe_log_interval: int = 250

    # --- Stage E: hyper-connections (single => Stage A)
    mhc_mode: str = 'single'
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_init_logit: float = 10.0
    mhc_init_scheme: str = 'e0'
    mhc_instrument: bool = False
    mhc_log_interval: int = 250

    # --- optimiser and schedule
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    max_iters: int = 600000
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True

    # --- system
    device: str = 'cuda'
    dtype: str = 'auto'                # 'auto' | float32 | bfloat16 | float16
    compile: bool = True
    backend: str = 'nccl'              # DDP only
    seed: int = 1337

    def model_kwargs(self):
        """The subset that GPTConfig accepts."""
        from .config import GPTConfig
        allowed = {f.name for f in fields(GPTConfig)}
        return {k: v for k, v in vars(self).items() if k in allowed}

    def arch_signature(self):
        return {k: getattr(self, k) for k in ARCH_KEYS}


def _coerce(name, value, target_type):
    """Accept int for float fields and list for tuple fields."""
    if target_type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if target_type is tuple and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _field_types():
    return {f.name: f.type for f in fields(RunSettings)}


def apply_config_file(settings, path):
    """Evaluate a config file against `settings` and return updated settings.

    The file runs with every current setting bound as a local, so relative
    edits work. `include(path)` pulls in another config file first, which is
    how the stage configs layer on top of the Stage A baseline.
    """
    known = _field_types()
    namespace = dict(vars(settings))

    def include(other):
        """Layer another config file into this one, evaluated first."""
        with open(other) as handle:
            exec(compile(handle.read(), other, 'exec'),
                 {'__builtins__': __builtins__}, namespace)

    namespace['include'] = include
    with open(path) as handle:
        exec(compile(handle.read(), path, 'exec'),
             {'__builtins__': __builtins__}, namespace)

    unknown = [k for k in namespace
               if k not in known and k != 'include' and not k.startswith('_')]
    if unknown:
        raise KeyError(f"{path} sets unknown setting(s): {sorted(unknown)}. "
                       f"A typo here would otherwise train the wrong architecture "
                       f"silently.")
    updates = {k: _coerce(k, namespace[k], known[k]) for k in known if k in namespace}
    return replace(settings, **updates)


def apply_overrides(settings, args):
    """Apply `path/to/config.py` and `--key=value` arguments, in order."""
    known = _field_types()
    for arg in args:
        if not arg.startswith('--'):
            if not os.path.isfile(arg):
                raise FileNotFoundError(f"config file not found: {arg}")
            settings = apply_config_file(settings, arg)
            print(f"config: {arg}")
            continue
        if '=' not in arg:
            raise ValueError(f"expected --key=value, got {arg!r}")
        key, raw = arg[2:].split('=', 1)
        if key not in known:
            raise KeyError(f"unknown setting --{key}. Known: {sorted(known)}")
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            value = raw
        value = _coerce(key, value, known[key])
        settings = replace(settings, **{key: value})
        print(f"override: {key} = {value!r}")
    return settings


def load_settings(argv):
    """Build settings from CLI arguments (config files then --key=value)."""
    return apply_overrides(RunSettings(), argv)
