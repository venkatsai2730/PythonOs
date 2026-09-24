"""Training entry point for the PythonOS-1B nano validation stages.

    python train.py configs/stage_a_nano.py
    python train.py configs/stage_c1_moe_loadbal.py --compile=False
    python train.py configs/stage_e3_mhc.py --mhc_init_logit=3.0 --out_dir=out-e3-l3

Multi-GPU (not needed at nano scale, but supported):

    torchrun --standalone --nproc_per_node=4 train.py configs/stage_a_nano.py

Arguments apply left to right: bare paths are config files, --key=value are
individual overrides. An unknown key is an error, not a silent no-op.
"""

import math
import os
import sys
import time
from contextlib import nullcontext

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from pythonos import GPT, GPTConfig
from pythonos.data import (data_dir_for, load_meta_vocab_size, make_get_batch,
                           verify_corpus)
from pythonos.settings import ARCH_KEYS, load_settings

CHECKPOINT_NAME = 'ckpt.pt'


# --------------------------------------------------------------------- setup

def init_distributed(cfg):
    """Returns (is_ddp, rank, local_rank, world_size, is_master, device)."""
    if int(os.environ.get('RANK', -1)) < 0:
        return False, 0, 0, 1, True, cfg.device
    init_process_group(backend=cfg.backend)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    return True, rank, local_rank, world, rank == 0, device


def resolve_dtype(name):
    if name != 'auto':
        return name
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return 'bfloat16'
    return 'float16' if torch.cuda.is_available() else 'float32'


def autocast_context(device_type, dtype_name):
    if device_type == 'cpu':
        return nullcontext()
    lookup = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
              'float16': torch.float16}
    return torch.amp.autocast(device_type=device_type, dtype=lookup[dtype_name])


def build_model(cfg, device, checkpoint):
    """Create the model, either fresh or restored from a checkpoint."""
    kwargs = cfg.model_kwargs()
    if checkpoint is not None:
        stored = checkpoint['model_args']
        # architecture must match the checkpoint, or the weights mean something
        # different from what the config claims
        drifted = {k: (stored[k], kwargs.get(k)) for k in ARCH_KEYS
                   if k in stored and k in kwargs and stored[k] != kwargs[k]}
        if drifted:
            detail = ', '.join(f"{k}: ckpt={a!r} config={b!r}"
                               for k, (a, b) in drifted.items())
            print(f"note: taking architecture from the checkpoint ({detail})")
        kwargs.update({k: v for k, v in stored.items() if k in kwargs})

    model = GPT(GPTConfig(**kwargs))
    if checkpoint is not None:
        state = checkpoint['model']
        # torch.compile prefixes parameter names; strip it so a compiled
        # checkpoint resumes uncompiled and vice versa
        prefix = '_orig_mod.'
        state = {(k[len(prefix):] if k.startswith(prefix) else k): v
                 for k, v in state.items()}
        model.load_state_dict(state)
    return model.to(device), kwargs


def learning_rate_at(step, cfg):
    """Linear warmup, then cosine decay to min_lr, then flat."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / (cfg.warmup_iters + 1)
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    span = max(cfg.lr_decay_iters - cfg.warmup_iters, 1)
    progress = (step - cfg.warmup_iters) / span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------- reporting

def report_moe(model, step, wandb_module):
    report = model.moe_report()
    if report is None:
        return
    # The aux-loss VALUE is not a collapse detector: it can read its perfect
    # minimum while load is fully collapsed. Only the counts are.
    #   dead      experts receiving zero tokens this batch
    #   max/mean  1.0 under uniform load, -> n_routed as traffic concentrates
    print(f"  [moe] step {step}: dead={report['total_dead']} "
          f"worst max/mean={report['worst_max_over_mean']:.2f} "
          f"lb={report['mean_load_balance']:.4f}")
    for entry in report['per_layer']:
        print(f"    L{entry['layer']:<2d} dead={entry['dead']:<2d} "
              f"max/mean={entry['max_over_mean']:.2f} counts={entry['counts']}")
    if wandb_module is not None:
        wandb_module.log({
            'iter': step,
            'moe/dead_experts': report['total_dead'],
            'moe/worst_max_over_mean': report['worst_max_over_mean'],
            'moe/load_balance': report['mean_load_balance'],
            **{f"moe/L{e['layer']}_max_over_mean": e['max_over_mean']
               for e in report['per_layer']},
        })


def report_mhc(model, step, wandb_module):
    report = model.mhc_report()
    if report is None:
        return
    # cos     -> 1.0 means the streams have become copies of each other
    # ||A-I|| -> 0 means mixing collapsed to independent per-stream residuals
    # rowdev  Sinkhorn row-sum residual: if it climbs, A is no longer doubly
    #         stochastic and mHC is not manifold-constrained at all
    row_dev = max(e['attn_mixing']['row_sum_dev'] for e in report['per_layer'])
    line = (f"  [mhc] step {step}: mode={report['mode']} "
            f"||A-I|| mean={report['mean_dev_from_identity']:.3e} "
            f"max={report['max_dev_from_identity']:.3e}")
    if 'mean_cos_sim' in report:
        line += (f" cos mean={report['mean_cos_sim']:.4f} "
                 f"max={report['max_cos_sim']:.4f}")
    if 'min_stream_grad_norm' in report:
        line += (f" grad=[{report['min_stream_grad_norm']:.2e},"
                 f"{report['max_stream_grad_norm']:.2e}]")
    print(line + f" rowdev={row_dev:.1e}")
    print(f"    readout={[round(v, 4) for v in report['readout']]}")
    for entry in report['per_layer']:
        extra = ''
        if 'cos_sim' in entry:
            extra += f" cos={entry['cos_sim']:.4f}"
        if 'grad_norms' in entry:
            extra += f" grad={[round(g, 5) for g in entry['grad_norms']]}"
        if 'stream_norms' in entry:
            extra += f" norms={[round(n, 3) for n in entry['stream_norms']]}"
        print(f"    L{entry['layer']:<2d} "
              f"||A-I||={entry['attn_mixing']['dev_from_identity']:.3e} "
              f"diag={entry['attn_mixing']['diag_mean']:.4f}{extra}")
    if row_dev > 1e-3:
        print(f"    WARNING: Sinkhorn row residual {row_dev:.1e} — A is no longer "
              f"doubly stochastic; raise mhc_sinkhorn_iters")
    if wandb_module is not None:
        payload = {'iter': step,
                   'mhc/dev_from_identity': report['mean_dev_from_identity'],
                   'mhc/sinkhorn_row_dev': row_dev}
        if 'mean_cos_sim' in report:
            payload['mhc/cos_sim'] = report['mean_cos_sim']
        if 'min_stream_grad_norm' in report:
            payload['mhc/min_stream_grad_norm'] = report['min_stream_grad_norm']
        wandb_module.log(payload)


# -------------------------------------------------------------------- main

def main(argv):
    cfg = load_settings(argv)
    is_ddp, _, local_rank, world_size, is_master, device = init_distributed(cfg)

    accum = cfg.gradient_accumulation_steps
    if is_ddp:
        if accum % world_size:
            raise ValueError(f"gradient_accumulation_steps={accum} must divide "
                             f"by world size {world_size}")
        accum //= world_size

    tokens_per_iter = accum * world_size * cfg.batch_size * cfg.block_size
    if is_master:
        os.makedirs(cfg.out_dir, exist_ok=True)
        print(f"tokens per iteration: {tokens_per_iter:,}")

    torch.manual_seed(cfg.seed + (local_rank if is_ddp else 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    dtype_name = resolve_dtype(cfg.dtype)
    amp = autocast_context(device_type, dtype_name)

    data_dir = data_dir_for(cfg.dataset)
    get_batch = make_get_batch(data_dir, cfg.block_size, cfg.batch_size,
                               device, device_type)
    if is_master:
        # a stage trained on regenerated data cannot be compared to any other
        print(verify_corpus(data_dir, strict=cfg.verify_corpus_strict))

    vocab = load_meta_vocab_size(data_dir)
    if vocab is not None:
        print(f"vocab_size={vocab} (from {data_dir}/meta.pkl)")
        cfg = cfg.__class__(**{**vars(cfg), 'vocab_size': vocab})
    elif cfg.vocab_size is None:
        print("no meta.pkl; defaulting vocab_size to 49152 (StarCoder2 BPE)")
        cfg = cfg.__class__(**{**vars(cfg), 'vocab_size': 49152})

    checkpoint = None
    start_step, best_val = 0, float('inf')
    if cfg.init_from == 'resume':
        path = os.path.join(cfg.out_dir, CHECKPOINT_NAME)
        print(f"resuming from {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        start_step = checkpoint['step']
        best_val = checkpoint['best_val_loss']
    elif cfg.init_from != 'scratch':
        raise ValueError(f"init_from must be 'scratch' or 'resume', "
                         f"got {cfg.init_from!r}")

    model, model_args = build_model(cfg, device, checkpoint)

    if cfg.use_moe and is_master:
        counts = model.param_report()
        print(f"MoE: {counts['total'] / 1e6:.2f}M total, "
              f"{counts['active'] / 1e6:.2f}M active "
              f"({counts['inactive'] / 1e6:.2f}M idle), "
              f"expert hidden {counts['expert_hidden']}")
        print(f"MoE: {cfg.n_shared_experts} shared + {cfg.n_routed_experts} routed, "
              f"top-{cfg.moe_top_k} => {cfg.n_shared_experts + cfg.moe_top_k} "
              f"active experts/token; layers below {cfg.moe_first_k_dense} stay dense")

    scaler = torch.amp.GradScaler(device_type, enabled=dtype_name == 'float16')
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), device_type)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None  # release the loaded tensors

    raw_model = model
    if cfg.compile:
        print("compiling model (first step will be slow)")
        model = torch.compile(model)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    wandb_module = None
    if cfg.wandb_log and is_master:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config=vars(cfg))
        wandb_module = wandb

    @torch.no_grad()
    def evaluate():
        """Mean loss on both splits. eval() mode excludes the aux losses."""
        model.eval()
        out = {}
        for split in ('train', 'val'):
            losses = torch.zeros(cfg.eval_iters)
            for i in range(cfg.eval_iters):
                batch_x, batch_y = get_batch(split)
                with amp:
                    _, loss = model(batch_x, batch_y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def save(step, val_loss):
        payload = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'step': step,
            'best_val_loss': val_loss,
            'settings': vars(cfg),
        }
        target = os.path.join(cfg.out_dir, CHECKPOINT_NAME)
        torch.save(payload, target)
        print(f"  saved checkpoint to {target}")

    model.train()
    inputs, targets = get_batch('train')
    smoothed_mfu = -1.0
    last_time = time.time()

    for step in range(start_step, cfg.max_iters + 1):
        lr = learning_rate_at(step, cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr

        if step % cfg.eval_interval == 0 and is_master:
            losses = evaluate()
            print(f"step {step}: train {losses['train']:.4f} val {losses['val']:.4f}")
            if wandb_module is not None:
                wandb_module.log({'iter': step, 'train/loss': losses['train'],
                                  'val/loss': losses['val'], 'lr': lr,
                                  'mfu': smoothed_mfu * 100})
            if step > start_step and (cfg.always_save_checkpoint
                                      or losses['val'] < best_val):
                best_val = min(best_val, losses['val'])
                save(step, best_val)
        if step == start_step and cfg.eval_only:
            break

        # Stage E instrumentation costs ~400ms per forward at nano shapes, so
        # only pay it on logged steps. This must be set before the forward: the
        # per-stream gradient norms come from a hook registered during it.
        if cfg.mhc_mode != 'single' and cfg.mhc_instrument:
            raw_model.mhc_capture = step % cfg.mhc_log_interval == 0

        for micro in range(accum):
            if is_ddp:
                # only synchronise gradients on the final micro-step
                model.require_backward_grad_sync = micro == accum - 1
            with amp:
                _, loss = model(inputs, targets)
                loss = loss / accum
            # prefetch the next batch while the device works through the forward
            inputs, targets = get_batch('train')
            scaler.scale(loss).backward()

        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        now = time.time()
        elapsed, last_time = now - last_time, now
        if step % cfg.log_interval == 0 and is_master:
            shown = loss.item() * accum
            if step > start_step + 5:
                mfu = raw_model.estimate_mfu(cfg.batch_size * accum, elapsed)
                smoothed_mfu = mfu if smoothed_mfu < 0 else 0.9 * smoothed_mfu + 0.1 * mfu
            print(f"step {step}: loss {shown:.4f} lr {lr:.2e} "
                  f"{elapsed * 1000:.0f}ms mfu {smoothed_mfu * 100:.2f}%")

        if cfg.use_moe and is_master and step % cfg.moe_log_interval == 0:
            report_moe(raw_model, step, wandb_module)
        if (cfg.mhc_mode != 'single' and is_master
                and step % cfg.mhc_log_interval == 0):
            report_mhc(raw_model, step, wandb_module)

    if is_ddp:
        destroy_process_group()


if __name__ == '__main__':
    main(sys.argv[1:])
