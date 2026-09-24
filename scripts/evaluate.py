"""
Evaluate a checkpoint: held-out loss, perplexity, bits per byte, throughput.

    python scripts/evaluate.py --ckpt out-dev-ecc/ckpt.pt
    python scripts/evaluate.py --ckpt out-stage-a/ckpt.pt --batches 200 --sample

WHY BITS PER BYTE IS REPORTED
-----------------------------
Cross-entropy in nats per token is only comparable between models sharing a
tokenizer. Stage F (BLT) consumes raw bytes and has no token vocabulary at
all, so its loss cannot be compared to a BPE stage's loss — a model with a
coarser tokenizer predicts fewer, harder tokens and looks worse on
loss-per-token while being no worse as a model.

Bits per byte normalises by the underlying text instead:

    bpb = loss_nats * n_tokens / (ln 2 * n_bytes)

That is comparable across tokenizers, and is the number to use when Stage F is
compared with Stages A-G. Reported here so the habit starts at Stage A.

TWO BASELINES, AND WHY THE SECOND ONE IS THE REAL TEST
------------------------------------------------------
uniform  ln(vocab_size). A model that has learnt nothing at all scores this.
         Beating it is not an achievement; it is nearly automatic.

unigram  cross-entropy of the held-out split under the training split's token
         frequencies (add-one smoothed). This is what raw frequency counting
         buys, with no context modelling whatsoever. On a corpus with a skewed
         token distribution — code and markdown, where a single space token
         can be over 10% of all tokens — the gap between uniform and unigram
         is large, and a model can look impressive against uniform while
         having learnt little more than "predict a space".

Read the model against the UNIGRAM number. That difference is the part
attributable to actually modelling context.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonos import GPT, GPTConfig                        # noqa: E402
from pythonos.data import data_dir_for, verify_corpus      # noqa: E402
from pythonos import tokenizer as tok                       # noqa: E402


def _tokenizer_for(encoding):
    """(encode_fn, decode_fn, true_vocab) for the encoding a corpus was
    actually built with, read from its manifest.json.

    Corpora built before the tokenizer switch (see pythonos/tokenizer.py)
    recorded encoding='gpt2'; this still decodes those correctly via
    tiktoken, an optional/legacy-only import. Anything else is assumed to be
    the current tokenizer. Using the checkpoint's own recorded encoding
    (rather than always assuming the current one) matters for both
    count_bytes() and --sample: decoding token ids with the wrong tokenizer
    produces garbage, not an error.
    """
    if encoding == 'gpt2':
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        return enc.encode_ordinary, enc.decode, enc.n_vocab
    return tok.encode_ordinary, tok.decode, tok.vocab_size()


def _encoding_from_manifest(manifest):
    """The corpus's actual tokenizer, inferred robustly across manifest
    schema versions.

    Manifests written before the StarCoder2 switch record only
    'tokenizer': 'tiktoken/gpt2' -- no separate 'encoding' key at all (see
    data/ecc_repo/manifest.json from before this change). Defaulting a
    missing 'encoding' straight to the current tokenizer would silently
    decode a GPT-2-era corpus's token ids with the wrong vocabulary: mostly
    out-of-range ids get dropped by count_bytes()'s guard rather than raise,
    so this would look like a working but wrong byte count, not a crash.
    """
    encoding = manifest.get('encoding')
    if encoding:
        return encoding
    if 'gpt2' in str(manifest.get('tokenizer', '')):
        return 'gpt2'
    return tok.ENCODING_NAME


def load_model(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**blob['model_args'])
    model = GPT(config)
    state = blob['model']
    prefix = '_orig_mod.'
    state = {(k[len(prefix):] if k.startswith(prefix) else k): v
             for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval(), blob, config


def count_bytes(token_ids, encoding=tok.ENCODING_NAME):
    """Exact UTF-8 byte length of the text these tokens decode to."""
    _, decode, true_vocab = _tokenizer_for(encoding)
    # padding rows above the real vocabulary never appear in the data, but
    # guard anyway so a corrupt file cannot crash the decoder
    usable = [int(t) for t in token_ids if int(t) < true_vocab]
    return len(decode(usable).encode('utf-8'))


def unigram_baseline(data_dir, vocab_size):
    """Cross-entropy of val under train's token frequencies, add-one smoothed.

    This is the bar a model must clear to have learnt anything about context
    rather than just which tokens are common.
    """
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    if not (os.path.isfile(train_path) and os.path.isfile(val_path)):
        return None
    train = np.memmap(train_path, dtype=np.uint16, mode='r')
    val = np.memmap(val_path, dtype=np.uint16, mode='r')
    counts = np.bincount(np.asarray(train, dtype=np.int64),
                         minlength=vocab_size).astype(np.float64)
    probs = (counts + 1.0) / (counts.sum() + vocab_size)
    return float(-np.log(probs[np.asarray(val, dtype=np.int64)]).mean())


@torch.no_grad()
def measure(model, tokens, block_size, batch_size, batches, device, seed=1234):
    """Mean loss over `batches` random windows, plus throughput."""
    generator = torch.Generator().manual_seed(seed)
    limit = len(tokens) - block_size - 1
    if limit <= 0:
        raise SystemExit(f"split has {len(tokens)} tokens, too few for "
                         f"block_size={block_size}")

    total_loss, counted, elapsed = 0.0, 0, 0.0
    for _ in range(batches):
        starts = torch.randint(limit, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(
            np.asarray(tokens[int(i):int(i) + block_size], dtype=np.int64))
            for i in starts]).to(device)
        y = torch.stack([torch.from_numpy(
            np.asarray(tokens[int(i) + 1:int(i) + 1 + block_size], dtype=np.int64))
            for i in starts]).to(device)
        start = time.perf_counter()
        _, loss = model(x, y)
        elapsed += time.perf_counter() - start
        total_loss += loss.item()
        counted += 1
    tokens_seen = counted * batch_size * block_size
    return total_loss / counted, tokens_seen / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset', default=None,
                        help='defaults to the dataset recorded in the checkpoint')
    parser.add_argument('--batches', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample', action='store_true',
                        help='also generate a short continuation')
    parser.add_argument('--sample-tokens', type=int, default=120)
    args = parser.parse_args()

    model, blob, config = load_model(args.ckpt, args.device)
    settings = blob.get('settings', {})
    dataset = args.dataset or settings.get('dataset')
    if dataset is None:
        raise SystemExit("checkpoint records no dataset; pass --dataset")

    data_dir = data_dir_for(dataset)
    print(f"checkpoint : {args.ckpt}")
    print(f"dataset    : {dataset}")
    print(f"step       : {blob.get('step', '?')}")
    print(verify_corpus(data_dir, strict=False))

    counts = model.param_report()
    print(f"params     : {counts['total'] / 1e6:.2f}M total, "
          f"{counts['active'] / 1e6:.2f}M active")

    results = {}
    for split in ('train', 'val'):
        path = os.path.join(data_dir, f'{split}.bin')
        if not os.path.isfile(path):
            continue
        tokens = np.memmap(path, dtype=np.uint16, mode='r')
        loss, throughput = measure(model, tokens, config.block_size,
                                   args.batch_size, args.batches, args.device)
        results[split] = {'loss': loss, 'tokens': len(tokens),
                          'throughput': throughput}

    uniform = math.log(config.vocab_size)
    unigram = unigram_baseline(data_dir, config.vocab_size)
    print("\nbaselines (nats/token)")
    print(f"  uniform  ln({config.vocab_size}) = {uniform:7.4f}   "
          f"a model that has learnt nothing")
    if unigram is not None:
        print(f"  unigram              = {unigram:7.4f}   "
              f"what token frequency alone buys")
    print(f"\n{'split':6s} {'loss':>8s} {'ppl':>10s} {'bits/tok':>9s} "
          f"{'bits/byte':>10s} {'vs uniform':>11s}")

    manifest_path = os.path.join(data_dir, 'manifest.json')
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)

    for split, info in results.items():
        loss = info['loss']
        bits_per_token = loss / math.log(2)
        # decode a bounded prefix and extrapolate: decoding 10M tokens is slow
        # and the tokens-per-byte ratio is stable across a large sample
        tokens = np.memmap(os.path.join(data_dir, f'{split}.bin'),
                           dtype=np.uint16, mode='r')
        probe = min(len(tokens), 200_000)
        probe_bytes = count_bytes(tokens[:probe], _encoding_from_manifest(manifest))
        bytes_per_token = probe_bytes / probe
        bits_per_byte = bits_per_token / bytes_per_token
        info.update(bits_per_token=bits_per_token, bits_per_byte=bits_per_byte,
                    bytes_per_token=bytes_per_token)
        print(f"{split:6s} {loss:8.4f} {math.exp(min(loss, 20)):10.2f} "
              f"{bits_per_token:9.4f} {bits_per_byte:10.4f} "
              f"{uniform - loss:+11.4f}")

    if unigram is not None and 'val' in results:
        gain = unigram - results['val']['loss']
        share = 100 * gain / unigram
        print(f"\nvs unigram (val): {gain:+.4f} nats/token — "
              f"{share:.1f}% better than frequency")
        print("  counting alone. This, not the margin over uniform, is the "
              "part\n  attributable to modelling context.")

    if 'train' in results and 'val' in results:
        gap = results['val']['loss'] - results['train']['loss']
        print(f"\ngeneralisation gap (val - train): {gap:+.4f} nats/token")
        if gap > 0.5:
            print("  large gap: the model is fitting the training split, which "
                  "is expected on a small single-project corpus")

    print(f"\nthroughput : {results['val']['throughput']:,.0f} tokens/sec "
          f"(forward only, {args.device})")
    print(f"bytes/token: {results['val']['bytes_per_token']:.3f}")

    if args.sample:
        print("\n--- sample continuation ---")
        # use the checkpoint's own recorded encoding, not always the current
        # one -- generating with the wrong tokenizer decodes to garbage, not
        # an error, so this must match what the corpus was actually built with
        encode, decode, true_vocab = _tokenizer_for(_encoding_from_manifest(manifest))
        prompt = "def "
        ids = torch.tensor([encode(prompt)], device=args.device)
        out = model.generate(ids, args.sample_tokens, temperature=0.8, top_k=40)
        text = decode([int(t) for t in out[0] if int(t) < true_vocab])
        print(text)
        print("--- end sample ---")


if __name__ == '__main__':
    main()
