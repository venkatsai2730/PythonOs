"""Build a tiny character-level corpus for the environment smoke test.

Purpose is to prove the training loop runs end to end — CUDA, checkpointing,
resume — not to train anything useful. Deliberately generates its own text
rather than downloading, so the smoke test needs no network and is byte-stable
across machines.

Writes train.bin / val.bin / meta.pkl in the same layout as the real corpus,
but no manifest.json: this dataset is not frozen and must never be used for a
stage comparison.

  $ python data/tiny_smoke/prepare.py
"""

import os
import pickle
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
TOKENS = 400_000
VAL_FRACTION = 0.1

# A small Python-flavoured vocabulary, so the smoke corpus at least resembles
# the real one in character distribution.
KEYWORDS = ['def', 'return', 'for', 'in', 'if', 'else', 'while', 'class',
            'import', 'from', 'try', 'except', 'with', 'as', 'lambda', 'yield',
            'None', 'True', 'False', 'self', 'range', 'len', 'print', 'append']
NAMES = ['result', 'total', 'index', 'value', 'items', 'node', 'left', 'right',
         'count', 'buffer', 'cache', 'queue', 'visited', 'graph', 'depth']
SYMBOLS = ['(', ')', '[', ']', ':', ',', ' = ', ' + ', ' - ', ' * ', '.', '\n',
           '\n    ', '\n        ', ' == ', ' < ', ' > ']


def build_text(target_chars):
    rng = random.Random(SEED)
    chunks = []
    size = 0
    while size < target_chars:
        piece = rng.choice(
            [rng.choice(KEYWORDS), rng.choice(NAMES), rng.choice(SYMBOLS),
             str(rng.randint(0, 999)), ' '])
        chunks.append(piece)
        size += len(piece)
    return ''.join(chunks)


def main():
    text = build_text(TOKENS)
    alphabet = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = [stoi[ch] for ch in text]

    split = int(len(encoded) * (1 - VAL_FRACTION))
    parts = {'train': encoded[:split], 'val': encoded[split:]}

    import numpy as np
    for name, ids in parts.items():
        path = os.path.join(HERE, f'{name}.bin')
        np.asarray(ids, dtype=np.uint16).tofile(path)
        print(f"{name}: {len(ids):,} tokens -> {path}")

    with open(os.path.join(HERE, 'meta.pkl'), 'wb') as handle:
        pickle.dump({'vocab_size': len(alphabet), 'stoi': stoi, 'itos': itos},
                    handle)
    print(f"vocab size {len(alphabet)}; wrote meta.pkl")
    print("NOT frozen — smoke tests only, never for stage comparisons.")


if __name__ == '__main__':
    main()
