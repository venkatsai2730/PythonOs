"""
Build a token corpus from a local source-repository checkout.

Used here to run the pipeline end to end on real text before a GPU is
available. Point it at a git checkout:

  git clone --depth 1 https://github.com/affaan-m/ecc.git /c/Temp/ecc
  python data/ecc_repo/prepare.py --source /c/Temp/ecc

WHAT THIS CORPUS IS, AND IS NOT
-------------------------------
This is a single project's repository. It is suitable for proving the data
pipeline, training loop and eval harness work on real text. It is NOT the
Stage A corpus and must never be used for a stage comparison:

  - far too small (single-digit millions of tokens against a 300-500M target)
  - not a Python corpus; it is dominated by project markdown
  - highly redundant, being one project's docs, so held-out loss will look
    better than the model deserves

Two choices here exist specifically to keep the evaluation honest:

  file-level split  train and validation never share a file. Splitting by
                    token offset instead would put the first half of a
                    document in train and the rest in validation, and the
                    reported loss would be measuring memorisation.
  exact dedup       identical files are kept once. Repos carry duplicated
                    licence headers, generated docs and vendored copies; left
                    in, they appear on both sides of the split.

Writes train.bin, val.bin, meta.pkl and manifest.json, same layout as the
Stage A corpus, so train.py needs no changes.
"""

import argparse
import hashlib
import json
import os
import pickle
import random

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
VAL_FRACTION = 0.05

# Text formats worth training on. Deliberately excludes lock files, minified
# bundles and anything binary.
TEXT_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs', '.rs', '.go', '.java',
    '.c', '.h', '.cpp', '.hpp', '.sh', '.bash', '.sql', '.md', '.rst', '.txt',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.css', '.html',
}
SKIP_DIRS = {
    '.git', 'node_modules', '.next', 'dist', 'build', '__pycache__', '.venv',
    'venv', '.mypy_cache', '.pytest_cache', 'target', 'vendor', '.turbo',
}
SKIP_NAMES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml', 'Cargo.lock',
    'poetry.lock', 'composer.lock',
}
MAX_FILE_BYTES = 1_000_000   # skip generated monsters
MIN_FILE_BYTES = 16          # skip near-empty stubs


def collect_files(source):
    """Every candidate text file, in a deterministic order."""
    found = []
    for root, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name in SKIP_NAMES:
                continue
            if os.path.splitext(name)[1].lower() not in TEXT_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if MIN_FILE_BYTES <= size <= MAX_FILE_BYTES:
                found.append(path)
    return found


def read_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return handle.read()
    except (UnicodeDecodeError, OSError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True,
                        help='path to the repository checkout')
    parser.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    parser.add_argument('--name', default='ecc_repo',
                        help='recorded in the manifest for provenance')
    args = parser.parse_args()

    source = os.path.abspath(args.source)
    if not os.path.isdir(source):
        raise SystemExit(f"source directory not found: {source}")

    paths = collect_files(source)
    print(f"found {len(paths):,} candidate text files under {source}")

    documents, seen_hashes = [], set()
    duplicates = skipped = 0
    by_extension = {}
    for path in paths:
        text = read_text(path)
        if text is None:
            skipped += 1
            continue
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        ext = os.path.splitext(path)[1].lower()
        by_extension[ext] = by_extension.get(ext, 0) + len(text)
        documents.append(text)

    print(f"  {len(documents):,} unique documents "
          f"({duplicates:,} exact duplicates dropped, {skipped:,} unreadable)")
    chars = sum(len(d) for d in documents)
    print(f"  {chars / 1e6:.2f} MB of text")
    print("  largest contributors:")
    for ext, size in sorted(by_extension.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {ext:8s} {size / 1e6:6.2f} MB  ({100 * size / chars:4.1f}%)")

    # Shuffle whole documents, then split on a document boundary, so no file
    # appears on both sides.
    random.Random(SEED).shuffle(documents)
    cut = max(1, int(len(documents) * (1 - args.val_fraction)))
    parts = {'train': documents[:cut], 'val': documents[cut:]}

    encoder = tiktoken.get_encoding('gpt2')
    separator = encoder.eot_token
    written, digests = {}, {}
    for split, docs in parts.items():
        ids = []
        for doc in docs:
            ids.extend(encoder.encode_ordinary(doc))
            ids.append(separator)   # document boundary
        array = np.asarray(ids, dtype=np.uint16)
        path = os.path.join(HERE, f'{split}.bin')
        array.tofile(path)
        written[split] = len(array)
        digests[split] = hashlib.sha256(array.tobytes()).hexdigest()
        print(f"{split}: {len(docs):,} docs -> {len(array):,} tokens")

    # 50257 padded to a multiple of 64: the padding rows are never emitted but
    # make the output matmul meaningfully faster on tensor cores
    padded_vocab = 50304
    with open(os.path.join(HERE, 'meta.pkl'), 'wb') as handle:
        pickle.dump({'vocab_size': padded_vocab,
                     'true_vocab_size': encoder.n_vocab,
                     'encoding': 'gpt2'}, handle)

    manifest = {
        'source': args.name,
        'source_path': source,
        'tokenizer': 'tiktoken/gpt2',
        'vocab_size': padded_vocab,
        'seed': SEED,
        'documents': len(documents),
        'duplicates_dropped': duplicates,
        'characters': chars,
        'train_tokens': written['train'],
        'val_tokens': written['val'],
        'train_sha256': digests['train'],
        'val_sha256': digests['val'],
        'split': 'by document, so train and val share no file',
        'frozen': True,
        'note': ('Pipeline validation corpus. NOT the Stage A corpus: too '
                 'small, single-project, markdown-dominated. Never use for a '
                 'stage comparison.'),
    }
    with open(os.path.join(HERE, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)

    total = written['train'] + written['val']
    print(f"\ntotal {total:,} tokens ({chars / max(total, 1):.2f} chars/token)")
    print(f"train sha256 {digests['train'][:32]}...")
    print(f"wrote train.bin, val.bin, meta.pkl, manifest.json to {HERE}")


if __name__ == '__main__':
    main()
