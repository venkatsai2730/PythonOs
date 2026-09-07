"""Corpus loading, and the freeze check that makes cross-stage claims real.

Stages A-G must all train on byte-identical data in identical order. The
manifest check is what makes that verifiable rather than assumed: it compares
each .bin against the sha256 recorded when the slice was frozen, and refuses
to train on a mismatch.

Stage F (BLT) will not use this loader — it consumes raw bytes through an
entropy patcher. Expect a separate module rather than a flag here.
"""

import hashlib
import json
import os
import pickle

import numpy as np
import torch

_HASH_CHUNK = 8 << 20  # 8 MiB


def data_dir_for(dataset, root='data'):
    return os.path.join(root, dataset)


def load_meta_vocab_size(data_dir):
    """Vocabulary size recorded by the dataset's prepare.py, or None."""
    path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as handle:
        return pickle.load(handle)['vocab_size']


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus(data_dir, strict=True):
    """Check the .bin files against the frozen manifest.

    Returns a human-readable status. With strict=True a hash mismatch raises:
    silently training a later stage on regenerated data invalidates every
    cross-stage comparison, and it cannot be detected after the fact.

    Note this reads both files in full — roughly 800 MB for a 400M-token
    slice, so a few seconds at startup.
    """
    manifest_path = os.path.join(data_dir, 'manifest.json')
    if not os.path.isfile(manifest_path):
        return 'no manifest (unfrozen dataset — fine for smoke tests)'

    with open(manifest_path) as handle:
        manifest = json.load(handle)

    mismatches = []
    for split in ('train', 'val'):
        recorded = manifest.get(f'{split}_sha256')
        path = os.path.join(data_dir, f'{split}.bin')
        if not recorded or not os.path.isfile(path):
            continue
        actual = _file_digest(path)
        if actual != recorded:
            mismatches.append(
                f"{split}.bin is {actual[:16]}..., manifest says {recorded[:16]}...")

    if mismatches:
        detail = '\n  '.join(mismatches)
        message = (f"CORPUS MISMATCH — this is not the frozen slice:\n  {detail}\n"
                   f"Cross-stage comparisons require byte-identical data. Restore "
                   f"the frozen .bin files, or re-run every stage on the new slice.")
        if strict:
            raise RuntimeError(message)
        return message

    tokens = manifest.get('train_tokens')
    counted = f"{tokens:,}" if isinstance(tokens, int) else '?'
    return f"corpus verified against manifest ({counted} train tokens)"


class TokenBatcher:
    """Samples fixed-length windows from a flat token file.

    The .bin files are flat uint16 arrays, memory-mapped rather than loaded,
    so corpora far larger than RAM cost nothing to open. The map is reopened
    per batch: holding one open across many reads lets the OS page cache grow
    without bound in this access pattern.

    Targets are inputs shifted one position left, which is the whole of
    next-token prediction.
    """

    def __init__(self, data_dir, block_size, batch_size, device, device_type):
        self.data_dir = data_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.pin = device_type == 'cuda'

    def _open(self, split):
        name = 'train.bin' if split == 'train' else 'val.bin'
        return np.memmap(os.path.join(self.data_dir, name), dtype=np.uint16, mode='r')

    def __call__(self, split):
        tokens = self._open(split)
        span = self.block_size
        highest = len(tokens) - span - 1
        if highest <= 0:
            raise RuntimeError(
                f"{split}.bin holds {len(tokens)} tokens, too few for "
                f"block_size={span}")
        starts = torch.randint(highest, (self.batch_size,))

        def window(offset):
            return torch.from_numpy(
                np.asarray(tokens[offset:offset + span], dtype=np.int64))

        inputs = torch.stack([window(int(i)) for i in starts])
        targets = torch.stack([window(int(i) + 1) for i in starts])

        if self.pin:
            # pinned memory lets the copy overlap with compute
            inputs = inputs.pin_memory().to(self.device, non_blocking=True)
            targets = targets.pin_memory().to(self.device, non_blocking=True)
            return inputs, targets
        return inputs.to(self.device), targets.to(self.device)


def make_get_batch(data_dir, block_size, batch_size, device, device_type):
    """Convenience wrapper returning a callable batcher."""
    return TokenBatcher(data_dir, block_size, batch_size, device, device_type)
