"""
Prepare the FROZEN Python code corpus slice for PythonOS-1B nano validation.

Streams a Python subset of the-stack-dedup, tokenises with GPT-2 BPE, and
writes train.bin / val.bin / meta.pkl plus a manifest.json recording exactly
what was produced.

This slice is frozen: Stages A-G must all train on the byte-identical output of
a single run of this script. Re-running it with different settings invalidates
every cross-stage comparison. The manifest exists so you can prove, later, that
a given run used this slice.

  $ python data/pythonos_code/prepare.py                  # ~400M tokens
  $ python data/pythonos_code/prepare.py --target 50e6    # small trial slice

the-stack-dedup is a gated dataset: accept the terms on its HF page and
`huggingface-cli login` first. --dataset codeparrot/codeparrot-clean is an
ungated fallback.
"""

import argparse
import hashlib
import json
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_FRACTION = 0.0005  # ~200k tokens of val at a 400M-token target
SEED = 1337

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=float, default=400e6,
                    help="target total tokens (train + val)")
parser.add_argument("--dataset", default="bigcode/the-stack-dedup",
                    help="HF dataset id; use codeparrot/codeparrot-clean if ungated")
parser.add_argument("--data-dir", default="data/python",
                    help="dataset subdirectory (the-stack-dedup only)")
args = parser.parse_args()

target_tokens = int(args.target)
val_tokens = max(int(target_tokens * VAL_FRACTION), 100_000)

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token  # 50256, document separator

load_kwargs = dict(split="train", streaming=True)
if "the-stack" in args.dataset:
    load_kwargs["data_dir"] = args.data_dir
print(f"streaming {args.dataset} ({load_kwargs.get('data_dir', 'default')})")
ds = load_dataset(args.dataset, **load_kwargs).shuffle(seed=SEED, buffer_size=10_000)

# GPT-2 vocab is 50257, so uint16 (max 65535) is safe
train_path = os.path.join(HERE, "train.bin")
val_path = os.path.join(HERE, "val.bin")

# val is filled first, then train — a single pass, no shuffling across the split
# boundary, so the two never share a document
written = {"val": 0, "train": 0}
docs = {"val": 0, "train": 0}
hashers = {"val": hashlib.sha256(), "train": hashlib.sha256()}

CONTENT_KEYS = ("content", "text")

with open(val_path, "wb") as f_val, open(train_path, "wb") as f_train, \
        tqdm(total=target_tokens, unit="tok", unit_scale=True) as bar:
    handles = {"val": f_val, "train": f_train}
    for doc in ds:
        split = "val" if written["val"] < val_tokens else "train"

        content = next((doc[k] for k in CONTENT_KEYS if doc.get(k)), None)
        if not content:
            continue

        ids = enc.encode_ordinary(content)
        ids.append(eot)
        arr = np.asarray(ids, dtype=np.uint16)

        buf = arr.tobytes()
        handles[split].write(buf)
        hashers[split].update(buf)
        written[split] += len(arr)
        docs[split] += 1
        bar.update(len(arr))

        if written["val"] + written["train"] >= target_tokens:
            break

# 50257 padded up to the nearest multiple of 64. train.py reads vocab_size from
# here; the padding rows are never emitted by the tokenizer but make the lm_head
# matmul meaningfully faster on tensor cores.
PADDED_VOCAB = 50304
meta = {"vocab_size": PADDED_VOCAB, "true_vocab_size": enc.n_vocab, "encoding": "gpt2"}
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    import pickle
    pickle.dump(meta, f)

manifest = {
    "dataset": args.dataset,
    "data_dir": load_kwargs.get("data_dir"),
    "tokenizer": "tiktoken/gpt2",
    "vocab_size": PADDED_VOCAB,
    "true_vocab_size": enc.n_vocab,
    "seed": SEED,
    "target_tokens": target_tokens,
    "train_tokens": written["train"],
    "val_tokens": written["val"],
    "train_docs": docs["train"],
    "val_docs": docs["val"],
    "train_sha256": hashers["train"].hexdigest(),
    "val_sha256": hashers["val"].hexdigest(),
    "frozen": True,
    "note": "Stages A-G must train on this exact slice. Verify sha256 before each run.",
}
with open(os.path.join(HERE, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\ntrain: {written['train']:,} tokens across {docs['train']:,} docs")
print(f"val:   {written['val']:,} tokens across {docs['val']:,} docs")
print(f"train sha256: {manifest['train_sha256']}")
print(f"wrote train.bin, val.bin, meta.pkl, manifest.json to {HERE}")
print("\nThis slice is now FROZEN. Do not re-run for Stages B-G.")
