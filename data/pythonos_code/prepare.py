"""
Prepare the FROZEN Python code corpus slice for PythonOS-1B nano validation.

Streams a Python subset of the-stack-dedup, tokenises with StarCoder2 BPE
(see pythonos/tokenizer.py for why this and not GPT-2/cl100k/o200k), and
writes train.bin / val.bin / meta.pkl plus a manifest.json recording exactly
what was produced.

This slice is frozen: Stages A-G must all train on the byte-identical output of
a single run of this script. Re-running it with different settings invalidates
every cross-stage comparison. The manifest exists so you can prove, later, that
a given run used this slice.

  $ python data/pythonos_code/prepare.py                  # ~400M tokens (Stage A)
  $ python data/pythonos_code/prepare.py --target 10e6    # dev slice, minutes

The default source (codeparrot/codeparrot-clean) is ungated. the-stack-dedup
is larger but requires accepting its terms and huggingface-cli login first.

"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pythonos import tokenizer as tok

HERE = os.path.dirname(os.path.abspath(__file__))
# Held-out fraction. Floored at 100k tokens so a small dev slice still has a
# usable validation set; at the full 400M target this is ~200k tokens.
VAL_FRACTION = 0.02
SEED = 1337

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=float, default=400e6,
                    help="target total tokens (train + val)")
parser.add_argument("--dataset", default="codeparrot/codeparrot-clean",
                    help="HF dataset id. The default is ungated. "
                         "bigcode/the-stack-dedup is larger but requires "
                         "accepting its terms and huggingface-cli login.")
parser.add_argument("--split", default="train",
                    help="dataset split to stream")
parser.add_argument("--data-dir", default="data/python",
                    help="dataset subdirectory (the-stack-dedup only)")
args = parser.parse_args()

target_tokens = int(args.target)
val_tokens = max(int(target_tokens * VAL_FRACTION), 100_000)

eot = tok.eot_token()
true_vocab = tok.vocab_size()

load_kwargs = dict(split=args.split, streaming=True)
if "the-stack" in args.dataset:
    load_kwargs["data_dir"] = args.data_dir
print(f"streaming {args.dataset} ({load_kwargs.get('data_dir', 'default')})")
ds = load_dataset(args.dataset, **load_kwargs).shuffle(seed=SEED, buffer_size=10_000)

# StarCoder2 vocab is 49,152, so uint16 (max 65535) is safe
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

        ids = tok.encode_ordinary(content)
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

# StarCoder2's 49,152 is already a multiple of 64 -- no padding step needed
# the way GPT-2's 50,257 -> 50,304 was. train.py reads vocab_size from here.
meta = {"vocab_size": true_vocab, "true_vocab_size": true_vocab,
        "encoding": tok.ENCODING_NAME}
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    import pickle
    pickle.dump(meta, f)

manifest = {
    "dataset": args.dataset,
    "data_dir": load_kwargs.get("data_dir"),
    "tokenizer": f"huggingface-tokenizers/{tok.HF_TOKENIZER_ID}",
    "vocab_size": true_vocab,
    "true_vocab_size": true_vocab,
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
