"""
Prepare the FROZEN four-domain corpus slice for PythonOS-1B nano validation.

Unlike data/pythonos_code (a single generic-Python slice), this builds the
corpus that matches the *use case*: the four target domains, mixed in fixed
proportions, so a nano run actually sees DSA / ML / Agentic / Theory text.

  DSA      competitive-programming problems + solutions
  ML       Python / PyTorch code
  Agentic  tool-calling / function-calling dialogues (JSON-shaped)
  Theory   math + CS prose

Output is byte-identical in FORMAT to every other slice here — a flat uint16
GPT-2-BPE token stream with EOT document separators, plus meta.pkl and
manifest.json — so train.py needs no changes. The manifest additionally records
per-domain token counts for provenance.

  $ python data/pythonos_multidomain/prepare.py                 # ~200M tokens
  $ python data/pythonos_multidomain/prepare.py --target 20e6   # T4 proof slice

IMPORTANT
- This slice is FROZEN once written. Stages A-G must all train on the
  byte-identical output of a *single* run. Re-running with different settings
  or a different mix invalidates every cross-stage comparison — that is the one
  discipline the whole harness exists to protect.
- Domain labels are recorded in the manifest but NOT fed to the model. Label-
  driven expert routing (the doc's Stage 2) is a separate, later change; keeping
  the .bin unlabelled here means this corpus stays drop-in for the current
  train.py and every existing stage config.
- A domain whose dataset fails to load (gated / renamed / offline) is SKIPPED
  with a warning rather than aborting the build; the manifest records which
  domains actually contributed. Do not read a partial slice as a full one.

The default datasets are chosen to be ungated where possible. Swap any of them
via the DOMAINS table below; each entry only needs a HF id, a split, and an
extractor that turns one record into a text string (or None to drop it).
"""

import argparse
import hashlib
import json
import os
import pickle

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
VAL_FRACTION = 0.02          # held-out fraction, per domain
SHUFFLE_BUFFER = 10_000
MIN_CHARS = 32               # drop near-empty records


# --- per-domain extractors ---------------------------------------------------
# Each turns one dataset record into a training string, or None to skip it.
# Datasets differ in schema, so the extraction is explicit rather than guessed.

def _code(rec):
    return rec.get("content") or rec.get("text")


def _text(rec):
    return rec.get("text")


def _apps(rec):
    """APPS-style: a problem statement plus (JSON-encoded) solutions list."""
    question = rec.get("question") or rec.get("problem")
    solutions = rec.get("solutions")
    if isinstance(solutions, str):
        try:
            solutions = json.loads(solutions)
        except (json.JSONDecodeError, TypeError):
            solutions = [solutions]
    first = solutions[0] if isinstance(solutions, list) and solutions else None
    if not question:
        return None
    parts = [f"# Problem\n{question}"]
    if first:
        parts.append(f"\n\n# Solution\n{first}")
    return "".join(parts)


def _dialogue(rec):
    """Function-calling / agentic dialogue: join system + chat if present."""
    chunks = [rec.get("system"), rec.get("chat") or rec.get("conversations")
              or rec.get("text")]
    joined = "\n".join(str(c) for c in chunks if c)
    return joined or None


# Edit these to change the mix. `weight` is the share of the token budget.
# Weights are normalised, so they need not sum to 1.
DOMAINS = {
    "dsa": dict(dataset="codeparrot/apps", split="train", data_dir=None,
                extract=_apps, weight=0.30),
    "ml": dict(dataset="codeparrot/codeparrot-clean", split="train", data_dir=None,
               extract=_code, weight=0.30),
    "agentic": dict(dataset="glaiveai/glaive-function-calling-v2", split="train",
                    data_dir=None, extract=_dialogue, weight=0.20),
    "theory": dict(dataset="open-web-math/open-web-math", split="train",
                   data_dir=None, extract=_text, weight=0.20),
}


def stream_domain(spec):
    """Yield text records for one domain, or nothing if the dataset won't load."""
    from datasets import load_dataset
    kwargs = dict(split=spec["split"], streaming=True)
    if spec.get("data_dir"):
        kwargs["data_dir"] = spec["data_dir"]
    try:
        ds = load_dataset(spec["dataset"], **kwargs)
        ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ! could not load {spec['dataset']}: {exc}")
        print(f"  ! skipping this domain — the slice will be built without it")
        return
    extract = spec["extract"]
    for rec in ds:
        text = extract(rec)
        if text and len(text) >= MIN_CHARS:
            yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=200e6,
                        help="target total tokens (train + val), across all domains")
    args = parser.parse_args()
    target_tokens = int(args.target)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    total_weight = sum(d["weight"] for d in DOMAINS.values())
    quotas = {name: int(target_tokens * d["weight"] / total_weight)
              for name, d in DOMAINS.items()}

    train_path = os.path.join(HERE, "train.bin")
    val_path = os.path.join(HERE, "val.bin")
    hashers = {"train": hashlib.sha256(), "val": hashlib.sha256()}
    written = {"train": 0, "val": 0}
    per_domain = {name: {"train": 0, "val": 0, "docs": 0} for name in DOMAINS}
    seen_hashes = set()

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        handles = {"train": f_train, "val": f_val}
        for name, spec in DOMAINS.items():
            quota = quotas[name]
            val_quota = max(int(quota * VAL_FRACTION), 50_000)
            got = {"train": 0, "val": 0}
            print(f"[{name}] target {quota:,} tokens from {spec['dataset']}")

            for text in stream_domain(spec):
                # exact dedup across the whole corpus: repeated licence headers,
                # vendored copies and boilerplate otherwise land on both sides
                # of the split and inflate held-out performance
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                # fill this domain's val quota first, then its train quota, so a
                # whole document never straddles the split (no memorisation leak)
                split = "val" if got["val"] < val_quota else "train"

                ids = enc.encode_ordinary(text)
                ids.append(eot)
                buf = np.asarray(ids, dtype=np.uint16).tobytes()
                handles[split].write(buf)
                hashers[split].update(buf)

                n = len(ids)
                got[split] += n
                written[split] += n
                per_domain[name][split] += n
                per_domain[name]["docs"] += 1

                if got["train"] >= quota - val_quota and got["val"] >= val_quota:
                    break

            print(f"[{name}] wrote {got['train']:,} train + {got['val']:,} val "
                  f"across {per_domain[name]['docs']:,} docs")

    contributed = [n for n in DOMAINS if per_domain[n]["docs"] > 0]
    if not contributed:
        raise SystemExit("no domain produced any data — check dataset access")

    PADDED_VOCAB = 50304        # 50257 padded to a multiple of 64 (tensor cores)
    meta = {"vocab_size": PADDED_VOCAB, "true_vocab_size": enc.n_vocab,
            "encoding": "gpt2"}
    with open(os.path.join(HERE, "meta.pkl"), "wb") as handle:
        pickle.dump(meta, handle)

    manifest = {
        "corpus": "pythonos_multidomain",
        "tokenizer": "tiktoken/gpt2",
        "vocab_size": PADDED_VOCAB,
        "seed": SEED,
        "target_tokens": target_tokens,
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "train_sha256": hashers["train"].hexdigest(),
        "val_sha256": hashers["val"].hexdigest(),
        "domains": {name: {"dataset": DOMAINS[name]["dataset"],
                           "weight": DOMAINS[name]["weight"],
                           **per_domain[name]} for name in DOMAINS},
        "domains_contributed": contributed,
        "frozen": True,
        "note": ("Four-domain slice. Stages A-G must train on this exact slice; "
                 "verify sha256 before each run. Labels recorded but not fed to "
                 "the model."),
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    total = written["train"] + written["val"]
    print(f"\ntotal {total:,} tokens "
          f"({written['train']:,} train / {written['val']:,} val)")
    print(f"domains contributed: {', '.join(contributed)}")
    print(f"train sha256: {manifest['train_sha256'][:32]}...")
    print(f"wrote train.bin, val.bin, meta.pkl, manifest.json to {HERE}")
    print("\nThis slice is now FROZEN. Do not re-run for Stages B-G.")


if __name__ == "__main__":
    main()
