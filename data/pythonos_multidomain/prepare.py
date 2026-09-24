"""
Prepare the FROZEN four-domain corpus slice for PythonOS-1B nano validation.

Unlike data/pythonos_code (a single generic-Python slice), this builds the
corpus that matches the *use case*: the four target domains, mixed in fixed
proportions, so a nano run actually sees DSA / ML / Agentic / Theory text.

  DSA      worked algorithm implementations (git repo, see DOMAINS below)
  ML       Python / PyTorch code
  Agentic  tool-calling / function-calling dialogues (JSON-shaped)
  Theory   math + CS prose

Output is byte-identical in FORMAT to every other slice here — a flat uint16
token stream (StarCoder2 BPE; see pythonos/tokenizer.py) with EOT document
separators, plus meta.pkl and manifest.json — so train.py needs no changes
(it reads vocab_size from meta.pkl). The manifest additionally records
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
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from pythonos import tokenizer as tok

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
VAL_FRACTION = 0.02          # held-out fraction, per domain
# HF's streaming shuffle holds this many RAW records in memory at once, before
# our per-record extraction ever runs. deepmind/code_contests records are
# unusually heavy (each carries a `solutions` list that can hold hundreds of
# full programs, plus public/private/generated test cases we don't even use),
# and this was enough to get the process OOM-killed on a memory-constrained
# free Colab runtime (Kaggle's session had enough RAM to absorb the same
# target). 10,000 -> 2,000 made no measurable difference, which means the
# buffer wasn't the only thing scaling memory linearly here — so this cuts
# hard rather than guessing again in small steps. 200 records is a small
# enough window that memory should no longer be the constraint; if it still
# OOMs at this size, the buffer isn't the actual driver and the fix needs to
# be a different dataset for this domain, not a smaller number here.
# Trade-off: a smaller buffer means less global shuffling — records within
# any given ~200-record window of the source order end up adjacent more
# often. Fine for a nano validation slice; would matter more at full scale.
SHUFFLE_BUFFER = 200
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


def _code_contests(rec):
    """DeepMind CodeContests: problem description + a Python-looking solution.

    `solutions.language` is a ClassLabel whose int<->name mapping is not
    guaranteed to resolve under streaming, so the Python solution is picked by
    a light content heuristic (has `def`/`print(`, no `#include`) rather than
    trusting the label id. Good enough for a training-mix filter; this is not
    trying to be an exact per-language classifier.
    """
    description = rec.get("description")
    if not description:
        return None
    texts = (rec.get("solutions") or {}).get("solution") or []
    solution = next((t for t in texts if t and "#include" not in t
                     and ("def " in t or "print(" in t)), None)
    if solution is None and texts:
        solution = texts[0]
    parts = [f"# Problem\n{description}"]
    if solution:
        parts.append(f"\n\n# Solution\n{solution}")
    return "".join(parts)


def _dialogue(rec):
    """Function-calling / agentic dialogue: join system + chat if present."""
    chunks = [rec.get("system"), rec.get("chat") or rec.get("conversations")
              or rec.get("text")]
    joined = "\n".join(str(c) for c in chunks if c)
    return joined or None


# Edit these to change the mix. `weight` is the share of the token budget.
# Weights are normalised, so they need not sum to 1.
#
# Two source types:
#   'hf'   stream a Hugging Face dataset (needs dataset/split/extract; see
#          stream_domain()). Subject to HF Hub's network retries and its
#          streaming shuffle's memory footprint, which scales with how heavy
#          that dataset's own record schema is.
#   'git'  shallow-clone a plain git repo and read its files directly (needs
#          git_url; see stream_git_repo()). No parquet/Arrow materialisation,
#          no HF Hub throttling, no shuffle-buffer memory pressure — ordinary
#          files on disk. Chosen for dsa after deepmind/code_contests proved
#          unreliable across two cloud notebook platforms: Kaggle succeeded
#          once, but Colab's smaller RAM budget OOM-killed the process on
#          that dataset's records (each carries hundreds of solutions/test
#          cases per problem) even after the shuffle buffer was cut 50x.
#          TheAlgorithms/Python is organised into folders that map almost
#          exactly onto the doc's DSA topic list (dynamic_programming/,
#          graphs/, searches/, sorts/, backtracking/, greedy_methods/, ...),
#          MIT licensed, and pure Python. Trade-off: this gives worked
#          algorithm implementations, not paired (problem statement, verified
#          solution) examples the way code_contests did — a different, not
#          strictly worse, training signal for a nano validation corpus.
DOMAINS = {
    "dsa": dict(source="git",
                git_url="https://github.com/TheAlgorithms/Python.git",
                weight=0.30),
    "ml": dict(source="hf", dataset="codeparrot/codeparrot-clean", split="train",
               data_dir=None, extract=_code, weight=0.30),
    "agentic": dict(source="hf", dataset="glaiveai/glaive-function-calling-v2",
                    split="train", data_dir=None, extract=_dialogue, weight=0.20),
    "theory": dict(source="hf", dataset="open-web-math/open-web-math",
                   split="train", data_dir=None, extract=_text, weight=0.20),
}

GIT_TEXT_EXTENSIONS = {'.py', '.md', '.rst'}
GIT_SKIP_DIRS = {'.git', '__pycache__', '.github', 'node_modules'}
GIT_MAX_FILE_BYTES = 200_000
GIT_MIN_FILE_BYTES = MIN_CHARS


def stream_domain(spec):
    """Yield text records for one domain, or nothing if the dataset won't load.

    Two failure points are handled separately, because they need different
    responses:

      load-time  the dataset id is gone/renamed/gated/uses an unsupported
                 loading script. Nothing has been collected yet, so this
                 domain contributes 0 docs — logged and skipped.

      mid-stream a network read (a parquet shard fetch) times out and
                 huggingface_hub's own retry logic exhausts its attempts.
                 HF Hub throttles unauthenticated requests harder, which is
                 exactly why this shows up as a real, recurring failure here
                 rather than a one-off — see the "unauthenticated requests"
                 warning printed at the start of a run; setting HF_TOKEN
                 reduces how often this triggers, but does not guarantee it
                 away. Either way, records already yielded for this domain by
                 this point are real and already written to the .bin file —
                 discarding them over one bad shard would throw away good
                 data for no reason. So this domain simply stops here with
                 whatever it already collected, instead of crashing the
                 whole corpus build.
    """
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
    iterator = iter(ds)
    while True:
        try:
            rec = next(iterator)
        except StopIteration:
            return
        except Exception as exc:                                # noqa: BLE001
            print(f"  ! network error mid-stream for {spec['dataset']}: {exc}")
            print(f"  ! stopping this domain here — keeping what was already "
                  f"collected rather than losing the whole build over it")
            return
        text = extract(rec)
        if text and len(text) >= MIN_CHARS:
            yield text


def stream_git_repo(git_url, cache_dir):
    """Yield file contents from a shallow clone of a git repo.

    Reused (and simplified) from data/ecc_repo/prepare.py's file-collection
    pattern. A clone failure (bad URL, no network, git missing) is handled
    the same way an HF load failure is: log it, skip this domain, let the
    rest of the build carry on.
    """
    import hashlib
    import subprocess

    if not os.path.isdir(os.path.join(cache_dir, '.git')):
        print(f"  cloning {git_url} (shallow)...")
        try:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            subprocess.run(
                ['git', 'clone', '--depth', '1', git_url, cache_dir],
                check=True, capture_output=True, text=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError) as exc:
            print(f"  ! could not clone {git_url}: {exc}")
            print(f"  ! skipping this domain — the slice will be built without it")
            return
    else:
        print(f"  reusing existing clone at {cache_dir}")

    seen = set()
    for root, dirs, files in os.walk(cache_dir):
        dirs[:] = sorted(d for d in dirs if d not in GIT_SKIP_DIRS)
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in GIT_TEXT_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if not (GIT_MIN_FILE_BYTES <= size <= GIT_MAX_FILE_BYTES):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as handle:
                    text = handle.read()
            except (UnicodeDecodeError, OSError):
                continue
            digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            yield text


def records_for(name, spec):
    """Dispatch to the right generator for this domain's source type."""
    if spec.get('source') == 'git':
        cache_dir = os.path.join(HERE, '_git_cache', name)
        return stream_git_repo(spec['git_url'], cache_dir)
    return stream_domain(spec)


def source_label(spec):
    """Human-readable source, for progress printing and the manifest."""
    return spec.get('dataset') or spec.get('git_url')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=200e6,
                        help="target total tokens (train + val), across all domains")
    args = parser.parse_args()
    target_tokens = int(args.target)

    eot = tok.eot_token()
    true_vocab = tok.vocab_size()

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
            print(f"[{name}] target {quota:,} tokens from {source_label(spec)}")

            for text in records_for(name, spec):
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

                ids = tok.encode_ordinary(text)
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

    # StarCoder2's vocab (49,152) is already a multiple of 64 — see
    # pythonos/tokenizer.py for why this tokenizer and not GPT-2/cl100k/o200k.
    meta = {"vocab_size": true_vocab, "true_vocab_size": true_vocab,
            "encoding": tok.ENCODING_NAME}
    with open(os.path.join(HERE, "meta.pkl"), "wb") as handle:
        pickle.dump(meta, handle)

    manifest = {
        "corpus": "pythonos_multidomain",
        "tokenizer": f"huggingface-tokenizers/{tok.HF_TOKENIZER_ID}",
        "vocab_size": true_vocab,
        "seed": SEED,
        "target_tokens": target_tokens,
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "train_sha256": hashers["train"].hexdigest(),
        "val_sha256": hashers["val"].hexdigest(),
        "domains": {name: {"source": source_label(DOMAINS[name]),
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
    # We `break` out of each domain's streaming iterator as soon as its quota
    # is met, which can leave huggingface_hub/fsspec background retry threads
    # alive and mid-request. Python's normal interpreter shutdown can then
    # race one of them and crash with "Fatal Python error: PyGILState_Release"
    # — harmless (train.bin/val.bin/manifest.json are already written and
    # correct by this point) but alarming, and it can make an otherwise
    # successful Kaggle cell look like it failed. Skip the race.
    os._exit(0)
