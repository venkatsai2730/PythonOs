"""
Build an SFT dataset for Fireworks' Managed Training, from the SAME source as
the nano harness's agentic domain (glaiveai/glaive-function-calling-v2).

This is a SEPARATE experiment from the nano architecture-validation harness,
not a change to it. It exists to answer a different question: does our
prepared data produce real learning signal on a large, capable model, using
free training compute -- a large-scale sanity check on the data itself, not
on any architecture choice.

WHY A NEW SCRIPT, NOT data/pythonos_multidomain/prepare.py
------------------------------------------------------------
prepare.py's _dialogue() extractor flattens a whole conversation into one
string of continuous text, for next-token-prediction PRETRAINING -- that is
the right shape for train.py, and the wrong shape for SFT. Fireworks' SFT API
needs role-structured conversational JSONL (system/user/assistant/tool turns,
see https://docs.fireworks.ai/fine-tuning/fine-tuning-models), which is a
different, incompatible target format. Rather than bend prepare.py to emit
two different shapes from the same extractor, this is its own small script.

ONLY the agentic domain is used here. DSA/ML/Theory, as built in
pythonos_multidomain, are raw continuation-style text (algorithm files, code
snippets, encyclopedia prose) with no natural prompt/response split --
turning them into genuine instruction data needs real synthesis (inventing
prompts from docstrings/titles), which is separate, bigger, and judgment-
heavy work, not something to bundle into a first test. Agentic is used
because its source data is already naturally conversational.

  $ python data/fireworks_sft/build_agentic_sft.py --target 5000

Output: one dataset.jsonl, Fireworks' documented schema --
{"messages": [{"role": ..., "content": ...}, ...]} per line. Upload it with
`firectl create dataset` (see the command this script prints at the end).
"""

import argparse
import hashlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
MIN_MESSAGES = 2       # a system-only or single-turn record isn't useful SFT data
MIN_CONTENT_CHARS = 2  # drop empty turns left over after stripping markers

ROLE_MAP = {"USER": "user", "ASSISTANT": "assistant", "FUNCTION RESPONSE": "tool"}
# Splits right before each role marker at the start of a line, keeping the
# marker with its content in the same chunk -- robust to the content itself
# containing blank lines (JSON blobs, multi-paragraph answers), unlike a
# naive split on the "\n\n\n" separator alone.
TURN_RE = re.compile(r'\n*(?=^(?:USER|ASSISTANT|FUNCTION RESPONSE): )', re.MULTILINE)
PREFIX_RE = re.compile(r'^(USER|ASSISTANT|FUNCTION RESPONSE):\s*(.*)', re.DOTALL)


def to_messages(rec):
    """One glaive record -> Fireworks' messages list, or [] if unusable.

    Verified against real streamed records before this was written: handles
    multi-turn function-calling sequences correctly, including the
    FUNCTION RESPONSE -> tool-role mapping and <|endoftext|> stripping.
    """
    system = re.sub(r'^(SYSTEM:\s*)+', '', rec.get("system") or "").strip()
    chat = rec.get("chat") or ""
    messages = [{"role": "system", "content": system}] if system else []

    for chunk in TURN_RE.split(chat):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = PREFIX_RE.match(chunk)
        if not match:
            continue
        role, content = match.group(1), match.group(2).strip()
        content = content.replace("<|endoftext|>", "").strip()
        if len(content) >= MIN_CONTENT_CHARS:
            messages.append({"role": ROLE_MAP[role], "content": content})

    return messages if len(messages) >= MIN_MESSAGES else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000,
                        help="number of conversations to write")
    args = parser.parse_args()

    from datasets import load_dataset
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train",
                      streaming=True).shuffle(seed=SEED, buffer_size=2000)

    out_path = os.path.join(HERE, "dataset.jsonl")
    seen_hashes = set()
    written = 0

    with open(out_path, "w", encoding="utf-8") as handle:
        iterator = iter(ds)
        while written < args.target:
            try:
                rec = next(iterator)
            except StopIteration:
                print(f"  ! source exhausted at {written:,}/{args.target:,} "
                      f"-- writing what was collected")
                break
            except Exception as exc:                            # noqa: BLE001
                print(f"  ! network error mid-stream: {exc}")
                print(f"  ! stopping here, keeping what was already collected")
                break

            messages = to_messages(rec)
            if not messages:
                continue

            # exact dedup, same discipline as the other corpus builders
            digest = hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            handle.write(json.dumps({"messages": messages}) + "\n")
            written += 1

    print(f"wrote {written:,} conversations to {out_path}")
    print("\nNext: upload and launch via the Fireworks CLI, e.g.")
    print(f"  firectl create dataset agentic-sft-test {out_path}")
    print("  firectl create sftj --base-model <muse-glimmer-30b-id> "
          "--dataset agentic-sft-test --wandb-project pythonos-agentic-test")


if __name__ == "__main__":
    main()
