"""
Prove-then-add sweep: run the dense baseline and each architecture block on the
SAME frozen corpus, same seed, same iteration budget, and emit one table of
val-loss deltas + parameter counts.

This is the fundable artifact. For each block it answers the only question that
matters at nano scale: on byte-identical data, does this technique *measurably*
help, and at what parameter cost? Attribution is clean because exactly one thing
changes per row.

  python scripts/proof_sweep.py --dataset pythonos_multidomain --max_iters 2000

What it does NOT prove: domain dominance or beating any external model — that
needs scale + post-training. It proves each block is correct, stable, and worth
(or not worth) its parameters. Report it as exactly that.

Notes
- Every run uses the same --max_iters / --dataset / seed, so the comparison is
  valid even at a reduced budget. A shorter budget gives noisier deltas, not
  biased ones. For a headline number, re-run at the full 6100 iters.
- `compare_against` follows STAGES.md: B3 (MLA) is compared to B1, not the dense
  baseline, because MLA presumes RoPE; comparing to A would confound the two.
- torch.compile is disabled for every run (it needs a C++ toolchain and is
  often slower than it saves at nano shapes); --t4 additionally uses float16.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (label, config file, what it should be compared against). Order = prove order.
EXPERIMENTS = [
    ("A_dense",     "configs/stage_a_nano.py",        None),
    ("B1_rope_nope","configs/stage_b1_rope_nope.py",  "A_dense"),
    ("B2_swa",      "configs/stage_b2_swa.py",        "A_dense"),
    ("B3_mla",      "configs/stage_b3_mla.py",        "B1_rope_nope"),
    ("B4_kvshare",  "configs/stage_b4_kvshare.py",    "A_dense"),
    ("C1_moe",      "configs/stage_c1_moe_loadbal.py","A_dense"),
    ("E2_hc",       "configs/stage_e2_hc.py",         "A_dense"),
    ("E3_mhc",      "configs/stage_e3_mhc.py",        "E2_hc"),
]

RE_PARAMS = re.compile(r"number of parameters:\s*([\d.]+)M")
RE_MOE = re.compile(r"MoE:\s*([\d.]+)M total,\s*([\d.]+)M active")
RE_EVAL = re.compile(r"step\s+(\d+):\s*train\s+([\d.]+)\s+val\s+([\d.]+)")


def run_one(label, config, dataset, max_iters, device, t4):
    out_dir = os.path.join("runs", f"proof_{label}")
    cmd = [sys.executable, "train.py", config,
           f"--dataset={dataset}",
           f"--max_iters={max_iters}",
           f"--lr_decay_iters={max_iters}",
           f"--out_dir={out_dir}",
           f"--device={device}",
           # torch.compile needs a C++ toolchain (MSVC `cl` on Windows) and is
           # often slower than it saves on a T4 — off everywhere for the sweep.
           "--compile=False",
           "--wandb_log=False"]
    if t4:
        cmd += ["--dtype=float16"]
    print(f"\n{'=' * 70}\n[{label}] {' '.join(cmd)}\n{'=' * 70}")

    total = active = final_val = final_train = None
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        if m := RE_PARAMS.search(line):
            total = active = float(m.group(1))
        if m := RE_MOE.search(line):
            total, active = float(m.group(1)), float(m.group(2))
        if m := RE_EVAL.search(line):
            final_train, final_val = float(m.group(2)), float(m.group(3))
    proc.wait()
    ok = proc.returncode == 0 and final_val is not None
    return {"label": label, "config": config, "ok": ok,
            "total_params_m": total, "active_params_m": active,
            "final_train": final_train, "final_val": final_val}


def write_report(results, dataset, max_iters):
    by_label = {r["label"]: r for r in results}
    compare = {label: cmp for label, _, cmp in EXPERIMENTS}

    lines = [
        "# Nano prove-then-add results",
        "",
        f"- corpus: `{dataset}` (frozen, same slice for every row)",
        f"- budget: {max_iters} iters, seed 1337, identical for every row",
        "- val loss is pure cross-entropy (aux losses excluded at eval)",
        "- **lower val is better**; delta is vs the `compare` column",
        "- nano proves *correctness + per-param effect*, NOT domain dominance",
        "",
        "| block | total (M) | active (M) | final val | compare | delta val | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, _, cmp in EXPERIMENTS:
        r = by_label.get(label)
        if not r or not r["ok"]:
            lines.append(f"| {label} | — | — | **failed** | {cmp or '—'} | — | ⚠️ rerun |")
            continue
        base = by_label.get(cmp)
        if base and base.get("final_val") is not None:
            delta = r["final_val"] - base["final_val"]
            verdict = "✅ helps" if delta < -0.01 else (
                "➖ neutral" if delta <= 0.01 else "❌ hurts")
            dtxt = f"{delta:+.3f}"
        else:
            dtxt, verdict = "—", "baseline" if cmp is None else "—"
        lines.append(
            f"| {label} | {r['total_params_m']:.1f} | {r['active_params_m']:.1f} "
            f"| {r['final_val']:.3f} | {cmp or '—'} | {dtxt} | {verdict} |")

    lines += [
        "",
        "## How to read this",
        "- `delta val < 0` on **param-matched** rows (B2, C1-active) = a real win.",
        "- B3/B4 remove params — read the param columns before the loss; a loss",
        "  rise there may be capacity, not the technique. Compare B3 to B1.",
        "- C1: compare **active** params to A_dense; equal active + lower val is",
        "  the clean MoE win.",
        "- E3 (mHC): a near-zero delta is the expected near-identity collapse.",
        "  Confirm at the full 6100-iter budget before concluding it cannot learn.",
    ]
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    md = os.path.join(ROOT, "runs", "proof_results.md")
    js = os.path.join(ROOT, "runs", "proof_results.json")
    with open(md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(js, "w", encoding="utf-8") as handle:
        json.dump({"dataset": dataset, "max_iters": max_iters,
                   "results": results}, handle, indent=2)
    # the .md file is utf-8; the Windows console may be cp1252, so print safely
    body = "\n".join(lines)
    try:
        print("\n" + body)
    except UnicodeEncodeError:
        print("\n" + body.encode("ascii", "replace").decode("ascii"))
    print(f"\nwrote {md} and {js}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pythonos_multidomain")
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--t4", action="store_true",
                        help="use float16 (16GB T4; bfloat16 needs Ampere+)")
    parser.add_argument("--only", nargs="*",
                        help="run only these labels (default: all)")
    args = parser.parse_args()

    todo = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    results = [run_one(label, config, args.dataset, args.max_iters,
                       args.device, args.t4)
               for label, config, _ in todo]
    write_report(results, args.dataset, args.max_iters)


if __name__ == "__main__":
    main()
