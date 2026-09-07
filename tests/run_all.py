"""
Run every gate. This is what you run before spending GPU hours, and after
touching anything in pythonos/.

$ python tests/run_all.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

GATES = [
    ('Stage A invariance', 'tests/test_stage_a_invariance.py'),
    ('Stage B correctness', 'tests/test_stage_b.py'),
    ('Stage C correctness', 'tests/test_stage_c.py'),
    ('Stage E correctness', 'tests/test_stage_e.py'),
    ('Overfit sanity (all variants)', 'tests/sanity_overfit.py'),
]

results = []
for name, path in GATES:
    print(f"\n{'=' * 70}\n  {name}  ({path})\n{'=' * 70}")
    proc = subprocess.run([sys.executable, path], cwd=ROOT)
    results.append((name, proc.returncode == 0))

print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
for name, ok in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")

failed = [n for n, ok in results if not ok]
if failed:
    raise SystemExit(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
print("\nAll gates passed.")
