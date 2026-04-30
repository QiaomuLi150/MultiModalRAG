from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
RUN_EVAL = PROJECT_DIR / "scripts" / "run_eval.py"


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    runs = manifest.get("runs") or []
    if not runs:
        raise SystemExit("No runs found in batch manifest.")

    python_bin = args.python_bin or sys.executable
    print("Batch start")
    print(f"- manifest: {args.manifest}")
    print(f"- run_count: {len(runs)}")
    print(f"- python: {python_bin}")

    for index, run in enumerate(runs, start=1):
        label = str(run["run_label"])
        cmd = [python_bin, str(RUN_EVAL)] + [str(arg) for arg in run["args"]]
        print(f"\n[{index}/{len(runs)}] {label}")
        print(" ".join(cmd))
        start = time.perf_counter()
        completed = subprocess.run(cmd, cwd=str(PROJECT_DIR), check=False)
        elapsed = time.perf_counter() - start
        print(f"exit_code={completed.returncode} elapsed_s={elapsed:.2f}")
        if completed.returncode != 0 and not args.keep_going:
            raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequence of MultiModalRAG eval jobs from a JSON manifest.")
    parser.add_argument("--manifest", required=True, help="Path to batch manifest JSON.")
    parser.add_argument("--python-bin", help="Python executable to use for child runs.")
    parser.add_argument("--keep-going", action="store_true", help="Continue even if one run fails.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
