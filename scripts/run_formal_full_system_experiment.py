from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "eval" / "manifests" / "formal_full_system_v1.json"
RUN_BATCH = PROJECT_DIR / "scripts" / "run_eval_batch.py"


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest or DEFAULT_MANIFEST)
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    cmd = [
        sys.executable,
        str(RUN_BATCH),
        "--manifest",
        str(manifest),
    ]
    if args.keep_going:
        cmd.append("--keep-going")

    print("Formal full-system experiment")
    print(f"- manifest: {manifest}")
    print(f"- project_dir: {PROJECT_DIR}")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(PROJECT_DIR), check=False)
    raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the formal MultiModalRAG full-system experiment suite.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional custom manifest path. Defaults to eval/manifests/formal_full_system_v1.json.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue through later runs even if one run fails.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
