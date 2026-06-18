"""Run gprMax geometry-only builds for generated input files.

This is intentionally separate from B-scan generation: it only builds the model
geometry and writes files requested by #geometry_view commands.
"""


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def add_gprmax_to_path(gprmax_root: Path) -> None:
    sys.path.insert(0, str(gprmax_root.resolve()))


def iter_inputs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.in"))


def geometry_done(input_path: Path) -> bool:
    text = input_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#geometry_view:"):
            continue
        parts = line.split()
        if len(parts) != 12:
            return False
        base_name = parts[-2]
        mode = parts[-1].lower()
        suffix = ".vti" if mode == "n" else ".vtp"
        if not (input_path.parent / f"{base_name}{suffix}").exists():
            return False
    return True


def run_one(input_path: Path, overwrite: bool) -> dict:
    try:
        from gprMax.gprMax import api
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("gprMax."):
            raise RuntimeError(
                "gprMax cannot be imported because its compiled extension modules are missing. "
                "Build/install gprMax for this Python environment before running geometry-only simulation."
            ) from exc
        raise

    if not overwrite and geometry_done(input_path):
        return {"input": str(input_path), "status": "skipped"}

    api(str(input_path), n=1, geometry_only=True, gpu=None)
    return {"input": str(input_path), "status": "done"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry-only gprMax builds for .in files.")
    parser.add_argument("--input-dir", type=Path, default=Path("gprMax-master/domain1_crack_inputs"))
    parser.add_argument("--gprmax-root", type=Path, default=Path("gprMax-master"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd().resolve()
    input_dir = (workspace_root / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir
    gprmax_root = (workspace_root / args.gprmax_root).resolve() if not args.gprmax_root.is_absolute() else args.gprmax_root

    add_gprmax_to_path(gprmax_root)
    inputs = iter_inputs(input_dir)
    if args.limit is not None:
        inputs = inputs[: args.limit]

    if not inputs:
        raise FileNotFoundError(f"No .in files found under {input_dir}")

    results = []
    original_cwd = Path.cwd()
    try:
        os.chdir(gprmax_root)
        for index, input_path in enumerate(inputs, start=1):
            print(f"[{index}/{len(inputs)}] geometry-only {input_path.name}")
            results.append(run_one(input_path, args.overwrite))
    finally:
        os.chdir(original_cwd)

    status_path = input_dir / "geometry_status.json"
    status_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
