"""Run 70-step B-scan simulations for Domain 1 cavity input files.

The source .in files keep #geometry_view commands for manual geometry checks.
For B-scan simulation, this script reads each .in file without its final line
before passing it to gprMax, so n=70 does not produce geometry VTI files.
"""


from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


NUM_SCANS = 70
RX_NUMBER = 1
RX_COMPONENT = "Ez"


def add_gprmax_to_path(gprmax_root: Path) -> None:
    sys.path.insert(0, str(gprmax_root.resolve()))


class NamedStringIO(io.StringIO):
    def __init__(self, content: str, name: str) -> None:
        super().__init__(content)
        self.name = name


def read_without_last_line(source: Path) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[:-1]) + "\n"


def save_bscan_png(data: np.ndarray, path: Path, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmax = float(np.max(np.abs(data)))
    if vmax == 0:
        vmax = 1.0
    normalized = np.clip((data + vmax) / (2 * vmax), 0.0, 1.0)
    image = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image.save(path)


def run_one(
    input_path: Path,
    out_dir: Path,
    use_gpu: bool,
    overwrite: bool,
    image_size: int,
    keep_merged: bool,
    geometry_fixed: bool,
) -> dict:
    try:
        from gprMax.gprMax import api
        from tools.outputfiles_merge import merge_files
        from tools.plot_Bscan import get_output_data
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("gprMax."):
            raise RuntimeError(
                "gprMax cannot be imported because its compiled extension modules are missing. "
                "Build/install gprMax for this Python environment before running B-scan simulation."
            ) from exc
        raise

    stem = input_path.stem
    npy_path = out_dir / "bscan_npy" / f"{stem}.npy"
    txt_path = out_dir / "bscan_txt" / f"{stem}.txt"
    png_path = out_dir / "bscan_png" / f"{stem}.png"
    input_stem = input_path.with_suffix("")
    merged_path = input_stem.with_name(input_stem.name + "_merged.out")

    if npy_path.exists() and png_path.exists() and not overwrite:
        return {"input": str(input_path), "status": "skipped", "npy": str(npy_path), "png": str(png_path)}

    if overwrite and merged_path.exists():
        merged_path.unlink()

    gpu_arg = [0] if use_gpu else None
    input_content = read_without_last_line(input_path)
    with NamedStringIO(input_content, str(input_path)) as input_file:
        api(input_file, n=NUM_SCANS, geometry_only=False, geometry_fixed=geometry_fixed, gpu=gpu_arg)
    merge_files(str(input_stem), removefiles=True)

    output_data, dt = get_output_data(str(merged_path), RX_NUMBER, RX_COMPONENT)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, output_data)
    np.savetxt(txt_path, output_data, delimiter=" ")
    save_bscan_png(output_data, png_path, image_size)

    if not keep_merged and merged_path.exists():
        merged_path.unlink()

    return {
        "input": str(input_path),
        "status": "done",
        "npy": str(npy_path),
        "txt": str(txt_path),
        "png": str(png_path),
        "shape": list(output_data.shape),
        "dt": float(dt),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Domain 1 cavity B-scan simulations.")
    parser.add_argument("--input-dir", type=Path, default=Path("gprMax-master/domain1_cavity_inputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain1_cavity_outputs"))
    parser.add_argument("--gprmax-root", type=Path, default=Path("gprMax-master"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-merged", action="store_true")
    parser.add_argument(
        "--rebuild-geometry-each-scan",
        action="store_true",
        help="Disable geometry_fixed; only use this if geometry changes during the 70 scans.",
    )
    parser.add_argument("--image-size", type=int, default=321)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd().resolve()
    input_dir = (workspace_root / args.input_dir).resolve() if not args.input_dir.is_absolute() else args.input_dir
    out_dir = (workspace_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    gprmax_root = (workspace_root / args.gprmax_root).resolve() if not args.gprmax_root.is_absolute() else args.gprmax_root

    add_gprmax_to_path(gprmax_root)

    inputs = sorted(input_dir.glob("*.in"))
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise FileNotFoundError(f"No .in files found under {input_dir}")

    results = []
    original_cwd = Path.cwd()
    try:
        os.chdir(gprmax_root)
        for index, input_path in enumerate(inputs, start=1):
            print(f"[{index}/{len(inputs)}] simulate {input_path.name}")
            results.append(
                run_one(
                    input_path=input_path,
                    out_dir=out_dir,
                    use_gpu=args.use_gpu,
                    overwrite=args.overwrite,
                    image_size=args.image_size,
                    keep_merged=args.keep_merged,
                    geometry_fixed=not args.rebuild_geometry_each_scan,
                )
            )
    finally:
        os.chdir(original_cwd)

    status_path = out_dir / "bscan_status.json"
    status_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")


if __name__ == "__main__":
    main()
