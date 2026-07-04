"""Convert gprMax normal geometry .vti files to PNG material previews.

The converter reads the raw appended UInt32 Material array from gprMax VTI
ImageData files. It is intentionally lightweight and does not require VTK.
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def parse_header(raw: bytes) -> tuple[int, int, int, dict[int, str], int]:
    appended_marker = raw.index(b"<AppendedData")
    underscore = raw.index(b"_", appended_marker)
    header = raw[:underscore].decode("utf-8", errors="ignore")

    extent_match = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', header)
    if not extent_match:
        raise ValueError("Could not find WholeExtent in VTI header.")
    xs, xf, ys, yf, zs, zf = map(int, extent_match.groups())
    nx = xf - xs
    ny = yf - ys
    nz = zf - zs

    materials = {}
    for name, value in re.findall(r'<Material name="([^"]+)">(\d+)</Material>', raw.decode("utf-8", errors="ignore")):
        materials[int(value)] = name

    return nx, ny, nz, materials, underscore + 1


def read_material_array(path: Path) -> tuple[np.ndarray, dict[int, str]]:
    raw = path.read_bytes()
    nx, ny, nz, materials, data_start = parse_header(raw)

    block_size = struct.unpack_from("<I", raw, data_start)[0]
    expected = nx * ny * nz * np.dtype("<u4").itemsize
    if block_size != expected:
        raise ValueError(f"Unexpected material block size in {path}: {block_size} != {expected}")

    offset = data_start + 4
    material = np.frombuffer(raw, dtype="<u4", count=nx * ny * nz, offset=offset)
    material = material.reshape((nz, ny, nx))[0]
    return material, materials


def colorize(material: np.ndarray, materials: dict[int, str]) -> Image.Image:
    image = np.zeros((*material.shape, 3), dtype=np.uint8)

    for value in np.unique(material):
        name = materials.get(int(value), "")
        if name == "cavity_material":
            color = (220, 30, 30)
        elif name == "free_space":
            color = (245, 245, 245)
        elif name == "pec":
            color = (20, 20, 20)
        elif name.startswith("|mysoil_"):
            soil_index = int(re.search(r"_(\d+)\|", name).group(1))
            shade = 70 + int(150 * soil_index / 20)
            color = (shade, max(45, shade - 45), 30)
        else:
            color = (80, 110, 160)
        image[material == value] = color

    # Put shallow ground at the top of the preview.
    image = np.flipud(image)
    return Image.fromarray(image, mode="RGB")


def convert_one(path: Path, out_dir: Path, scale: float) -> Path:
    material, materials = read_material_array(path)
    image = colorize(material, materials)
    if scale != 1.0:
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        image = image.resize((width, height), Image.Resampling.NEAREST)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.png"
    image.save(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert gprMax VTI geometry views to PNG previews.")
    parser.add_argument("--vti-dir", type=Path, default=Path("gprMax-master/domain1_cavity_inputs/geometry_views"))
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain1_cavity_inputs/geometry_png"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.vti_dir.glob("*.vti"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No .vti files found under {args.vti_dir}")

    for index, path in enumerate(files, start=1):
        out_path = convert_one(path, args.out_dir, args.scale)
        print(f"[{index}/{len(files)}] {path.name} -> {out_path}")


if __name__ == "__main__":
    main()
