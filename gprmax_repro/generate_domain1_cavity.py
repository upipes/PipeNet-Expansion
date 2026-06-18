"""Generate Domain 1 cavity gprMax input files for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DOMAIN_X = 2.0
DOMAIN_Y = 1.0
DOMAIN_Z = 0.002
GROUND_TOP = 0.8
SOURCE_Z = 0.001
SAMPLES = 100
OBJECT_X_MIN = 0.12
OBJECT_X_MAX = 1.5


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def header(sample_id: int) -> list[str]:
    return [
        "#title: Domain 1 - Cavity in Ideal Sandy Loam with Peplinski Modeling",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 15e-9",
        "",
        "#soil_peplinski: 0.8 0.1 1.5 2.66 0.01 0.05 my_soil",
        f"#fractal_box: 0 0 0 2.0 0.8 0.002 1.3 1 1 1 20 my_soil mysoil {sample_id}",
        "",
        "#material: 1.0 0.0 1.0 0.0 cavity_material",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain1_cavity_{sample_id:03d}_geometry"
    return [
        "",
        "#waveform: ricker 1 800e6 my_wave",
        f"#hertzian_dipole: z 0.1 0.8 {SOURCE_Z} my_wave",
        f"#rx: 0.3 0.8 {SOURCE_Z}",
        "#src_steps: 0.02 0 0",
        "#rx_steps: 0.02 0 0",
        f"#geometry_view: 0 0 0 2.0 1.0 0.002 0.002 0.002 0.002 {geometry_name} n",
    ]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def cylinder_line(x: float, y: float, radius: float) -> str:
    x = clamp(x, OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
    y = clamp(y, radius + 0.05, GROUND_TOP - radius - 0.08)
    return (
        f"#cylinder: {fmt(x)} {fmt(y)} 0 "
        f"{fmt(x)} {fmt(y)} 0.002 {fmt(radius)} cavity_material"
    )


def box_line(cx: float, cy: float, width: float, height: float) -> str:
    x1 = clamp(cx - width / 2, OBJECT_X_MIN, OBJECT_X_MAX - 0.04)
    y1 = clamp(cy - height / 2, 0.08, GROUND_TOP - 0.08)
    x2 = clamp(x1 + width, x1 + 0.04, OBJECT_X_MAX)
    y2 = clamp(y1 + height, y1 + 0.025, GROUND_TOP - 0.04)
    return f"#box: {fmt(x1)} {fmt(y1)} 0 {fmt(x2)} {fmt(y2)} 0.002 cavity_material"


def sample_center(rng: random.Random) -> tuple[float, float]:
    return rng.uniform(0.35, 1.30), rng.uniform(0.20, 0.62)


def cylinder_cluster(rng: random.Random) -> list[str]:
    cx, cy = sample_center(rng)
    parts = rng.randint(3, 6)
    lines = []
    for _ in range(parts):
        radius = rng.uniform(0.045, 0.13)
        x = cx + rng.uniform(-0.22, 0.22)
        y = cy + rng.uniform(-0.14, 0.14)
        lines.append(cylinder_line(x, y, radius))
    return lines


def box_cluster(rng: random.Random) -> list[str]:
    cx, cy = sample_center(rng)
    parts = rng.randint(3, 5)
    lines = []
    for _ in range(parts):
        width = rng.uniform(0.10, 0.34)
        height = rng.uniform(0.045, 0.18)
        x = cx + rng.uniform(-0.22, 0.22)
        y = cy + rng.uniform(-0.12, 0.12)
        lines.append(box_line(x, y, width, height))
    return lines


def mixed_irregular(rng: random.Random) -> list[str]:
    cx, cy = sample_center(rng)
    parts = rng.randint(4, 7)
    lines = []
    for index in range(parts):
        x = cx + rng.uniform(-0.26, 0.26)
        y = cy + rng.uniform(-0.16, 0.16)
        if index % 2 == 0:
            lines.append(cylinder_line(x, y, rng.uniform(0.04, 0.11)))
        else:
            lines.append(box_line(x, y, rng.uniform(0.08, 0.26), rng.uniform(0.04, 0.14)))
    return lines


def separated_cavities(rng: random.Random) -> list[str]:
    parts = rng.randint(2, 4)
    anchors = sorted(rng.uniform(0.35, 1.35) for _ in range(parts))
    lines = []
    for x in anchors:
        y = rng.uniform(0.22, 0.62)
        if rng.random() < 0.65:
            lines.append(cylinder_line(x, y, rng.uniform(0.05, 0.12)))
        else:
            lines.append(box_line(x, y, rng.uniform(0.10, 0.28), rng.uniform(0.05, 0.15)))
    return lines


def cavity_geometry(sample_id: int, rng: random.Random) -> tuple[str, list[str]]:
    mode = sample_id % 4
    if mode == 0:
        return "cylinder_cluster", cylinder_cluster(rng)
    if mode == 1:
        return "box_cluster", box_cluster(rng)
    if mode == 2:
        return "mixed_irregular", mixed_irregular(rng)
    return "separated_cavities", separated_cavities(rng)


def make_input(sample_id: int, rng: random.Random) -> tuple[str, str]:
    shape_mode, geometry = cavity_geometry(sample_id, rng)
    lines = header(sample_id)
    lines.extend(geometry)
    lines.extend(footer(sample_id))
    return shape_mode, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + sample_id)
        shape_mode, content = make_input(sample_id, rng)
        path = out_dir / f"domain1_cavity_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": sample_id,
                "random_seed": seed + sample_id,
                "shape_mode": shape_mode,
                "input": str(path.as_posix()),
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Domain 1 cavity input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain1_cavity_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 1 cavity input files under {args.out_dir}")


if __name__ == "__main__":
    main()
