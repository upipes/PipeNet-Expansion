"""Generate Domain 4 cavity gprMax input files for layered road structure."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


DOMAIN_X = 2.0
DOMAIN_Y = 1.0
DOMAIN_Z = 0.002
SOURCE_Z = 0
SAMPLES = 100
FRACTAL_SEED_START = 1500
OBJECT_X_MIN = 0.12
OBJECT_X_MAX = 1.5
OBJECT_Y_MIN = 0.18
OBJECT_Y_MAX = 0.62
INTERFACE_Y = 0.55


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 4 - Cavity in Layered Road Structure with Peplinski Modeling",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 20e-9",
        "",
        "#material: 5.0 0.01 1.0 0.0 asphalt",
        "#material: 7.0 0.005 1.0 0.0 gravel_base",
        "#soil_peplinski: 0.4 0.3 1.5 2.66 0.15 0.30 native_soil",
        "",
        "#box: 0 0.7 0 2.0 0.8 0.002 asphalt",
        "#box: 0 0.55 0 2.0 0.7 0.002 gravel_base",
        f"#fractal_box: 0 0 0 2.0 0.55 0.002 1.7 1 1 1 10 native_soil native_soil_bd {fractal_seed}",
        "",
        f"#add_surface_roughness: 0 0 0 2.0 0.55 0 2.2 1 1 -0.001 0.001 native_soil_bd {fractal_seed}",
        "",
        "#material: 1.0 0.0 1.0 0.0 cavity_material",
        "",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain4_cavity_{sample_id:03d}_geometry"
    return [
        "",
        "#waveform: ricker 1 800e6 my_wave",
        f"#hertzian_dipole: z 0.1 0.8 {SOURCE_Z} my_wave",
        f"#rx: 0.3 0.8 {SOURCE_Z}",
        "#src_steps: 0.02 0 0",
        "#rx_steps: 0.02 0 0",
        f"#geometry_view: 0 0 0 2.0 1.0 0.002 0.002 0.002 0.002 {geometry_name} n",
    ]


def cylinder_line(x: float, y: float, radius: float) -> str:
    x = clamp(x, OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
    y = clamp(y, OBJECT_Y_MIN + radius, OBJECT_Y_MAX - radius)
    return f"#cylinder: {fmt(x)} {fmt(y)} 0 {fmt(x)} {fmt(y)} 0.002 {fmt(radius)} cavity_material"


def box_line(x1: float, y1: float, x2: float, y2: float) -> str:
    x1 = clamp(x1, OBJECT_X_MIN, OBJECT_X_MAX - 0.05)
    y1 = clamp(y1, OBJECT_Y_MIN, OBJECT_Y_MAX - 0.025)
    x2 = clamp(x2, x1 + 0.05, OBJECT_X_MAX)
    y2 = clamp(y2, y1 + 0.025, OBJECT_Y_MAX)
    return f"#box: {fmt(x1)} {fmt(y1)} 0 {fmt(x2)} {fmt(y2)} 0.002 cavity_material"


def cavity_center(rng: random.Random) -> tuple[float, float]:
    if rng.random() < 0.58:
        y = rng.uniform(INTERFACE_Y - 0.07, INTERFACE_Y + 0.025)
    else:
        y = rng.uniform(0.24, INTERFACE_Y - 0.04)
    x = rng.uniform(0.34, 1.28)
    return x, y


def cavity_cylinder_cluster(rng: random.Random) -> list[str]:
    cx, cy = cavity_center(rng)
    parts = rng.randint(2, 4)
    lines = []
    for _ in range(parts):
        radius = rng.uniform(0.035, 0.11)
        x = cx + rng.uniform(-0.10, 0.10)
        y = cy + rng.uniform(-0.035, 0.035)
        lines.append(cylinder_line(x, y, radius))
    return lines


def cavity_box_cluster(rng: random.Random) -> list[str]:
    cx, cy = cavity_center(rng)
    parts = rng.randint(2, 4)
    lines = []
    for _ in range(parts):
        width = rng.uniform(0.07, 0.28)
        height = rng.uniform(0.035, 0.12)
        x = cx + rng.uniform(-0.12, 0.12)
        y = cy + rng.uniform(-0.04, 0.04)
        lines.append(box_line(x - width / 2, y - height / 2, x + width / 2, y + height / 2))
    return lines


def cavity_mixed_irregular(rng: random.Random) -> list[str]:
    cx, cy = cavity_center(rng)
    parts = rng.randint(3, 5)
    lines = []
    for idx in range(parts):
        x = cx + rng.uniform(-0.14, 0.14)
        y = cy + rng.uniform(-0.05, 0.05)
        if idx % 2 == 0:
            lines.append(cylinder_line(x, y, rng.uniform(0.035, 0.10)))
        else:
            width = rng.uniform(0.07, 0.24)
            height = rng.uniform(0.035, 0.10)
            lines.append(box_line(x - width / 2, y - height / 2, x + width / 2, y + height / 2))
    return lines


def cavity_geometry(sample_id: int, rng: random.Random) -> tuple[str, list[str]]:
    mode = sample_id % 3
    if mode == 0:
        return "cavity_cylinder_cluster", cavity_cylinder_cluster(rng)
    if mode == 1:
        return "cavity_box_cluster", cavity_box_cluster(rng)
    return "cavity_mixed_irregular", cavity_mixed_irregular(rng)


def make_input(sample_id: int, rng: random.Random) -> tuple[str, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    cavity_mode, cavity_lines = cavity_geometry(sample_id, rng)
    lines = header(fractal_seed)
    lines.extend(cavity_lines)
    lines.extend(footer(sample_id))
    return cavity_mode, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 70000 + sample_id)
        cavity_mode, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain4_cavity_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "surface_seed": fractal_seed,
                "random_seed": seed + 70000 + sample_id,
                "cavity_mode": cavity_mode,
                "input": str(path.as_posix()),
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Domain 4 cavity input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain4_cavity_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 4 cavity input files under {args.out_dir}")


if __name__ == "__main__":
    main()
