"""Generate Domain 3 cavity gprMax input files with variable urban-fill clutter."""

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
SOURCE_Z = 0
SAMPLES = 100
FRACTAL_SEED_START = 1000
OBJECT_X_MIN = 0.2
OBJECT_X_MAX = 1.5
OBJECT_Y_MIN = 0.2
OBJECT_Y_MAX = 0.72


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 3 - Cavity in Urban Fill with Clutter with Peplinski Modeling",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 20e-9",
        "",
        "#soil_peplinski: 0.4 0.3 1.6 2.66 0.10 0.25 my_soil",
        f"#fractal_box: 0 0 0 2.0 0.8 0.002 2.1 1 1 1 50 my_soil mysoil {fractal_seed}",
        "",
        f"#add_surface_roughness: 0 0 0 2.0 0.8 0 2.2 1 1 -0.001 0.001 mysoil {fractal_seed}",
        "",
        "#material: 7.0 0.001 1.0 0.0 rock_material",
        "#material: 1.0 0.0 1.0 0.0 air",
        "#material: 15 0.05 1 0 high_moisture_soil",
        "#material: 6 0.01 1 0 concrete_material",
        "#material: 1.0 0.0 1.0 0.0 cavity_material",
        "",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain3_cavity_{sample_id:03d}_geometry"
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


def cylinder_line(x: float, y: float, radius: float, material: str) -> str:
    x = clamp(x, OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
    y = clamp(y, OBJECT_Y_MIN + radius, OBJECT_Y_MAX - radius)
    return f"#cylinder: {fmt(x)} {fmt(y)} 0 {fmt(x)} {fmt(y)} 0.002 {fmt(radius)} {material}"


def box_line(x1: float, y1: float, x2: float, y2: float, material: str) -> str:
    x1 = clamp(x1, OBJECT_X_MIN, OBJECT_X_MAX - 0.02)
    y1 = clamp(y1, OBJECT_Y_MIN, OBJECT_Y_MAX - 0.02)
    x2 = clamp(x2, x1 + 0.02, OBJECT_X_MAX)
    y2 = clamp(y2, y1 + 0.02, OBJECT_Y_MAX)
    return f"#box: {fmt(x1)} {fmt(y1)} 0 {fmt(x2)} {fmt(y2)} 0.002 {material}"


def rock_clutter(rng: random.Random) -> list[str]:
    count = rng.randint(5, 10)
    lines = []
    clustered = rng.random() < 0.45
    if clustered:
        cx = rng.uniform(0.35, 1.30)
        cy = rng.uniform(0.28, 0.62)
    for _ in range(count):
        radius = rng.uniform(0.01, 0.03)
        if clustered and rng.random() < 0.65:
            x = cx + rng.uniform(-0.20, 0.20)
            y = cy + rng.uniform(-0.12, 0.12)
        else:
            x = rng.uniform(OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
            y = rng.uniform(OBJECT_Y_MIN + radius, OBJECT_Y_MAX - radius)
        lines.append(cylinder_line(x, y, radius, "rock_material"))
    return lines


def concrete_clutter(rng: random.Random) -> list[str]:
    draw = rng.random()
    if draw < 0.60:
        count = 0
    elif draw < 0.90:
        count = 1
    else:
        count = 2

    lines = []
    for _ in range(count):
        width = rng.uniform(0.04, 0.11)
        height = rng.uniform(0.02, 0.04)
        x1 = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - width)
        y1 = rng.uniform(OBJECT_Y_MIN, OBJECT_Y_MAX - height)
        lines.append(box_line(x1, y1, x1 + width, y1 + height, "concrete_material"))
        if rng.random() < 0.35:
            # Add one adjacent small block to make the concrete patch slightly irregular.
            dx = rng.uniform(-0.03, 0.03)
            dy = rng.uniform(-0.02, 0.02)
            lines.append(box_line(x1 + dx, y1 + height * 0.45 + dy, x1 + dx + width * 0.55, y1 + height * 1.25 + dy, "concrete_material"))
    return lines


def air_clutter(rng: random.Random) -> list[str]:
    count = rng.randint(0, 3)
    lines = []
    used: list[tuple[float, float]] = []
    attempts = 0
    while len(lines) < count and attempts < 100:
        attempts += 1
        radius = rng.uniform(0.005, 0.015)
        x = rng.uniform(OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
        y = rng.uniform(OBJECT_Y_MIN + radius, OBJECT_Y_MAX - radius)
        if all(abs(x - ux) > 0.18 or abs(y - uy) > 0.08 for ux, uy in used):
            used.append((x, y))
            lines.append(cylinder_line(x, y, radius, "air"))
    return lines


def high_moisture_soil(rng: random.Random) -> list[str]:
    lines = []
    count = 1 if rng.random() < 0.82 else 2
    for _ in range(count):
        width = rng.uniform(0.10, 1.10)
        height = rng.uniform(0.03, 0.08)
        x1 = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - width)
        y1 = rng.uniform(0.56, OBJECT_Y_MAX - height)
        lines.append(box_line(x1, y1, x1 + width, y1 + height, "high_moisture_soil"))
        if rng.random() < 0.45:
            notch_width = rng.uniform(0.06, min(0.20, width))
            notch_height = rng.uniform(0.015, height * 0.65)
            nx1 = clamp(x1 + rng.uniform(0, max(0.01, width - notch_width)), OBJECT_X_MIN, OBJECT_X_MAX - notch_width)
            ny1 = clamp(y1 + rng.uniform(-0.015, 0.015), OBJECT_Y_MIN, OBJECT_Y_MAX - notch_height)
            lines.append(box_line(nx1, ny1, nx1 + notch_width, ny1 + notch_height, "high_moisture_soil"))
    return lines


def domain_clutter(rng: random.Random) -> list[str]:
    lines = []
    lines.extend(rock_clutter(rng))
    lines.extend(concrete_clutter(rng))
    lines.extend(air_clutter(rng))
    lines.extend(high_moisture_soil(rng))
    return lines


def cavity_cylinder_cluster(rng: random.Random) -> list[str]:
    cx = rng.uniform(0.45, 1.18)
    cy = rng.uniform(0.28, 0.56)
    parts = rng.randint(2, 4)
    lines = []
    for _ in range(parts):
        radius = rng.uniform(0.055, 0.14)
        x = cx + rng.uniform(-0.12, 0.12)
        y = cy + rng.uniform(-0.08, 0.08)
        lines.append(cylinder_line(x, y, radius, "cavity_material"))
    return lines


def cavity_box_cluster(rng: random.Random) -> list[str]:
    cx = rng.uniform(0.45, 1.18)
    cy = rng.uniform(0.28, 0.56)
    parts = rng.randint(2, 4)
    lines = []
    for _ in range(parts):
        width = rng.uniform(0.12, 0.34)
        height = rng.uniform(0.06, 0.18)
        x = cx + rng.uniform(-0.12, 0.12)
        y = cy + rng.uniform(-0.08, 0.08)
        lines.append(box_line(x - width / 2, y - height / 2, x + width / 2, y + height / 2, "cavity_material"))
    return lines


def cavity_mixed_irregular(rng: random.Random) -> list[str]:
    cx = rng.uniform(0.45, 1.16)
    cy = rng.uniform(0.28, 0.56)
    parts = rng.randint(3, 5)
    lines = []
    for idx in range(parts):
        x = cx + rng.uniform(-0.16, 0.16)
        y = cy + rng.uniform(-0.10, 0.10)
        if idx % 2 == 0:
            lines.append(cylinder_line(x, y, rng.uniform(0.055, 0.13), "cavity_material"))
        else:
            width = rng.uniform(0.12, 0.28)
            height = rng.uniform(0.055, 0.15)
            lines.append(box_line(x - width / 2, y - height / 2, x + width / 2, y + height / 2, "cavity_material"))
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
    lines.extend(domain_clutter(rng))
    lines.append("")
    lines.extend(cavity_lines)
    lines.extend(footer(sample_id))
    return cavity_mode, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 40000 + sample_id)
        cavity_mode, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain3_cavity_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "surface_seed": fractal_seed,
                "random_seed": seed + 40000 + sample_id,
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
    parser = argparse.ArgumentParser(description="Generate Domain 3 cavity input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain3_cavity_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 3 cavity input files under {args.out_dir}")


if __name__ == "__main__":
    main()
