"""Generate Domain 4 crack gprMax input files for layered road structure."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from generate_domain4_cavity import (
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    OBJECT_X_MAX,
    OBJECT_X_MIN,
    SOURCE_Z,
    SAMPLES,
    clamp,
    fmt,
)


FRACTAL_SEED_START = 1600
CRACK_MIN_THICKNESS = 0.008
CRACK_MAX_THICKNESS = 0.022
ASPHALT_TOP = 0.8
ASPHALT_BOTTOM = 0.7
GRAVEL_BOTTOM = 0.55
NATIVE_SOIL_BOTTOM = 0.18


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 4 - Crack in Layered Road Structure with Peplinski Modeling",
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
        "#material: 1.0 0.0 1.0 0.0 crack_material",
        "",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain4_crack_{sample_id:03d}_geometry"
    return [
        "",
        "#waveform: ricker 1 800e6 my_wave",
        f"#hertzian_dipole: z 0.1 0.8 {SOURCE_Z} my_wave",
        f"#rx: 0.3 0.8 {SOURCE_Z}",
        "#src_steps: 0.02 0 0",
        "#rx_steps: 0.02 0 0",
        f"#geometry_view: 0 0 0 2.0 1.0 0.002 0.002 0.002 0.002 {geometry_name} n",
    ]


def crack_box(x1: float, y1: float, x2: float, y2: float) -> str:
    x1 = clamp(x1, OBJECT_X_MIN, OBJECT_X_MAX - CRACK_MIN_THICKNESS)
    x2 = clamp(x2, x1 + CRACK_MIN_THICKNESS, OBJECT_X_MAX)
    y1 = clamp(y1, NATIVE_SOIL_BOTTOM, ASPHALT_TOP - CRACK_MIN_THICKNESS)
    y2 = clamp(y2, y1 + CRACK_MIN_THICKNESS, ASPHALT_TOP)
    return f"#box: {fmt(x1)} {fmt(y1)} 0 {fmt(x2)} {fmt(y2)} 0.002 crack_material"


def vertical_or_inclined_surface_crack(rng: random.Random, through: bool) -> list[str]:
    thickness = rng.uniform(CRACK_MIN_THICKNESS, 0.018)
    segments = rng.randint(4, 8)
    y_top = rng.uniform(0.782, ASPHALT_TOP - 0.002)
    if through:
        y_bottom = rng.uniform(0.655, 0.705)
    else:
        y_bottom = rng.uniform(0.715, 0.755)
    x_top = rng.uniform(0.25, 1.35)
    total_dx = rng.uniform(-0.10, 0.10)
    lines = []

    for idx in range(segments):
        t1 = idx / segments
        t2 = (idx + 1) / segments
        xa = x_top + total_dx * t1 + rng.uniform(-0.006, 0.006)
        xb = x_top + total_dx * t2 + rng.uniform(-0.006, 0.006)
        x_mid = (xa + xb) / 2
        y_high = y_top + (y_bottom - y_top) * t1
        y_low = y_top + (y_bottom - y_top) * t2
        lines.append(crack_box(x_mid - thickness / 2, min(y_high, y_low), x_mid + thickness / 2, max(y_high, y_low)))
    return lines


def asphalt_to_gravel_slanted_crack(rng: random.Random) -> list[str]:
    thickness = rng.uniform(CRACK_MIN_THICKNESS, 0.018)
    segments = rng.randint(12, 18)
    x_start = rng.uniform(0.25, 1.25)
    y_start = rng.uniform(0.775, ASPHALT_TOP - 0.002)
    x_end = clamp(x_start + rng.uniform(0.12, 0.32) * (-1 if rng.random() < 0.5 else 1), 0.20, 1.45)
    y_end = rng.uniform(0.64, 0.69)
    lines = []

    for idx in range(segments):
        t1 = idx / segments
        t2 = (idx + 1) / segments
        x1 = x_start + (x_end - x_start) * t1
        x2 = x_start + (x_end - x_start) * t2
        y1 = y_start + (y_end - y_start) * t1
        y2 = y_start + (y_end - y_start) * t2
        x_mid = (x1 + x2) / 2 + rng.uniform(-0.004, 0.004)
        y_mid = (y1 + y2) / 2 + rng.uniform(-0.004, 0.004)
        step_width = max(thickness, abs(x2 - x1) + thickness)
        step_height = max(thickness, abs(y2 - y1) + thickness)
        lines.append(crack_box(x_mid - step_width / 2, y_mid - step_height / 2, x_mid + step_width / 2, y_mid + step_height / 2))
    return lines


def internal_horizontal_crack(rng: random.Random) -> list[str]:
    in_gravel = rng.random() < 0.45
    total_length = rng.uniform(0.35, 1.05)
    thickness = rng.uniform(0.010, CRACK_MAX_THICKNESS)
    x_start = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - total_length)
    segments = rng.randint(5, 11)
    seg_len = total_length / segments
    if in_gravel:
        base_y = rng.uniform(0.575, 0.665)
    else:
        base_y = rng.uniform(0.28, 0.52)
    slope = rng.uniform(-0.035, 0.035)
    lines = []

    for idx in range(segments):
        progress = idx / max(1, segments - 1)
        x1 = x_start + idx * seg_len
        x2 = x_start + (idx + 1) * seg_len + rng.uniform(-0.006, 0.006)
        y_mid = base_y + slope * (progress - 0.5) + rng.uniform(-0.006, 0.006)
        lines.append(crack_box(x1, y_mid - thickness / 2, x2, y_mid + thickness / 2))

    if rng.random() < 0.35:
        branch_len = rng.uniform(0.10, 0.24)
        bx1 = rng.uniform(x_start + total_length * 0.25, x_start + total_length - branch_len)
        by = base_y + rng.choice([-1, 1]) * rng.uniform(0.018, 0.040)
        lines.append(crack_box(bx1, by - thickness / 2, bx1 + branch_len, by + thickness / 2))
    return lines


def crack_geometry(sample_id: int, rng: random.Random) -> tuple[str, list[str]]:
    mode = sample_id % 4
    if mode == 0:
        return "surface_through_asphalt", vertical_or_inclined_surface_crack(rng, through=True)
    if mode == 1:
        return "asphalt_only", vertical_or_inclined_surface_crack(rng, through=False)
    if mode == 2:
        return "asphalt_to_gravel_top", asphalt_to_gravel_slanted_crack(rng)
    return "internal_base_or_native_soil", internal_horizontal_crack(rng)


def make_input(sample_id: int, rng: random.Random) -> tuple[str, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    crack_mode, crack_lines = crack_geometry(sample_id, rng)
    lines = header(fractal_seed)
    lines.extend(crack_lines)
    lines.extend(footer(sample_id))
    return crack_mode, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 80000 + sample_id)
        crack_mode, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain4_crack_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "surface_seed": fractal_seed,
                "random_seed": seed + 80000 + sample_id,
                "crack_mode": crack_mode,
                "input": str(path.as_posix()),
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Domain 4 crack input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain4_crack_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 4 crack input files under {args.out_dir}")


if __name__ == "__main__":
    main()
