"""Generate Domain 3 crack gprMax input files with variable urban-fill clutter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from generate_domain3_cavity import (
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    OBJECT_X_MAX,
    OBJECT_X_MIN,
    OBJECT_Y_MAX,
    OBJECT_Y_MIN,
    SOURCE_Z,
    SAMPLES,
    box_line,
    clamp,
    domain_clutter,
    fmt,
)


FRACTAL_SEED_START = 1100
CRACK_MIN_THICKNESS = 0.02
CRACK_MAX_THICKNESS = 0.05


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 3 - Crack in Urban Fill with Clutter with Peplinski Modeling",
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
        "#material: 1.0 0.0 1.0 0.0 crack_material",
        "",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain3_crack_{sample_id:03d}_geometry"
    return [
        "",
        "#waveform: ricker 1 800e6 my_wave",
        f"#hertzian_dipole: z 0.1 0.8 {SOURCE_Z} my_wave",
        f"#rx: 0.3 0.8 {SOURCE_Z}",
        "#src_steps: 0.02 0 0",
        "#rx_steps: 0.02 0 0",
        f"#geometry_view: 0 0 0 2.0 1.0 0.002 0.002 0.002 0.002 {geometry_name} n",
    ]


def crack_box(x1: float, y_mid: float, x2: float, thickness: float) -> str:
    thickness = clamp(thickness, 0.021, 0.049)
    y1 = y_mid - thickness / 2
    y2 = y_mid + thickness / 2
    y1 = clamp(y1, OBJECT_Y_MIN, OBJECT_Y_MAX - thickness)
    y2 = y1 + thickness
    return box_line(x1, y1, x2, y2, "crack_material")


def horizontal_crack(rng: random.Random) -> list[str]:
    total_length = rng.uniform(0.35, 1.05)
    thickness = rng.uniform(CRACK_MIN_THICKNESS, CRACK_MAX_THICKNESS)
    x1 = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - total_length)
    x2 = x1 + total_length
    y_mid = rng.uniform(0.28, 0.62)

    lines = [crack_box(x1, y_mid, x2, thickness)]
    if rng.random() < 0.35:
        branch_length = rng.uniform(0.10, 0.24)
        branch_x1 = rng.uniform(x1 + total_length * 0.25, x2 - branch_length)
        direction = -1 if rng.random() < 0.5 else 1
        branch_y = y_mid + direction * rng.uniform(0.025, 0.055)
        lines.append(crack_box(branch_x1, branch_y, branch_x1 + branch_length, rng.uniform(0.02, 0.035)))
    return lines


def curved_crack(rng: random.Random) -> list[str]:
    segments = rng.randint(5, 10)
    total_length = rng.uniform(0.35, 0.95)
    x_start = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - total_length)
    seg_length = total_length / segments
    base_y = rng.uniform(0.30, 0.62)
    slope = rng.uniform(-0.07, 0.07)
    amplitude = rng.uniform(0.015, 0.055)
    phase = rng.uniform(0, 3.14159)
    thickness = rng.uniform(CRACK_MIN_THICKNESS, CRACK_MAX_THICKNESS)

    lines = []
    for idx in range(segments):
        x1 = x_start + idx * seg_length
        x2 = x_start + (idx + 1) * seg_length + rng.uniform(-0.008, 0.008)
        progress = idx / max(1, segments - 1)
        y_mid = base_y + slope * (progress - 0.5) + amplitude * math.sin(phase + progress * math.pi)
        y_mid += rng.uniform(-0.008, 0.008)
        lines.append(crack_box(x1, y_mid, x2, thickness * rng.uniform(0.88, 1.05)))
    return lines


def branched_crack(rng: random.Random) -> list[str]:
    lines = curved_crack(rng)
    branch_count = 1 if rng.random() < 0.75 else 2

    for _ in range(branch_count):
        branch_length = rng.uniform(0.12, 0.32)
        thickness = rng.uniform(CRACK_MIN_THICKNESS, 0.038)
        x1 = rng.uniform(OBJECT_X_MIN + 0.12, OBJECT_X_MAX - branch_length)
        y_base = rng.uniform(0.32, 0.60)
        direction = -1 if rng.random() < 0.5 else 1
        steps = rng.randint(2, 4)
        step_len = branch_length / steps
        for idx in range(steps):
            bx1 = x1 + idx * step_len
            bx2 = x1 + (idx + 1) * step_len
            y_mid = y_base + direction * (idx + 1) * rng.uniform(0.012, 0.025)
            lines.append(crack_box(bx1, y_mid, bx2, thickness))
    return lines


def crack_geometry(sample_id: int, rng: random.Random) -> tuple[str, list[str]]:
    mode = sample_id % 3
    if mode == 0:
        return "horizontal_crack", horizontal_crack(rng)
    if mode == 1:
        return "curved_crack", curved_crack(rng)
    return "branched_crack", branched_crack(rng)


def make_input(sample_id: int, rng: random.Random) -> tuple[str, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    crack_mode, crack_lines = crack_geometry(sample_id, rng)
    lines = header(fractal_seed)
    lines.extend(domain_clutter(rng))
    lines.append("")
    lines.extend(crack_lines)
    lines.extend(footer(sample_id))
    return crack_mode, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 50000 + sample_id)
        crack_mode, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain3_crack_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "surface_seed": fractal_seed,
                "random_seed": seed + 50000 + sample_id,
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
    parser = argparse.ArgumentParser(description="Generate Domain 3 crack input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain3_crack_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 3 crack input files under {args.out_dir}")


if __name__ == "__main__":
    main()
