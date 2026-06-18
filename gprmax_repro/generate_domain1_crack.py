"""Generate Domain 1 crack gprMax input files for manual inspection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


DOMAIN_X = 2.0
DOMAIN_Y = 1.0
DOMAIN_Z = 0.002
GROUND_TOP = 0.8
SOURCE_Z = 0.001
SAMPLES = 100
FRACTAL_SEED_START = 100
OBJECT_X_MIN = 0.12
OBJECT_X_MAX = 1.5
CRACK_WIDTH_MIN = 0.02
CRACK_WIDTH_MAX = 0.05


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 1 - Crack in Ideal Sandy Loam with Peplinski Modeling",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 15e-9",
        "",
        "#soil_peplinski: 0.8 0.1 1.5 2.66 0.01 0.05 my_soil",
        f"#fractal_box: 0 0 0 2.0 0.8 0.002 1.3 1 1 1 20 my_soil mysoil {fractal_seed}",
        "",
        "#material: 1.0 0.0 1.0 0.0 crack_material",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain1_crack_{sample_id:03d}_geometry"
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


def box_line(x1: float, y_center: float, length: float, width: float) -> str:
    x1 = clamp(x1, OBJECT_X_MIN, OBJECT_X_MAX - 0.04)
    x2 = clamp(x1 + length, x1 + 0.035, OBJECT_X_MAX)
    half_width = width / 2
    y1 = clamp(y_center - half_width, 0.08, GROUND_TOP - 0.08)
    y2 = clamp(y_center + half_width, y1 + CRACK_WIDTH_MIN, GROUND_TOP - 0.04)
    return f"#box: {fmt(x1)} {fmt(y1)} 0 {fmt(x2)} {fmt(y2)} 0.002 crack_material"


def horizontal_crack(rng: random.Random) -> list[str]:
    length = rng.uniform(0.35, 0.85)
    x1 = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - length)
    y = rng.uniform(0.20, 0.62)
    width = rng.uniform(CRACK_WIDTH_MIN, CRACK_WIDTH_MAX)
    return [box_line(x1, y, length, width)]


def gently_curved_crack(rng: random.Random, direction: int) -> list[str]:
    total_length = rng.uniform(0.42, 0.90)
    segments = rng.randint(6, 10)
    segment_length = total_length / segments
    x_start = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - total_length)
    y_base = rng.uniform(0.24, 0.58)
    amplitude = rng.uniform(0.025, 0.10)
    width = rng.uniform(CRACK_WIDTH_MIN, CRACK_WIDTH_MAX)
    phase = rng.uniform(-0.5, 0.5)

    lines = []
    for idx in range(segments):
        x1 = x_start + idx * segment_length
        t = idx / max(1, segments - 1)
        curve = direction * amplitude * (t - 0.5)
        smooth = 0.025 * math.sin(2 * math.pi * (t + phase))
        y = y_base + curve + smooth
        lines.append(box_line(x1, y, segment_length * rng.uniform(0.92, 1.08), width * rng.uniform(0.9, 1.1)))
    return lines


def branched_crack(rng: random.Random) -> list[str]:
    lines = gently_curved_crack(rng, rng.choice((-1, 1)))
    branch_count = rng.randint(1, 3)

    for _ in range(branch_count):
        base = rng.choice(lines)
        parts = base.split()
        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[4])
        y2 = float(parts[5])
        bx = rng.uniform(x1, x2)
        by = (y1 + y2) / 2
        branch_length = rng.uniform(0.10, 0.24)
        branch_width = rng.uniform(CRACK_WIDTH_MIN, CRACK_WIDTH_MAX)
        direction = rng.choice((-1, 1))
        pieces = rng.randint(2, 4)
        for idx in range(pieces):
            px = bx + idx * branch_length / pieces
            py = by + direction * (idx + 1) * rng.uniform(0.012, 0.028)
            lines.append(box_line(px, py, branch_length / pieces, branch_width))
    return lines


def segmented_near_horizontal(rng: random.Random) -> list[str]:
    total_length = rng.uniform(0.40, 0.80)
    segments = rng.randint(5, 9)
    segment_length = total_length / segments
    x_start = rng.uniform(OBJECT_X_MIN, OBJECT_X_MAX - total_length)
    y = rng.uniform(0.20, 0.62)
    width = rng.uniform(CRACK_WIDTH_MIN, CRACK_WIDTH_MAX)
    slope = rng.uniform(-0.06, 0.06)

    lines = []
    for idx in range(segments):
        x1 = x_start + idx * segment_length
        jitter = rng.uniform(-0.018, 0.018)
        y_center = y + slope * idx / max(1, segments - 1) + jitter
        lines.append(box_line(x1, y_center, segment_length * rng.uniform(0.80, 1.05), width * rng.uniform(0.9, 1.1)))
    return lines


def crack_geometry(sample_id: int, rng: random.Random) -> tuple[str, list[str]]:
    mode = sample_id % 4
    if mode == 0:
        return "horizontal", horizontal_crack(rng)
    if mode == 1:
        return "gently_curved_up", gently_curved_crack(rng, 1)
    if mode == 2:
        return "gently_curved_down", gently_curved_crack(rng, -1)
    return "branched_or_segmented", branched_crack(rng) if rng.random() < 0.55 else segmented_near_horizontal(rng)


def make_input(sample_id: int, rng: random.Random) -> tuple[str, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    shape_mode, geometry = crack_geometry(sample_id, rng)
    lines = header(fractal_seed)
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
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain1_crack_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
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
    parser = argparse.ArgumentParser(description="Generate Domain 1 crack input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain1_crack_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 1 crack input files under {args.out_dir}")


if __name__ == "__main__":
    main()
