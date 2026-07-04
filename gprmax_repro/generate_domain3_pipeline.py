"""Generate Domain 3 pipeline gprMax input files with variable urban-fill clutter."""

from __future__ import annotations

import argparse
import csv
import json
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
    clamp,
    cylinder_line,
    domain_clutter,
    fmt,
)


FRACTAL_SEED_START = 1200
PIPELINE_MIN_RADIUS = 0.05
PIPELINE_MAX_RADIUS = 0.28
PIPELINE_MAX_COUNT = 3


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 3 - Pipeline in Urban Fill with Clutter with Peplinski Modeling",
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
        "#material: 1.0 1e6 1.0 0.0 metal_material",
        "",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain3_pipeline_{sample_id:03d}_geometry"
    return [
        "",
        "#waveform: ricker 1 800e6 my_wave",
        f"#hertzian_dipole: z 0.1 0.8 {SOURCE_Z} my_wave",
        f"#rx: 0.3 0.8 {SOURCE_Z}",
        "#src_steps: 0.02 0 0",
        "#rx_steps: 0.02 0 0",
        f"#geometry_view: 0 0 0 2.0 1.0 0.002 0.002 0.002 0.002 {geometry_name} n",
    ]


def pipeline_radius(rng: random.Random) -> float:
    if rng.random() < 0.10:
        return rng.uniform(0.05, 0.095)
    if rng.random() < 0.72:
        return rng.uniform(0.10, 0.18)
    return rng.uniform(0.18, PIPELINE_MAX_RADIUS)


def pipeline_count(rng: random.Random) -> int:
    draw = rng.random()
    if draw < 0.56:
        return 1
    if draw < 0.86:
        return 2
    return 3


def pipeline_geometry(rng: random.Random) -> list[str]:
    target_count = pipeline_count(rng)
    pipes: list[tuple[float, float, float]] = []
    attempts = 0

    while len(pipes) < target_count and attempts < 250:
        attempts += 1
        radius = pipeline_radius(rng)
        x = rng.uniform(OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
        y_low = max(OBJECT_Y_MIN + radius, 0.24)
        y_high = OBJECT_Y_MAX - radius
        if y_low >= y_high:
            continue
        y = rng.uniform(y_low, y_high)

        if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 > radius + pr + 0.035 for px, py, pr in pipes):
            pipes.append((x, y, radius))

    if not pipes:
        radius = 0.11
        pipes.append((0.85, 0.45, radius))

    return [
        f"#cylinder: {fmt(clamp(x, OBJECT_X_MIN + r, OBJECT_X_MAX - r))} "
        f"{fmt(clamp(y, OBJECT_Y_MIN + r, OBJECT_Y_MAX - r))} 0 "
        f"{fmt(clamp(x, OBJECT_X_MIN + r, OBJECT_X_MAX - r))} "
        f"{fmt(clamp(y, OBJECT_Y_MIN + r, OBJECT_Y_MAX - r))} 0.002 "
        f"{fmt(r)} metal_material"
        for x, y, r in pipes
    ]


def make_input(sample_id: int, rng: random.Random) -> tuple[int, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    pipeline_lines = pipeline_geometry(rng)
    lines = header(fractal_seed)
    lines.extend(domain_clutter(rng))
    lines.append("")
    lines.extend(pipeline_lines)
    lines.extend(footer(sample_id))
    return len(pipeline_lines), "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 60000 + sample_id)
        pipeline_count_value, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain3_pipeline_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "surface_seed": fractal_seed,
                "random_seed": seed + 60000 + sample_id,
                "pipeline_count": pipeline_count_value,
                "input": str(path.as_posix()),
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Domain 3 pipeline input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain3_pipeline_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 3 pipeline input files under {args.out_dir}")


if __name__ == "__main__":
    main()
