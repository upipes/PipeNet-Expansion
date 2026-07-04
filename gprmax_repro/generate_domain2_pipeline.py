"""Generate Domain 2 pipeline gprMax input files for manual inspection."""

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
SOURCE_Z = 0
SAMPLES = 100
FRACTAL_SEED_START = 700
OBJECT_X_MIN = 0.12
OBJECT_X_MAX = 1.5
PIPE_RADIUS_MIN = 0.05
PIPE_RADIUS_MAX = 0.30


def fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def header(fractal_seed: int) -> list[str]:
    return [
        "#title: Domain 2 - Pipeline in Wet Silty Clay with Peplinski Modeling",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        "#dx_dy_dz: 0.002 0.002 0.002",
        "#time_window: 20e-9",
        "",
        "#soil_peplinski: 0.2 0.7 1.9 2.7 0.25 0.45 my_soil",
        f"#fractal_box: 0 0 0 2.0 0.8 0.002 1.9 1 1 1 30 my_soil mysoil {fractal_seed}",
        "",
        "#material: 1.0 1e6 1.0 0.0 metal_material",
    ]


def footer(sample_id: int) -> list[str]:
    geometry_name = f"geometry_views/domain2_pipeline_{sample_id:03d}_geometry"
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
    return (
        f"#cylinder: {fmt(x)} {fmt(y)} 0 "
        f"{fmt(x)} {fmt(y)} 0.002 {fmt(radius)} metal_material"
    )


def overlaps(candidate: tuple[float, float, float], pipes: list[tuple[float, float, float]]) -> bool:
    x, y, radius = candidate
    for px, py, pr in pipes:
        distance = math.hypot(x - px, y - py)
        if distance <= radius + pr + 0.035:
            return True
    return False


def sample_pipe(rng: random.Random, large_ok: bool) -> tuple[float, float, float]:
    if large_ok and rng.random() < 0.18:
        radius = rng.uniform(0.18, PIPE_RADIUS_MAX)
    else:
        radius = rng.uniform(PIPE_RADIUS_MIN, 0.18)

    x = rng.uniform(OBJECT_X_MIN + radius, OBJECT_X_MAX - radius)
    y = rng.uniform(max(0.16, radius + 0.06), GROUND_TOP - radius - 0.08)
    y = clamp(y, radius + 0.06, GROUND_TOP - radius - 0.08)
    return x, y, radius


def pipeline_geometry(rng: random.Random) -> tuple[int, list[str]]:
    pipe_count = rng.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.18, 0.07], k=1)[0]
    pipes: list[tuple[float, float, float]] = []

    attempts = 0
    while len(pipes) < pipe_count and attempts < 500:
        attempts += 1
        candidate = sample_pipe(rng, large_ok=(pipe_count <= 2))
        if not overlaps(candidate, pipes):
            pipes.append(candidate)

    if not pipes:
        pipes.append(sample_pipe(rng, large_ok=True))

    return len(pipes), [cylinder_line(x, y, radius) for x, y, radius in pipes]


def make_input(sample_id: int, rng: random.Random) -> tuple[int, str]:
    fractal_seed = FRACTAL_SEED_START + sample_id
    pipe_count, geometry = pipeline_geometry(rng)
    lines = header(fractal_seed)
    lines.extend(geometry)
    lines.extend(footer(sample_id))
    return pipe_count, "\n".join(lines)


def generate(out_dir: Path, samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "geometry_views").mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample_id in range(samples):
        rng = random.Random(seed + 30000 + sample_id)
        pipe_count, content = make_input(sample_id, rng)
        fractal_seed = FRACTAL_SEED_START + sample_id
        path = out_dir / f"domain2_pipeline_{sample_id:03d}.in"
        path.write_text(content, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "fractal_seed": fractal_seed,
                "random_seed": seed + 30000 + sample_id,
                "pipe_count": pipe_count,
                "input": str(path.as_posix()),
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Domain 2 pipeline input files.")
    parser.add_argument("--out-dir", type=Path, default=Path("gprMax-master/domain2_pipeline_inputs"))
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out_dir, args.samples, args.seed)
    print(f"Generated {args.samples} Domain 2 pipeline input files under {args.out_dir}")


if __name__ == "__main__":
    main()
