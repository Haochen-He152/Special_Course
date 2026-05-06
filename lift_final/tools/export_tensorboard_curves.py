# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export TensorBoard scalar curves from one skrl run to CSV and PNG files.

Example:
    python github/Special_Course/lift_final/tools/export_tensorboard_curves.py

Or with Isaac Lab's Python:
    ./isaaclab.sh -p github/Special_Course/lift_final/tools/export_tensorboard_curves.py

By default this exports the run:
    logs/skrl/franka_lift/2026-05-06_20-30-47_ppo_torch

The script reads all scalar tags from TensorBoard event files and writes one CSV and one PNG per tag.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_LOGDIR = Path(
    "/root/autodl-tmp/projects/IsaacLab/logs/skrl/franka_lift/2026-05-06_20-30-47_ppo_torch"
)
DEFAULT_OUTPUT_DIR = DEFAULT_LOGDIR / "exported_curves"


def sanitize_filename(name: str) -> str:
    """Convert a TensorBoard tag into a safe file name."""
    name = name.strip().replace("\\", "/")
    name = re.sub(r"[^A-Za-z0-9._/-]+", "_", name)
    name = name.replace("/", "__")
    return name.strip("._") or "curve"


def export_scalar_csv(output_path: Path, scalar_events) -> None:
    """Write scalar events to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "wall_time", "value"])
        for event in scalar_events:
            writer.writerow([event.step, event.wall_time, event.value])


def export_scalar_png(output_path: Path, tag: str, scalar_events) -> None:
    """Write scalar events to a PNG line plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(steps, values, linewidth=1.6)
    ax.set_title(tag)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all TensorBoard scalar curves from one run.")
    parser.add_argument(
        "--logdir",
        type=Path,
        default=DEFAULT_LOGDIR,
        help="Path to one TensorBoard/skrl run directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV and PNG files will be saved.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only export CSV files. Useful if matplotlib is not installed.",
    )
    args = parser.parse_args()

    if not args.logdir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {args.logdir}")

    event_files = sorted(args.logdir.rglob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {args.logdir}")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.output_dir / "csv"
    png_dir = args.output_dir / "png"
    csv_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_png:
        png_dir.mkdir(parents=True, exist_ok=True)

    accumulator = EventAccumulator(str(args.logdir))
    accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    if not scalar_tags:
        raise RuntimeError(f"No scalar curves found in TensorBoard logs under: {args.logdir}")

    print(f"[INFO] Log directory: {args.logdir}")
    print(f"[INFO] Event files: {len(event_files)}")
    print(f"[INFO] Scalar curves: {len(scalar_tags)}")
    print(f"[INFO] Output directory: {args.output_dir}")

    exported_count = 0
    for tag in scalar_tags:
        scalar_events = accumulator.Scalars(tag)
        if not scalar_events:
            continue

        safe_name = sanitize_filename(tag)
        export_scalar_csv(csv_dir / f"{safe_name}.csv", scalar_events)
        if not args.no_png:
            export_scalar_png(png_dir / f"{safe_name}.png", tag, scalar_events)
        exported_count += 1

    print(f"[INFO] Exported {exported_count} scalar curves.")
    print(f"[INFO] CSV files: {csv_dir}")
    if not args.no_png:
        print(f"[INFO] PNG files: {png_dir}")


if __name__ == "__main__":
    main()
