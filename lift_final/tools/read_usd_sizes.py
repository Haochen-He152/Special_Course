# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print original bounding-box sizes for the USD assets used by lift_final.

Run this script in an Isaac Sim / USD Python environment, for example with Isaac Lab's python launcher.

Examples:
    ./isaaclab.sh -p lift_final/tools/read_usd_sizes.py
    ./isaaclab.sh -p lift_final/tools/read_usd_sizes.py --nucleus-dir /Isaac
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class UsdAsset:
    name: str
    relative_path: str


USD_ASSETS = [
    UsdAsset("OBJECT_A_sugar_box", "Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"),
    UsdAsset("OBJECT_B_cracker_box", "Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"),
    UsdAsset("OBJECT_C_tomato_soup_can", "Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"),
    UsdAsset("OBJECT_D_mustard_bottle", "Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"),
    UsdAsset("OBJECT_E_tuna_fish_can", "Props/YCB/Axis_Aligned_Physics/007_tuna_fish_can.usd"),
    UsdAsset("OBJECT_F_bleach_cleanser", "Props/YCB/Axis_Aligned_Physics/021_bleach_cleanser.usd"),
    UsdAsset("OBJECT_G_large_marker", "Props/YCB/Axis_Aligned_Physics/040_large_marker.usd"),
    UsdAsset("KLT_Bin", "Props/KLT_Bin/small_KLT.usd"),
]


def resolve_nucleus_dir(cli_nucleus_dir: str | None) -> str:
    """Resolve the Isaac Nucleus root used by Isaac Lab."""
    if cli_nucleus_dir:
        return cli_nucleus_dir.rstrip("/\\")

    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        return ISAAC_NUCLEUS_DIR.rstrip("/\\")
    except Exception as exc:
        raise RuntimeError(
            "Could not import ISAAC_NUCLEUS_DIR from Isaac Lab. Run in an Isaac Lab environment "
            "or pass --nucleus-dir explicitly."
        ) from exc


def compute_usd_bbox_size(usd_path: str, purposes: list[str]) -> tuple[float, float, float]:
    """Compute world-aligned bounding-box size for a USD file."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    prim = stage.GetDefaultPrim()
    if not prim:
        prim = stage.GetPseudoRoot()

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned_range = bbox.ComputeAlignedRange()
    size = aligned_range.GetSize()
    return float(size[0]), float(size[1]), float(size[2])


def resolve_purposes(purpose_names: list[str]) -> list[str]:
    """Convert CLI purpose names to USD tokens."""
    from pxr import UsdGeom

    purposes = []
    for purpose in purpose_names:
        if purpose == "default":
            purposes.append(UsdGeom.Tokens.default_)
        else:
            purposes.append(getattr(UsdGeom.Tokens, purpose))
    return purposes


def main() -> None:
    parser = argparse.ArgumentParser(description="Read original USD bounding-box sizes for lift_final assets.")
    parser.add_argument(
        "--nucleus-dir",
        default=None,
        help="Override Isaac Nucleus root. Defaults to isaaclab.utils.assets.ISAAC_NUCLEUS_DIR.",
    )
    parser.add_argument(
        "--purposes",
        nargs="+",
        default=["default", "render"],
        choices=["default", "render", "proxy", "guide"],
        help="USD purposes included in the bounding-box computation.",
    )
    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
    except Exception:
        args = parser.parse_args()

    nucleus_dir = resolve_nucleus_dir(args.nucleus_dir)
    purposes = resolve_purposes(args.purposes)

    print(f"Isaac Nucleus dir: {nucleus_dir}")
    print(f"Purposes: {', '.join(args.purposes)}")
    print()
    print(f"{'asset':<28} {'size_x_m':>12} {'size_y_m':>12} {'size_z_m':>12}  usd_path")
    print("-" * 110)

    for asset in USD_ASSETS:
        usd_path = f"{nucleus_dir}/{asset.relative_path}"
        size_x, size_y, size_z = compute_usd_bbox_size(usd_path, purposes)
        print(f"{asset.name:<28} {size_x:12.6f} {size_y:12.6f} {size_z:12.6f}  {usd_path}")

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
