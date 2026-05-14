# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Read original bounding-box sizes for selected Isaac Sim USD assets.

Run this script in an Isaac Sim / USD Python environment.

Examples:
    ./isaaclab.sh -p usd_size_reader/read_usd_sizes.py
    ./isaaclab.sh -p usd_size_reader/read_usd_sizes.py --nucleus-dir /path/to/Isaac
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class UsdAsset:
    name: str
    relative_paths: tuple[str, ...]


@dataclass(frozen=True)
class UsdInfo:
    size: tuple[float, float, float]
    mass_kg: float | None
    densities: tuple[float, ...]
    has_rigid_body: bool
    has_collision: bool


def ycb_asset(file_name: str) -> tuple[str, ...]:
    """Return the selected YCB asset path."""
    return (f"Props/YCB/Axis_Aligned/{file_name}",)


USD_ASSETS = [
    UsdAsset("OBJECT_A_master_chef_can", ycb_asset("002_master_chef_can.usd")),
    UsdAsset("OBJECT_B_sugar_box", ycb_asset("004_sugar_box.usd")),
    UsdAsset("OBJECT_C_tomato_soup_can", ycb_asset("005_tomato_soup_can.usd")),
    UsdAsset("OBJECT_D_mustard_bottle", ycb_asset("006_mustard_bottle.usd")),
    UsdAsset("OBJECT_E_tuna_fish_can", ycb_asset("007_tuna_fish_can.usd")),
    UsdAsset("OBJECT_F_scissors", ycb_asset("037_scissors.usd")),
    UsdAsset("OBJECT_G_large_marker", ycb_asset("040_large_marker.usd")),
    UsdAsset("KLT_Bin", ("Props/KLT_Bin/small_KLT.usd",)),
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
            "Could not import ISAAC_NUCLEUS_DIR from Isaac Lab. "
            "Run this script with Isaac Lab's Python launcher or pass --nucleus-dir."
        ) from exc


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


def _read_float_attr(attr) -> float | None:
    value = attr.Get()
    return None if value is None else float(value)


def read_usd_info(usd_path: str, purposes: list[str]) -> UsdInfo:
    """Read bounding-box and authored physics information from a USD file."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    prim = stage.GetDefaultPrim()
    if not prim:
        prim = stage.GetPseudoRoot()

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    bbox = bbox_cache.ComputeWorldBound(prim)
    aligned_range = bbox.ComputeAlignedRange()
    bbox_size = aligned_range.GetSize()

    masses = []
    densities = set()
    has_rigid_body = False
    has_collision = False

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass = _read_float_attr(mass_api.GetMassAttr())
            density = _read_float_attr(mass_api.GetDensityAttr())
            if mass is not None:
                masses.append(mass)
            if density is not None:
                densities.add(density)

        has_rigid_body = has_rigid_body or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_collision = has_collision or prim.HasAPI(UsdPhysics.CollisionAPI)

    return UsdInfo(
        size=(float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])),
        mass_kg=sum(masses) if masses else None,
        densities=tuple(sorted(densities)),
        has_rigid_body=has_rigid_body,
        has_collision=has_collision,
    )


def format_optional_float(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "-"


def format_densities(densities: tuple[float, ...]) -> str:
    return ",".join(f"{density:.6g}" for density in densities) if densities else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read original USD bounding-box sizes.")
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on the first USD that cannot be opened. By default, failed assets are reported and skipped.",
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
    print(
        f"{'asset':<28} {'size_x_m':>12} {'size_y_m':>12} {'size_z_m':>12}"
        f" {'mass_kg':>12} {'density':>14} {'rigid_body':>10} {'collision':>10}  usd_path"
    )
    print("-" * 160)

    for asset in USD_ASSETS:
        errors = []
        for relative_path in asset.relative_paths:
            usd_path = f"{nucleus_dir}/{relative_path}"
            try:
                usd_info = read_usd_info(usd_path, purposes)
                size_x, size_y, size_z = usd_info.size
                print(
                    f"{asset.name:<28} {size_x:12.6f} {size_y:12.6f} {size_z:12.6f}"
                    f" {format_optional_float(usd_info.mass_kg):>12}"
                    f" {format_densities(usd_info.densities):>14}"
                    f" {str(usd_info.has_rigid_body):>10}"
                    f" {str(usd_info.has_collision):>10}  {usd_path}"
                )
                break
            except Exception as exc:
                errors.append((usd_path, exc))
        else:
            if args.strict:
                raise RuntimeError(f"All candidate USD paths failed for {asset.name}: {errors}")
            print(
                f"{asset.name:<28} {'FAILED':>12} {'FAILED':>12} {'FAILED':>12}"
                f" {'FAILED':>12} {'FAILED':>14} {'FAILED':>10} {'FAILED':>10}  {asset.relative_paths[0]}"
            )
            for usd_path, exc in errors:
                print(f"{'':<28} tried: {usd_path}")
                print(f"{'':<28} error: {exc}")

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
