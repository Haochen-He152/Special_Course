# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""List USD assets under Isaac's YCB Axis_Aligned_Physics folder.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift_final/tools/list_axis_aligned_physics_assets.py

The script first tries ``omni.client`` when available. If that module is not present, it falls back to HTTP
listing for the public Isaac assets S3 URL used by ``ISAAC_NUCLEUS_DIR``.
"""

from __future__ import annotations

import argparse
import posixpath
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


YCB_PHYSICS_RELATIVE_DIR = "Props/YCB/Axis_Aligned_Physics"


def resolve_nucleus_dir(cli_nucleus_dir: str | None) -> str:
    """Resolve the Isaac Nucleus root used by Isaac Lab."""
    if cli_nucleus_dir:
        return cli_nucleus_dir.rstrip("/")

    try:
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        return ISAAC_NUCLEUS_DIR.rstrip("/")
    except Exception as exc:
        raise RuntimeError(
            "Could not import ISAAC_NUCLEUS_DIR from Isaac Lab. Run with ./isaaclab.sh -p or pass --nucleus-dir."
        ) from exc


def list_with_omni_client(asset_dir: str) -> list[str]:
    """List assets with omni.client if it is available in the current Isaac Sim environment."""
    import omni.client

    result, entries = omni.client.list(asset_dir)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"omni.client.list failed for {asset_dir}: {result}")

    return sorted(entry.relative_path for entry in entries if entry.relative_path.endswith(".usd"))


def list_public_s3_http(asset_dir: str) -> list[str]:
    """List public S3-backed Isaac asset URLs with the S3 list-bucket API."""
    parsed = urllib.parse.urlparse(asset_dir)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError(f"HTTP fallback only supports http(s) asset dirs, got: {asset_dir}")

    bucket_url = f"{parsed.scheme}://{parsed.netloc}/"
    prefix = parsed.path.strip("/") + "/"
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "delimiter": "/"})
    list_url = f"{bucket_url}?{query}"

    with urllib.request.urlopen(list_url, timeout=30) as response:
        xml_text = response.read()

    root = ET.fromstring(xml_text)
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"

    assets = []
    for contents in root.findall(f"{namespace}Contents"):
        key = contents.findtext(f"{namespace}Key")
        if key and key.endswith(".usd"):
            assets.append(posixpath.basename(key))
    return sorted(assets)


def main() -> None:
    parser = argparse.ArgumentParser(description="List USD files in Props/YCB/Axis_Aligned_Physics.")
    parser.add_argument(
        "--nucleus-dir",
        default=None,
        help="Override Isaac Nucleus root. Defaults to isaaclab.utils.assets.ISAAC_NUCLEUS_DIR.",
    )
    parser.add_argument("--show-paths", action="store_true", help="Print full USD paths instead of only file names.")
    args = parser.parse_args()

    nucleus_dir = resolve_nucleus_dir(args.nucleus_dir)
    asset_dir = f"{nucleus_dir}/{YCB_PHYSICS_RELATIVE_DIR}"

    try:
        assets = list_with_omni_client(asset_dir)
        method = "omni.client"
    except Exception as omni_exc:
        try:
            assets = list_public_s3_http(asset_dir)
            method = "public S3 HTTP"
        except Exception as http_exc:
            raise RuntimeError(
                f"Could not list assets under {asset_dir}.\n"
                f"omni.client error: {omni_exc}\n"
                f"HTTP fallback error: {http_exc}"
            ) from http_exc

    print(f"Isaac Nucleus dir: {nucleus_dir}")
    print(f"Asset directory: {asset_dir}")
    print(f"Listing method: {method}")
    print(f"USD asset count: {len(assets)}")
    print()

    for index, asset_name in enumerate(assets, start=1):
        if args.show_paths:
            print(f"{index:03d}  {asset_dir}/{asset_name}")
        else:
            print(f"{index:03d}  {asset_name}")


if __name__ == "__main__":
    main()
