# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record one video showing Franka and the six lift2 grocery objects in one env.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift2/tools/record_groceries_initial.py --headless

This script is only a visualization helper. It does not register or change any
training task.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Record one initial-state video with six groceries on one table.")
parser.add_argument("--video-length", type=int, default=180, help="Number of environment steps to record.")
parser.add_argument("--warmup-steps", type=int, default=0, help="Zero-action warmup steps before recording.")
parser.add_argument("--progress-interval", type=int, default=30, help="Print progress every N recorded steps.")
parser.add_argument(
    "--output-dir",
    type=str,
    default="github/Special_Course/lift2/outputs/groceries_initial_video",
    help="Directory for video output.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(1.45, -1.25, 1.15),
    metavar=("X", "Y", "Z"),
    help="Camera eye position in world coordinates.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=(0.47, 0.0, 0.05),
    metavar=("X", "Y", "Z"),
    help="Camera target position in world coordinates.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.assets import RigidObjectCfg

import lift2.config.franka  # noqa: F401  # registers the lift2 Franka tasks
from lift2.config.franka.joint_pos_env_cfg import (
    FrankaBlackCubeLargeRandomLiftEnvCfg,
    GROCERY_OBJECTS,
)


OBJECT_ORDER = [
    "sugar_box",
    "tomato_soup_can",
    "mustard_bottle",
    "white_cube",
    "black_cube",
    "small_tomato_soup_can",
]

OBJECT_Y_POSITIONS = {
    "sugar_box": -0.30,
    "tomato_soup_can": -0.18,
    "mustard_bottle": -0.06,
    "white_cube": 0.07,
    "black_cube": 0.18,
    "small_tomato_soup_can": 0.30,
}

SCENE_ATTR_NAMES = {
    "sugar_box": "preview_sugar_box",
    "tomato_soup_can": "preview_tomato_soup_can",
    "mustard_bottle": "preview_mustard_bottle",
    "white_cube": "preview_white_cube",
    "small_tomato_soup_can": "preview_small_tomato_soup_can",
}


def _object_pose(object_name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    object_cfg = GROCERY_OBJECTS[object_name]
    z = object_cfg["initial_pos"][2]
    return (0.45, OBJECT_Y_POSITIONS[object_name], z), object_cfg["initial_rot"]


class FrankaSixGroceriesPreviewEnvCfg(FrankaBlackCubeLargeRandomLiftEnvCfg):
    """Preview env with one controllable black cube plus five extra grocery objects."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics = False
        self.commands.object_pose.debug_vis = False
        self.observations.policy.enable_corruption = False

        # Keep the primary env object deterministic so the preview layout is stable.
        black_pos, black_rot = _object_pose("black_cube")
        black_cfg = GROCERY_OBJECTS["black_cube"]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/black_cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=black_pos, rot=black_rot),
            spawn=black_cfg["spawn"],
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        # Add the remaining groceries as scene assets, arranged in a row on the table.
        for object_name, scene_attr_name in SCENE_ATTR_NAMES.items():
            object_pos, object_rot = _object_pose(object_name)
            object_cfg = GROCERY_OBJECTS[object_name]
            setattr(
                self.scene,
                scene_attr_name,
                RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                    init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_rot),
                    spawn=object_cfg["spawn"],
                ),
            )


def _unpack_step(step_result):
    """Handle both 4-value and 5-value gym step APIs."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    return step_result


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Recording one table with six groceries:", flush=True)
    for object_name in OBJECT_ORDER:
        object_pos, object_rot = _object_pose(object_name)
        object_cfg = GROCERY_OBJECTS[object_name]
        print(f"  - {object_name}: pos={object_pos}, rot={object_rot}, usd={object_cfg['spawn'].usd_path}", flush=True)
    print(f"[INFO] Video output directory: {output_dir}", flush=True)

    env_cfg = FrankaSixGroceriesPreviewEnvCfg()
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make("Isaac-Lift2-BlackCubeLargeRandom-Franka-v0", cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length + args_cli.warmup_steps,
        "name_prefix": "lift2_six_groceries_initial_state",
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    env.reset(seed=args_cli.seed)

    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            _unpack_step(env.step(actions))

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            _unpack_step(env.step(actions))
        if args_cli.progress_interval > 0 and (step + 1) % args_cli.progress_interval == 0:
            print(f"[INFO] Recorded {step + 1}/{args_cli.video_length} steps.", flush=True)

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved lift2 six-groceries initial video under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
