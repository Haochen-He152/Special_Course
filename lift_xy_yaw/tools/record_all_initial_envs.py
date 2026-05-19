# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record short headless videos for all lift_xy_yaw initial object scenes.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift_xy_yaw/tools/record_all_initial_envs.py --headless

The script records one video for each single-object Franka lift task and stores
the videos under ``outputs/initial_env_videos/<object_name>`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Record initial videos for all lift_xy_yaw objects.")
parser.add_argument("--video-length", type=int, default=180, help="Number of environment steps to record per object.")
parser.add_argument("--warmup-steps", type=int, default=0, help="Zero-action warmup steps before recording.")
parser.add_argument("--progress-interval", type=int, default=30, help="Print progress every N recorded steps.")
parser.add_argument(
    "--output-dir",
    type=str,
    default="github/Special_Course/lift_xy_yaw/outputs/initial_env_videos",
    help="Root directory for video outputs.",
)
parser.add_argument("--seed", type=int, default=42, help="Base environment seed.")
parser.add_argument(
    "--objects",
    type=str,
    nargs="*",
    default=[
        "SugarBox",
        "TomatoSoupCan",
        "MustardBottle",
        "WhiteCube",
        "BlackCube",
        "SmallTomatoSoupCan",
    ],
    help="Object task names to record. Use names such as SugarBox or TomatoSoupCan.",
)
parser.add_argument(
    "--random-actions",
    action="store_true",
    help="Use small random actions instead of zero actions.",
)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(1.7, -1.2, 1.35),
    metavar=("X", "Y", "Z"),
    help="Camera eye position in world coordinates.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=(0.5, 0.0, 0.05),
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

import lift_xy_yaw.config.franka  # noqa: F401  # registers the lift_xy_yaw Franka tasks
from lift_xy_yaw.config.franka.joint_pos_env_cfg import (
    FrankaBlackCubeLiftEnvCfg_PLAY,
    FrankaMustardBottleLiftEnvCfg_PLAY,
    FrankaSmallTomatoSoupCanLiftEnvCfg_PLAY,
    FrankaSugarBoxLiftEnvCfg_PLAY,
    FrankaTomatoSoupCanLiftEnvCfg_PLAY,
    FrankaWhiteCubeLiftEnvCfg_PLAY,
    SINGLE_OBJECTS,
)


TASKS = {
    "SugarBox": ("Isaac-Lift-XY-Yaw-SugarBox-Franka-Play-v0", FrankaSugarBoxLiftEnvCfg_PLAY, "sugar_box"),
    "TomatoSoupCan": (
        "Isaac-Lift-XY-Yaw-TomatoSoupCan-Franka-Play-v0",
        FrankaTomatoSoupCanLiftEnvCfg_PLAY,
        "tomato_soup_can",
    ),
    "MustardBottle": (
        "Isaac-Lift-XY-Yaw-MustardBottle-Franka-Play-v0",
        FrankaMustardBottleLiftEnvCfg_PLAY,
        "mustard_bottle",
    ),
    "WhiteCube": ("Isaac-Lift-XY-Yaw-WhiteCube-Franka-Play-v0", FrankaWhiteCubeLiftEnvCfg_PLAY, "white_cube"),
    "BlackCube": ("Isaac-Lift-XY-Yaw-BlackCube-Franka-Play-v0", FrankaBlackCubeLiftEnvCfg_PLAY, "black_cube"),
    "SmallTomatoSoupCan": (
        "Isaac-Lift-XY-Yaw-SmallTomatoSoupCan-Franka-Play-v0",
        FrankaSmallTomatoSoupCanLiftEnvCfg_PLAY,
        "small_tomato_soup_can",
    ),
}


def _unpack_step(step_result):
    """Handle both 4-value and 5-value gym step APIs."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    return step_result


def _record_one_object(object_task_name: str, output_root: Path) -> None:
    if object_task_name not in TASKS:
        valid_names = ", ".join(TASKS)
        raise ValueError(f"Unknown object task '{object_task_name}'. Valid names: {valid_names}")

    task_id, env_cfg_cls, object_key = TASKS[object_task_name]
    output_dir = output_root / object_key
    output_dir.mkdir(parents=True, exist_ok=True)

    object_cfg = SINGLE_OBJECTS[object_key]
    print(f"[INFO] Recording {object_task_name}", flush=True)
    print(f"  task id: {task_id}", flush=True)
    print(f"  usd path: {object_cfg['spawn'].usd_path}", flush=True)
    print(f"  initial position: {object_cfg['initial_pos']}", flush=True)
    print(f"  output directory: {output_dir}", flush=True)

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.object_pose.debug_vis = False

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length + args_cli.warmup_steps,
        "name_prefix": f"lift_xy_yaw_{object_key}_initial_env",
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    action_shape = env.action_space.shape
    actions = torch.zeros(action_shape, device=env.unwrapped.device)

    env.reset(seed=args_cli.seed)

    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            if args_cli.random_actions:
                actions = 0.05 * (2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0)
            _unpack_step(env.step(actions))

    for step in range(args_cli.video_length):
        with torch.inference_mode():
            if args_cli.random_actions:
                actions = 0.05 * (2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0)
            _unpack_step(env.step(actions))
        if args_cli.progress_interval > 0 and (step + 1) % args_cli.progress_interval == 0:
            print(f"[INFO] {object_task_name}: recorded {step + 1}/{args_cli.video_length} steps.", flush=True)

    env.close()


def main() -> None:
    output_root = Path(args_cli.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for object_task_name in args_cli.objects:
        _record_one_object(object_task_name, output_root)

    simulation_app.close()
    print(f"[INFO] Saved all initial-environment videos under: {output_root}", flush=True)


if __name__ == "__main__":
    main()
