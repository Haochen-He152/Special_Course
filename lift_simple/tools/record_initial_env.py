# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a short headless video of the initial lift_simple grocery environment.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift_simple/tools/record_initial_env.py --headless

The script creates one environment with the current lift_simple object collection.

It resets the scene, applies zero actions for a few seconds, and saves an mp4 under
``outputs/initial_env_video`` by default. The server must support Isaac Sim headless camera rendering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Record a short video of the current lift_simple initial environment.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Simple-Franka-Play-v0", help="Gym task id to preview.")
parser.add_argument("--video-length", type=int, default=180, help="Number of environment steps to record.")
parser.add_argument("--warmup-steps", type=int, default=0, help="Extra zero-action steps before the recorded loop.")
parser.add_argument("--progress-interval", type=int, default=30, help="Print progress every N recorded steps.")
parser.add_argument("--output-dir", type=str, default="outputs/initial_env_video", help="Directory for video output.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument(
    "--random-actions",
    action="store_true",
    help="Use small random actions instead of zero actions. Useful for checking that recording progresses.",
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

"""Rest everything follows."""

import gymnasium as gym
import torch

import lift_simple.config.franka  # noqa: F401  # registers the lift_simple Franka tasks
from lift_simple.config.franka.joint_pos_env_cfg import (
    GROCERY_INITIAL_POSES,
    GROCERIES,
    FrankaCubeLiftEnvCfg_PLAY,
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

    print("[INFO] Current lift_simple objects:", flush=True)
    for object_name, spawn_cfg in GROCERIES.items():
        print(f"  - {object_name}: {spawn_cfg.usd_path}", flush=True)
        print(f"    initial position: {GROCERY_INITIAL_POSES[object_name]}", flush=True)
    print(f"[INFO] Video output directory: {output_dir}", flush=True)

    env_cfg = FrankaCubeLiftEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.object_pose.debug_vis = False

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length + args_cli.warmup_steps,
        "name_prefix": "lift_simple_initial_env",
        "disable_logger": True,
    }
    print("[INFO] Recording video with Gymnasium RecordVideo wrapper.", flush=True)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    action_shape = env.action_space.shape
    actions = torch.zeros(action_shape, device=env.unwrapped.device)

    print("[INFO] Resetting environment...", flush=True)
    env.reset(seed=args_cli.seed)

    print(f"[INFO] Running {args_cli.warmup_steps} warmup steps...", flush=True)
    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            if args_cli.random_actions:
                actions = 0.05 * (2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0)
            _unpack_step(env.step(actions))

    print(f"[INFO] Recording {args_cli.video_length} steps...", flush=True)
    for step in range(args_cli.video_length):
        with torch.inference_mode():
            if args_cli.random_actions:
                actions = 0.05 * (2.0 * torch.rand(action_shape, device=env.unwrapped.device) - 1.0)
            _unpack_step(env.step(actions))
        if args_cli.progress_interval > 0 and (step + 1) % args_cli.progress_interval == 0:
            print(f"[INFO] Recorded {step + 1}/{args_cli.video_length} steps.", flush=True)

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved initial-environment video under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
