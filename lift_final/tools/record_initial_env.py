# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record a short headless video of the initial lift_final environment.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift_final/tools/record_initial_env.py --headless

The script creates one environment, resets it, applies zero actions for a few seconds, and saves an mp4 under
``outputs/initial_env_video`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Record a short video of the lift_final initial environment.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Groceries-Franka-Play-v0", help="Gym task id to preview.")
parser.add_argument("--video-length", type=int, default=180, help="Number of environment steps to record.")
parser.add_argument("--output-dir", type=str, default="outputs/initial_env_video", help="Directory for video output.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
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

import lift_final.config.franka  # noqa: F401  # registers the lift_final Franka tasks
from lift_final.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg_PLAY


def _unpack_step(step_result):
    """Handle both 4-value and 5-value gym step APIs."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    return step_result


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = FrankaCubeLiftEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": "lift_final_initial_env",
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.reset(seed=args_cli.seed)
    action_shape = env.action_space.shape
    actions = torch.zeros(action_shape, device=env.unwrapped.device)

    for _ in range(args_cli.video_length):
        with torch.inference_mode():
            _unpack_step(env.step(actions))

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved initial-environment video under: {output_dir}")


if __name__ == "__main__":
    main()
