# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record one video showing six parallel envs with one different object per table.

This preview uses the lift_simple object collection so that all six object assets
exist in a single Gym task. After reset, the script manually assigns one visible
target object to each of the six envs and hides the non-target objects high above
their own env origins.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift_single/tools/record_six_envs_initial.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OBJECT_NAMES = [
    "sugar_box",
    "tomato_soup_can",
    "mustard_bottle",
    "white_cube",
    "black_cube",
    "small_tomato_soup_can",
]


parser = argparse.ArgumentParser(description="Record one video with six envs and six different objects.")
parser.add_argument("--video-length", type=int, default=180, help="Number of environment steps to record.")
parser.add_argument("--warmup-steps", type=int, default=0, help="Zero-action warmup steps before recording.")
parser.add_argument("--progress-interval", type=int, default=30, help="Print progress every N recorded steps.")
parser.add_argument(
    "--output-dir",
    type=str,
    default="github/Special_Course/lift_single/outputs/six_envs_initial_video",
    help="Directory for video output.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--env-spacing", type=float, default=2.5, help="Spacing between parallel environments.")
parser.add_argument(
    "--object-x",
    type=float,
    default=0.45,
    help="Local table-frame x coordinate for each visible object.",
)
parser.add_argument(
    "--object-y",
    type=float,
    default=0.0,
    help="Local table-frame y coordinate for each visible object.",
)
parser.add_argument(
    "--hidden-z",
    type=float,
    default=200.0,
    help="Local z coordinate used to hide non-target objects.",
)
parser.add_argument(
    "--camera-eye",
    type=float,
    nargs=3,
    default=(4.0, -5.0, 5.0),
    metavar=("X", "Y", "Z"),
    help="Camera eye position in world coordinates.",
)
parser.add_argument(
    "--camera-target",
    type=float,
    nargs=3,
    default=(1.8, 1.0, 0.0),
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

import lift_simple.config.franka  # noqa: F401  # registers the lift_simple Franka task
from lift_simple.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg_PLAY, GROCERIES


def _unpack_step(step_result):
    """Handle both 4-value and 5-value gym step APIs."""
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    return step_result


def _force_one_object_per_env(env) -> None:
    """Place one different target object on each table and hide all other objects."""
    object_collection = env.scene["object"]
    env_ids = torch.arange(len(OBJECT_NAMES), dtype=torch.long, device=object_collection.device)
    object_ids, selected_object_names = object_collection.find_objects(OBJECT_NAMES, preserve_order=True)
    object_ids = object_ids.to(device=object_collection.device)

    object_state = object_collection.data.default_object_state[env_ids][:, object_ids].clone()
    env_origins = env.scene.env_origins[env_ids]

    hidden_pos = torch.tensor(
        (0.0, 0.0, args_cli.hidden_z),
        dtype=object_state.dtype,
        device=object_collection.device,
    )
    object_state[..., :3] = env_origins[:, None, :] + hidden_pos
    object_state[..., 3:7] = torch.tensor(
        (1.0, 0.0, 0.0, 0.0),
        dtype=object_state.dtype,
        device=object_collection.device,
    )
    object_state[..., 7:] = 0.0

    for env_index in range(len(OBJECT_NAMES)):
        target_state = object_collection.data.default_object_state[env_index, object_ids[env_index]].clone()
        target_state[:3] += env_origins[env_index]
        target_state[0] = env_origins[env_index, 0] + args_cli.object_x
        target_state[1] = env_origins[env_index, 1] + args_cli.object_y
        target_state[3:7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            dtype=object_state.dtype,
            device=object_collection.device,
        )
        target_state[7:] = 0.0
        object_state[env_index, env_index] = target_state

    object_collection.write_object_state_to_sim(object_state, env_ids=env_ids, object_ids=object_ids)

    env._target_object_names = selected_object_names
    env._target_object_choice_ids = torch.arange(env.num_envs, dtype=torch.long, device=object_collection.device)
    env._target_object_ids = object_ids.clone()

    print("[INFO] Forced six-env object layout:", flush=True)
    for env_index, object_name in enumerate(selected_object_names):
        local_pos = (args_cli.object_x, args_cli.object_y, float(object_state[env_index, env_index, 2] - env_origins[env_index, 2]))
        print(f"  env_{env_index}: {object_name}, local position={local_pos}", flush=True)


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Objects shown in this video:", flush=True)
    for object_name in OBJECT_NAMES:
        print(f"  - {object_name}: {GROCERIES[object_name].usd_path}", flush=True)
    print(f"[INFO] Video output directory: {output_dir}", flush=True)

    env_cfg = FrankaCubeLiftEnvCfg_PLAY()
    env_cfg.scene.num_envs = len(OBJECT_NAMES)
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make("Isaac-Lift-Simple-Franka-Play-v0", cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    video_kwargs = {
        "video_folder": str(output_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length + args_cli.warmup_steps,
        "name_prefix": "lift_six_envs_initial",
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    action_shape = env.action_space.shape
    actions = torch.zeros(action_shape, device=env.unwrapped.device)

    print("[INFO] Resetting environment...", flush=True)
    env.reset(seed=args_cli.seed)
    _force_one_object_per_env(env.unwrapped)

    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            _unpack_step(env.step(actions))

    print(f"[INFO] Recording {args_cli.video_length} steps...", flush=True)
    for step in range(args_cli.video_length):
        with torch.inference_mode():
            _unpack_step(env.step(actions))
        if args_cli.progress_interval > 0 and (step + 1) % args_cli.progress_interval == 0:
            print(f"[INFO] Recorded {step + 1}/{args_cli.video_length} steps.", flush=True)

    env.close()
    simulation_app.close()
    print(f"[INFO] Saved six-env initial video under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
