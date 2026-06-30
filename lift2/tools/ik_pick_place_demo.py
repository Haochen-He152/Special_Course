# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place the lift2 grocery objects with absolute-pose IK.

Run from an Isaac Lab checkout, for example:

    ./isaaclab.sh -p github/Special_Course/lift2/tools/ik_pick_place_demo.py --object sugar_box
    ./isaaclab.sh -p github/Special_Course/lift2/tools/ik_pick_place_demo.py --object all --headless
    ./isaaclab.sh -p github/Special_Course/lift2/tools/ik_pick_place_demo.py

This is a scripted control demo. It reuses the lift2 Franka scene and grocery
assets, but it does not load a trained policy.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


OBJECT_ORDER = [
    "sugar_box",
    "tomato_soup_can",
    "mustard_bottle",
    "white_cube",
    "black_cube",
    "small_tomato_soup_can",
]


parser = argparse.ArgumentParser(description="Scripted IK pick-place demo for the lift2 grocery objects.")
parser.add_argument(
    "--object",
    choices=[*OBJECT_ORDER, "all"],
    default=None,
    help="Object to manipulate. If omitted, the script prompts for a selection.",
)
parser.add_argument("--seed", type=int, default=7, help="Random seed for start and goal samples.")
parser.add_argument("--max-steps", type=int, default=900, help="Maximum env steps per object.")
parser.add_argument("--hold-steps", type=int, default=45, help="Steps spent at each state-machine waypoint.")
parser.add_argument("--settle-steps", type=int, default=40, help="Zero/hold-action steps after reset.")
parser.add_argument("--start-x", type=float, nargs=2, default=(0.38, 0.58), metavar=("MIN", "MAX"))
parser.add_argument("--start-y", type=float, nargs=2, default=(-0.22, 0.22), metavar=("MIN", "MAX"))
parser.add_argument("--goal-x", type=float, nargs=2, default=(0.38, 0.58), metavar=("MIN", "MAX"))
parser.add_argument("--goal-y", type=float, nargs=2, default=(-0.22, 0.22), metavar=("MIN", "MAX"))
parser.add_argument("--min-start-goal-dist", type=float, default=0.18, help="Minimum XY distance between samples.")
parser.add_argument("--lift-height", type=float, default=0.34, help="World z used while carrying the object.")
parser.add_argument(
    "--grasp-z-offset",
    type=float,
    default=0.0,
    help="Extra offset added to the per-object grasp z. Use a negative value to close lower.",
)
parser.add_argument(
    "--release-z-offset",
    type=float,
    default=0.0,
    help="Extra offset added to the per-object release z.",
)
parser.add_argument("--camera-eye", type=float, nargs=3, default=(1.35, -1.05, 0.95), metavar=("X", "Y", "Z"))
parser.add_argument("--camera-target", type=float, nargs=3, default=(0.48, 0.0, 0.05), metavar=("X", "Y", "Z"))
parser.add_argument("--no-record-video", action="store_true", help="Disable video recording for quick debugging.")
parser.add_argument(
    "--output-dir",
    type=str,
    default="github/Special_Course/lift2/outputs/ik_pick_place_demo",
    help="Directory used for recorded videos.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _select_object_from_prompt() -> str:
    print("[INFO] Select the object to pick and place:", flush=True)
    for index, object_name in enumerate(OBJECT_ORDER, start=1):
        print(f"  {index}. {object_name}", flush=True)
    print(f"  {len(OBJECT_ORDER) + 1}. all", flush=True)

    valid_names = {name.lower(): name for name in [*OBJECT_ORDER, "all"]}
    while True:
        selected = input("Object name or number: ").strip().lower()
        if selected.isdigit():
            index = int(selected)
            if 1 <= index <= len(OBJECT_ORDER):
                return OBJECT_ORDER[index - 1]
            if index == len(OBJECT_ORDER) + 1:
                return "all"
        if selected in valid_names:
            return valid_names[selected]
        print("[WARN] Invalid selection. Please enter a listed name or number.", flush=True)


args_cli.object = args_cli.object or _select_object_from_prompt()
args_cli.record_video = not args_cli.no_record_video
args_cli.enable_cameras = args_cli.enable_cameras or args_cli.record_video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.assets import RigidObjectCfg

import lift2.config.franka  # noqa: F401
from lift2.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg
from lift2.config.franka.joint_pos_env_cfg import GROCERY_OBJECTS


GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
DOWNWARD_EE_QUAT = (0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True)
class GraspTuning:
    grasp_z: float
    approach_z: float
    release_z: float


GRASP_TUNING = {
    "sugar_box": GraspTuning(grasp_z=0.100, approach_z=0.28, release_z=0.105),
    "tomato_soup_can": GraspTuning(grasp_z=0.100, approach_z=0.27, release_z=0.105),
    "mustard_bottle": GraspTuning(grasp_z=0.105, approach_z=0.30, release_z=0.110),
    "white_cube": GraspTuning(grasp_z=0.050, approach_z=0.23, release_z=0.060),
    "black_cube": GraspTuning(grasp_z=0.040, approach_z=0.22, release_z=0.050),
    "small_tomato_soup_can": GraspTuning(grasp_z=0.075, approach_z=0.24, release_z=0.080),
}


class FrankaGroceryIKPickPlaceEnvCfg(FrankaCubeLiftEnvCfg):
    """One-env IK scene with a selected lift2 grocery object."""

    object_name = "sugar_box"

    def __post_init__(self):
        super().__post_init__()

        object_cfg = GROCERY_OBJECTS[self.object_name]
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics = False
        self.commands.object_pose.debug_vis = False
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 30.0

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_cfg["initial_pos"], rot=object_cfg["initial_rot"]),
            spawn=object_cfg["spawn"],
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }


def _unpack_step(step_result):
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, terminated | truncated, info
    return step_result


def _sample_point(rng: random.Random, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[float, float]:
    return rng.uniform(*x_range), rng.uniform(*y_range)


def _sample_start_and_goal(rng: random.Random) -> tuple[tuple[float, float], tuple[float, float]]:
    start = _sample_point(rng, tuple(args_cli.start_x), tuple(args_cli.start_y))
    goal = _sample_point(rng, tuple(args_cli.goal_x), tuple(args_cli.goal_y))
    for _ in range(100):
        dist = ((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2) ** 0.5
        if dist >= args_cli.min_start_goal_dist:
            return start, goal
        goal = _sample_point(rng, tuple(args_cli.goal_x), tuple(args_cli.goal_y))
    return start, goal


def _make_action(env, pose: tuple[float, float, float, float, float, float, float], gripper: float) -> torch.Tensor:
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    if action.ndim == 1:
        action[:7] = torch.tensor(pose, device=env.unwrapped.device)
        action[-1] = gripper
    else:
        action[:, :7] = torch.tensor(pose, device=env.unwrapped.device)
        action[:, -1] = gripper
    return action


def _set_object_pose(env, object_name: str, xy: tuple[float, float]) -> None:
    scene_object = env.unwrapped.scene["object"]
    object_cfg = GROCERY_OBJECTS[object_name]
    root_pose = scene_object.data.default_root_state[:, :7].clone()
    root_pose[:, 0] = xy[0]
    root_pose[:, 1] = xy[1]
    root_pose[:, 2] = object_cfg["initial_pos"][2]
    root_pose[:, 3:7] = torch.tensor(object_cfg["initial_rot"], device=root_pose.device)
    root_velocity = torch.zeros_like(scene_object.data.default_root_state[:, 7:])
    scene_object.write_root_pose_to_sim(root_pose)
    scene_object.write_root_velocity_to_sim(root_velocity)
    env.unwrapped.scene.write_data_to_sim()


def _run_steps(env, action: torch.Tensor, count: int) -> bool:
    done = False
    for _ in range(count):
        with torch.inference_mode():
            _, _, done_tensor, _ = _unpack_step(env.step(action))
        done = bool(torch.any(done_tensor).item()) if torch.is_tensor(done_tensor) else bool(done_tensor)
        if done:
            break
    return done


def _pose(x: float, y: float, z: float) -> tuple[float, float, float, float, float, float, float]:
    return (x, y, z, *DOWNWARD_EE_QUAT)


def run_object(object_name: str, rng: random.Random) -> None:
    env_cfg_cls = type(
        f"Franka{object_name.title().replace('_', '')}IKPickPlaceEnvCfg",
        (FrankaGroceryIKPickPlaceEnvCfg,),
        {"object_name": object_name},
    )
    env_cfg = env_cfg_cls()
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make("Isaac-Lift2-Cube-Franka-IK-Abs-v0", cfg=env_cfg, render_mode="rgb_array")
    env.unwrapped.sim.set_camera_view(args_cli.camera_eye, args_cli.camera_target)

    if args_cli.record_video:
        output_dir = Path(args_cli.output_dir).expanduser().resolve() / object_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {object_name}: recording video to {output_dir}", flush=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(output_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.max_steps,
            name_prefix=f"lift2_ik_pick_place_{object_name}",
            disable_logger=True,
        )

    start_xy, goal_xy = _sample_start_and_goal(rng)
    tuning = GRASP_TUNING[object_name]

    env.reset(seed=args_cli.seed)
    _set_object_pose(env, object_name, start_xy)

    grasp_z = tuning.grasp_z + args_cli.grasp_z_offset
    release_z = tuning.release_z + args_cli.release_z_offset
    pre_grasp = _pose(start_xy[0], start_xy[1], tuning.approach_z)
    grasp = _pose(start_xy[0], start_xy[1], grasp_z)
    lift = _pose(start_xy[0], start_xy[1], args_cli.lift_height)
    transfer = _pose(goal_xy[0], goal_xy[1], args_cli.lift_height)
    pre_place = _pose(goal_xy[0], goal_xy[1], tuning.approach_z)
    place = _pose(goal_xy[0], goal_xy[1], release_z)

    print(
        f"[INFO] {object_name}: start=({start_xy[0]:.3f}, {start_xy[1]:.3f}), "
        f"goal=({goal_xy[0]:.3f}, {goal_xy[1]:.3f}), grasp_z={grasp_z:.3f}, release_z={release_z:.3f}",
        flush=True,
    )

    sequence = [
        ("open above object", pre_grasp, GRIPPER_OPEN, args_cli.settle_steps + args_cli.hold_steps),
        ("descend", grasp, GRIPPER_OPEN, args_cli.hold_steps),
        ("close", grasp, GRIPPER_CLOSE, args_cli.hold_steps),
        ("lift", lift, GRIPPER_CLOSE, args_cli.hold_steps),
        ("transfer", transfer, GRIPPER_CLOSE, args_cli.hold_steps * 2),
        ("above goal", pre_place, GRIPPER_CLOSE, args_cli.hold_steps),
        ("lower", place, GRIPPER_CLOSE, args_cli.hold_steps),
        ("release", place, GRIPPER_OPEN, args_cli.hold_steps),
        ("retreat", pre_place, GRIPPER_OPEN, args_cli.hold_steps),
    ]

    steps = 0
    for label, target_pose, gripper, count in sequence:
        print(f"[INFO] {object_name}: {label}", flush=True)
        done = _run_steps(env, _make_action(env, target_pose, gripper), count)
        steps += count
        if done or steps >= args_cli.max_steps:
            break

    final_pos = env.unwrapped.scene["object"].data.root_pos_w[0].detach().cpu()
    dist_to_goal = torch.linalg.norm(final_pos[:2] - torch.tensor(goal_xy)).item()
    print(
        f"[INFO] {object_name}: final_xy=({final_pos[0]:.3f}, {final_pos[1]:.3f}), "
        f"goal_xy_error={dist_to_goal:.3f} m",
        flush=True,
    )
    env.close()


def main() -> None:
    rng = random.Random(args_cli.seed)
    object_names = OBJECT_ORDER if args_cli.object == "all" else [args_cli.object]
    for object_name in object_names:
        run_object(object_name, rng)
    simulation_app.close()


if __name__ == "__main__":
    main()
