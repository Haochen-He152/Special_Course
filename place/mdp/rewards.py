# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_pos_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    return target_pos_w


def _gripper_open_fraction(
    env: ManagerBasedRLEnv,
    open_width: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    finger_joint_names: list[str] = ["panda_finger.*"],
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    finger_joint_ids, _ = robot.find_joints(finger_joint_names, preserve_order=True)
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    return torch.clamp(finger_pos.mean(dim=1) / open_width, min=0.0, max=1.0)


def ee_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the end effector for moving close to the placement target."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    target_pos_w = _target_pos_w(env, command_name, robot_cfg)
    distance = torch.norm(ee_pos_w - target_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def release_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    open_width: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening the gripper only when the object is near the placement target."""
    object: RigidObject = env.scene[object_cfg.name]
    target_pos_w = _target_pos_w(env, command_name, robot_cfg)
    near_target = (torch.norm(object.data.root_pos_w[:, :3] - target_pos_w, dim=1) < threshold).float()
    gripper_open = _gripper_open_fraction(env, open_width, robot_cfg)
    return near_target * gripper_open
