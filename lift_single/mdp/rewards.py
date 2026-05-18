# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float = 0.04,
    height_offset: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    if height_offset is not None:
        height_threshold = object.data.default_root_state[:, 2] + height_offset
    else:
        height_threshold = minimal_height
    return torch.where(object.data.root_pos_w[:, 2] > height_threshold, 1.0, 0.0)


def object_lift_height(
    env: ManagerBasedRLEnv,
    target_height: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for lifting the object above its reset height."""
    object: RigidObject = env.scene[object_cfg.name]
    object_height = object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]
    return torch.clamp(object_height / target_height, min=0.0, max=1.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def gripper_closed_when_near_object(
    env: ManagerBasedRLEnv,
    std: float = 0.08,
    open_width: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    finger_joint_names: list[str] = ["panda_finger.*"],
) -> torch.Tensor:
    """Reward closing the gripper only when the end-effector is near the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    near_object = 1 - torch.tanh(torch.norm(object_pos_w - ee_w, dim=1) / std)

    finger_joint_ids, _ = robot.find_joints(finger_joint_names, preserve_order=True)
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper_closed = 1.0 - torch.clamp(finger_pos.mean(dim=1) / open_width, min=0.0, max=1.0)
    return near_object * gripper_closed


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    goal_height_offset: float = 0.20,
    minimal_height: float = 0.04,
    height_offset: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the object for staying close to the goal above its current x/y position."""
    object: RigidObject = env.scene[object_cfg.name]

    goal_pos_w = object.data.root_pos_w[:, :3].clone()
    goal_pos_w[:, 2] = object.data.default_root_state[:, 2] + goal_height_offset
    distance = torch.norm(goal_pos_w - object.data.root_pos_w[:, :3], dim=1)

    if height_offset is not None:
        height_threshold = object.data.default_root_state[:, 2] + height_offset
    else:
        height_threshold = minimal_height
    return (object.data.root_pos_w[:, 2] > height_threshold) * (1 - torch.tanh(distance / std))
