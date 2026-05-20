# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_yaw_sin_cos(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Yaw of the object encoded as sin/cos."""
    object: RigidObject = env.scene[object_cfg.name]
    qw, qx, qy, qz = object.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def object_goal_pose_above_object_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    goal_height_offset: float = 0.20,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Goal pose above the object in the robot's root frame.

    The goal follows the object's current x/y position and uses the object's default z plus an offset.
    The orientation is fixed to the identity quaternion to match the official generated command dimension.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    goal_pos_w = object.data.root_pos_w[:, :3].clone()
    goal_pos_w[:, 2] = object.data.default_root_state[:, 2] + goal_height_offset
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_w)
    goal_quat_b = torch.zeros(env.num_envs, 4, dtype=goal_pos_b.dtype, device=goal_pos_b.device)
    goal_quat_b[:, 0] = 1.0
    return torch.cat((goal_pos_b, goal_quat_b), dim=-1)
