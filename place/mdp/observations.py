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
    """Current object position in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object.data.root_pos_w[:, :3],
    )
    return object_pos_b


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Current object pose in the robot root frame as position and quaternion."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object.data.root_pos_w[:, :3],
        object.data.root_quat_w,
    )
    return torch.cat((object_pos_b, object_quat_b), dim=-1)


def target_position_command(env: ManagerBasedRLEnv, command_name: str = "object_pose") -> torch.Tensor:
    """Target placement position command, without the unused orientation part."""
    return env.command_manager.get_command(command_name)[:, :3]


def target_pose_command(env: ManagerBasedRLEnv, command_name: str = "object_pose") -> torch.Tensor:
    """Target placement pose command as position and quaternion."""
    return env.command_manager.get_command(command_name)
