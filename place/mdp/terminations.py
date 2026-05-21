# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .rewards import _gripper_open_fraction, _target_pos_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def place_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    velocity_threshold: float,
    gripper_open_threshold: float,
    open_width: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success when the object is near the target, released, and nearly still."""
    object: RigidObject = env.scene[object_cfg.name]
    target_pos_w = _target_pos_w(env, command_name, robot_cfg)
    object_pos_w = object.data.root_pos_w[:, :3]
    if hasattr(object.data, "root_lin_vel_w"):
        object_lin_vel_w = object.data.root_lin_vel_w
    else:
        object_lin_vel_w = object.data.root_state_w[:, 7:10]

    near_target = torch.norm(object_pos_w - target_pos_w, dim=1) < distance_threshold
    object_stable = torch.norm(object_lin_vel_w, dim=1) < velocity_threshold
    gripper_open = _gripper_open_fraction(env, open_width, robot_cfg) > gripper_open_threshold
    return near_target & object_stable & gripper_open
