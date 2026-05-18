# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_target_choice_ids(env: ManagerBasedRLEnv, object: RigidObjectCollection) -> torch.Tensor:
    """Return per-env target choice indices, defaulting to the first object before reset."""
    if hasattr(env, "_target_object_choice_ids"):
        return env._target_object_choice_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=object.device)


def _get_target_object_ids(env: ManagerBasedRLEnv, object: RigidObjectCollection) -> torch.Tensor:
    """Return per-env object indices for the current target object."""
    if hasattr(env, "_target_object_ids"):
        return env._target_object_ids
    return torch.zeros(env.num_envs, dtype=torch.long, device=object.device)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of all objects in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObjectCollection = env.scene[object_cfg.name]
    object_pos_w = object.data.object_pos_w
    num_objects = object_pos_w.shape[1]
    robot_pos_w = robot.data.root_pos_w[:, None, :].expand(-1, num_objects, -1).reshape(-1, 3)
    robot_quat_w = robot.data.root_quat_w[:, None, :].expand(-1, num_objects, -1).reshape(-1, 4)
    object_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, object_pos_w.reshape(-1, 3))
    return object_pos_b.reshape(env.num_envs, -1)


def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The current target object's position in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObjectCollection = env.scene[object_cfg.name]
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=object.device)
    target_object_ids = _get_target_object_ids(env, object)
    object_pos_w = object.data.object_pos_w[env_ids, target_object_ids]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def target_object_id_one_hot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-hot encoded target object id for the current episode."""
    object: RigidObjectCollection = env.scene[object_cfg.name]
    target_choice_ids = _get_target_choice_ids(env, object)
    return torch.nn.functional.one_hot(target_choice_ids, num_classes=object.num_objects).to(torch.float32)


def target_object_yaw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Yaw of the current target object encoded as sin/cos."""
    object: RigidObjectCollection = env.scene[object_cfg.name]
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=object.device)
    target_object_ids = _get_target_object_ids(env, object)
    object_quat_w = object.data.object_quat_w[env_ids, target_object_ids]

    qw, qx, qy, qz = object_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def target_object_orientation_one_hot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-hot encoded coarse target object orientation.

    The three classes indicate which target-object local axis is most aligned with the world z-axis:
    local x, local y, or local z. The sign is ignored, so upside-down variants share the same class.
    """
    object: RigidObjectCollection = env.scene[object_cfg.name]
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=object.device)
    target_object_ids = _get_target_object_ids(env, object)
    object_quat_w = object.data.object_quat_w[env_ids, target_object_ids]

    qw, qx, qy, qz = object_quat_w.unbind(dim=-1)
    axis_z_alignment = torch.stack(
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
        dim=-1,
    ).abs()
    orientation_ids = torch.argmax(axis_z_alignment, dim=-1)
    return torch.nn.functional.one_hot(orientation_ids, num_classes=3).to(torch.float32)
