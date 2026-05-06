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


def target_object_id_one_hot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """One-hot encoded target object id for the current episode."""
    object: RigidObjectCollection = env.scene[object_cfg.name]
    target_choice_ids = _get_target_choice_ids(env, object)
    return torch.nn.functional.one_hot(target_choice_ids, num_classes=object.num_objects).to(torch.float32)
