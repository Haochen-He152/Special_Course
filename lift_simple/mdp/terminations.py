# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_object_pos_w(
    env: ManagerBasedRLEnv,
    object: RigidObjectCollection,
    object_name: str = "sugar_box",
) -> torch.Tensor:
    """Return the current target object's position from a rigid object collection."""
    if hasattr(env, "_target_object_ids"):
        env_ids = torch.arange(env.num_envs, device=object.device)
        return object.data.object_pos_w[env_ids, env._target_object_ids, :3]
    object_ids, _ = object.find_objects(object_name, preserve_order=True)
    return object.data.object_pos_w[:, object_ids[0], :3]


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    object_name: str = "sugar_box",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObjectCollection = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    object_pos_w = _get_object_pos_w(env, object, object_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object_pos_w, dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def object_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_name: str = "sugar_box",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the target object falls below the minimum height."""
    object: RigidObjectCollection = env.scene[object_cfg.name]
    object_pos_w = _get_object_pos_w(env, object, object_name)
    return object_pos_w[:, 2] < minimum_height
