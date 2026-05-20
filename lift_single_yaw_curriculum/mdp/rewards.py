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


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from a quaternion in ``(w, x, y, z)`` format."""
    qw, qx, qy, qz = quat_wxyz.unbind(dim=-1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


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


def gripper_closed_when_far_from_object(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.12,
    open_width: float = 0.04,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    finger_joint_names: list[str] = ["panda_finger.*"],
) -> torch.Tensor:
    """Penalize closing the gripper before the end-effector is near the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    far_from_object = (torch.norm(object_pos_w - ee_w, dim=1) > near_threshold).float()

    finger_joint_ids, _ = robot.find_joints(finger_joint_names, preserve_order=True)
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper_closed = 1.0 - torch.clamp(finger_pos.mean(dim=1) / open_width, min=0.0, max=1.0)
    return far_from_object * gripper_closed


def gripper_open_after_object_moves(
    env: ManagerBasedRLEnv,
    moved_height_threshold: float = 0.015,
    open_width: float = 0.04,
    open_threshold: float = 0.03,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_joint_names: list[str] = ["panda_finger.*"],
) -> torch.Tensor:
    """Penalize opening the gripper after the object has moved upward from its reset height."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_lifted = (object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2]) > moved_height_threshold

    finger_joint_ids, _ = robot.find_joints(finger_joint_names, preserve_order=True)
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper_open = (finger_pos.mean(dim=1) > open_threshold).float()
    return object_lifted.float() * gripper_open


def object_between_fingers_and_gripper_closed(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.20,
    balance_std: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    finger_body_names: list[str] = ["panda_leftfinger", "panda_rightfinger"],
) -> torch.Tensor:
    """Reward balanced finger-object distances near the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    finger_body_ids, _ = robot.find_bodies(finger_body_names, preserve_order=True)
    finger_pos_w = robot.data.body_pos_w[:, finger_body_ids, :]
    left_finger_pos_w = finger_pos_w[:, 0, :]
    right_finger_pos_w = finger_pos_w[:, 1, :]

    object_pos_w = object.data.root_pos_w[:, :3]
    finger_midpoint_w = 0.5 * (left_finger_pos_w + right_finger_pos_w)
    near_fingers = (torch.norm(object_pos_w - finger_midpoint_w, dim=1) < near_threshold).float()

    left_distance = torch.norm(object_pos_w - left_finger_pos_w, dim=1)
    right_distance = torch.norm(object_pos_w - right_finger_pos_w, dim=1)
    distance_balance = torch.abs(left_distance - right_distance)
    centered_between_fingers = 1.0 - torch.tanh(distance_balance / balance_std)

    return near_fingers * centered_between_fingers


def ee_object_yaw_alignment(
    env: ManagerBasedRLEnv,
    near_threshold: float = 0.18,
    std: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward aligning the end-effector yaw with the object's yaw near the object.

    A parallel gripper is symmetric under a 180 degree yaw flip, so the reward uses
    ``cos(2 * yaw_error)`` instead of ``cos(yaw_error)``.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(object_pos_w - ee_w, dim=1)
    near_object = (distance < near_threshold).float()
    proximity_reward = 1.0 - torch.tanh(distance / std)

    object_yaw = _yaw_from_quat_wxyz(object.data.root_quat_w)
    ee_yaw = _yaw_from_quat_wxyz(ee_frame.data.target_quat_w[..., 0, :])
    yaw_error = ee_yaw - object_yaw
    alignment_reward = 0.5 * (torch.cos(2.0 * yaw_error) + 1.0)
    return near_object * proximity_reward * alignment_reward


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
