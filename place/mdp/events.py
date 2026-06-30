# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_root_state_uniform_absolute(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object pose using absolute per-environment table coordinates."""
    object: RigidObject = env.scene[asset_cfg.name]
    env_ids = env_ids.to(device=object.device, dtype=torch.long)

    root_state = object.data.default_root_state[env_ids].clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)

    for dim, key in enumerate(["x", "y", "z"]):
        if key in pose_range:
            root_state[:, dim] = env.scene.env_origins[env_ids, dim] + samples[:, dim]

    orientation_delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], orientation_delta)

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    root_state[:, 7:] += samples

    object.write_root_state_to_sim(root_state, env_ids=env_ids)


def reset_object_root_state_to_ee(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_range: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Reset the object into the gripper using the end-effector frame pose."""
    object: RigidObject = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    env_ids = env_ids.to(device=object.device, dtype=torch.long)

    root_state = object.data.default_root_state[env_ids].clone()
    ee_pos_w = ee_frame.data.target_pos_w[env_ids, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[env_ids, 0, :]

    offset_pos = torch.tensor(position_offset, device=object.device, dtype=root_state.dtype).repeat(len(env_ids), 1)
    offset_quat = math_utils.quat_from_euler_xyz(
        torch.full((len(env_ids),), orientation_offset[0], device=object.device, dtype=root_state.dtype),
        torch.full((len(env_ids),), orientation_offset[1], device=object.device, dtype=root_state.dtype),
        torch.full((len(env_ids),), orientation_offset[2], device=object.device, dtype=root_state.dtype),
    )

    root_state[:, :3], root_state[:, 3:7] = math_utils.combine_frame_transforms(
        ee_pos_w,
        ee_quat_w,
        offset_pos,
        offset_quat,
    )

    if velocity_range is None:
        velocity_range = {}
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    root_state[:, 7:] = samples

    object.write_root_state_to_sim(root_state, env_ids=env_ids)
