# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_object_collection_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_names: list[str] | None = None,
    randomize_target: bool = False,
    target_object_names: list[str] | None = None,
    orientation_choices: dict[str, list[tuple[float, float, float]]]
    | list[tuple[float, float, float]]
    | None = None,
):
    """Reset rigid objects in a collection with uniform pose and velocity noise."""
    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    env_ids = env_ids.to(device=object_collection.device, dtype=torch.long)

    if object_names is None:
        object_ids = torch.arange(object_collection.num_objects, device=object_collection.device)
        selected_object_names = object_collection.object_names
    else:
        object_ids, selected_object_names = object_collection.find_objects(object_names, preserve_order=True)

    object_state = object_collection.data.default_object_state[env_ids][:, object_ids].clone()
    object_state[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object_collection.device)
    samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), object_state.shape[1], 6),
        device=object_collection.device,
    )

    object_state[..., :3] += samples[..., :3]
    orientations_delta = math_utils.quat_from_euler_xyz(samples[..., 3], samples[..., 4], samples[..., 5])
    object_state[..., 3:7] = math_utils.quat_mul(object_state[..., 3:7], orientations_delta)

    if orientation_choices is not None:
        for index, object_name in enumerate(selected_object_names):
            if isinstance(orientation_choices, dict):
                choices = orientation_choices.get(object_name, orientation_choices.get("*"))
            else:
                choices = orientation_choices
            if not choices:
                continue

            choices_tensor = torch.tensor(choices, dtype=torch.float, device=object_collection.device)
            choice_ids = torch.randint(len(choices), (len(env_ids),), device=object_collection.device)
            chosen_euler = choices_tensor[choice_ids]
            discrete_orientation_delta = math_utils.quat_from_euler_xyz(
                chosen_euler[:, 0], chosen_euler[:, 1], chosen_euler[:, 2]
            )
            object_state[:, index, 3:7] = math_utils.quat_mul(
                object_state[:, index, 3:7], discrete_orientation_delta
            )

    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object_collection.device)
    samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), object_state.shape[1], 6),
        device=object_collection.device,
    )
    object_state[..., 7:] += samples

    object_collection.write_object_state_to_sim(object_state, env_ids=env_ids, object_ids=object_ids)

    if randomize_target:
        if target_object_names is None:
            target_object_names = object_collection.object_names
        target_object_ids, target_object_names = object_collection.find_objects(target_object_names, preserve_order=True)
        target_choices = torch.randint(
            len(target_object_ids),
            (len(env_ids),),
            device=object_collection.device,
        )

        if not hasattr(env, "_target_object_choice_ids"):
            env._target_object_choice_ids = torch.zeros(env.num_envs, dtype=torch.long, device=object_collection.device)
        if not hasattr(env, "_target_object_ids"):
            env._target_object_ids = torch.zeros(env.num_envs, dtype=torch.long, device=object_collection.device)

        env._target_object_names = target_object_names
        env._target_object_choice_ids[env_ids] = target_choices
        env._target_object_ids[env_ids] = target_object_ids[target_choices]
