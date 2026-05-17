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
    absolute_position: bool = False,
    min_separation: float = 0.0,
    max_resample_attempts: int = 50,
    spawn_only_target: bool = False,
    hidden_position: tuple[float, float, float] = (-10.0, 0.0, 0.5),
    orientation_choices: dict[str, list[tuple[float, float, float]]]
    | list[tuple[float, float, float]]
    | None = None,
):
    """Reset rigid objects in a collection with uniform pose and velocity noise.

    When ``absolute_position`` is true, sampled x/y/z values are interpreted as
    table/world coordinates relative to each environment origin instead of
    offsets from the objects' default poses.

    When ``min_separation`` is positive, x/y samples are rejected and resampled
    until all selected objects in each environment are at least that far apart
    in the table plane. This is a simple center-distance check intended to avoid
    invalid initial overlap between groceries.
    """
    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]
    env_ids = env_ids.to(device=object_collection.device, dtype=torch.long)

    if object_names is None:
        object_ids = torch.arange(object_collection.num_objects, device=object_collection.device)
        selected_object_names = object_collection.object_names
    else:
        object_ids, selected_object_names = object_collection.find_objects(object_names, preserve_order=True)

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

    if spawn_only_target:
        if not hasattr(env, "_target_object_ids"):
            raise RuntimeError("spawn_only_target=True requires randomize_target=True.")

        hidden_pos = torch.tensor(hidden_position, dtype=object_state.dtype, device=object_collection.device)
        object_state[..., :3] = env.scene.env_origins[env_ids].unsqueeze(1) + hidden_pos
        object_state[..., 7:] = 0.0

        local_target_ids = env._target_object_ids[env_ids]
        if object_names is not None:
            object_id_lookup = torch.full(
                (object_collection.num_objects,),
                -1,
                dtype=torch.long,
                device=object_collection.device,
            )
            object_id_lookup[object_ids] = torch.arange(len(object_ids), device=object_collection.device)
            local_target_ids = object_id_lookup[local_target_ids]
            if (local_target_ids < 0).any():
                raise RuntimeError("The randomized target object must be included in object_names.")

        env_origins = env.scene.env_origins[env_ids]
        target_state = object_collection.data.default_object_state[env_ids, env._target_object_ids[env_ids]].clone()
        target_state[:, :3] += env_origins
        if absolute_position:
            for dim, key in enumerate(["x", "y", "z"]):
                if key in pose_range:
                    target_state[:, dim] = env_origins[:, dim] + samples[:, 0, dim]
        else:
            target_state[:, :3] += samples[:, 0, :3]

        orientations_delta = math_utils.quat_from_euler_xyz(samples[:, 0, 3], samples[:, 0, 4], samples[:, 0, 5])
        target_state[:, 3:7] = math_utils.quat_mul(target_state[:, 3:7], orientations_delta)

        if orientation_choices is not None:
            for object_id, object_name in enumerate(object_collection.object_names):
                if isinstance(orientation_choices, dict):
                    choices = orientation_choices.get(object_name, orientation_choices.get("*"))
                else:
                    choices = orientation_choices
                if not choices:
                    continue

                matching_envs = torch.nonzero(env._target_object_ids[env_ids] == object_id, as_tuple=False).squeeze(-1)
                if len(matching_envs) == 0:
                    continue
                choices_tensor = torch.tensor(choices, dtype=torch.float, device=object_collection.device)
                choice_ids = torch.randint(len(choices), (len(matching_envs),), device=object_collection.device)
                chosen_euler = choices_tensor[choice_ids]
                discrete_orientation_delta = math_utils.quat_from_euler_xyz(
                    chosen_euler[:, 0], chosen_euler[:, 1], chosen_euler[:, 2]
                )
                target_state[matching_envs, 3:7] = math_utils.quat_mul(
                    target_state[matching_envs, 3:7], discrete_orientation_delta
                )

        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        velocity_ranges = torch.tensor(velocity_range_list, device=object_collection.device)
        velocity_samples = math_utils.sample_uniform(
            velocity_ranges[:, 0],
            velocity_ranges[:, 1],
            (len(env_ids), 6),
            device=object_collection.device,
        )
        target_state[:, 7:] += velocity_samples
        object_state[torch.arange(len(env_ids), device=object_collection.device), local_target_ids] = target_state

        object_collection.write_object_state_to_sim(object_state, env_ids=env_ids, object_ids=object_ids)
        return

    if absolute_position:
        env_origins = env.scene.env_origins[env_ids].unsqueeze(1)
        for dim, key in enumerate(["x", "y", "z"]):
            if key in pose_range:
                object_state[..., dim] = env_origins[..., dim] + samples[..., dim]
    else:
        object_state[..., :3] += samples[..., :3]

    if min_separation > 0.0 and object_state.shape[1] > 1:
        xy_ranges = torch.tensor(
            [pose_range.get("x", (0.0, 0.0)), pose_range.get("y", (0.0, 0.0))],
            device=object_collection.device,
        )
        if absolute_position and "x" in pose_range and "y" in pose_range:
            env_origins = env.scene.env_origins[env_ids].unsqueeze(1)
            unresolved = torch.ones(len(env_ids), dtype=torch.bool, device=object_collection.device)
            for _ in range(max_resample_attempts):
                candidate_xy = math_utils.sample_uniform(
                    xy_ranges[:, 0],
                    xy_ranges[:, 1],
                    (unresolved.sum(), object_state.shape[1], 2),
                    device=object_collection.device,
                )
                pairwise_dist = torch.cdist(candidate_xy, candidate_xy)
                eye = torch.eye(object_state.shape[1], dtype=torch.bool, device=object_collection.device).unsqueeze(0)
                pairwise_dist = pairwise_dist.masked_fill(eye, min_separation)
                valid = torch.all(pairwise_dist >= min_separation, dim=(1, 2))

                unresolved_ids = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                valid_ids = unresolved_ids[valid]
                if len(valid_ids) > 0:
                    object_state[valid_ids, :, :2] = env_origins[valid_ids, :, :2] + candidate_xy[valid]
                    unresolved[valid_ids] = False
                if not unresolved.any():
                    break
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
