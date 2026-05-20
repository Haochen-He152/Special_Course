# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


_STAGED_SINGLE_OBJECT_TASKS = {
    # Stage 1: fixed cube position and fixed yaw.
    "CubeFixed": "FrankaCubeFixedLiftEnvCfg",
    # Stage 2: random cube XY, fixed yaw.
    "CubeRandomXY": "FrankaCubeRandomXYLiftEnvCfg",
    # Stage 3: random XY, fixed yaw, for the remaining six objects.
    "SugarBoxRandomXY": "FrankaSugarBoxRandomXYLiftEnvCfg",
    "TomatoSoupCanRandomXY": "FrankaTomatoSoupCanRandomXYLiftEnvCfg",
    "MustardBottleRandomXY": "FrankaMustardBottleRandomXYLiftEnvCfg",
    "WhiteCubeRandomXY": "FrankaWhiteCubeRandomXYLiftEnvCfg",
    "BlackCubeRandomXY": "FrankaBlackCubeRandomXYLiftEnvCfg",
    "SmallTomatoSoupCanRandomXY": "FrankaSmallTomatoSoupCanRandomXYLiftEnvCfg",
    # Stage 4: random XY and random yaw, available for all seven objects.
    "CubeRandomXYYaw": "FrankaCubeRandomXYYawLiftEnvCfg",
    "SugarBoxRandomXYYaw": "FrankaSugarBoxRandomXYYawLiftEnvCfg",
    "TomatoSoupCanRandomXYYaw": "FrankaTomatoSoupCanRandomXYYawLiftEnvCfg",
    "MustardBottleRandomXYYaw": "FrankaMustardBottleRandomXYYawLiftEnvCfg",
    "WhiteCubeRandomXYYaw": "FrankaWhiteCubeRandomXYYawLiftEnvCfg",
    "BlackCubeRandomXYYaw": "FrankaBlackCubeRandomXYYawLiftEnvCfg",
    "SmallTomatoSoupCanRandomXYYaw": "FrankaSmallTomatoSoupCanRandomXYYawLiftEnvCfg",
}


for object_task_name, env_cfg_class_name in _STAGED_SINGLE_OBJECT_TASKS.items():
    skrl_cfg_name = "skrl_cube_fixed_ppo_cfg.yaml" if object_task_name == "CubeFixed" else "skrl_ppo_cfg.yaml"

    gym.register(
        id=f"Isaac-Lift-Single-Yaw-Curriculum-{object_task_name}-Franka-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:{env_cfg_class_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:{skrl_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
        disable_env_checker=True,
    )

    gym.register(
        id=f"Isaac-Lift-Single-Yaw-Curriculum-{object_task_name}-Franka-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:{env_cfg_class_name}_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:{skrl_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
        disable_env_checker=True,
    )
