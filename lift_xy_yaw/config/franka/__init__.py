# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


_SINGLE_OBJECT_TASKS = {
    "Cube": "FrankaCubeLiftEnvCfg",
    "SugarBox": "FrankaSugarBoxLiftEnvCfg",
    "TomatoSoupCan": "FrankaTomatoSoupCanLiftEnvCfg",
    "MustardBottle": "FrankaMustardBottleLiftEnvCfg",
    "WhiteCube": "FrankaWhiteCubeLiftEnvCfg",
    "BlackCube": "FrankaBlackCubeLiftEnvCfg",
    "SmallTomatoSoupCan": "FrankaSmallTomatoSoupCanLiftEnvCfg",
}


for object_task_name, env_cfg_class_name in _SINGLE_OBJECT_TASKS.items():
    gym.register(
        id=f"Isaac-Lift-XY-Yaw-{object_task_name}-Franka-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:{env_cfg_class_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
        disable_env_checker=True,
    )

    gym.register(
        id=f"Isaac-Lift-XY-Yaw-{object_task_name}-Franka-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:{env_cfg_class_name}_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
            "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        },
        disable_env_checker=True,
    )
