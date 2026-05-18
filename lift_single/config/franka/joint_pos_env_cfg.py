# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp
from ...lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


SINGLE_OBJECTS = {
    "cube": {
        "initial_pos": (0.5, 0.0, 0.055),
        "spawn": UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    },
    "sugar_box": {
        "initial_pos": (0.45, 0.0, 0.10),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            scale=(0.701, 0.823, 1.108),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    },
    "tomato_soup_can": {
        "initial_pos": (0.45, 0.0, 0.10),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(1.109, 1.080, 1.108),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
        ),
    },
    "mustard_bottle": {
        "initial_pos": (0.45, 0.0, 0.10),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            scale=(0.625, 1.098, 0.858),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    },
    "white_cube": {
        "initial_pos": (0.45, 0.0, 0.055),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.833333, 0.833333, 0.833333),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        ),
    },
    "black_cube": {
        "initial_pos": (0.45, 0.0, 0.055),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    },
    "small_tomato_soup_can": {
        "initial_pos": (0.45, 0.0, 0.10),
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(0.739, 0.687, 0.738),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.14),
        ),
    },
}


@configclass
class FrankaSingleObjectLiftEnvCfg(LiftEnvCfg):
    object_name = "cube"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        self.commands.object_pose.body_name = "panda_hand"

        object_cfg = SINGLE_OBJECTS[self.object_name]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_cfg["initial_pos"], rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=object_cfg["spawn"],
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "cube"


@configclass
class FrankaDefaultCubeLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "cube"


@configclass
class FrankaSugarBoxLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "sugar_box"


@configclass
class FrankaTomatoSoupCanLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "tomato_soup_can"


@configclass
class FrankaMustardBottleLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "mustard_bottle"


@configclass
class FrankaWhiteCubeLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "white_cube"


@configclass
class FrankaBlackCubeLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "black_cube"


@configclass
class FrankaSmallTomatoSoupCanLiftEnvCfg(FrankaSingleObjectLiftEnvCfg):
    object_name = "small_tomato_soup_can"


@configclass
class FrankaSingleObjectLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "cube"


@configclass
class FrankaDefaultCubeLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "cube"


@configclass
class FrankaSugarBoxLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "sugar_box"


@configclass
class FrankaTomatoSoupCanLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "tomato_soup_can"


@configclass
class FrankaMustardBottleLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "mustard_bottle"


@configclass
class FrankaWhiteCubeLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "white_cube"


@configclass
class FrankaBlackCubeLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "black_cube"


@configclass
class FrankaSmallTomatoSoupCanLiftEnvCfg_PLAY(FrankaSingleObjectLiftEnvCfg_PLAY):
    object_name = "small_tomato_soup_can"
