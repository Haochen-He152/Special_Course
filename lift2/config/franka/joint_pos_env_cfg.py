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


UPRIGHT_ROT = (1.0, 0.0, 0.0, 0.0)
TOMATO_CAN_UPRIGHT_ROT = (0.7071068, 0.7071068, 0.0, 0.0)
YAW_90_ROT = (0.7071068, 0.0, 0.0, 0.7071068)

GROCERY_OBJECTS = {
    "sugar_box": {
        "initial_pos": (0.45, 0.0, 0.10),
        "initial_rot": YAW_90_ROT,
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
            scale=(0.701, 0.823, 1.108),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    },
    "tomato_soup_can": {
        "initial_pos": (0.45, 0.0, 0.10),
        "initial_rot": TOMATO_CAN_UPRIGHT_ROT,
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(1.109, 1.080, 1.108),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
        ),
    },
    "mustard_bottle": {
        "initial_pos": (0.45, 0.0, 0.10),
        "initial_rot": YAW_90_ROT,
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            scale=(0.625, 1.098, 0.858),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    },
    "white_cube": {
        "initial_pos": (0.45, 0.0, 0.055),
        "initial_rot": (1.0, 0.0, 0.0, 0.0),
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
        "initial_rot": (1.0, 0.0, 0.0, 0.0),
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
        "initial_rot": TOMATO_CAN_UPRIGHT_ROT,
        "spawn": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(0.739, 0.687, 0.738),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.14),
        ),
    },
}


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
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
        )

        # Listens to the required transforms
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
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeSmallRandomLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.reset_object_position.params["pose_range"].update(
            {
                "x": (-0.08, 0.08),
                "y": (-0.13, 0.13),
            }
        )


@configclass
class FrankaCubeLargeRandomLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.events.reset_object_position.params["pose_range"].update(
            {
                "x": (-0.2, 0.2),
                "y": (-0.25, 0.25),
            }
        )


@configclass
class FrankaGroceryLargeRandomLiftEnvCfg(FrankaCubeLargeRandomLiftEnvCfg):
    object_name = "sugar_box"

    def __post_init__(self):
        super().__post_init__()

        object_cfg = GROCERY_OBJECTS[self.object_name]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_cfg["initial_pos"], rot=object_cfg["initial_rot"]),
            spawn=object_cfg["spawn"],
        )


@configclass
class FrankaSugarBoxLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "sugar_box"


@configclass
class FrankaTomatoSoupCanLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "tomato_soup_can"

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"]["yaw"] = (0.0, 0.0)


@configclass
class FrankaMustardBottleLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "mustard_bottle"


@configclass
class FrankaWhiteCubeLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "white_cube"


@configclass
class FrankaBlackCubeLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "black_cube"


@configclass
class FrankaBlackCubeSmallRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "black_cube"

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"].update(
            {
                "x": (-0.08, 0.08),
                "y": (-0.13, 0.13),
            }
        )


@configclass
class FrankaSmallTomatoSoupCanLargeRandomLiftEnvCfg(FrankaGroceryLargeRandomLiftEnvCfg):
    object_name = "small_tomato_soup_can"

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_position.params["pose_range"]["yaw"] = (0.0, 0.0)


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
