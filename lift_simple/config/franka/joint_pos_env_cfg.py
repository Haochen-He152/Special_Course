# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp
from ...lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


GROCERIES = {
    "sugar_box": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        scale=(0.701, 0.823, 1.108),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    "tomato_soup_can": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        scale=(0.739, 0.687, 0.738),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
    ),
    "mustard_bottle": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        scale=(0.625, 1.098, 0.858),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    "white_cube": sim_utils.UsdFileCfg(
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
    "black_cube": sim_utils.UsdFileCfg(
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
    "small_tomato_soup_can": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        scale=(0.739, 0.687, 0.738),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.14),
    ),
}

GROCERY_INITIAL_POSES = {
    "sugar_box": (0.45, -0.14, 0.10),
    "tomato_soup_can": (0.45, 0.00, 0.10),
    "mustard_bottle": (0.45, 0.14, 0.10),
    "white_cube": (0.35, -0.21, 0.055),
    "black_cube": (0.35, -0.12, 0.055),
    "small_tomato_soup_can": (0.35, 0.08, 0.10),
}

TRASH_BIN_SIZE = (0.197843, 0.296635, 0.146360)
GROUND_Z = -1.05
TABLE_TOP_Z = 0.0
TRASH_BIN_STAND_HEIGHT = TABLE_TOP_Z - GROUND_Z - TRASH_BIN_SIZE[2]
TRASH_BIN_STAND_POS = (0.78, 0.42, GROUND_Z + TRASH_BIN_STAND_HEIGHT / 2.0)
TRASH_BIN_POS = (
    TRASH_BIN_STAND_POS[0],
    TRASH_BIN_STAND_POS[1],
    TABLE_TOP_Z - TRASH_BIN_SIZE[2] / 2.0,
)


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # RigidObjectCollection with multiple distinct USD assets follows the bin_packing example and avoids
        # physics replication so each grocery asset is parsed with its own rigid-body properties.
        self.scene.replicate_physics = False

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

        # Set groceries as objects
        self.scene.object = RigidObjectCollectionCfg(
            rigid_objects={
                name: RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=GROCERY_INITIAL_POSES[name],
                        rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                    spawn=spawn_cfg,
                )
                for name, spawn_cfg in GROCERIES.items()
            }
        )

        self.scene.bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KLT_Bin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.70, -0.32, 0.08), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
        )

        self.scene.trash_bin_stand = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TrashBinStand",
            init_state=RigidObjectCfg.InitialStateCfg(pos=TRASH_BIN_STAND_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(TRASH_BIN_SIZE[0], TRASH_BIN_SIZE[1], TRASH_BIN_STAND_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            ),
        )

        self.scene.trash_bin = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TrashBin",
            init_state=RigidObjectCfg.InitialStateCfg(pos=TRASH_BIN_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
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
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
