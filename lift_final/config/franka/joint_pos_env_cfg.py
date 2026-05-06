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

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


GROCERIES = {
    "OBJECT_A": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_B": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/004_sugar_box.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_C": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_D": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_E": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_F": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/037_scissors.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
    "OBJECT_G": sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/040_large_marker.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=4),
    ),
}

GROCERY_INITIAL_POSES = {
    "OBJECT_A": (0.40, -0.18, 0.10),
    "OBJECT_B": (0.40, 0.00, 0.10),
    "OBJECT_C": (0.40, 0.18, 0.10),
    "OBJECT_D": (0.55, -0.18, 0.10),
    "OBJECT_E": (0.55, 0.00, 0.10),
    "OBJECT_F": (0.55, 0.18, 0.10),
    "OBJECT_G": (0.70, 0.00, 0.10),
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
