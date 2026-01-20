# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

X2_USD_PATH = str(Path(ISAAC_ASSET_DIR) / "X2_URDF-v1.3.0" / "usd" / "x2.usd")


def validate_x2_usd_path(usd_path: str | None = None) -> str:
    path = Path(usd_path or X2_USD_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "X2 USD asset not found at "
            f"'{path}'. Generate it from the URDF using the repo's URDF→USD conversion workflow "
            "and place it under legged_lab/assets/X2_URDF-v1.3.0/usd/ (expected x2.usd)."
        )
    return str(path)

    
X2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/X2_URDF-v1.3.0/usd/x2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "left_hip_pitch_joint": -0.3,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.61,
            "left_ankle_pitch_joint": -0.31,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.3,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.61,
            "right_ankle_pitch_joint": -0.31,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_hip_pitch_joint",
                "(left|right)_hip_roll_joint",
                "(left|right)_hip_yaw_joint",
                "(left|right)_knee_joint",
            ],
            effort_limit_sim={
                "(left|right)_hip_pitch_joint": 118,
                "(left|right)_hip_roll_joint": 118,
                "(left|right)_hip_yaw_joint": 118,
                "(left|right)_knee_joint": 118,
            },
            velocity_limit_sim={
                "(left|right)_hip_pitch_joint": 11.936,
                "(left|right)_hip_roll_joint": 11.936,
                "(left|right)_hip_yaw_joint": 11.936,
                "(left|right)_knee_joint": 11.936,
            },
            stiffness={
                "(left|right)_hip_roll_joint": 100,
                "(left|right)_hip_pitch_joint": 150,
                "(left|right)_hip_yaw_joint": 100,
                "(left|right)_knee_joint": 80,
            },
            damping={
                "(left|right)_hip_pitch_joint": 5,
                "(left|right)_hip_roll_joint": 4,
                "(left|right)_hip_yaw_joint": 4,
                "(left|right)_knee_joint": 10,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_ankle_pitch_joint",
                "(left|right)_ankle_roll_joint",
            ],
            effort_limit_sim={
                "(left|right)_ankle_pitch_joint": 36,
                "(left|right)_ankle_roll_joint": 24,
            },
            velocity_limit_sim={
                "(left|right)_ankle_pitch_joint": 13.087,
                "(left|right)_ankle_roll_joint": 15.077,
            },
            stiffness={
                "(left|right)_ankle_pitch_joint": 150,
                "(left|right)_ankle_roll_joint": 40,
            },
            damping={
                "(left|right)_ankle_pitch_joint": 8,
                "(left|right)_ankle_roll_joint": 4,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "(left|right)_shoulder_pitch_joint",
                "(left|right)_shoulder_roll_joint",
                "(left|right)_shoulder_yaw_joint",
                "(left|right)_elbow_joint",
            ],
            effort_limit_sim={
                "(left|right)_shoulder_pitch_joint": 25,
                "(left|right)_shoulder_roll_joint": 25,
                "(left|right)_shoulder_yaw_joint": 25,
                "(left|right)_elbow_joint": 25,
            },
            velocity_limit_sim={
                "(left|right)_shoulder_pitch_joint": 13.088,
                "(left|right)_shoulder_roll_joint": 13.088,
                "(left|right)_shoulder_yaw_joint": 15.077,
                "(left|right)_elbow_joint": 15.077,
            },
            stiffness={
                "(left|right)_shoulder_pitch_joint": 30,
                "(left|right)_shoulder_roll_joint": 20,
                "(left|right)_shoulder_yaw_joint": 20,
                "(left|right)_elbow_joint": 30,
            },
            damping={
                "(left|right)_shoulder_pitch_joint": 4,
                "(left|right)_shoulder_roll_joint": 4,
                "(left|right)_shoulder_yaw_joint": 4,
                "(left|right)_elbow_joint": 4,
            },
        
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 118,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 11.936,
            },
            stiffness={
                "waist_yaw_joint": 150,
            },
            damping={
                "waist_yaw_joint": 6,
            },
        ),
    },
)