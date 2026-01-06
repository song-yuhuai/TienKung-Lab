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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR

GP2_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/gp2_v2/usd/gp2.usd",
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
            "left_hip_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.17,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.2,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.17,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.2,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.97,
            "right_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.97,
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
                "(left|right)_hip_pitch_joint": 88,
                "(left|right)_hip_roll_joint": 88,
                "(left|right)_hip_yaw_joint": 88,
                "(left|right)_knee_joint": 139,
            },
            velocity_limit_sim={
                "(left|right)_hip_pitch_joint": 32,
                "(left|right)_hip_roll_joint": 32,
                "(left|right)_hip_yaw_joint": 32,
                "(left|right)_knee_joint": 20,
            },
            stiffness={
                "(left|right)_hip_roll_joint": 100,
                "(left|right)_hip_pitch_joint": 150,
                "(left|right)_hip_yaw_joint": 100,
                "(left|right)_knee_joint": 180,
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
                "(left|right)_ankle_pitch_joint": 50,
                "(left|right)_ankle_roll_joint": 50,
            },
            velocity_limit_sim={
                "(left|right)_ankle_pitch_joint": 37,
                "(left|right)_ankle_roll_joint": 37,
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
                "(left|right)_shoulder_pitch_joint": 37,
                "(left|right)_shoulder_roll_joint": 37,
                "(left|right)_shoulder_yaw_joint": 37,
                "(left|right)_elbow_joint": 37,
            },
            stiffness={
                "(left|right)_shoulder_pitch_joint": 90,
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
                "waist_yaw_joint": 50,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 20,
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