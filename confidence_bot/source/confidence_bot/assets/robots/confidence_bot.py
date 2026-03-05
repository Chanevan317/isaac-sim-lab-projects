# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ICT Bot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Path to the directory containing this Python script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the relative path to the USD file
USD_PATH = os.path.join(ROOT_DIR, "confidence_bot.usd")

CONFIDENCE_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=0.6,
            max_angular_velocity=3.5,
            max_depenetration_velocity=1000.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "right_front_wheel_joint": 0.0,
            "left_front_wheel_joint": 0.0,
            "right_back_wheel_joint": 0.0,
            "left_back_wheel_joint": 0.0,
        },

    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_front_wheel_joint",
                "left_front_wheel_joint",
                "right_back_wheel_joint",
                "left_back_wheel_joint"
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=0.0,
            damping=10000.0,
        ),
    },
)