# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ICT Bot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg

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

    sensors={
        "tiled_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/confidence_bot/body", # Path inside your USD
            update_period=0.016, # 60 FPS (1/60 = 0.016s)
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.21, 0.0, 0.75), # X=front of pole, Y=centered, Z=top of pole
                rot=(1.0, 0.0, 0.0, 0.0), # Looking straight forward
                convention="ros", # Uses standard ROS camera axes
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, # Adjust to match your 60° FOV
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        ),

        "imu": ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/confidence_bot/base", # Attach to the main chassis
            update_period=0.01, # 100Hz (matches typical real-world IMU rates)
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), 
                rot=(1.0, 0.0, 0.0, 0.0)
            ),
            gravity_bias=(0.0, 0.0, 9.81),
        ),
    },
)