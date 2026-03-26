# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot.tasks.e_corridor.ict_bot_corridor_env import CorridorEnvCfg

##
# Scene definition
##


@configclass
class IctBotCorridorEnvCfg(CorridorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 87 # [RelPos(3), Head(2), WheelVel(2), Lidar(72), IMU(6), LastAct(2)]
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.16
    max_linear_velocity = 1.0
    max_angular_velocity = 4.0
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]

    yaw_limit = 6.28  # allow max 360 degrees turns


@configclass
class IctBotCorridorEnvCfg_PLAY(IctBotCorridorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1

        self.fixed_play_level = 3

        # disable randomization for play
        self.observations.policy.enable_corruption = False