# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from confidence_bot.assets.robots.confidence_bot import CONFIDENCE_BOT_CFG
from confidence_bot.tasks.confidence_bot.conf_bot_env import ConfidenceBotEnvCfg

##
# Scene definition
##


@configclass
class ConfidenceBotEnvCfg(ConfidenceBotEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 13  #
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.51
    max_linear_velocity = 2.0
    max_angular_velocity = 8.0
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_front_wheel_joint",
        "left_back_wheel_joint",
        "right_front_wheel_joint",
        "right_backfront_wheel_joint",
    ]


@configclass
class ConfidenceBotEnvCfg_PLAY(ConfidenceBotEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False