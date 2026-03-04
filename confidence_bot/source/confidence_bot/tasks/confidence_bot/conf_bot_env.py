# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence

from confidence_bot.assets.robots.confidence_bot import CONFIDENCE_BOT_CFG
from confidence_bot.assets.apriltag.apriltag import APRILTAG_CFG

import isaaclab.sim as sim_utils

# import mdp
import confidence_bot.tasks.confidence_bot.mdp as mdp
from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

##
# Scene definition
##


@configclass
class ConfidenceBotSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # robots
    robot = CONFIDENCE_BOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # april tag
    april_tag = APRILTAG_CFG.replace(prim_path="{ENV_REGEX_NS}/AprilTag")


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_.*_wheel_joint", "right_.*_wheel_joint"],
        scale=5.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    


@configclass
class MyEventCfg:
    
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0), 
                "y": (0.0, 0.0), 
                "z": (0.3, 0.3),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),  # Random heading (Full 360 degrees)
            },
            "velocity_range": {}, # Sets all velocities to 0
        },
    )

    # 2. Randomize Camera Height (Z) and Tilt (Pitch)
    # We apply this to the 'tiled_camera' sensor attached to the robot
    randomize_camera_spec = EventTerm(
        func=mdp.reset_sensor_posture_uniform,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("robot", sensor_names=["tiled_camera"]),
            "pose_range": {
                "z": (0.375, 0.75),        # Half-pole to Full-pole height
                "pitch": (-0.087, 0.087),  # +/- 5 degrees in radians
            },
        },
    )

    # 3. Randomize Field of View (FOV)
    # This ensures the robot handles different focal lengths/zoom levels
    randomize_camera_fov = EventTerm(
        func=mdp.update_camera_intrinsic_matrix_uniform,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("robot", sensor_names=["tiled_camera"]),
            "fov_range": (50.0, 70.0), # 60 +/- 10 degrees
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds)


##
# Environment configuration
##


@configclass
class ConfidenceBotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for confidence bot."""

    # Scene settings
    scene: ConfidenceBotSceneCfg = ConfidenceBotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: MyEventCfg = MyEventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 30
        self.sim.dt = 1.0 / 60.0


##
# Environment class
##


class ConfidenceBotEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: ConfidenceBotEnvCfg
    
    def __init__(self, cfg: ConfidenceBotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find wheel joint indices
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
    
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset selected environments."""
        super()._reset_idx(env_ids)
        
        # Handle None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_resets = len(env_ids)
        
        # Reset wheel joint positions and velocities to zero
        num_wheels = len(self._wheel_indices)
        joint_pos = torch.zeros((num_resets, num_wheels), device=self.device)
        joint_vel = torch.zeros((num_resets, num_wheels), device=self.device)
        
        self.scene["robot"].write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._wheel_indices,
            env_ids=env_ids,
        )