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
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ActionTermCfg as ActTerm
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

    tiled_camera = TiledCameraCfg(
        # Notice the path: it points to a specific link INSIDE the robot's prim_path
        prim_path="{ENV_REGEX_NS}/Robot/confidence_bot/body/camera", 
        update_period=0.016, 
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.21, 0.0, 0.75), 
            rot=(0.5, 0.5, -0.5, -0.5), # Ensure this matches your desired tilt
            convention="ros",
        ),
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/confidence_bot/base", 
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 9.81),
    )



##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for the environment."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),  # Only resample on RESET
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6), # Random speed between 0.2 and 0.6 m/s
            lin_vel_y=(0.0, 0.0), # No sideways strafing
            ang_vel_z=(0.0, 0.0), # The AI will decide the steering, not the command
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_.*_wheel_joint", "right_.*_wheel_joint"],
        scale=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        imu_data = ObsTerm(
            func=mdp.get_imu_data, 
            params={"sensor_name": "imu"},
            noise=GaussianNoiseCfg(std=0.05), # Simulate motor vibration jitter
            scale=0.1,
        )

        tag_coords = ObsTerm(
            func=mdp.get_tag_pixel_coords,
            history_length=3,              # AI sees current + 2 past frames
            params={
                "sensor_name": "tiled_camera",
                "tag_cfg": SceneEntityCfg("april_tag"),
            },
            noise=GaussianNoiseCfg(std=0.02), # Simulate detection jitter
            clip=(-1.0, 1.0),
        )

        target_speed = ObsTerm(
            func=mdp.get_target_speed
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. THE CARROT: Stay centered on the Tag
    # We use a Gaussian-style reward so the 'pay' is highest at u=0
    track_tag_u = RewTerm(
        func=mdp.track_tag_center_u,
        weight=1.0,
        params={
            "std": 0.2,
            "tag_cfg": SceneEntityCfg("april_tag")
        },
    )

    # 2. PROGRESS: Keep moving forward
    # Reward the robot for having a positive velocity in the command direction
    forward_velocity = RewTerm(
        func=mdp.forward_velocity_tracking,
        weight=1.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity"
        },
    )

    # 3. STABILITY: Don't let the 1m pole wobble
    # Punish high lateral acceleration (y-axis) from the IMU
    penalize_pole_sway = RewTerm(
        func=mdp.log_penalty,
        weight=-0.1, # Negative weight makes it a penalty
        params={
            "sensor_cfg": SceneEntityCfg("imu"),
            "index": 1, # e.g., Penalizing Y-axis acceleration (side-to-side shaking)
        },
    )

    # 4. SMOOTHNESS: Punish rapid discrete steering changes
    # This prevents the -1 -> 1 -> -1 jitter (Bang-Bang control)
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    # 5. SUCCESS: Big bonus for reaching the "Disappearing Point"
    # When v (vertical) is close to -1 and tag is centered
    reach_tag_bonus = RewTerm(
        func=mdp.reach_tag_success,
        weight=50.0,
        params={
            "tag_cfg": SceneEntityCfg("april_tag"),
            "threshold_v": 0.85,
            "threshold_u": 0.15
        },
    )


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
                "z": (0.2, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),  # Random heading (Full 360 degrees)
            },
            "velocity_range": {}, # Sets all velocities to 0
        },
    )
    
    # Teleport the AprilTag to a random spot 1.5m to 3.0m in front of robot
    reset_tag = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("april_tag"),
            "pose_range": {
                "x": (1.5, 3.0), 
                "y": (-1.0, 1.0), 
                "z": (0.01, 0.01)
            },
            "velocity_range": {},
        },
    )

    # Randomize Camera Height (Z) and Tilt (Pitch)
    # We apply this to the 'tiled_camera' sensor attached to the robot
    randomize_camera_spec = EventTerm(
        func=mdp.reset_camera_posture_uniform,
        mode="reset",
        params={
            "sensor_name": "tiled_camera",
            "z_range": (0.375, 0.45),        
            "pitch_range": (-0.17, 0.17),  # +/- 10 degrees in radians
        },
    )

    # Randomize Field of View (FOV)
    # This ensures the robot handles different focal lengths/zoom levels
    randomize_camera_fov = EventTerm(
        func=mdp.update_camera_fov_uniform,
        mode="reset",
        params={
            "sensor_name": "tiled_camera",
            "fov_range": (50.0, 70.0), # 60 +/- 10 degrees
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    reached_target = DoneTerm(
        func=mdp.reached_tag_visual, 
        params={
            "tag_cfg": SceneEntityCfg("april_tag"),
            "threshold_v": 0.85, 
            "threshold_u": 0.15,
            "sensor_name": "tiled_camera",
        }
    )


##
# Environment configuration
##


@configclass
class ConfidenceBotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for confidence bot."""

    # Scene settings
    scene: ConfidenceBotSceneCfg = ConfidenceBotSceneCfg(num_envs=32, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: MyEventCfg = MyEventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    wheel_dof_name: str = ".*_wheel_joint"

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
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