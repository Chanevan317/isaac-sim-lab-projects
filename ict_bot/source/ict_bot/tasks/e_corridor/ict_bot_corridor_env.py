# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from ict_bot.assets.markers.target_cone import TARGET_CONE_CFG
from ict_bot.tasks.a_move_straight.ict_bot_env import MoveStraightSceneCfg
from ict_bot.tasks.a_move_straight.ict_bot_env import ActionsCfg as MoveStraightActionsCfg
import isaaclab.sim as sim_utils

import os
from ict_bot import ICT_BOT_ASSETS_DIR


# import mdp
import ict_bot.tasks.e_corridor.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns, ContactSensorCfg, ImuCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurTerm
from isaaclab.utils import configclass


##
# Scene definition
##


@configclass
class CorridorEnvSceneCfg(MoveStraightSceneCfg):
    """Configuration for the scene."""

    def __post_init__(self):
        super().__post_init__()

    # corridor scene asset
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacles",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "corridor.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Raycaster configuration for obstacle avoidance
    raycaster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacles", 
                is_shared=True, 
                merge_prim_meshes=True, 
                track_mesh_transforms=False
            )
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0), 
            horizontal_fov_range=(0.0, 360.0), 
            horizontal_res=5.0
        ),
        max_distance=4.0,
        debug_vis=True,
    )

    # Contact sensors to detect the collision with the base of the robot
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base", # Matches all robot parts
        update_period=0.0, # Update every physics step
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/obstacles"], # Only report contacts with obstacles
        visualizer_cfg=True,
    )

    # IMU sensor
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base", 
        update_period=0.01,
        gravity_bias=(0.0, 0.0, 9.81),
    )



##
# MDP settings
##


@configclass
class ActionsCfg(MoveStraightActionsCfg):
    """Action specifications for the MDP."""

    wheel_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint", "left_wheel_joint"],
        scale=5.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 1. Targeting (Essential for navigation)
        rel_target = ObsTerm(
            func=mdp.rel_target_pos, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [3]

        heading = ObsTerm(
            func=mdp.heading_error, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [2]

        # 2. Proprioception (Fixes the "weird" movement/speed control)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel
        )   # [2] - Required for real-world motor control
        
        # 3. Environment (Reduced to history_length=1 for stability)
        lidar = ObsTerm(
            func=mdp.lidar_distances, 
            params={"sensor_cfg": SceneEntityCfg("raycaster")},
            history_length=1 # Remove history here; the LSTM/RNN or Last Action handles timing better
        )   # [72]

        # 4. Stability & Physics (Essential for real-world IMU)
        imu = ObsTerm(
            func=mdp.imu_observations, 
            params={"sensor_cfg": SceneEntityCfg("imu")}
        )   # [6]
        
        # 5. Smoothness
        last_action = ObsTerm(
            func=mdp.last_action
        )   # [2]

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- POSITIVE MOTIVATION ---
    progress = RewTerm(
        func=mdp.reward_gated_progress_exponential,
        weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    
    forward_vel = RewTerm(
        func=mdp.forward_velocity_reward, 
        weight=1.0, 
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    reached = RewTerm(
        func=mdp.target_reached_reward,
        weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    # --- NEGATIVE CONSTRAINTS ---

    wall_dist = RewTerm(
        func=mdp.lidar_proximity_penalty,
        weight=-15.0,
        params={"sensor_cfg": SceneEntityCfg("raycaster")}
    )

    stability = RewTerm(
        func=mdp.imu_stability_phased,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("imu")}
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2_phased,
        weight=1.0,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_penalty_phased,
        weight=1.0 
    )

    alive = RewTerm(
        func=mdp.is_alive_phased,
        weight=1.0,
    )


@configclass
class MyEventCfg():
    """Event specifications for the MDP."""

    reset_robot_base = EventTerm(
        func=mdp.reset_robot_base_curriculum,
        mode="reset",
        params={
            "yaw_range": 0.0,
            "lidar_enabled": False,
            "curr_level": 1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_target_position = EventTerm(
        func=mdp.reset_target_marker_location,
        mode="reset",
        params={
            "y_range": (-0.1, 0.1), 
            "x_range": (0.9, 1.1)
        },
    )


@configclass
class TerminationsCfg():
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    no_progress = DoneTerm(
        func=mdp.stagnation_termination,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    reached_termination = DoneTerm(
        func=mdp.target_reached_termination, 
        params={"robot_cfg": SceneEntityCfg("robot")} 
    )

    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor"), 
            "threshold": 1.0, # Force in Newtons
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # This will check the success rate every 100 steps
    adaptive_task = CurTerm(
        func=mdp.adaptive_curriculum,
    )



##
# Environment configuration
##


@configclass
class CorridorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot."""

    # Scene settings
    scene: CorridorEnvSceneCfg = CorridorEnvSceneCfg(num_envs=4096, env_spacing=20.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: MyEventCfg = MyEventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    target_marker_cfg = TARGET_CONE_CFG

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 45.0
        # simulation settings
        self.sim.dt = 1.0 / 100.0


##
# Environment class
##


class CorridorEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: CorridorEnvCfg
    
    def __init__(self, cfg: CorridorEnvCfg, render_mode: str | None = None, **kwargs):

        self.target_pos = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)
        self.prev_tgt_dist = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        # Access the event parameters
        target_params = self.cfg.events.reset_target_position.params
        # We'll assume you add these new keys to your EventCfg (see below)
        robot_params = self.cfg.events.reset_robot_base.params

        # Initialize Curriculum Variables from Config
        # This allows PLAY mode to override them
        self.active_y_range = target_params.get("y_range", (-0.1, 0.1))
        self.active_x_pos = target_params.get("x_range", (0.9, 1.1))
        self.spawn_yaw_range = robot_params.get("yaw_range", 0.0)
        self.curr_level = robot_params.get("curr_level", 1)
        
        # Initialize Curriculum/Success Trackers
        # self.extras["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["success_rate"] = torch.tensor(0.0, device=self.device) # Change to scalar for mean tracking
        self.level_up_timer = 0  # Track steps spent above success threshold

        # timer to terminate if robot does not move
        self.stagnation_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Find wheel joint indices
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Use the config from the EnvCfg
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)


    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset selected environments."""
        super()._reset_idx(env_ids)
        
        # Handle None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 2. Immediately update the visual marker position for the reset envs
        if self.sim.has_gui():
            # We must pass the positions for ALL envs, or use a slice. 
            # The most robust way is to update all or use the visualizer's internal state.
            self.target_marker.visualize(self.target_pos)

        self.stagnation_timer[env_ids] = 0.0

        current_root_pos = self.scene["robot"].data.root_pos_w[env_ids]
        self.prev_tgt_dist[env_ids] = torch.norm(self.target_pos[env_ids] - current_root_pos, dim=-1)
        
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


    def _step_impl(self, actions: torch.Tensor):
        # Calculate 2D distance by zeroing out the Z difference
        target_rel_world = self.target_pos - self.scene["robot"].data.root_pos_w
        target_rel_world[:, 2] = 0.0 
        self.prev_tgt_dist[:] = torch.norm(target_rel_world, dim=-1)
        
        # Perform physics step
        super()._step_impl(actions)
        
        # Update visualization markers so they follow the 'target_pos' logic
        if self.sim.has_gui():
            self._set_debug_vis_impl(True)