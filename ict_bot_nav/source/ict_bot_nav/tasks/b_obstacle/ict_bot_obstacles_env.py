# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from ict_bot_nav.assets.markers.target_cone import TARGET_CONE_CFG
from ict_bot_nav.assets.obstacles.static_obstacle import CUBE_OBSTACLE_CFG, CYLINDER_OBSTACLE_CFG
from ict_bot_nav.tasks.a_navigation.ict_bot_navigation_env import NavigationEnvSceneCfg
from ict_bot_nav.tasks.a_navigation.ict_bot_navigation_env import ActionsCfg as NavigationActionsCfg
from ict_bot_nav.tasks.a_navigation.ict_bot_navigation_env import MyEventCfg as NavigationEventCfg
from ict_bot_nav.tasks.a_navigation.ict_bot_navigation_env import TerminationsCfg as NavigationTerminationsCfg

import isaaclab.sim as sim_utils

import os
from ict_bot_nav import ICT_BOT_ASSETS_DIR


# import mdp
import ict_bot_nav.tasks.b_obstacle.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


##
# Scene definition
##


@configclass
class ObstacleEnvSceneCfg(NavigationEnvSceneCfg):
    """Configuration for the scene."""

    def __post_init__(self):
        super().__post_init__()

    static_cube = CUBE_OBSTACLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/obstacles/cube"
    )

    static_cylinder = CYLINDER_OBSTACLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/obstacles/cylinder"
    )


##
# MDP settings
##


@configclass
class ActionsCfg(NavigationActionsCfg):
    """Action specifications for the MDP."""


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Targeting (Essential for navigation)
        rel_target = ObsTerm(
            func=mdp.rel_target_pos, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [3]

        heading = ObsTerm(
            func=mdp.heading_error, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [2]

        # Proprioception (Fixes the "weird" movement/speed control)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel
        )   # [2] - Required for real-world motor control

        # Smoothness
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
        func=mdp.velocity_toward_target,
        weight=1.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    speed_bonus = RewTerm(
        func=mdp.reward_forward_speed,
        weight=70.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    reached = RewTerm(
        func=mdp.target_reached_reward,
        weight=50.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    # --- NEGATIVE CONSTRAINTS ---

    backward = RewTerm(
        func=mdp.penalize_backwards_movement,
        weight=-5.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.25,
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=-0.5,
    )


@configclass
class MyEventCfg(NavigationEventCfg):
    """Event specifications for the MDP."""

    reset_target_position = EventTerm(
        func=mdp.reset_target_marker_location,
        mode="reset",
        params={
            "x_range": (1.0, 3.0),
            "y_range": (-1.0, 1.0), 
        },
    )


@configclass
class TerminationsCfg(NavigationTerminationsCfg):
    """Termination terms for the MDP."""



##
# Environment configuration
##


@configclass
class ObstacleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot."""

    # Scene settings
    scene: ObstacleEnvSceneCfg = ObstacleEnvSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: MyEventCfg = MyEventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    target_marker_cfg = TARGET_CONE_CFG

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 40.0
        # simulation settings
        self.sim.dt = 1.0 / 100.0


##
# Environment class
##


class ObstacleEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: ObstacleEnvCfg
    
    def __init__(self, cfg: ObstacleEnvCfg, render_mode: str | None = None, **kwargs):

        self.target_pos = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)
        self.prev_tgt_dist = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize Curriculum/Success Trackers
        # self.extras["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["success_rate"] = torch.tensor(0.0, device=self.device) # Change to scalar for mean tracking

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
        diff = self.target_pos[env_ids] - current_root_pos
        diff[:, 2] = 0.0  # ignore height
        self.prev_tgt_dist[env_ids] = torch.norm(diff, dim=-1)
        
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
        
        # Perform physics step
        super()._step_impl(actions)
        
        # Update visualization markers so they follow the 'target_pos' logic
        if self.sim.has_gui():
            self._set_debug_vis_impl(True)