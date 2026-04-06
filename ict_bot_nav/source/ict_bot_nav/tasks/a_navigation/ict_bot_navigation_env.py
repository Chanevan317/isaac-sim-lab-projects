# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from ict_bot_nav.assets.markers.target_cone import TARGET_CONE_CFG
from ict_bot_nav.assets.robots.ict_bot import ICT_BOT_CFG
from .carrot import place_carrot, update_carrot
import isaaclab.sim as sim_utils

import os
from ict_bot_nav import ICT_BOT_ASSETS_DIR


# import mdp
import ict_bot_nav.tasks.a_navigation.mdp as mdp
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns, ContactSensorCfg
from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
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
class NavigationEnvSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    def __post_init__(self):
        super().__post_init__()

    # world
    ground_plane = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # corridor scene asset
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/obstacles",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "corridor.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # robots
    robot: ArticulationCfg = ICT_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
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


##
# MDP settings
##


@configclass
class ActionsCfg:
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

        # Targeting (Essential for navigation)
        rel_target = ObsTerm(
            func=mdp.rel_target_pos, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [2]

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

        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"num_beams": 72}
        )   # [72] - 360° scan with 5° resolution, max distance 4m, kept zero for the navigation task

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
        weight=50.0,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    heading = RewTerm(
        func=mdp.reward_heading_alignment,
        weight=2.0,
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
        weight=-0.5,
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=-1.0,
    )


@configclass
class MyEventCfg():
    """Event specifications for the MDP."""

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0), 
                "y": (-0.3, 0.3), 
                "z": (0.1, 0.1),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.1415, 3.1415),
            },
            "velocity_range": {}
        },
    )

    # reset_target_position = EventTerm(
    #     func=mdp.reset_target_marker_location,
    #     mode="reset",
    #     params={
    #         "min_distance": 1.0,
    #         "max_distance": 2.5,
    #     },
    # )

    # randomize_wheel_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*wheel_joint"]),
    #         "static_friction_range": (0.5, 1.5),
    #         "dynamic_friction_range": (0.4, 1.2),
    #         "restitution_range": (0.0, 0.1),
    #         "num_buckets": 250,
    #     }
    # )

    # # Robot mass randomization — accounts for payload, battery charge variation
    # randomize_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #         "mass_distribution_params": (0.8, 1.2),  # ±20% of nominal mass
    #         "operation": "scale",
    #     }
    # )

    # Push randomization — random impulses simulate bumps and disturbances
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(3.0, 6.0),  # random push every 3–6 seconds
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "velocity_range": {
    #             "x": (-0.3, 0.3),
    #             "y": (-0.3, 0.3),
    #             "yaw": (-0.5, 0.5),
    #         }
    #     }
    # )


@configclass
class TerminationsCfg():
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    no_progress = DoneTerm(
        func=mdp.stagnation_termination,
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    # reached_termination = DoneTerm(
    #     func=mdp.target_reached_termination, 
    #     params={"robot_cfg": SceneEntityCfg("robot")} 
    # )



##
# Environment configuration
##


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot."""

    # Scene settings
    scene: NavigationEnvSceneCfg = NavigationEnvSceneCfg(num_envs=4096, env_spacing=20.0)
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
        self.episode_length_s = 25.0
        # simulation settings
        self.sim.dt = 1.0 / 100.0


##
# Environment class
##


class NavigationEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: NavigationEnvCfg
    

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        # Must initialize before super().__init__ — managers read these
        self.target_pos = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)
        self.prev_tgt_dist = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.carrot_forward_dir = torch.zeros((cfg.scene.num_envs, 2), device=cfg.sim.device)
        self.carrot_timer = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.carrot_advance_count = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        # Post-init state
        self.stagnation_timer = torch.zeros(self.num_envs, device=self.device)
        self.extras["log"] = {}

        # Wheel joints
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Visualization
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)


    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset carrot state for these envs
        self.carrot_timer[env_ids] = 0.0
        self.carrot_advance_count[env_ids] = 0.0  # reset counter at episode start
        place_carrot(self, env_ids)

        # Reset stagnation
        self.stagnation_timer[env_ids] = 0.0

        # Reset wheels
        n = len(env_ids)
        nw = len(self._wheel_indices)
        self.scene["robot"].write_joint_state_to_sim(
            torch.zeros((n, nw), device=self.device),
            torch.zeros((n, nw), device=self.device),
            joint_ids=self._wheel_indices,
            env_ids=env_ids,
        )

        if self.sim.has_gui():
            self.target_marker.visualize(self.target_pos)


    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # --- carrot update — after physics, before rewards ---
        update_carrot(self)
        if self.sim.has_gui():
            self.target_marker.visualize(self.target_pos)
        # -----------------------------------------------------

        # post-step counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # rewards — uses updated carrot position
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # reset terminated envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # command manager
        self.command_manager.compute(dt=self.step_dt)

        # interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # observations — after reset so reset envs get correct obs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras