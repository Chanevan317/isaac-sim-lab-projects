# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from ict_bot_nav.assets.markers.target_cone import TARGET_CONE_CFG
from ict_bot_nav.assets.obstacles.object_obstacle import ObstacleSetCfg
from ict_bot_nav.assets.obstacles.object_obstacle import CUBE_SMALL_CFG, CUBE_MEDIUM_CFG, CUBE_LARGE_CFG, CYLINDER_SMALL_CFG, CYLINDER_MEDIUM_CFG, CYLINDER_LARGE_CFG
from ict_bot_nav.assets.robots.ict_bot import ICT_BOT_CFG
from .carrot import place_carrot, update_carrot
from .obstacle_manager import ObstacleManager
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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
            ),
            color=(0.5, 0.5, 1.0), # Set color as (R, G, B) between 0.0 and 1.0 
        ),
    )

    # corridor scene asset
    corridor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/corridor",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "corridor.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # corridor and obstacle assets
    cube_small      = CUBE_SMALL_CFG.replace(prim_path="{ENV_REGEX_NS}/CubeSmall")
    cube_medium     = CUBE_MEDIUM_CFG.replace(prim_path="{ENV_REGEX_NS}/CubeMedium")
    cube_large      = CUBE_LARGE_CFG.replace(prim_path="{ENV_REGEX_NS}/CubeLarge")
    cylinder_small  = CYLINDER_SMALL_CFG.replace(prim_path="{ENV_REGEX_NS}/CylinderSmall")
    cylinder_medium = CYLINDER_MEDIUM_CFG.replace(prim_path="{ENV_REGEX_NS}/CylinderMedium")
    cylinder_large  = CYLINDER_LARGE_CFG.replace(prim_path="{ENV_REGEX_NS}/CylinderLarge")

    # robot
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
                prim_expr="{ENV_REGEX_NS}/corridor", 
                is_shared=True, 
                merge_prim_meshes=True, 
                track_mesh_transforms=False
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CubeSmall",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CylinderSmall",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CubeMedium",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CylinderMedium",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CubeLarge",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/CylinderLarge",
                is_shared=False, 
                merge_prim_meshes=False, 
                track_mesh_transforms=True
            ),
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, 
            vertical_fov_range=(0.0, 0.0), 
            horizontal_fov_range=(0.0, 360.0), 
            horizontal_res=2.0,  # 180 beams for full 360° coverage
        ),
        update_period=0.05,
        max_distance=3.0,
        debug_vis=True,
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ict_bot_01/link_base",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/corridor",   
            "{ENV_REGEX_NS}/CubeSmall",
            "{ENV_REGEX_NS}/CubeMedium",
            "{ENV_REGEX_NS}/CubeLarge",
            "{ENV_REGEX_NS}/CylinderSmall",
            "{ENV_REGEX_NS}/CylinderMedium",
            "{ENV_REGEX_NS}/CylinderLarge",
        ], 
        visualizer_cfg=None,
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
        scale=15.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):

        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={
                "sensor_cfg": SceneEntityCfg("raycaster"), 
            }
        )   # [180] — 180 beams
        
        # Targeting (Essential for navigation)
        rel_target = ObsTerm(
            func=mdp.rel_line_dist,
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [2]

        heading = ObsTerm(
            func=mdp.heading_to_line, 
            params={"robot_cfg": SceneEntityCfg("robot")}
        )   # [2]

        # Proprioception (Fixes the "weird" movement/speed control)
        joint_vel = ObsTerm(
            func=mdp.joint_velocity,
            params={"robot_cfg": SceneEntityCfg("robot", joint_names=["left_wheel_joint", "right_wheel_joint"])}
        )   # [2] - Required for real-world motor control

        robot_vel = ObsTerm(
            func=mdp.root_lin_vel_b_2d,    # body-frame linear velocity [vx, vy]
            params={"robot_cfg": SceneEntityCfg("robot")}
        )  # [3]

        robot_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_b_z,    # body-frame angular velocity [wz]
            params={"robot_cfg": SceneEntityCfg("robot")}
        )  # [3]


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- POSITIVE MOTIVATION ---

    velocity_toward_carrot = RewTerm(
        func=mdp.reward_velocity_toward_carrot,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
        }
    )

    carrot_pass = RewTerm(
        func=mdp.reward_carrot_pass,
        weight=12.0,             # large sparse bonus — clear peak signal for PPO
    )

    # --- NEGATIVE CONSTRAINTS ---

    proximity_penalty = RewTerm(
        func=mdp.lidar_proximity_penalty,
        weight=-3.0,                    # negative — quadratic output is always positive
        params={
            "sensor_cfg": SceneEntityCfg("raycaster"),
        }
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,   # built-in Isaac Lab term, fires -1 on termination step
        weight=-50.0,
    )

    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=-0.01,
    # )


@configclass
class MyEventCfg():
    """Event specifications for the MDP."""

    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (0.0, 1.0),
                "y":     (-0.75, 0.75),
                "z":     (0.1, 0.1),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (1.0, 2.14),
            },
            "velocity_range": {
                "linear":  {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
                "angular": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            }
        },
    )

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
class CurriculumCfg:
    # spawn_yaw = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "param_name": "pose_range.yaw",
    #         "start_value": (-0.5, 0.5),      # ~±28° — mostly facing target
    #         "end_value":   (-3.1415, 3.1415), # full random — any direction
    #         "num_steps":   5,                 # 5 curriculum steps to full random
    #     }
    # )

    obstacle_difficulty = CurrTerm(
        func=mdp.obstacle_curriculum_term,
        params={"cfg": mdp.ObstacleCurriculumCfg(
            eval_window=1000,
            consecutive_windows_to_promote=2,
            success_key="goal_reached",
        )},
    )


@configclass
class TerminationsCfg():
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the robot touches anything with a force > 1.0 N
    collision_termination = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor"), 
            "threshold": 1.0 # Force in Newtons
        }
    )



##
# Environment configuration
##


@configclass
class NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot."""

    # Scene settings
    scene: NavigationEnvSceneCfg = NavigationEnvSceneCfg(num_envs=2048, env_spacing=21.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: MyEventCfg = MyEventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    wheel_dof_name: list[str] = ["right_wheel_joint", "left_wheel_joint"]
    target_marker_cfg = TARGET_CONE_CFG

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 100.0
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solver_velocity_iteration_count = 1


##
# Environment class
##


class NavigationEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: NavigationEnvCfg
    

    def __init__(self, cfg: NavigationEnvCfg, render_mode: str | None = None, **kwargs):
        # Must initialize before super().__init__ — managers read these
        self.target_pos       = torch.zeros((cfg.scene.num_envs, 3), device=cfg.sim.device)
        self.prev_tgt_dist    = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.episode_start_x  = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.stagnation_timer = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.carrot_pass_count = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        super().__init__(cfg, render_mode, **kwargs)

        self.extras.setdefault("episode", {})["goal_reached"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # Post-init state
        self.extras["log"] = {}

        # Wheel joints
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)

        # Visualization
        self.target_marker = VisualizationMarkers(self.cfg.target_marker_cfg)

        self.obstacle_manager = ObstacleManager(
            env=self,
            assets=[
                self.scene["cube_large"],
                self.scene["cube_medium"],
                self.scene["cube_small"],
                self.scene["cylinder_large"],
                self.scene["cylinder_medium"],
                self.scene["cylinder_small"],
            ],
            cfg=ObstacleSetCfg(
                max_obstacles=4,          # matches curriculum ceiling
                shapes=[0, 1, 2, 3, 4, 5],
                corridor_half_width=1.5,
                spawn_radius=4.0,
                min_obstacle_spacing=0.3,
                max_speed=0.8,
            ),
        )


    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        self.stagnation_timer[env_ids] = 0.0

        # Success signal for curriculum — robot passed >= 3 carrots this episode
        self.extras["episode"]["goal_reached"][env_ids] = (
            self.carrot_pass_count[env_ids] >= 3.0
        ).float()

        # Log pass count for training monitor before reset
        # (training.py reads this before _reset_idx via snapshot — see below)

        self.episode_start_x[env_ids] = self.scene["robot"].data.root_pos_w[env_ids, 0]

        # if hasattr(self, "lidar_prev") and self.lidar_prev is not None:
        #     self.lidar_prev[env_ids] = 0.0

        if hasattr(self, "_lidar_history") and self._lidar_history is not None:
            self._lidar_history[env_ids] = 0.0

        place_carrot(self, env_ids)   # resets carrot_pass_count[env_ids] to 0.0

        if hasattr(self, "_prev_carrot_pass_count"):
            self._prev_carrot_pass_count[env_ids] = 0.0

        self.obstacle_manager.reset(env_ids)

        n  = len(env_ids)
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

        self.obstacle_manager.step()

        # post-step counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # rewards — uses updated carrot position
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

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

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # observations — after reset so reset envs get correct obs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras