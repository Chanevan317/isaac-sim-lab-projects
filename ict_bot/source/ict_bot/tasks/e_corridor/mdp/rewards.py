from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reward_navigate_to_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)
    h_error = heading_error(env, robot_cfg)
    cos_angle = h_error[:, 1]

    # --- Progress reward, gated by alignment ---
    dist_delta = (env.prev_tgt_dist - current_dist) / env.step_dt
    if hasattr(env, "reset_buf"):
        dist_delta = torch.where(env.reset_buf, torch.zeros_like(dist_delta), dist_delta)
    env.prev_tgt_dist = current_dist.clone()

    alignment_gate = torch.sigmoid(3.0 * cos_angle)
    progress = torch.clamp(dist_delta, min=0.0) * alignment_gate * 20.0

    # --- Heading improvement ---
    if not hasattr(env, "prev_heading_cos"):
        env.prev_heading_cos = cos_angle.clone()
        heading_imp = torch.zeros(env.num_envs, device=env.device)
    else:
        improvement = cos_angle - env.prev_heading_cos
        if hasattr(env, "reset_buf"):
            improvement = torch.where(env.reset_buf, torch.zeros_like(improvement), improvement)
        env.prev_heading_cos = cos_angle.clone()
        heading_imp = torch.clamp(improvement, min=0.0) * 80.0

    # --- Backward penalty only when already well-aligned ---
    robot = env.scene[robot_cfg.name]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    forward_speed = -local_vel[:, 1]
    badly_misaligned = cos_angle < -0.3
    backward_penalty = torch.where(
        (forward_speed < 0) & (~badly_misaligned),
        forward_speed * 10.0,
        torch.zeros_like(forward_speed)
    )

    return progress + heading_imp + backward_penalty


def target_reached_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    reached = check_target_reached(env, robot_cfg)
    terminating_envs = env.reset_buf
    if terminating_envs.any():
        successes = reached[terminating_envs].float().mean()
        env.extras["success_rate"] = 0.98 * env.extras.get(
            "success_rate", torch.tensor(0.0, device=env.device)
        ) + 0.02 * successes
    return reached.float() * 3000.0




def imu_stability_phased(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    if level < 2:
        return torch.zeros(env.num_envs, device=env.device)

    imu_data = imu_observations(env, sensor_cfg)
    yaw_rate = torch.abs(imu_data[:, 2])
    
    # Allow a small amount of natural turning (0.5 rad/s) before penalizing
    excess_spin = torch.clamp(yaw_rate - 0.5, min=0.0)
    return -0.5 * excess_spin


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold=0.25):
    level = getattr(env, "curr_level", 1)
    if level < 2:
        return torch.zeros(env.num_envs, device=env.device)
    
    lidar_values = lidar_distances(env, sensor_cfg, max_distance=1.0)
    min_dist, _ = torch.min(lidar_values, dim=-1)
    
    # Soft buffer penalty
    penalty = torch.where(min_dist < (threshold / 0.5), -2.0, 0.0)
    return penalty