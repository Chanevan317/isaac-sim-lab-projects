from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.e_corridor.mdp.observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from isaaclab.utils.math import quat_inv, quat_apply
from isaaclab.envs.mdp import action_rate_l2, joint_vel_l2

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reward_gated_progress_exponential(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)
    dist_delta = (env.prev_tgt_dist - current_dist) / env.step_dt
    
    # PHASE 1: Pure distance reduction. No questions asked.
    if level == 1:
        return torch.clamp(dist_delta, min=0.0) * 15.0 

    # PHASE 2+: Re-introduce gates for precision
    pow_val = 2.0 if level == 2 else 4.0
    speed_thresh = 0.05 if level == 2 else 0.2
    
    h_error = heading_error(env, robot_cfg)
    alignment_gate = torch.pow(torch.clamp(h_error[:, 1], min=0.0), pow_val)
    speed_gate = torch.sigmoid(10.0 * (dist_delta - speed_thresh))
    
    return torch.clamp(dist_delta, min=0.0) * alignment_gate * speed_gate * 20.0


def forward_velocity_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    robot = env.scene[robot_cfg.name]
    
    # Local -Y is forward
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    forward_speed = -local_vel[:, 1] # Positive = Forward, Negative = Backward
    
    # Get alignment (1.0 = perfect, 0.0 = perpendicular, -1.0 = backwards)
    h_error = heading_error(env, robot_cfg)
    alignment = h_error[:, 1]
    
    if level == 1:
        # PHASE 1: Simple Carrot & Stick
        # If moving forward, reward is boosted by alignment. 
        # If moving backward, it's a flat penalty to kill the habit.
        reward = torch.where(forward_speed > 0, forward_speed * alignment, forward_speed * 2.0)
        weight = 15.0
    elif level == 2:
        # PHASE 2+: The Alignment Gate
        # Robot ONLY gets forward reward if alignment > 0.5 (approx 60 degrees)
        gate = torch.clamp(alignment, min=0.0)
        reward = forward_speed * torch.pow(gate, 2.0)
        weight = 10.0 if level == 2 else 5.0
    else:
        gate = torch.clamp(alignment, min=0.0)
        reward = forward_speed * torch.pow(gate, 4.0)
        weight = 5.0
    
    return reward * weight


def target_reached_reward_phased(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, distance: float = 0.3):
    level = getattr(env, "curr_level", 1)
    
    # Calculate distance (Ignoring Z)
    robot = env.scene[robot_cfg.name]
    dist = torch.norm(env.target_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    
    # 30cm threshold
    reached = (dist < distance).float()

    # Hardcoded Logic
    if level == 1: prize = 500.0
    elif level == 2: prize = 1000.0
    elif level == 3: prize = 5000.0
    else: prize = 15000.0 # Level 4
        
    return reached * prize


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.25):
    """Penalizes getting too close to walls based on LIDAR."""
    # Use your normalized lidar_distances [0, 1]
    # 0.0 is a hit, 1.0 is clear (at max_distance=0.5m)
    lidar_values = lidar_distances(env, sensor_cfg, max_distance=1.0)
    
    # Find the closest point in the entire scan
    min_dist, _ = torch.min(lidar_values, dim=-1)
    
    # If min_dist < threshold (e.g., 0.3/0.5 = 0.6 normalized), apply penalty
    # This creates a "soft" buffer around the robot
    penalty = torch.where(min_dist < (threshold / 0.5), -1.0, 0.0)
    return penalty


def imu_stability_phased(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    
    # Disable for Phase 1 & 2
    if level <= 2:
        return torch.zeros(env.num_envs, device=env.device)

    imu_data = imu_observations(env, sensor_cfg)
    yaw_rate = torch.abs(imu_data[:, 2])
    
    # Penalize only when level is high
    weight = -0.025 if level == 3 else -0.5
    return weight * torch.clamp(yaw_rate - 0.1, min=0.0)


def action_rate_l2_phased(env: ManagerBasedRLEnv):
    level = getattr(env, "curr_level", 1)
    
    if level == 1:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Standard penalty for Level 2+
    penalty = action_rate_l2(env)
    
    # You can also scale it by level here if you want
    weight = 0.01 if level == 2 else 0.05
    return penalty * weight


def is_alive_phased(env: ManagerBasedRLEnv):
    level = getattr(env, "curr_level", 1)
    
    # Increasing weights directly
    if level == 1: weight = -0.5
    elif level == 2: weight = -2.0
    else: weight = -5.0 # Level 3 & 4
        
    return torch.ones(env.num_envs, device=env.device) * weight


def joint_vel_penalty_phased(env: ManagerBasedRLEnv):
    """Penalizes high wheel speeds to prevent slipping, active from Phase 2."""
    level = getattr(env, "curr_level", 1)
    
    # Phase 1: No penalty (let it spin to discover movement)
    if level == 1:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Built-in MDP function for L2 norm of joint velocities
    # This penalizes the square of the wheel speeds
    raw_penalty = joint_vel_l2(env)
    
    # Phase 2: Light penalty | Phase 3-4: Stronger penalty
    weight = -0.0001 if level == 2 else -0.0005
    
    return raw_penalty * weight