from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def rel_target_dist(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Scalar distance to waypoint target, ignoring height differences."""
    local_pos = rel_target_pos(env, robot_cfg)  # [N, 2]
    dist = torch.norm(local_pos, dim=-1, keepdim=True)  # [N, 1]
    return dist


def rel_target_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Target position in robot's USD local frame."""
    robot = env.scene[robot_cfg.name]
    target = env.target_pos
    
    # Calculate world-space vector
    pos_w = target - robot.data.root_pos_w
    
    # CRITICAL: Ignore height differences for 2D corridor navigation
    pos_w[:, 2] = 0.0 
    
    # Rotate into robot's local frame
    q_inv = quat_inv(robot.data.root_quat_w)
    local_pos = quat_apply(q_inv, pos_w) # [N, 3]
    return local_pos[:, :2] # drop z, return [N, 2]


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Angle to target where 0.0 rad is the -Y axis (The Face)."""
    local_pos = rel_target_pos(env, robot_cfg)
    
    # Using your -Y logic: Side is X, Forward is -Y
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    
    # Return as a 2-element vector per environment
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def lidar_scan(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, num_beams: int = 72):
    """
    Returns stacked [lidar_t, lidar_t-1] normalised to [0,1] with Gaussian noise.
    Returns zeros if sensor not present (navigation-only stage).
    Shape: [N, 144]
    """
    # --- Guard: return zeros if sensor not in scene (Stage 1 navigation) ---
    if sensor_cfg.name not in env.scene.keys():
        if not hasattr(env, "lidar_prev") or env.lidar_prev is None:
            env.lidar_prev = torch.zeros(env.num_envs, num_beams, device=env.device)
        return torch.zeros(env.num_envs, num_beams * 2, device=env.device)

    # --- Get current scan from ray caster ---
    sensor = env.scene[sensor_cfg.name]
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
    hits = sensor.data.ray_hits_w                                 # [N, num_beams, 3]
    distances = torch.norm(hits - robot_pos, dim=-1)              # [N, num_beams]

    # --- Clamp and normalise ---
    max_range = 8.0
    distances = torch.clamp(distances, 0.0, max_range) / max_range

    # --- Gaussian noise for sim-to-real ---
    noise = torch.randn_like(distances) * 0.02
    lidar_t = torch.clamp(distances + noise, 0.0, 1.0)

    # --- Initialise prev on first call ---
    if not hasattr(env, "lidar_prev") or env.lidar_prev is None:
        env.lidar_prev = lidar_t.clone()

    lidar_t1 = env.lidar_prev.clone()
    env.lidar_prev = lidar_t.clone()

    return torch.cat([lidar_t, lidar_t1], dim=-1)  # [N, 144]
