from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


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
    return quat_apply(q_inv, pos_w)


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Angle to target where 0.0 rad is the -Y axis (The Face)."""
    local_pos = rel_target_pos(env, robot_cfg)
    
    # Using your -Y logic: Side is X, Forward is -Y
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    
    # Return as a 2-element vector per environment
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def lidar_distances(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_distance: float = 2.0):
    """Returns normalized Lidar distances [0, 1]. 0 is hit, 1 is clear."""
    raycaster = env.scene[sensor_cfg.name]
    
    # ray_hits_w is [num_envs, num_rays, 3]
    # pos_w is [num_envs, 3]
    hit_positions = raycaster.data.ray_hits_w
    
    # Unsqueeze pos_w to [num_envs, 1, 3] for broadcasting
    sensor_pos = raycaster.data.pos_w.unsqueeze(1)
    
    # Calculate Euclidean distance
    hit_dist = torch.norm(hit_positions - sensor_pos, dim=-1)
    
    # Normalize: 0.0 is a hit, 1.0 is a clear path or max range
    # Increased max_distance to 2.0m so the robot sees obstacles earlier
    return torch.clamp(hit_dist, max=max_distance) / max_distance


def imu_observations(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Returns IMU data: [ang_vel, lin_acc]."""
    # 1. Access the sensor data from the scene
    imu_sensor = env.scene[sensor_cfg.name]
    
    # 2. Get the readings
    # ang_vel_b is in the sensor's body frame
    # lin_acc_b (linear acceleration) includes gravity bias by default (+9.81 on Z)
    ang_vel = imu_sensor.data.ang_vel_b
    lin_acc = imu_sensor.data.lin_acc_b
    
    # 3. Concatenate and return
    # Resulting shape: [num_envs, 6]
    return torch.cat([ang_vel, lin_acc], dim=-1)