from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import get_tag_pixel_coords

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def track_tag_center_u(env: ManagerBasedRLEnv, std: float, tag_cfg: SceneEntityCfg) -> torch.Tensor:
    # 1. Get current pixel coordinates directly
    coords = get_tag_pixel_coords(env, tag_cfg)
    u_coords = coords[:, 0]
    visible = coords[:, 2]
    
    # 2. Gaussian reward: Only give reward if the tag is actually visible
    reward = torch.exp(-torch.square(u_coords) / (2 * std**2))
    return reward * visible


def forward_velocity_tracking(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    # Use the name from robot_cfg to find the asset
    robot = env.scene[robot_cfg.name]
    
    target_vx = env.command_manager.get_command(command_name)[:, 0]
    actual_vx = robot.data.root_lin_vel_w[:, 0]
    
    return torch.exp(-torch.abs(target_vx - actual_vx) / 0.25)


def log_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, index: int) -> torch.Tensor:
    # 1. Access the actual sensor instance from the scene using the config's name
    sensor = env.scene[sensor_cfg.name]
    
    # 2. Access the raw data buffer. 
    # For an IMU, this is usually .data.accel_w or .data.ang_vel_w
    # If your observation 'obs_term_name' was a concatenated IMU signal, 
    # we need to be specific about which buffer we are indexing into.
    
    # Example: if you are penalizing linear acceleration:
    val = sensor.data.lin_acc_b[:, index]
    
    return torch.abs(val)


def reach_tag_success(env: ManagerBasedRLEnv, tag_cfg: SceneEntityCfg, threshold_v: float = 0.85, threshold_u: float = 0.15) -> torch.Tensor:
    # 1. Get coordinates
    coords = get_tag_pixel_coords(env, tag_cfg)
    u, v, visible = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # 2. Condition: Bottom (v >= 0.85), Centered, and Visible
    is_reached = (v >= threshold_v) & (u.abs() <= threshold_u) & (visible > 0.5)
    return is_reached.float()
