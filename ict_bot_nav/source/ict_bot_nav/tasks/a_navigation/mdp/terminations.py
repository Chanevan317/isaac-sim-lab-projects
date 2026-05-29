from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from .observations import lidar_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def stagnation_termination(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    min_lin: float = 0.1,    
    time_limit: float = 3.0,    
):
    """Terminate if robot makes no LINEAR progress for time_limit seconds.
    Angular movement alone does not reset the timer — prevents wall-spinning exploit.
    """
    if not hasattr(env, "stagnation_timer"):
        env.stagnation_timer = torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    linear_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)

    # Only linear speed matters — angular spinning does not count as progress
    not_moving = linear_speed < min_lin

    env.stagnation_timer = torch.where(
        not_moving,
        env.stagnation_timer + env.step_dt,
        torch.zeros_like(env.stagnation_timer),
    )

    return env.stagnation_timer >= time_limit


def lidar_collision(env, sensor_cfg, threshold: float = 0.25):
    scan = lidar_scan(env, sensor_cfg, num_beams=72)[:, :72]
    too_close = (scan < (threshold / 8.0)).float()
    return too_close.sum(dim=-1) >= 3  # at least 3 beams must agree


def tipping_termination(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    max_pitch_deg: float = 20.0,
    max_roll_deg: float = 20.0,
) -> torch.Tensor:
    """
    Sim-only termination — no real sensor required.
    Reads roll and pitch directly from physics quaternion state.
    Terminates when robot tips beyond threshold in either axis.
    Prevents LiDAR-blinding exploit where robot tips to blind its own scan plane.
    """
    robot = env.scene[robot_cfg.name]
    w = robot.data.root_quat_w[:, 0]
    x = robot.data.root_quat_w[:, 1]
    y = robot.data.root_quat_w[:, 2]
    z = robot.data.root_quat_w[:, 3]

    # Roll — rotation around X axis
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch — rotation around Y axis
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)

    max_roll  = max_roll_deg  * math.pi / 180.0
    max_pitch = max_pitch_deg * math.pi / 180.0

    return (torch.abs(roll) > max_roll) | (torch.abs(pitch) > max_pitch)