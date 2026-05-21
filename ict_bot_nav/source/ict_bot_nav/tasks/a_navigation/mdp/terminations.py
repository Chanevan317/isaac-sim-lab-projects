from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def stagnation_termination(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    min_lin: float = 0.02,
    min_ang: float = 0.05,
    time_limit: float = 10.0,
):
    """Terminates episode if robot is stuck for longer than time_limit seconds."""
    if not hasattr(env, "stagnation_timer"):
        env.stagnation_timer = torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]
    linear_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    angular_speed = torch.abs(robot.data.root_ang_vel_w[:, 2])

    stuck = (linear_speed < min_lin) & (angular_speed < min_ang)

    env.stagnation_timer = torch.where(
        stuck,
        env.stagnation_timer + env.step_dt,
        torch.zeros_like(env.stagnation_timer),
    )

    return env.stagnation_timer >= time_limit


def lidar_collision(env, sensor_cfg, threshold: float = 0.25):
    sensor = env.scene[sensor_cfg.name]
    hits   = sensor.data.ray_hits_w      # [N, B, 3]
    origin = sensor.data.pos_w           # [N, 3]

    # Distance from sensor origin to each hit point
    diff  = hits - origin.unsqueeze(1)   # [N, B, 3]
    dists = torch.norm(diff, dim=-1)     # [N, B]

    min_dist = dists.min(dim=-1).values  # [N]
    return min_dist < threshold          # [N] bool