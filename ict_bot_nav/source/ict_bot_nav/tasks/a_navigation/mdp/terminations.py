from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def stagnation_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Terminates episode if robot stops moving for too long."""
    robot = env.scene[robot_cfg.name]

    linear_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    angular_speed = torch.abs(robot.data.root_ang_vel_w[:, 2])

    min_lin, min_ang, time_limit = 0.02, 0.05, 10.0

    stuck = (linear_speed < min_lin) & (angular_speed < min_ang)

    env.stagnation_timer = torch.where(
        stuck,
        env.stagnation_timer + env.step_dt,
        torch.zeros_like(env.stagnation_timer)
    )

    if hasattr(env, "reset_buf"):
        env.stagnation_timer = torch.where(
            env.reset_buf,
            torch.zeros_like(env.stagnation_timer),
            env.stagnation_timer
        )

    result = env.stagnation_timer > time_limit
    env.stagnation_timer = torch.where(
        result,
        torch.zeros_like(env.stagnation_timer),
        env.stagnation_timer
    )

    return result