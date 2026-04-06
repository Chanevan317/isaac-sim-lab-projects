from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def target_reached_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    return check_target_reached(env, robot_cfg)


def stagnation_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    
    linear_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    angular_speed = torch.abs(robot.data.root_ang_vel_w[:, 2]) # Yaw rate
    
    min_lin, min_ang, time_limit = 0.02, 0.05, 10.0
    
    # 3. Are we stuck? (Not moving fast enough linearly AND not turning fast enough)
    stuck = (linear_speed < min_lin) & (angular_speed < min_ang)
    
    # 4. Initialize buffer if it doesn't exist (safety catch)
    if not hasattr(env, "stagnation_timer"):
        env.stagnation_timer = torch.zeros(env.num_envs, device=env.device)
        
    # 5. Accumulate timer, but CRITICALLY: Reset to 0 if the env was reset this step!
    env.stagnation_timer = torch.where(
        stuck, 
        env.stagnation_timer + env.step_dt, 
        torch.zeros_like(env.stagnation_timer)
    )
    
    # Force timer to 0 for environments that just reset (prevents instant-death loops)
    if hasattr(env, "reset_buf"):
        env.stagnation_timer = torch.where(env.reset_buf, 0.0, env.stagnation_timer)
    
    result = env.stagnation_timer > time_limit
    # Clear timer for envs that just triggered stagnation termination
    env.stagnation_timer = torch.where(result, torch.zeros_like(env.stagnation_timer), env.stagnation_timer)

    return result