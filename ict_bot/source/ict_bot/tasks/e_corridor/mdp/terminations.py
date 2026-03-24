from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminates episodes at 30s for Ph1-2, and 45s for Ph3+."""
    # Convert current step buffer to seconds
    # (episode_length_buf counts steps, so we multiply by dt * decimation)
    current_time_s = env.episode_length_buf * env.step_dt
    
    # Determine the limit based on the level
    # Default to 30s if level is 1 or 2
    limit = 30.0
    if getattr(env, "curr_level", 1) >= 3:
        limit = 45.0
    if getattr(env, "curr_level", 1) >= 5:
        limit = 60
        
    # Return a boolean tensor of environments that exceeded their specific limit
    return current_time_s >= limit


def target_reached_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    return check_target_reached(env, robot_cfg)


def stagnation_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    level = getattr(env, "curr_level", 1)
    robot = env.scene[robot_cfg.name]
    
    # 1. Calculate actual movement (Kinematics) instead of distance to goal
    # This allows the robot to drive sideways to dodge walls in Phase 6
    linear_speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    angular_speed = torch.abs(robot.data.root_ang_vel_w[:, 2]) # Yaw rate
    
    # 2. Phase-based thresholds
    if level == 1:
        # Lenient: Just don't be completely frozen
        min_lin, min_ang, time_limit = 0.01, 0.05, 5.0 
    elif level == 2:
        min_lin, min_ang, time_limit = 0.02, 0.1, 4.0
    elif level in [3, 4]:
        # THE PIVOT PHASES: It might have 0 linear speed while doing a 180-turn.
        # As long as it is spinning (angular_speed), it is NOT stuck.
        min_lin, min_ang, time_limit = 0.02, 0.2, 8.0
    else: # Phase 5 & 6 (Sensors & Corridors)
        # Demand purposeful movement. Don't let it just vibrate in place.
        min_lin, min_ang, time_limit = 0.05, 0.1, 5.0

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
    
    return env.stagnation_timer > time_limit