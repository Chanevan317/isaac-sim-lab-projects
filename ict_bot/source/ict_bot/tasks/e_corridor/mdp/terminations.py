from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.e_corridor.mdp.observations import rel_target_pos

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
    # if getattr(env, "curr_level", 1) >= 2:
    #     limit = 45.0
    if getattr(env, "curr_level", 1) >= 3:
        limit = 45.0
    # if getattr(env, "curr_level", 1) >= 4:
    #     limit = 100
        
    # Return a boolean tensor of environments that exceeded their specific limit
    return current_time_s >= limit


def stagnation_termination(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 0.005, time_limit: float = 2.0):
    """
    Terminates the episode if the robot doesn't move closer to the target.
    threshold: Minimum distance (meters) to be considered 'moving'.
    time_limit: Seconds allowed to be stuck before reset.
    """

    level = getattr(env, "curr_level", 1)

    if level == 2:
        threshold, time_limit = 0.01, 2.0
    elif level >= 3: # Phase 3 & 4
        threshold, time_limit = 0.05, 1.0

    robot = env.scene[robot_cfg.name]
    current_dist = torch.norm(env.target_pos[:, :2] - robot.data.root_pos_w[:, :2], dim=-1)
    
    # Calculate progress
    progress = env.prev_tgt_dist - current_dist
    stuck = progress < threshold
    
    # Update timer using torch.where to maintain tensor shape
    env.stagnation_timer = torch.where(
        stuck, 
        env.stagnation_timer + env.step_dt, 
        torch.zeros_like(env.stagnation_timer)
    )
    
    return env.stagnation_timer > time_limit