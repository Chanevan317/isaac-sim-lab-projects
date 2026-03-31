from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_target_marker_location(env: ManagerBasedRLEnv, env_ids: torch.Tensor, y_range: tuple[float, float], x_range: tuple[float, float]):
    num_resets = len(env_ids)
    device = env.device
    
    # 1. Check if the curriculum has updated these values on the 'env'
    # Otherwise, use the 'y_range' and 'x_range' passed from the config
    current_y_range = getattr(env, "active_y_range", y_range)
    current_x_range = getattr(env, "active_x_pos", x_range)
    
    # 2. Randomize as usual
    y_local = torch.empty(num_resets, device=device).uniform_(*current_y_range)
    x_local = torch.empty(num_resets, device=device).uniform_(*current_x_range)
    
    # 3. Apply to world space
    env.target_pos[env_ids, 0] = x_local + env.scene.env_origins[env_ids, 0]
    env.target_pos[env_ids, 1] = y_local + env.scene.env_origins[env_ids, 1]
    env.target_pos[env_ids, 2] = env.scene.env_origins[env_ids, 2] + 0.25