from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


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


def randomize_obstacle_system(env, env_ids, cube_cfg, cyl_cfg):
    """
    Randomizes between a Cube and a Cylinder, scales them, and places 
    one in the path and hides the other.
    """
    num_resets = len(env_ids)
    device = env.device

    # 1. Access the Assets
    cube = env.scene["obstacle_cube"]
    cyl = env.scene["obstacle_cyl"]

    # 2. Randomly choose which shape is ACTIVE (0 = Cube, 1 = Cylinder)
    choice = torch.randint(0, 2, (num_resets,), device=device)

    # 3. Generate Random Scales (X and Y between 0.2m and 0.6m for a 15cm robot)
    # We keep Z at 1.0 since it's a 2D LiDAR
    random_scales = torch.rand((num_resets, 3), device=device) * 0.4 + 0.2
    random_scales[:, 2] = 1.0 

    # 4. Calculate Positions
    # Get the current target positions (assuming target is already reset)
    target_pos = env.scene["target"].data.root_pos_w[env_ids]
    
    # Place obstacle at 50% of the distance to target with a small Y-jitter
    # This ensures it's usually in the way but not always perfectly centered
    jitter = (torch.rand((num_resets,), device=device) - 0.5) * 0.8 # +/- 0.4m
    obs_pos = target_pos * 0.5
    obs_pos[:, 1] += jitter 
    obs_pos[:, 2] = 0.25 # Keep it on the ground

    # 5. The "Storage" Position (Your safe spot outside corridor walls)
    hidden_pos = torch.tensor([3.0, 2.5, 0.25], device=device).repeat(num_resets, 1)

    # 6. Apply logic: If choice == 0, Cube is at obs_pos, Cyl is hidden (and vice versa)
    cube_final_pos = torch.where(choice.unsqueeze(1) == 0, obs_pos, hidden_pos)
    cyl_final_pos = torch.where(choice.unsqueeze(1) == 1, obs_pos, hidden_pos)

    # 7. Write to Simulation
    # Set Positions
    cube.write_root_pose_to_sim(torch.cat([cube_final_pos, torch.zeros((num_resets, 4), device=device)], dim=-1), env_ids)
    cyl.write_root_pose_to_sim(torch.cat([cyl_final_pos, torch.zeros((num_resets, 4), device=device)], dim=-1), env_ids)
    
    # Set Scales (This assumes your framework supports runtime scaling via root_physx_view)
    # If not supported at runtime, you can set these in the 'startup' mode instead.
    # cube.set_prop_scales(random_scales, env_ids)