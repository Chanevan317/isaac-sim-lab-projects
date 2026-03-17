from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reset_robot_base_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, yaw_range: float = 0.0, lidar_enabled: bool = False, curr_level: int = 1):
    """Wrapper that pulls dynamic yaw/pos from env and calls the built-in reset."""

    # FRONT is -Y, TARGET is +X. 
    # To point local -Y to world +X, base yaw must be -1.5708 rad (-90 deg)
    base_yaw = 1.5708 
    
    # Pull dynamic range from the env (set by your curriculum function)
    yaw_range = getattr(env, "spawn_yaw_range", 0.0) 
    
    # Define the dynamic pose range
    pose_range = {
        "x": (0.0, 0.0), 
        "y": (0.0, 0.0), 
        "z": (0.1, 0.1),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (base_yaw - yaw_range, base_yaw + yaw_range),
    }

    velocity_range = {}

    # Call the built-in Isaac Lab function with our dynamic range
    return reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)


def adaptive_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, threshold: float = 0.85):
    current_success = torch.mean(env.extras.get("success_rate", torch.tensor(0.0)))
    print(f"DEBUG: Level: {getattr(env, 'curr_level', 'N/A')} | Success: {current_success:.4f}")

    if not hasattr(env, "curr_level"): env.curr_level = 1

    if current_success >= threshold:
        # Phase 1 -> 2: Add 45-degree orientation randomness
        if env.curr_level == 1:
            env.spawn_yaw_range = 0.785 # +/- 45 deg
            env.active_x_pos = (1.0, 2.0)
            env.curr_level = 2
            print(f">>> Level 2: Orientation Enabled (Success: {current_success:.2f})")
            env.extras["success_rate"] = torch.zeros_like(env.extras["success_rate"])
            
        # Phase 2 -> 3: 10m Distance + Full 360 Orientation
        elif env.curr_level == 2:
            env.active_y_range = (-0.25, 0.25)
            env.active_x_pos = (2.0, 3.0)
            env.spawn_yaw_range = 3.1415 # Full 360
            env.curr_level = 3
            print(f">>> Level 3: 10m Distance + Full Heading")
            env.extras["success_rate"] = torch.zeros_like(env.extras["success_rate"])

        # Phase 3 -> 4: Unlock LIDAR & Wide Corridor
        elif env.curr_level == 3:
            env.active_y_range = (-1.35, 1.35)
            env.lidar_enabled = True # This "unmasks" the observation function!
            env.curr_level = 4
            print(f">>> Level 4: LIDAR Observation UNLOCKED")
            env.extras["success_rate"] = torch.zeros_like(env.extras["success_rate"])