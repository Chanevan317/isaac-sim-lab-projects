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
    # current_success = torch.mean(env.extras.get("success_rate", torch.tensor(0.0)))
    # print(f"DEBUG: Level: {getattr(env, 'curr_level', 'N/A')} | Success: {current_success:.4f}")

    if not hasattr(env, "curr_level"): env.curr_level = 1
    if not hasattr(env, "level_up_timer"): env.level_up_timer = 0

    current_success = env.extras.get("success_rate", torch.tensor(0.0, device=env.device)).item()
    print(f"DEBUG: Level: {getattr(env, 'curr_level', 'N/A')} | Success: {current_success:.4f}")

    # Stability Check
    if current_success >= threshold:
        env.level_up_timer += 1
    else:
        env.level_up_timer = 0

    # Only level up if it's been successful for a sustained period
    if env.curr_level <= 2:
        ready_to_level_up = (env.level_up_timer > 1000)
    elif env.curr_level <= 4:
        ready_to_level_up = (env.level_up_timer > 3000)
    else:
        ready_to_level_up = (env.level_up_timer > 7500)

    if ready_to_level_up:
        # Reset the timer for the next level
        env.level_up_timer = 0

        # Phase 1 -> 2: Add 45-degree orientation randomness
        if env.curr_level == 1:
            env.active_y_range = (-0.25, 0.25)
            env.spawn_yaw_range = 0.785 # +/- 45 deg
            env.active_x_pos = (1.0, 2.0)
            env.curr_level = 2
            print(f">>> Level 2: Orientation 90 deg")
        
        # Phase 2 -> 3: Add 45-degree orientation randomness
        elif env.curr_level == 2:
            env.active_y_range = (-0.75, 0.75)
            env.spawn_yaw_range = 1.8326 # +/- 105 deg
            env.active_x_pos = (1.5, 3.0)
            env.curr_level = 3
            print(f">>> Level 3: Orientation 210 deg")

        # Phase 3 -> 4:  Full 360 Orientation
        elif env.curr_level == 3:
            env.active_y_range = (-1.35, 1.35)
            env.spawn_yaw_range = 3.1415 # Full 360
            env.curr_level = 4
            print(f">>> Level 4: Full Heading")

        # Phase 4 -> 5: Unlock IMU
        elif env.curr_level == 4:
            env.curr_level = 5
            print(f">>> Level 5: IMU Observation unlocked")

        # Phase 5 -> 6: Unlock Lidar
        elif env.curr_level == 5:
            env.lidar_enabled = True 
            env.curr_level = 6
            print(f">>> Level 6: LIDAR Observation unlocked")


        if hasattr(env, "runner"):
            agent = env.runner.agent
            
            # --- Professional Hyperparameter Strategy ---
            if env.curr_level <= 2:
                # Base learning for simple forward movement
                new_lr = 3e-4 
                new_entropy = 0.01 
            elif env.curr_level == 3:
                # Step up: Moderate boost to handle wider target angles
                new_lr = 4e-4 
                new_entropy = 0.02
            elif env.curr_level == 4:
                # The Pivot: High energy and exploration to discover the 180-degree turn
                new_lr = 5e-4 
                new_entropy = 0.05
            else: # Level 5 and 6 (Sensors/IMU/Lidar)
                # Mastery: Lower the values to "cool down" the policy. 
                # This stops the jitter and allows the robot to learn precise sensor-based steering.
                new_lr = 3e-4 
                new_entropy = 0.02

            # 1. Update the Optimizer (Current Weights)
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = new_lr
                
            # 2. Update the Scheduler (Prevents immediate reversal)
            if hasattr(agent, "scheduler") and agent.scheduler is not None:
                # Set the internal base learning rate for the KLAdaptiveLR scheduler
                agent.scheduler._lr = new_lr
                # Reset the multiplier so it starts fresh at 1.0 for the new level
                if hasattr(agent.scheduler, "_multiplier"):
                    agent.scheduler._multiplier = 1.0

            # 3. Update Entropy (Internal skrl attribute)
            if hasattr(agent, "_entropy_loss_scale"):
                agent._entropy_loss_scale = new_entropy
                
            print(f">>> LEVEL UP: {env.curr_level} | Setting LR: {new_lr}, Entropy: {new_entropy}")


        # When leveling up, wipe the success rate history 
        # so the new level starts from 0.0
        env.extras["success_rate"] = torch.tensor(0.0, device=env.device)
        print(f">>> LEVEL UP TO {env.curr_level} | Progress Reset.")