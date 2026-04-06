from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def adaptive_curriculum(env: ManagerBasedRLEnv, env_ids: torch.Tensor, threshold: float = 0.9):
    # Check if a fixed level is defined in the config
    # We look for 'fixed_play_level' which we just added above
    fixed_level = getattr(env.cfg, "fixed_play_level", None)

    if fixed_level is not None:
        env.curr_level = fixed_level
    else:
        # Normal training logic: Initialize if not present
        if not hasattr(env, "curr_level"): env.curr_level = 1

    # Standard setup
    if not hasattr(env, "level_up_timer"): env.level_up_timer = 0
    
    # If we are in "Fixed Level" mode, we EXIT early 
    # This prevents the timer from running or the LR from changing
    if fixed_level is not None:
        return 

    # if not hasattr(env, "curr_level"): env.curr_level = 1
    # if not hasattr(env, "level_up_timer"): env.level_up_timer = 0
    if not hasattr(env, "curriculum_step_counter"): env.curriculum_step_counter = 0

    current_success = env.extras.get("success_rate", torch.tensor(0.0, device=env.device)).item()
    print(f"DEBUG: Level: {getattr(env, 'curr_level', 'N/A')} | Success: {current_success:.4f}")

    # Increment a true timestep counter (this function is called every step via reset events,
    # but we want to count env steps, not individual env resets)
    env.curriculum_step_counter += 1

    # Stability check in actual env steps
    if current_success >= threshold:
        env.level_up_timer += 1
    else:
        env.level_up_timer = 0

    if env.curr_level == 1:
        ready_to_level_up = (env.level_up_timer > 500) 
    elif env.curr_level == 2:
        ready_to_level_up = (env.level_up_timer > 1500)  
    else:
        ready_to_level_up = (env.level_up_timer > 3000)   

    if ready_to_level_up:

        # Reset the timer for the next level
        env.level_up_timer = 0

        # Phase 1 -> 2: Add static obstacles
        if env.curr_level == 1:
            env.curr_level = 2
            print(f">>> Level 2: Static Obstacle Avoidance")
        
        # Phase 2 -> 3: Add dynamic obstacles
        elif env.curr_level == 2:
            env.curr_level = 3
            print(f">>> Level 3: Dynamic Obstable Avoidance")


        if hasattr(env, "runner"):
            agent = env.runner.agent
            
            # --- Professional Hyperparameter Strategy ---
            if env.curr_level == 2:
                new_lr = 4e-4 
                new_entropy = 0.01 
            elif env.curr_level == 3:
                new_lr = 4e-4 
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