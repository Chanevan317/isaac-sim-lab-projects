from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def rel_target_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Target position in robot's USD local frame."""
    robot = env.scene[robot_cfg.name]
    target = env.target_pos
    
    # Calculate world-space vector
    pos_w = target - robot.data.root_pos_w
    
    # CRITICAL: Ignore height differences for 2D corridor navigation
    pos_w[:, 2] = 0.0 
    
    # Rotate into robot's local frame
    q_inv = quat_inv(robot.data.root_quat_w)
    return quat_apply(q_inv, pos_w)


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Angle to target where 0.0 rad is the -Y axis (The Face)."""
    local_pos = rel_target_pos(env, robot_cfg)
    
    # Using your -Y logic: Side is X, Forward is -Y
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    
    # Return as a 2-element vector per environment
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)