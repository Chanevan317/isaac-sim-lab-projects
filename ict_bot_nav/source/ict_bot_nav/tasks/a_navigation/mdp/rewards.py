from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.utils.math import quat_apply
from .observations import lidar_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def reward_velocity_toward_carrot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Rewards progress toward crossing the carrot line using absolute waypoint geometry.
    """
    robot = env.scene[robot_cfg.name]

    # ---- 1. Fetch Exact Corridor Forward (Tangent) ----
    e_idx = env.spawn_end
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(env.waypoint_idx, max=max_idx)
    
    corridor_fwd = env.static_tangents[e_idx, w_idx_safe]  # [N, 2]

    # ---- 2. Signed Distance to Line ----
    target_pos = env.target_pos[:, :2]
    robot_pos = robot.data.root_pos_w[:, :2]
    to_carrot = target_pos - robot_pos
    
    # Project to_carrot onto the tangent. Positive = line ahead.
    dx = (to_carrot * corridor_fwd).sum(dim=-1)

    # ---- 3. Velocity Toward Line ----
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    vel_toward_line = (vel_xy * corridor_fwd).sum(dim=-1)
    
    # Sign: positive if moving toward line, negative if moving away
    vel_toward_line = vel_toward_line * torch.sign(dx)

    # ---- 4. Robot Heading Alignment ----
    # Robot front is local -Y
    forward_local = torch.zeros_like(robot.data.root_pos_w)
    forward_local[:, 1] = -1.0
    forward_world = quat_apply(robot.data.root_quat_w, forward_local)[:, :2]
    
    cos_heading = (forward_world * corridor_fwd).sum(dim=-1)

    # ---- 5. Combined Reward ----
    max_speed = 0.5
    vel_normalised = vel_toward_line / max_speed
    speed = torch.norm(vel_xy, dim=-1)
    moving = (speed > 0.05).float()

    positive = torch.clamp(vel_normalised, min=0.0) * torch.clamp(cos_heading, min=0.0)
    wrong_heading  = torch.clamp(-cos_heading, min=0.0) * 0.5
    wrong_velocity = torch.clamp(-vel_normalised, min=0.0)
    stationary = (1.0 - moving) * 0.5

    return (positive - wrong_heading - wrong_velocity - stationary) * 10.0



def reward_carrot_pass(env: ManagerBasedRLEnv):
    """Sparse bonus, fires once per carrot reached."""
    if not hasattr(env, "_prev_carrot_pass_count"):
        env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    delta = env.carrot_pass_count - env._prev_carrot_pass_count
    env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    return torch.clamp(delta, min=0.0)



def lidar_proximity_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    safe_dist: float = 1.0,
) -> torch.Tensor:
    
    beams_t   = lidar_scan(env, sensor_cfg)  # 180 beams 
    forward_min = beams_t[:, 180:].min(dim=-1).values * 4.0   # forward 180°
    global_min  = beams_t.min(dim=-1).values * 4.0            # all directions

    forward_ratio = torch.clamp((safe_dist - forward_min) / safe_dist, 0.0, 1.0)
    global_ratio  = torch.clamp((safe_dist - global_min)  / safe_dist, 0.0, 1.0)

    return forward_ratio ** 2 * 1.3 + global_ratio ** 2  # forward weighted 1.3x                          # [N], range [0, 1]
