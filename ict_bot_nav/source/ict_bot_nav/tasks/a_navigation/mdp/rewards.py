from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.utils.math import quat_apply
from .observations import lidar_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



# def reward_velocity_toward_carrot(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """
#     Rewards progress toward crossing the carrot line.
    
#     Two components:
#     1. Velocity component crossing the line (world X direction of corridor)
#     2. Robot heading alignment with the line-crossing direction
    
#     Unlike point-goal reward, the robot can be at any Y position —
#     only forward X progress matters.
#     """
#     robot = env.scene[robot_cfg.name]

#     # ---- Corridor forward direction ----
#     # The line is perpendicular to corridor X axis
#     # Crossing it requires net positive world X velocity
#     # We need to find the corridor X direction from env origins
#     # For a straight corridor aligned with world X, this is simply world X
#     # For general deployment, use the direction from robot spawn to carrot

#     # Direction from robot to nearest point on line (corridor forward direction)
#     line_x = env.target_pos[:, 0]        # [N]
#     dx = line_x - robot.data.root_pos_w[:, 0]               # [N] signed dist to line

#     # Corridor forward is the direction along which the line must be crossed
#     # For straight corridor = world X unit vector
#     corridor_fwd = torch.zeros_like(robot.data.root_pos_w[:, :2])
#     corridor_fwd[:, 0] = 1.0  # world +X is corridor forward

#     # ---- Velocity toward line ----
#     vel_xy = robot.data.root_lin_vel_w[:, :2]
#     vel_toward_line = (vel_xy * corridor_fwd).sum(dim=-1)    # world X velocity [N]
#     # Sign: positive if moving toward line, negative if moving away
#     vel_toward_line = vel_toward_line * torch.sign(dx)       # [N]

#     # ---- Robot heading alignment with corridor forward ----
#     forward_local = torch.zeros_like(robot.data.root_pos_w)
#     forward_local[:, 1] = -1.0
#     forward_world = quat_apply(robot.data.root_quat_w, forward_local)
#     cos_heading = (forward_world[:, :2] * corridor_fwd).sum(dim=-1)  # [N]

#     # ---- Combined reward ----
#     max_speed = 0.5
#     vel_normalised = vel_toward_line / max_speed
#     speed = torch.norm(vel_xy, dim=-1)
#     moving = (speed > 0.05).float()

#     # Positive reward: moving toward line while facing corridor
#     positive = torch.clamp(vel_normalised, min=0.0) * torch.clamp(cos_heading, min=0.0)

#     # Negative penalties — each wrong dimension penalised independently
#     wrong_heading  = torch.clamp(-cos_heading, min=0.0) * 0.5   # facing away
#     wrong_velocity = torch.clamp(-vel_normalised, min=0.0)       # moving away

#     # Stationary penalty
#     stationary = (1.0 - moving) * 0.5

#     return (positive - wrong_heading - wrong_velocity - stationary) * 10.0



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
    safe_dist: float = 0.75,
) -> torch.Tensor:
    
    beams_t   = lidar_scan(env, sensor_cfg)  # 180 beams 
    forward_min = beams_t[:, 90:].min(dim=-1).values * 4.0   # forward 180°
    global_min  = beams_t.min(dim=-1).values * 4.0            # all directions

    forward_ratio = torch.clamp((safe_dist - forward_min) / safe_dist, 0.0, 1.0)
    global_ratio  = torch.clamp((safe_dist - global_min)  / safe_dist, 0.0, 1.0)

    return forward_ratio ** 2 * 1.3 + global_ratio ** 2  # forward weighted 2x                          # [N], range [0, 1]
