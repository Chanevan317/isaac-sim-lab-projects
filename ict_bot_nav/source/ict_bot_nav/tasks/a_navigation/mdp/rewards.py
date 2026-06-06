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
#     robot = env.scene[robot_cfg.name]

#     to_carrot = env.target_pos[:, :2] - robot.data.root_pos_w[:, :2]
#     dist = torch.norm(to_carrot, dim=-1, keepdim=True).clamp(min=1e-6)
#     to_carrot_unit = to_carrot / dist

#     vel_xy = robot.data.root_lin_vel_w[:, :2]
#     # print(f"vel_xy[0]: {vel_xy[0]}")
#     # print(f"root_pos[0]: {robot.data.root_pos_w[0]}")
#     # print(f"target_pos[0]: {env.target_pos[0]}")
#     vel_toward = (vel_xy * to_carrot_unit).sum(dim=-1)

#     forward_local = torch.zeros_like(robot.data.root_pos_w)
#     forward_local[:, 1] = -1.0
#     forward_world = quat_apply(robot.data.root_quat_w, forward_local)
#     cos_heading = (forward_world[:, :2] * to_carrot_unit).sum(dim=-1)

#     # Scale vel to [0, 10] range before multiplying
#     # max_speed ≈ 0.5 m/s → normalise then amplify
#     max_speed = 0.5
#     vel_normalised = vel_toward / max_speed          # [-1, 1]
    
#     c = 0.05
#     return (vel_normalised * cos_heading - c) * 10.0


def reward_velocity_toward_carrot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Rewards progress toward crossing the carrot line.
    
    Two components:
    1. Velocity component crossing the line (world X direction of corridor)
    2. Robot heading alignment with the line-crossing direction
    
    Unlike point-goal reward, the robot can be at any Y position —
    only forward X progress matters.
    """
    robot = env.scene[robot_cfg.name]

    # ---- Corridor forward direction ----
    # The line is perpendicular to corridor X axis
    # Crossing it requires net positive world X velocity
    # We need to find the corridor X direction from env origins
    # For a straight corridor aligned with world X, this is simply world X
    # For general deployment, use the direction from robot spawn to carrot

    # Direction from robot to nearest point on line (corridor forward direction)
    line_x = env.target_pos[:, 0]        # [N]
    dx = line_x - robot.data.root_pos_w[:, 0]               # [N] signed dist to line

    # Corridor forward is the direction along which the line must be crossed
    # For straight corridor = world X unit vector
    corridor_fwd = torch.zeros_like(robot.data.root_pos_w[:, :2])
    corridor_fwd[:, 0] = 1.0  # world +X is corridor forward

    # ---- Velocity toward line ----
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    vel_toward_line = (vel_xy * corridor_fwd).sum(dim=-1)    # world X velocity [N]
    # Sign: positive if moving toward line, negative if moving away
    vel_toward_line = vel_toward_line * torch.sign(dx)       # [N]

    # ---- Robot heading alignment with corridor forward ----
    forward_local = torch.zeros_like(robot.data.root_pos_w)
    forward_local[:, 1] = -1.0
    forward_world = quat_apply(robot.data.root_quat_w, forward_local)
    cos_heading = (forward_world[:, :2] * corridor_fwd).sum(dim=-1)  # [N]

    # ---- Combined reward ----
    max_speed = 0.5
    vel_normalised = vel_toward_line / max_speed
    speed = torch.norm(vel_xy, dim=-1)
    moving = (speed > 0.05).float()

    # Positive reward: moving toward line while facing corridor
    positive = torch.clamp(vel_normalised, min=0.0) * torch.clamp(cos_heading, min=0.0)

    # Negative penalties — each wrong dimension penalised independently
    wrong_heading  = torch.clamp(-cos_heading, min=0.0) * 0.5   # facing away
    wrong_velocity = torch.clamp(-vel_normalised, min=0.0)       # moving away

    # Stationary penalty
    stationary = (1.0 - moving) * 0.5

    return (positive - wrong_heading - wrong_velocity - stationary) * 10.0


def reward_carrot_pass(env: ManagerBasedRLEnv):
    """Sparse bonus, fires once per carrot reached."""
    if not hasattr(env, "_prev_carrot_pass_count"):
        env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    delta = env.carrot_pass_count - env._prev_carrot_pass_count
    env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    return torch.clamp(delta, min=0.0)


def reward_corridor_clearance(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, safe_dist: float = 0.7):
    """
    Signed clearance reward using forward 180°.
    Positive = clear ahead (> safe_dist), zero = at threshold, negative = too close.
    Range: [-1, +1]
    """
    # Use current frame only — sector_t is first 18 values
    stacked  = lidar_scan(env, sensor_cfg, num_beams=72)
    beams_t = stacked[:, :72]

    # Forward 180° = beams 36-71 (robot front is -Y = beam 54 centre)
    forward_beams = beams_t[:, 36:]                        # [N, 36]
    min_forward_m = forward_beams.min(dim=-1).values * 3.0 # metres

    # Normalised distance ratio: 1.0 at contact, 0.0 at safe_dist
    ratio   = torch.clamp((safe_dist - min_forward_m) / safe_dist, 0.0, 1.0)

    # Quadratic penalty term: 0 at safe_dist, 1.0 at contact
    penalty = ratio ** 2                                       # [N]

    # Clearance bonus: linear, 0 at safe_dist, 1.0 at max_range
    bonus   = torch.clamp((min_forward_m - safe_dist) / (3.0 - safe_dist), 0.0, 1.0)

    # Combined: positive when clear, zero at threshold, negative (quadratic) when close
    return bonus - penalty                                     # [N], range [-1, +1]                     # [N]


# def lidar_proximity_penalty(env, sensor_cfg, safe_dist=0.5):
#     scan = lidar_scan(env, sensor_cfg, num_beams=72)[:, :72]  # [N, 72]

#     # Forward 180° sector — beams facing roughly forward
#     # For a 360° LiDAR with beam 54 at front, beams 36-72 cover 180° front sector
#     forward_sector = scan[:, 36:72]  # [N, 36]
#     min_dist = forward_sector.min(dim=-1).values   # [N]

#     min_dist_m = min_dist * 4.0

#     penalty = torch.clamp(
#         (safe_dist - min_dist_m) / safe_dist, min=0.0
#     ) ** 2

#     return penalty  # [N], range [0, 1]


# def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, danger_dist=0.3):
#     """
#     Soft proximity penalty using lidar_scan observation.
#     Uses current frame only [:, :72].
#     danger_dist in metres — converted to normalised units with max_range=8.0
#     """
#     scan = lidar_scan(env, sensor_cfg, num_beams=72)[:, :72]
#     threshold = danger_dist / 8.0                           # normalised
#     danger_beams = (scan < threshold).float()
#     return torch.clamp(danger_beams.sum(dim=-1), max=8.0)   # cap at 8 beams

