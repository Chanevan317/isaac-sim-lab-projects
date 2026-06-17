from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



# def rel_line_dist(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
#     """
#     Returns [forward_dist_to_line, lateral_offset_from_centreline].
#     forward_dist: positive = line ahead
#     lateral_offset: 0 = on centreline, ±1.0 = at line edge
#     Shape: [N, 2]
#     """
#     robot = env.scene[robot_cfg.name]

#     forward_local = torch.zeros(env.num_envs, 3, device=env.device)
#     forward_local[:, 1] = -1.0
#     forward_world = quat_apply(robot.data.root_quat_w, forward_local)[:, :2]

#     to_carrot = env.target_pos[:, :2] - robot.data.root_pos_w[:, :2]
#     forward_dist = (to_carrot * forward_world).sum(dim=-1, keepdim=True)  # [N, 1]

#     # Lateral offset from corridor centreline, normalised to [0, 1] at line edge
#     env_origins = env.scene.env_origins
#     lateral_offset = (
#         robot.data.root_pos_w[:, 1] - env_origins[:, 1]
#     ).unsqueeze(-1) / 1.0   # normalised by CORRIDOR_HALF_WIDTH=1.0

#     return torch.cat([forward_dist, lateral_offset], dim=-1)  # [N, 2]



# def heading_to_line(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
#     """
#     Angle from robot forward to the nearest point on the carrot line.
#     The nearest point on the perpendicular line is always directly
#     ahead — so this is the angle between robot forward and corridor forward.
#     This allows the robot to approach the line from any lateral position
#     without being pulled toward the carrot centre point.
#     Shape: [N, 2] — [sin, cos] of angle
#     """
#     robot = env.scene[robot_cfg.name]

#     # Nearest point on line at carrot_x is (carrot_x, robot_y)
#     line_target = torch.stack([
#         env.target_pos[:, 0],   # line X position
#         robot.data.root_pos_w[:, 1],              # robot current Y — nearest point on line
#         torch.zeros(env.num_envs, device=env.device)
#     ], dim=-1)  # [N, 3]

#     # Vector from robot to nearest line point
#     pos_w = line_target - robot.data.root_pos_w
#     pos_w[:, 2] = 0.0

#     # Rotate into robot local frame
#     q_inv = quat_inv(robot.data.root_quat_w)
#     local_pos = quat_apply(q_inv, pos_w)[:, :2]  # [N, 2]

#     # Angle in robot frame — forward is local -Y
#     angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
#     return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)



def rel_line_dist(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """
    Returns [forward_dist_to_line, lateral_offset_from_centreline].
    Uses the absolute waypoint geometry and lateral triggers.
    """
    robot = env.scene[robot_cfg.name]

    # Fetch geometry mappings
    e_idx = env.spawn_end
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(env.waypoint_idx, max=max_idx)
    
    corridor_fwd = env.static_tangents[e_idx, w_idx_safe]  # [N, 2]
    lateral_vec  = env.static_laterals[e_idx, w_idx_safe]  # [N, 2]
    local_triggers = env.static_triggers[e_idx, w_idx_safe]

    robot_pos = robot.data.root_pos_w[:, :2]
    target_pos = env.target_pos[:, :2]

    # 1. Forward Distance (projected onto tangent)
    to_carrot = target_pos - robot_pos
    forward_dist = (to_carrot * corridor_fwd).sum(dim=-1, keepdim=True)

    # 2. Lateral Offset (projected onto lateral vector from the trigger line)
    world_triggers = env.scene.env_origins[:, :2] + local_triggers
    vec_to_robot = robot_pos - world_triggers
    
    # Divided by 1.0 (LINE_HALF_WIDTH) to normalize
    lateral_offset = (vec_to_robot * lateral_vec).sum(dim=-1, keepdim=True) / 1.0

    return torch.cat([forward_dist, lateral_offset], dim=-1)  # [N, 2]



def heading_to_line(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """
    Angle from robot forward to the absolute corridor forward (tangent).
    Shape: [N, 2] — [sin, cos] of angle
    """
    robot = env.scene[robot_cfg.name]

    # Fetch exact corridor forward
    e_idx = env.spawn_end
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(env.waypoint_idx, max=max_idx)
    
    corridor_fwd = env.static_tangents[e_idx, w_idx_safe]  # [N, 2]

    # Pad to 3D to apply quaternion rotation
    pos_w = torch.cat([
        corridor_fwd, 
        torch.zeros(env.num_envs, 1, device=env.device)
    ], dim=-1)

    # Rotate into robot local frame
    q_inv = quat_inv(robot.data.root_quat_w)
    local_pos = quat_apply(q_inv, pos_w)[:, :2]  # [N, 2]

    # Angle in robot frame — forward is local -Y
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)



def joint_velocity(env, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_vel[:, robot_cfg.joint_ids]  # [N, 2] — raw rad/s

def root_lin_vel_b_2d(env, robot_cfg):
    return env.scene[robot_cfg.name].data.root_lin_vel_b[:, :2]  # [vx, vy]

def root_ang_vel_b_z(env, robot_cfg):
    return env.scene[robot_cfg.name].data.root_ang_vel_b[:, 2:3]  # [wz]



def lidar_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_beams: int = 180,
):
    """
    Returns only the current frame of normalized LiDAR scan data [0, 1].
    Temporal tracking is now handled downstream by the policy's recurrent GRU block.
    Shape: [N, num_beams]
    """
    # --- Guard ---
    if sensor_cfg.name not in env.scene.keys():
        return torch.zeros(env.num_envs, num_beams, device=env.device)

    # --- Raw scan ---
    sensor    = env.scene[sensor_cfg.name]
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
    hits      = sensor.data.ray_hits_w                            # [N, num_beams, 3]
    distances = torch.norm(hits - robot_pos, dim=-1)              # [N, num_beams]

    # --- Normalise ---
    max_range = 4.0
    distances = torch.clamp(distances, 0.0, max_range) / max_range
    noise     = torch.randn_like(distances) * 0.02
    lidar_t   = torch.clamp(distances + noise, 0.0, 1.0)          # [N, num_beams]

    return lidar_t



# def lidar_scan(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     num_beams: int = 180,
#     num_frames: int = 20,
# ):
#     """
#     Returns stacked [lidar_t, lidar_t-1, ..., lidar_t-(n-1)] normalised to [0,1].
#     At 20 Hz, 10 frames = 0.5 seconds of history.
#     At 40 Hz reference paper used 40 frames = 1 second — we match 0.5s at 20 Hz.
#     Shape: [N, num_beams * num_frames]
#     """
#     # --- Guard ---
#     if sensor_cfg.name not in env.scene.keys():
#         if not hasattr(env, "_lidar_history") or env._lidar_history is None:
#             env._lidar_history = torch.zeros(
#                 env.num_envs, num_frames, num_beams, device=env.device
#             )
#         return torch.zeros(env.num_envs, num_beams * num_frames, device=env.device)

#     # --- Raw scan ---
#     sensor    = env.scene[sensor_cfg.name]
#     robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)  # [N, 1, 3]
#     hits      = sensor.data.ray_hits_w                            # [N, num_beams, 3]
#     distances = torch.norm(hits - robot_pos, dim=-1)              # [N, num_beams]

#     # --- Normalise ---
#     max_range = 3.0
#     distances = torch.clamp(distances, 0.0, max_range) / max_range
#     noise     = torch.randn_like(distances) * 0.02
#     lidar_t   = torch.clamp(distances + noise, 0.0, 1.0)          # [N, num_beams]

#     # --- Initialise history buffer on first call ---
#     if not hasattr(env, "_lidar_history") or env._lidar_history is None:
#         env._lidar_history = lidar_t.unsqueeze(1).expand(
#             env.num_envs, num_frames, num_beams
#         ).clone()

#     # --- Shift history: drop oldest frame, prepend current ---
#     # _lidar_history shape: [N, num_frames, num_beams]
#     # index 0 = most recent (t), index num_frames-1 = oldest (t-(n-1))
#     env._lidar_history = torch.cat([
#         lidar_t.unsqueeze(1),              # [N, 1, num_beams] — new frame
#         env._lidar_history[:, :-1, :]      # [N, num_frames-1, num_beams] — drop oldest
#     ], dim=1)

#     # --- Flatten to [N, num_beams * num_frames] ---
#     return env._lidar_history.view(env.num_envs, -1)
