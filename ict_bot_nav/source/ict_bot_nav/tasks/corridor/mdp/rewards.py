from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def heading_alignment_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Absolute cos(angle) — constant signal for being aligned. Proven to work."""
    h_error = heading_error(env, robot_cfg)
    cos_angle = h_error[:, 1]
    return cos_angle  # −1.0 (facing away) to +1.0 (facing target)


def reward_gated_progress(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Port of your working reward — strict gate, penalise sideways driving."""
    robot = env.scene[robot_cfg.name]
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)

    dist_delta = env.prev_tgt_dist - current_dist
    if hasattr(env, "reset_buf"):
        dist_delta = torch.where(env.reset_buf, torch.zeros_like(dist_delta), dist_delta)

    env.prev_tgt_dist = current_dist.clone()

    h_error = heading_error(env, robot_cfg)
    cos_angle = h_error[:, 1]

    # Strict gate — only reward progress when well aligned
    # Loosened slightly from 0.95 to 0.7 to allow curved paths for 360° spawns
    gate = (cos_angle > 0.7).float()
    return torch.where(gate > 0, dist_delta * 5.0, -torch.abs(dist_delta) * 2.0)


def penalize_backwards_movement(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Direct port of your working backward penalty."""
    robot = env.scene[robot_cfg.name]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    vel_y = local_vel[:, 1]  # local +Y is backwards
    return torch.clamp(vel_y, min=0.0)


def target_reached_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Returns 1.0 on success, 0.0 otherwise. W
ImportError: /home/zuru-ubuntu/miniconda3/envs/chan_env/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEventeight is the prize value."""
    reached = check_target_reached(env, robot_cfg)

    terminating_envs = env.reset_buf
    if terminating_envs.any():
        successes = reached[terminating_envs].float().mean()
        env.extras["success_rate"] = 0.98 * env.extras.get(
            "success_rate", torch.tensor(0.0, device=env.device)
        ) + 0.02 * successes

    return reached.float()


def base_posture_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Returns excess pitch rate above 0.2 rad/s. Weight should be negative."""
    imu_data = imu_observations(env, sensor_cfg)
    pitch_rate = torch.abs(imu_data[:, 3])
    return torch.clamp(pitch_rate - 0.2, min=0.0)


def imu_stability(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Returns excess yaw rate above 0.5 rad/s. Weight should be negative."""
    level = getattr(env, "curr_level", 1)
    if level < 2:
        return torch.zeros(env.num_envs, device=env.device)

    imu_data = imu_observations(env, sensor_cfg)
    yaw_rate = torch.abs(imu_data[:, 2])
    return torch.clamp(yaw_rate - 0.5, min=0.0)


def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.5):
    """Returns 1.0 when closer than threshold, 0.0 otherwise. Weight should be negative."""
    level = getattr(env, "curr_level", 1)
    if level < 2:
        return torch.zeros(env.num_envs, device=env.device)

    lidar_values = lidar_distances(env, sensor_cfg, max_distance=1.0)
    min_dist, _ = torch.min(lidar_values, dim=-1)
    return (min_dist < threshold).float()


# def is_alive(env: ManagerBasedRLEnv):
#     """Returns 1.0 every step. Weight should be negative to act as time penalty."""
#     return torch.ones(env.num_envs, device=env.device)