from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply
from .observations import heading_error, rel_target_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def velocity_toward_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Velocity toward target BUT only when robot is facing the target.
    Rewards forward driving toward goal. Does not reward backing in."""
    robot = env.scene[robot_cfg.name]
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)

    target_dir = local_pos / (current_dist.unsqueeze(-1) + 1e-6)
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)

    # Velocity component toward target
    vel_toward = (local_vel[:, :2] * target_dir[:, :2]).sum(dim=-1)

    # Forward speed in local frame — negative Y is forward for this robot
    forward_speed = -local_vel[:, 1]

    # Only reward if BOTH conditions are true:
    # 1. velocity is toward the target (dot product positive)
    # 2. robot is moving forward (not reversing)
    # This blocks the backward-toward-target cheat
    active = (current_dist > 0.12).float()
    reward = torch.clamp(vel_toward, min=0.0) * torch.clamp(forward_speed, min=0.0) * active

    return reward


def reward_forward_speed(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Squared version of velocity_toward_target for speed incentive."""
    robot = env.scene[robot_cfg.name]
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)

    target_dir = local_pos / (current_dist.unsqueeze(-1) + 1e-6)
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)

    vel_toward = (local_vel[:, :2] * target_dir[:, :2]).sum(dim=-1)
    forward_speed = -local_vel[:, 1]

    active = (current_dist > 0.12).float()
    base = torch.clamp(vel_toward, min=0.0) * torch.clamp(forward_speed, min=0.0) * active

    return base ** 2


def penalize_backwards_movement(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Penalizes any backward movement regardless of direction to target."""
    robot = env.scene[robot_cfg.name]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    forward_speed = -local_vel[:, 1]  # negative Y is forward
    # Return magnitude of backward movement — weight handles sign
    return torch.clamp(-forward_speed, min=0.0)


def target_reached_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Returns 1.0 on success. Weight is the prize."""
    reached = check_target_reached(env, robot_cfg)

    if env.reset_buf.any():
        successes = reached[env.reset_buf].float().mean()
        env.extras["success_rate"] = (
            0.98 * env.extras.get("success_rate", torch.tensor(0.0, device=env.device))
            + 0.02 * successes
        )
        # This key appears automatically in Tensorboard via Isaac Lab's logger
        env.extras["log"] = env.extras.get("log", {})
        env.extras["log"]["success_rate"] = env.extras["success_rate"].item()

    return reached.float()
