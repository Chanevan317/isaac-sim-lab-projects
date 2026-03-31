from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply
from .observations import rel_target_pos

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def velocity_toward_target(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Velocity component pointing at target in m/s. Natural range 0–2."""
    robot = env.scene[robot_cfg.name]
    local_pos = rel_target_pos(env, robot_cfg)
    current_dist = torch.norm(local_pos, dim=-1)

    target_dir = local_pos / (current_dist.unsqueeze(-1) + 1e-6)
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    vel_toward = (local_vel[:, :2] * target_dir[:, :2]).sum(dim=-1)

    return torch.clamp(vel_toward, min=0.0)


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


def penalize_backwards_movement(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Returns backward speed magnitude. Weight should be negative."""
    robot = env.scene[robot_cfg.name]
    local_vel = quat_apply(quat_inv(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    vel_y = local_vel[:, 1]  # local +Y is backwards
    return torch.clamp(vel_y, min=0.0)