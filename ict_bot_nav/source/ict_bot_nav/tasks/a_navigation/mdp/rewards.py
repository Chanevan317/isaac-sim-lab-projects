from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.utils.math import quat_apply
from .observations import lidar_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


# --- POSITIVE ---

def reward_velocity_toward_carrot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]

    to_carrot = env.target_pos[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.norm(to_carrot, dim=-1, keepdim=True).clamp(min=1e-6)
    to_carrot_unit = to_carrot / dist

    vel_xy = robot.data.root_lin_vel_w[:, :2]
    # print(f"vel_xy[0]: {vel_xy[0]}")
    # print(f"root_pos[0]: {robot.data.root_pos_w[0]}")
    # print(f"target_pos[0]: {env.target_pos[0]}")
    vel_toward = (vel_xy * to_carrot_unit).sum(dim=-1)

    forward_local = torch.zeros_like(robot.data.root_pos_w)
    forward_local[:, 1] = -1.0
    forward_world = quat_apply(robot.data.root_quat_w, forward_local)
    cos_heading = (forward_world[:, :2] * to_carrot_unit).sum(dim=-1)

    # Scale vel to [0, 10] range before multiplying
    # max_speed ≈ 0.5 m/s → normalise then amplify
    max_speed = 0.5
    vel_normalised = vel_toward / max_speed          # [-1, 1]
    
    c = 0.05
    return (vel_normalised * cos_heading - c) * 10.0


def reward_carrot_pass(env: ManagerBasedRLEnv):
    """Sparse bonus, fires once per carrot reached."""
    if not hasattr(env, "_prev_carrot_pass_count"):
        env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    delta = env.carrot_pass_count - env._prev_carrot_pass_count
    env._prev_carrot_pass_count = env.carrot_pass_count.clone()
    return torch.clamp(delta, min=0.0)



# --- NEGATIVE ---

def lidar_proximity_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, danger_dist=0.3):
    """
    Soft proximity penalty using lidar_scan observation.
    Uses current frame only [:, :72].
    danger_dist in metres — converted to normalised units with max_range=8.0
    """
    scan = lidar_scan(env, sensor_cfg, num_beams=72)[:, :72]
    threshold = danger_dist / 8.0                           # normalised
    danger_beams = (scan < threshold).float()
    return torch.clamp(danger_beams.sum(dim=-1), max=8.0)   # cap at 8 beams

