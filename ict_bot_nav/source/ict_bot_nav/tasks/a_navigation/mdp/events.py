from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def reset_target_marker_location(env: ManagerBasedRLEnv, env_ids: torch.Tensor, min_distance: float = 1.0, max_distance: float = 3.0):
    """Spawns target at random distance and angle around each robot."""
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[env_ids]

    n = len(env_ids)
    # Random angle — full 360°
    angle = torch.rand(n, device=env.device) * 2 * torch.pi
    # Random distance
    dist = min_distance + torch.rand(n, device=env.device) * (max_distance - min_distance)

    env.target_pos[env_ids, 0] = robot_pos[:, 0] + dist * torch.cos(angle)
    env.target_pos[env_ids, 1] = robot_pos[:, 1] + dist * torch.sin(angle)
    env.target_pos[env_ids, 2] = 0.25  # fixed height for marker visibility