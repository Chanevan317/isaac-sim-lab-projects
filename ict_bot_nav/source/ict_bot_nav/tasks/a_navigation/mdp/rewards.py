from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import rel_target_pos, heading_error, lidar_distances, imu_observations
from .common import check_target_reached
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



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