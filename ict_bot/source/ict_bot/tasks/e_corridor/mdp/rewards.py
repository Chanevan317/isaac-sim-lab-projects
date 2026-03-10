from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

import torch



