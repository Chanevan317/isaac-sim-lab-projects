from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from .observations import lidar_scan

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


