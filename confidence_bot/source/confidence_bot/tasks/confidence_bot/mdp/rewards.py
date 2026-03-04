from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import heading_error_xaxis
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg



