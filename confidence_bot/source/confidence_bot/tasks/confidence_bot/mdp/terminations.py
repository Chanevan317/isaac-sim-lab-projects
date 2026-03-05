from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from .observations import get_tag_pixel_coords

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode terminates if max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def reached_tag_visual(env: ManagerBasedRLEnv, tag_cfg: SceneEntityCfg, threshold_v: float = -0.85, threshold_u: float = 0.15, sensor_name: str = "tiled_camera") -> torch.Tensor:
    """Terminates when the tag is centered and at the bottom of the frame."""

    # Get the current pixel coordinates
    # This calls your projection math and returns [u, v, visible]
    coords = get_tag_pixel_coords(env, tag_cfg)
    
    u = coords[:, 0]
    v = coords[:, 1]
    visible = coords[:, 2]

    # 3. Success Condition Logic:
    # Note: In Isaac Lab/ROS convention, v = 1.0 is the BOTTOM of the image.
    # So we check if v >= threshold_v (e.g., v >= 0.85)
    success = (v >= threshold_v) & (u.abs() <= threshold_u) & (visible > 0.5)

    return success