from __future__ import annotations
import torch

LOOKAHEAD_DIST = 2.0
MARKER_HEIGHT  = 0.25


def place_carrot(env, env_ids: torch.Tensor) -> None:
    """Called at episode reset only. Places carrot at lookahead ahead of robot."""
    if env_ids.dtype == torch.bool:
        idx = env_ids.nonzero(as_tuple=False).squeeze(-1)
    else:
        idx = env_ids
    if len(idx) == 0:
        return

    robot_pos   = env.scene["robot"].data.root_pos_w
    env_origins = env.scene.env_origins

    env.target_pos[idx, 0] = robot_pos[idx, 0] + LOOKAHEAD_DIST
    env.target_pos[idx, 1] = env_origins[idx, 1]
    env.target_pos[idx, 2] = env_origins[idx, 2] + MARKER_HEIGHT

    diff = env.target_pos[idx, :2] - robot_pos[idx, :2]
    env.prev_tgt_dist[idx] = torch.norm(diff, dim=-1)


def update_carrot(env) -> None:
    """
    Called every step after physics. Mirrors NAV2 pure pursuit behaviour:
    carrot is always LOOKAHEAD_DIST ahead of robot along corridor X axis,
    anchored to corridor centerline Y. Continuous — no reach threshold.
    """
    robot_pos   = env.scene["robot"].data.root_pos_w
    env_origins = env.scene.env_origins

    # Continuous update every step — matches NAV2 lookahead computation
    env.target_pos[:, 0] = robot_pos[:, 0] + LOOKAHEAD_DIST
    env.target_pos[:, 1] = env_origins[:, 1]
    env.target_pos[:, 2] = env_origins[:, 2] + MARKER_HEIGHT

    env.prev_tgt_dist = torch.norm(
        env.target_pos[:, :2] - robot_pos[:, :2], dim=-1
    )