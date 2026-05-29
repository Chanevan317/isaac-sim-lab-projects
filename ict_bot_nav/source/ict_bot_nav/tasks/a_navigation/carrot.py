from __future__ import annotations
import torch

LOOKAHEAD_DIST  = 2.0    # metres — where next carrot is placed after reach
REACH_THRESHOLD = 0.4    # metres — how close before carrot advances
MARKER_HEIGHT   = 0.25   # metres

def place_carrot(env, env_ids: torch.Tensor) -> None:
    """Called at episode reset. Places carrot at LOOKAHEAD_DIST ahead of robot."""
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

    # Initialise pass counter if not present
    if not hasattr(env, "carrot_pass_count"):
        env.carrot_pass_count = torch.zeros(env.num_envs, device=env.device)
    env.carrot_pass_count[idx] = 0.0


def update_carrot(env) -> None:
    robot_pos   = env.scene["robot"].data.root_pos_w
    env_origins = env.scene.env_origins

    # Save distance BEFORE any carrot movement — this is what reward uses
    prev_dist_before_update = torch.norm(
        env.target_pos[:, :2] - robot_pos[:, :2], dim=-1
    )

    diff    = env.target_pos[:, :2] - robot_pos[:, :2]
    dist    = torch.norm(diff, dim=-1)
    reached = dist < REACH_THRESHOLD

    env.carrot_pass_count += reached.float()

    new_x = robot_pos[:, 0] + LOOKAHEAD_DIST
    new_y = env_origins[:, 1]
    new_z = env_origins[:, 2] + MARKER_HEIGHT

    env.target_pos[:, 0] = torch.where(reached, new_x, env.target_pos[:, 0])
    env.target_pos[:, 1] = torch.where(reached, new_y, env.target_pos[:, 1])
    env.target_pos[:, 2] = torch.where(reached, new_z, env.target_pos[:, 2])

    # prev_tgt_dist = distance to the carrot that was active this step
    # On a non-reach step: distance to same carrot (robot moved slightly closer)
    # On a reach step: distance was near 0 — next step starts fresh at 2.0
    env.prev_tgt_dist = prev_dist_before_update