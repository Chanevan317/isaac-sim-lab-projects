from __future__ import annotations
import torch

LOOKAHEAD_DIST  = 1.5   # metres — matches Nav2 at 0.5 m/s, lookahead_time=1.5s
ADVANCE_OFFSET  = 0.25   # metres ahead of carrot — robot centre fully past before advance
MARKER_HEIGHT   = 0.25   # metres
LINE_HALF_WIDTH  = 1.0   # ±1m from centreline = 2m total valid line

def place_carrot(env, env_ids: torch.Tensor) -> None:
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

    if not hasattr(env, "carrot_pass_count"):
        env.carrot_pass_count = torch.zeros(env.num_envs, device=env.device)
    env.carrot_pass_count[idx] = 0.0


def update_carrot(env) -> None:
    robot_pos   = env.scene["robot"].data.root_pos_w
    env_origins = env.scene.env_origins

    prev_dist_before_update = torch.norm(
        env.target_pos[:, :2] - robot_pos[:, :2], dim=-1
    )

    # Line crossing — robot must pass carrot X
    crossed_line = robot_pos[:, 0] >= (env.target_pos[:, 0] - ADVANCE_OFFSET)

    # Must be within ±LINE_HALF_WIDTH of corridor centreline
    within_line = torch.abs(
        robot_pos[:, 1] - env_origins[:, 1]
    ) <= LINE_HALF_WIDTH

    reached = crossed_line & within_line

    env.carrot_pass_count += reached.float()

    new_x = robot_pos[:, 0] + LOOKAHEAD_DIST
    new_y = env_origins[:, 1]
    new_z = env_origins[:, 2] + MARKER_HEIGHT

    env.target_pos[:, 0] = torch.where(reached, new_x, env.target_pos[:, 0])
    env.target_pos[:, 1] = torch.where(reached, new_y, env.target_pos[:, 1])
    env.target_pos[:, 2] = torch.where(reached, new_z, env.target_pos[:, 2])

    env.prev_tgt_dist = prev_dist_before_update