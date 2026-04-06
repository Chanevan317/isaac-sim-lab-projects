from __future__ import annotations
import torch
from isaaclab.utils.math import quat_apply


def place_carrot(
    env,
    env_ids: torch.Tensor,
    fwd_dist_range: tuple[float, float] = (1.0, 2.5),
    lateral_std: float = 0.15,
    lateral_max: float = 0.35,
    marker_height: float = 0.25,
):
    """Place new carrot ahead of robot along corridor x axis."""
    if env_ids.dtype == torch.bool:
        idx = env_ids.nonzero(as_tuple=False).squeeze(-1)
    else:
        idx = env_ids

    n = len(idx)
    if n == 0:
        return

    robot_pos = env.scene["robot"].data.root_pos_w

    # Forward distance along corridor x axis
    fwd_dist = (
        fwd_dist_range[0]
        + torch.rand(n, device=env.device)
        * (fwd_dist_range[1] - fwd_dist_range[0])
    )

    # Lateral offset relative to each robot's y position — center biased
    lateral = torch.randn(n, device=env.device) * lateral_std
    lateral = torch.clamp(lateral, -lateral_max, lateral_max)

    # Place carrot in each env's local corridor frame
    env.target_pos[idx, 0] = robot_pos[idx, 0] + fwd_dist
    env.target_pos[idx, 1] = robot_pos[idx, 1] + lateral
    env.target_pos[idx, 2] = marker_height

    # Checkpoint line direction is always world +x for all envs
    env.carrot_forward_dir[idx, 0] = 1.0
    env.carrot_forward_dir[idx, 1] = 0.0

    # Update prev_tgt_dist
    diff = env.target_pos[idx] - robot_pos[idx]
    diff[:, 2] = 0.0
    env.prev_tgt_dist[idx] = torch.norm(diff, dim=-1)

    # Increment per-env advance counter
    # Do NOT reset here — reset happens at episode end in training script
    if not hasattr(env, "carrot_advance_count"):
        env.carrot_advance_count = torch.zeros(env.num_envs, device=env.device)
    env.carrot_advance_count[idx] += 1


def update_carrot(env):
    robot_pos = env.scene["robot"].data.root_pos_w

    to_carrot = env.target_pos[:, :2] - robot_pos[:, :2]
    longitudinal = to_carrot[:, 0]
    dist = torch.norm(to_carrot, dim=-1)

    env.carrot_timer += env.step_dt

    passed_plane     = longitudinal < 0.0
    very_close       = dist < 0.3
    timed_out_carrot = env.carrot_timer > 8.0

    advance_mask = passed_plane | very_close | timed_out_carrot

    if advance_mask.any():
        env.carrot_timer[advance_mask] = 0.0
        advance_idx = advance_mask.nonzero(as_tuple=False).squeeze(-1)

        # Track how carrot advanced for diagnostics
        if not hasattr(env, "advance_reason"):
            env.advance_reason = {"plane": 0, "close": 0, "timeout": 0}
        env.advance_reason["plane"]   += passed_plane[advance_mask].sum().item()
        env.advance_reason["close"]   += very_close[advance_mask].sum().item()
        env.advance_reason["timeout"] += timed_out_carrot[advance_mask].sum().item()

        place_carrot(env, advance_idx)

        if env.sim.has_gui():
            env.target_marker.visualize(env.target_pos)