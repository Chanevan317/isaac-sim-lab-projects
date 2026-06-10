# curriculum_obstacle.py
"""
Cumulative obstacle curriculum term for ICT-Bot Stage 1 (Run 1).

Plugs into Isaac Lab's CurriculumManager via the standard CurriculumTermCfg
interface. When Isaac Lab calls the term function at the end of each
curriculum evaluation window, it receives the environment and any kwargs
defined in the cfg. We read the rolling episode success rate from the env's
extras dict, then advance the curriculum level and push new parameters to
the ObstacleManager.

Curriculum schedule
-------------------
  Level 0  →  0 obstacles, speed 0          (warm-up)
  Level 1  →  1 obstacle,  speed 0          (single static)
  Level 2  →  2 obstacles, speed 0          (two static)
  Level 3  →  2 obstacles, speed 0–0.4      (two slow moving)
  Level 4  →  3 obstacles, speed 0.2–0.6    (three moderate speed)
  Level 5  →  4 obstacles, speed 0.4–0.8    (four full speed)

Promotion: success_rate >= promote_threshold for consecutive_windows_to_promote
           consecutive evaluation windows.
Demotion:  success_rate < demote_threshold — step back one level.
Cooldown:  minimum transition_cooldown episodes between any transition
           (prevents rapid oscillation with large env counts).

Key fixes vs previous version
------------------------------
- eval_window increased to 2000 (was 200) — prevents 20-step evaluation cycles
  with 4096 envs that caused rapid oscillation
- promotion and demotion are mutually exclusive (elif) — prevents same-call
  promote+demote oscillation
- transition_cooldown added — minimum episodes between any level change
- state stored on env object (not module globals) — supports multiple runs
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Schedule definition
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CurriculumLevel:
    """One row in the obstacle curriculum schedule."""
    obstacle_count: int
    max_speed: float
    promote_threshold: float = 0.70
    demote_threshold: float  = 0.40


OBSTACLE_SCHEDULE: list[CurriculumLevel] = [
    CurriculumLevel(obstacle_count=0, max_speed=0.0, promote_threshold=0.60),  # 0 — warm-up
    CurriculumLevel(obstacle_count=1, max_speed=0.1),   # 1 — single static
    CurriculumLevel(obstacle_count=2, max_speed=0.2),   # 2 — two static
    CurriculumLevel(obstacle_count=2, max_speed=0.4),   # 3 — two slow moving
    CurriculumLevel(obstacle_count=3, max_speed=0.6),   # 4 — three moderate speed
    CurriculumLevel(obstacle_count=4, max_speed=0.8),   # 5 — four full speed
]


# ---------------------------------------------------------------------------
# Curriculum configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ObstacleCurriculumCfg:
    """Configuration for the obstacle curriculum term.

    Parameters
    ----------
    eval_window:
        Number of completed episodes over which success rate is averaged
        before a level transition is considered.
        With 4096 envs completing ~10 episodes/step, 2000 episodes = ~200
        simulation steps between evaluations — enough for meaningful signal.
    consecutive_windows_to_promote:
        How many consecutive evaluation windows must exceed the promote
        threshold before the level advances. Prevents lucky streaks.
    success_key:
        Key under env.extras["episode"] where the per-episode success
        signal is stored. Must be set in _reset_idx before place_carrot.
    play_level:
        If set, locks the curriculum at this level (for evaluation/play).
    transition_cooldown:
        Minimum number of completed episodes between any level transition.
        Prevents rapid oscillation with large environment counts.
        At 4096 envs, 5000 episodes = ~500 simulation steps of stability.
    """
    eval_window: int                    = 1000  
    consecutive_windows_to_promote: int = 2
    success_key: str                    = "goal_reached"
    play_level: int | None              = None
    transition_cooldown: int            = 5000   # episodes


# ---------------------------------------------------------------------------
# Term function
# ---------------------------------------------------------------------------

def obstacle_curriculum_term(
    env: "ManagerBasedRLEnv",
    env_ids,
    cfg: ObstacleCurriculumCfg,
) -> dict:
    """Evaluate success rate and advance/demote the obstacle curriculum level."""

    # ---- Initialise per-env state on first call ----------------------------
    if not hasattr(env, "_obs_curr_level"):
        env._obs_curr_level       = 0
        env._obs_curr_consecutive = 0
        env._obs_curr_successes   = []
        env._obs_curr_cooldown    = 0   # episodes remaining in cooldown

    # ---- Play mode: lock at requested level --------------------------------
    if cfg.play_level is not None:
        if env._obs_curr_level != cfg.play_level:
            env._obs_curr_level = cfg.play_level
            _apply_curriculum(env, cfg.play_level)
        return {"obstacle_level": env._obs_curr_level}

    # ---- Collect episode outcomes ------------------------------------------
    import torch

    ep_extras = env.extras.get("episode", {})
    successes  = ep_extras.get(cfg.success_key, None)

    n_new = len(env_ids)

    if successes is not None:
        if isinstance(successes, torch.Tensor):
            if successes.ndim == 0:
                env._obs_curr_successes.extend([successes.item()] * n_new)
            elif successes.shape[0] == env.num_envs:
                env._obs_curr_successes.extend(successes[env_ids].tolist())
            else:
                env._obs_curr_successes.extend(successes.tolist())
        else:
            env._obs_curr_successes.extend([float(successes)] * n_new)

    # Tick down cooldown
    env._obs_curr_cooldown = max(0, env._obs_curr_cooldown - n_new)

    # ---- Wait for minimum window size -------------------------------------
    if len(env._obs_curr_successes) < cfg.eval_window:
        return {"obstacle_level": env._obs_curr_level}

    # ---- Compute rolling success rate -------------------------------------
    window       = env._obs_curr_successes[-cfg.eval_window:]
    success_rate = sum(window) / len(window)
    level        = OBSTACLE_SCHEDULE[env._obs_curr_level]

    # ---- Promotion and demotion (mutually exclusive) ----------------------
    if env._obs_curr_cooldown == 0:

        if success_rate >= level.promote_threshold:
            env._obs_curr_consecutive += 1
            if env._obs_curr_consecutive >= cfg.consecutive_windows_to_promote:
                if env._obs_curr_level < len(OBSTACLE_SCHEDULE) - 1:
                    env._obs_curr_level      += 1
                    env._obs_curr_consecutive = 0
                    env._obs_curr_cooldown    = cfg.transition_cooldown
                    _apply_curriculum(env, env._obs_curr_level)
                    env._obs_curr_last_event = (
                        f"PROMOTED → L{env._obs_curr_level} "
                        f"(sr={success_rate:.2f}, "
                        f"obs={OBSTACLE_SCHEDULE[env._obs_curr_level].obstacle_count}, "
                        f"spd={OBSTACLE_SCHEDULE[env._obs_curr_level].max_speed:.1f})"
                    )
                else:
                    # Already at max level — stay here
                    env._obs_curr_consecutive = 0

        elif success_rate < level.demote_threshold:
            # elif — only fires if promotion did NOT fire
            if env._obs_curr_level > 0:
                env._obs_curr_level      -= 1
                env._obs_curr_consecutive = 0
                env._obs_curr_cooldown    = cfg.transition_cooldown
                _apply_curriculum(env, env._obs_curr_level)
                env._obs_curr_last_event = (
                    f"DEMOTED → L{env._obs_curr_level} (sr={success_rate:.2f})"
                )
            else:
                env._obs_curr_consecutive = 0

        else:
            # Between thresholds — reset consecutive counter
            env._obs_curr_consecutive = 0

    # ---- Trim buffer -------------------------------------------------------
    env._obs_curr_successes = env._obs_curr_successes[-(cfg.eval_window * 4):]

    return {
        "obstacle_level":        env._obs_curr_level,
        "obstacle_success_rate": success_rate,
        "obstacle_count":        OBSTACLE_SCHEDULE[env._obs_curr_level].obstacle_count,
        "obstacle_max_speed":    OBSTACLE_SCHEDULE[env._obs_curr_level].max_speed,
        "obstacle_cooldown":     env._obs_curr_cooldown,
    }


def _apply_curriculum(env: "ManagerBasedRLEnv", level: int) -> None:
    """Push new curriculum parameters to the ObstacleManager."""
    schedule = OBSTACLE_SCHEDULE[level]
    if hasattr(env, "obstacle_manager"):
        env.obstacle_manager.set_curriculum_params(
            obstacle_count=schedule.obstacle_count,
            max_speed=schedule.max_speed,
        )
    else:
        raise AttributeError(
            "env.obstacle_manager not found. Instantiate ObstacleManager in "
            "your env __init__ and assign it to self.obstacle_manager before "
            "the curriculum term is called."
        )