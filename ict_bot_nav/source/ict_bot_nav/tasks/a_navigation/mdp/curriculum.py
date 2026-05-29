# curriculum_obstacle.py
"""
Cumulative obstacle curriculum term for ICT-Bot Stage 1 (Run 1).

Plugs into Isaac Lab's CurriculumManager via the standard `CurriculumTermCfg`
interface.  When Isaac Lab calls the term function at the end of each
curriculum evaluation window, it receives the environment and any kwargs
defined in the cfg.  We read the rolling episode success rate from the env's
extras dict, then advance the curriculum level and push new parameters to
the ObstacleManager.

Curriculum schedule
-------------------
The schedule below reproduces the three-stage structure validated in the
literature (174% / 65% / 61% improvement across stages) while staying in a
single continuous PPO run:

  Level 0  →  0 obstacles, speed 0          (warm-up, always passes)
  Level 1  →  1-2 obstacles, speed 0        (static, sparse)
  Level 2  →  3-4 obstacles, speed 0        (static, dense)
  Level 3  →  3-4 obstacles, speed 0-0.4    (slow moving)
  Level 4  →  5-6 obstacles, speed 0.2-0.6  (moderate density + speed)
  Level 5  →  7-8 obstacles, speed 0.4-0.8  (full Stage 1 difficulty)

Promotion threshold: success_rate ≥ 0.7 for two consecutive windows.
Demotion: if success_rate < 0.4, step back one level (prevents policy collapse).

Usage in your env cfg
---------------------
    from isaaclab.managers import CurriculumTermCfg
    from curriculum_obstacle import obstacle_curriculum_term, ObstacleCurriculumCfg

    @configclass
    class MyCurriculumCfg:
        obstacle_difficulty = CurriculumTermCfg(
            func=obstacle_curriculum_term,
            params={"cfg": ObstacleCurriculumCfg()},
        )

Then in your env's __init__, after super().__init__():
    self.obstacle_manager = ObstacleManager(self, cube_obj, cyl_obj, obstacle_set_cfg)

The term function retrieves it from env.obstacle_manager.
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
    promote_threshold: float = 0.70   # success rate to advance
    demote_threshold: float  = 0.40   # success rate to step back


OBSTACLE_SCHEDULE: list[CurriculumLevel] = [
    CurriculumLevel(obstacle_count=0, max_speed=0.0, promote_threshold=0.60),  # 0 — warm-up
    CurriculumLevel(obstacle_count=1, max_speed=0.0),   # 1 — single static
    CurriculumLevel(obstacle_count=2, max_speed=0.0),   # 2 — two static
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
    consecutive_windows_to_promote:
        How many consecutive windows must exceed the promote threshold before
        the level actually advances.  Prevents lucky streaks from skipping.
    success_key:
        Key under which the per-episode success signal is stored in
        env.extras["episode"].  Must be set by your termination/reward manager.
    """
    eval_window: int = 200
    consecutive_windows_to_promote: int = 2
    success_key: str = "goal_reached"
    play_level: int | None = None


# ---------------------------------------------------------------------------
# Term function — Isaac Lab calls this at curriculum evaluation frequency
# ---------------------------------------------------------------------------

def obstacle_curriculum_term(
    env: "ManagerBasedRLEnv",
    env_ids,
    cfg: ObstacleCurriculumCfg,
) -> dict:
    # Attach state to env instead of module globals
    if not hasattr(env, "_obs_curr_level"):
        env._obs_curr_level       = 0
        env._obs_curr_consecutive = 0
        env._obs_curr_successes   = []

    # ---- Play mode: lock at requested level, skip all training logic ----
    if cfg.play_level is not None:
        if env._obs_curr_level != cfg.play_level:
            env._obs_curr_level = cfg.play_level
            _apply_curriculum(env, cfg.play_level)
        return {"obstacle_level": env._obs_curr_level}
    # ---------------------------------------------------------------------

    ep_extras = env.extras.get("episode", {})
    successes = ep_extras.get(cfg.success_key, None)

    if successes is not None:
        import torch
        if isinstance(successes, torch.Tensor):
            if successes.ndim == 0:
                env._obs_curr_successes.extend([successes.item()] * len(env_ids))
            else:
                if successes.ndim == 0:
                    env._obs_curr_successes.extend([successes.item()] * len(env_ids))
                elif successes.shape[0] == env.num_envs:
                    env._obs_curr_successes.extend(successes[env_ids].tolist())
                else:
                    # Fallback: successes already sliced to env_ids length
                    env._obs_curr_successes.extend(successes.tolist())
        else:
            env._obs_curr_successes.extend([float(successes)] * len(env_ids))

    if len(env._obs_curr_successes) < cfg.eval_window:
        return {"obstacle_level": env._obs_curr_level}

    window       = env._obs_curr_successes[-cfg.eval_window:]
    success_rate = sum(window) / len(window)
    level        = OBSTACLE_SCHEDULE[env._obs_curr_level]

    if success_rate >= level.promote_threshold:
        env._obs_curr_consecutive += 1
        if env._obs_curr_consecutive >= cfg.consecutive_windows_to_promote:
            if env._obs_curr_level < len(OBSTACLE_SCHEDULE) - 1:
                env._obs_curr_level      += 1
                env._obs_curr_consecutive = 0
                _apply_curriculum(env, env._obs_curr_level)
                print(f"[ObstacleCurriculum] PROMOTED to level {env._obs_curr_level} "
                        f"(success_rate={success_rate:.2f})  "
                        f"→ obstacles={OBSTACLE_SCHEDULE[env._obs_curr_level].obstacle_count}, "
                        f"max_speed={OBSTACLE_SCHEDULE[env._obs_curr_level].max_speed:.2f} m/s")
    else:
        env._obs_curr_consecutive = 0

    if success_rate < level.demote_threshold and env._obs_curr_level > 0:
        env._obs_curr_level      -= 1   
        env._obs_curr_consecutive = 0
        _apply_curriculum(env, env._obs_curr_level)
        print(f"[ObstacleCurriculum] DEMOTED to level {env._obs_curr_level} "
                f"(success_rate={success_rate:.2f})")

    env._obs_curr_successes = env._obs_curr_successes[-cfg.eval_window * 4:]

    return {
        "obstacle_level":        env._obs_curr_level,
        "obstacle_success_rate": success_rate,
        "obstacle_count":        OBSTACLE_SCHEDULE[env._obs_curr_level].obstacle_count,
        "obstacle_max_speed":    OBSTACLE_SCHEDULE[env._obs_curr_level].max_speed,
    }


def _apply_curriculum(env: "ManagerBasedRLEnv", level: int) -> None:
    """Push the new curriculum parameters to the ObstacleManager."""
    schedule = OBSTACLE_SCHEDULE[level]
    if hasattr(env, "obstacle_manager"):
        env.obstacle_manager.set_curriculum_params(
            obstacle_count=schedule.obstacle_count,
            max_speed=schedule.max_speed,
        )
    else:
        raise AttributeError(
            "env.obstacle_manager not found.  "
            "Instantiate ObstacleManager in your env __init__ and assign it to "
            "self.obstacle_manager before the curriculum term is called."
        )