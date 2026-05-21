"""
ObstacleManager — per-environment obstacle lifecycle for ICT-Bot Stage 1.

Six obstacle variants (3 cube sizes × 2 shapes) are pre-spawned per
environment. At each reset, active slots are assigned a random variant
and placed within spawn_radius of the robot. Inactive slots are parked
below the floor. Velocity-driven bouncing keeps obstacles in-corridor.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from ict_bot_nav.assets.obstacles.object_obstacle import SHAPE_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject
    from ict_bot_nav.assets.obstacles.object_obstacle import ObstacleSetCfg

PARK_Z  = -10.0
PARK_XY = (1000.0, 1000.0)

# shape_id → asset index in self._assets list
# 0=cube_small, 1=cube_medium, 2=cube_large
# 3=cyl_small,  4=cyl_medium,  5=cyl_large
NUM_VARIANTS = 6


class ObstacleManager:
    """
    Manages six kinematic obstacle variants for one vectorised env batch.

    Parameters
    ----------
    env:
        The IsaacLab ManagerBasedRLEnv instance.
    assets:
        List of six RigidObject handles in order:
        [cube_small, cube_medium, cube_large,
         cyl_small,  cyl_medium,  cyl_large]
    cfg:
        ObstacleSetCfg instance.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        assets: list[RigidObject],
        cfg: ObstacleSetCfg,
    ):
        assert len(assets) == NUM_VARIANTS, \
            f"Expected {NUM_VARIANTS} asset handles, got {len(assets)}"

        self.env     = env
        self._assets = assets   # indexed by shape_id
        self.cfg     = cfg

        self.num_envs: int = env.num_envs
        self.device        = env.device

        # Curriculum-controlled
        self._active_count: int  = 0
        self._max_speed: float   = 0.0

        n = self.num_envs
        m = cfg.max_obstacles
        # LOCAL corridor-frame state
        self._pos      = torch.zeros(n, m, 3, device=self.device)
        self._vel      = torch.zeros(n, m, 3, device=self.device)
        self._shape_id = torch.full((n, m), -1, dtype=torch.long, device=self.device)
        # -1 = parked

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------

    def set_curriculum_params(self, obstacle_count: int, max_speed: float) -> None:
        self._active_count = max(0, min(obstacle_count, self.cfg.max_obstacles))
        self._max_speed    = max(0.0, min(max_speed, self.cfg.max_speed))

    # ------------------------------------------------------------------
    # Environment hooks
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        cfg     = self.cfg
        n_reset = len(env_ids)
        m       = cfg.max_obstacles

        new_pos      = torch.zeros(n_reset, m, 3, device=self.device)
        new_vel      = torch.zeros(n_reset, m, 3, device=self.device)
        new_shape_id = torch.full((n_reset, m), -1, dtype=torch.long, device=self.device)

        robot_pos_w = self.env.scene["robot"].data.root_pos_w

        for i, eid in enumerate(env_ids.tolist()):
            robot_x = robot_pos_w[eid, 0].item()
            robot_y = robot_pos_w[eid, 1].item()

            placed: list[tuple[float, float, int]] = []

            # Shuffle available shapes so assignment is random but unique per env
            available_shapes = random.sample(cfg.shapes, min(len(cfg.shapes), m))

            for obs_idx in range(m):
                if obs_idx >= self._active_count:
                    new_pos[i, obs_idx] = torch.tensor(
                        [PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device
                    )
                    continue

                shape_id = available_shapes[obs_idx]
                new_shape_id[i, obs_idx] = shape_id

                xy = self._sample_valid_xy(
                    robot_x, robot_y, shape_id, placed, cfg
                )
                placed.append((xy[0], xy[1], shape_id))

                new_pos[i, obs_idx] = torch.tensor(
                    [xy[0], xy[1], 0.35], device=self.device
                )

                if self._max_speed > 0.0:
                    speed = random.uniform(0.1, self._max_speed)
                    angle = random.uniform(0.0, 2.0 * math.pi)
                    new_vel[i, obs_idx, 0] = speed * math.cos(angle)
                    new_vel[i, obs_idx, 1] = speed * math.sin(angle)

        self._pos[env_ids]      = new_pos
        self._vel[env_ids]      = new_vel
        self._shape_id[env_ids] = new_shape_id

        self._write_to_sim(env_ids)

    def step(self) -> None:
        """Integrate positions in local frame and bounce off corridor walls."""
        if self._active_count == 0 or self._max_speed == 0.0:
            return

        dt     = self.env.physics_dt
        cfg    = self.cfg
        active = self._active_count

        self._pos[:, :active, :2] += self._vel[:, :active, :2] * dt

        # Bounce X — stay within spawn radius bounds along corridor
        x      = self._pos[:, :active, 0]
        vx     = self._vel[:, :active, 0]
        x_lo   = 0.5
        x_hi   = cfg.corridor_half_width * 2 + 2.0  # generous corridor length
        flip_x = (x < x_lo) | (x > x_hi)
        self._vel[:, :active, 0] = torch.where(flip_x, -vx, vx)
        self._pos[:, :active, 0] = torch.clamp(x, x_lo, x_hi)

        # Bounce Y — stay within corridor width
        y      = self._pos[:, :active, 1]
        vy     = self._vel[:, :active, 1]
        margin = 0.5  # keep away from walls
        y_lo   = -(cfg.corridor_half_width - margin)
        y_hi   =   cfg.corridor_half_width - margin
        flip_y = (y < y_lo) | (y > y_hi)
        self._vel[:, :active, 1] = torch.where(flip_y, -vy, vy)
        self._pos[:, :active, 1] = torch.clamp(y, y_lo, y_hi)

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._write_to_sim(all_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_sim(self, env_ids: torch.Tensor) -> None:
        """Convert local positions to world frame and write to each asset."""
        n           = len(env_ids)
        env_origins = self.env.scene.env_origins  # [N, 3]

        world_pos = self._pos[env_ids].clone()  # [n, m, 3]
        world_pos[:, :, 0] += env_origins[env_ids, 0].unsqueeze(1)
        world_pos[:, :, 1] += env_origins[env_ids, 1].unsqueeze(1)
        world_pos[:, :, 2] += env_origins[env_ids, 2].unsqueeze(1)

        park = torch.tensor([PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device)
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        for variant_id in range(NUM_VARIANTS):
            asset = self._assets[variant_id]

            # For each env in env_ids, find first slot with this variant
            pos_list = []
            vel_list = []

            for i in range(n):
                eid        = env_ids[i]
                shape_row  = self._shape_id[eid]           # [m]
                matches    = (shape_row == variant_id).nonzero(as_tuple=False)

                if len(matches) > 0:
                    slot = int(matches[0, 0])
                    pos_list.append(world_pos[i, slot])
                    vel_list.append(self._vel[eid, slot])
                else:
                    pos_list.append(park)
                    vel_list.append(torch.zeros(3, device=self.device))

            pos_t = torch.stack(pos_list, dim=0)           # [n, 3]
            vel_t = torch.stack(vel_list, dim=0)           # [n, 3]

            pose = torch.cat(
                [pos_t, quat_id.unsqueeze(0).expand(n, -1)], dim=-1
            )   # [n, 7]
            twist = torch.cat(
                [vel_t, torch.zeros(n, 3, device=self.device)], dim=-1
            )   # [n, 6]

            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(twist, env_ids=env_ids)

    @staticmethod
    def _sample_valid_xy(
        robot_x: float,
        robot_y: float,
        shape_id: int,
        placed: list[tuple[float, float, int]],
        cfg: ObstacleSetCfg,
        max_attempts: int = 60,
    ) -> tuple[float, float]:
        """
        Sample XY in LOCAL corridor frame within spawn_radius of robot,
        within corridor bounds, and not overlapping already-placed obstacles.
        """
        r_spawn = cfg.spawn_radius
        y_lo    = -(cfg.corridor_half_width - 0.5)
        y_hi    =   cfg.corridor_half_width - 0.5
        # Keep obstacles ahead of robot in X (at least 0.5 m ahead)
        x_lo    = robot_x + 0.5
        x_hi    = robot_x + r_spawn

        own_radius = SHAPE_RADIUS[shape_id]

        for _ in range(max_attempts):
            x = random.uniform(x_lo, x_hi)
            y = random.uniform(y_lo, y_hi)

            # Check spacing against all already-placed obstacles
            valid = True
            for px, py, pid in placed:
                other_radius = SHAPE_RADIUS[pid]
                min_dist = own_radius + other_radius + cfg.min_obstacle_spacing
                if math.hypot(x - px, y - py) < min_dist:
                    valid = False
                    break

            if valid:
                return (x, y)

        # Fallback: return best effort even if spacing not met
        return (
            random.uniform(x_lo, x_hi),
            random.uniform(y_lo, y_hi),
        )