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
NUM_VARIANTS = 6

# L-corridor geometry — LOCAL frame (relative to env origin)
# Horizontal segment: robot moving in +x direction, y centered at 4.5
# Vertical segment:   robot moving in -y direction, x centered at 4.5
CORRIDOR_HALF_WIDTH = 1.5   # metres — 3m total width

# Segment bounds in LOCAL frame
SEG_H = {  # horizontal: x in [-3, 4.5], y in [3.0, 6.0]
    "x_lo": -3.0, "x_hi": 4.5,
    "y_lo":  3.0, "y_hi": 6.0,
    "y_center": 4.5,
}
SEG_V = {  # vertical: x in [3.0, 6.0], y in [-3, 4.5]
    "x_lo":  3.0, "x_hi": 6.0,
    "y_lo": -3.0, "y_hi": 4.5,
    "x_center": 4.5,
}
CORNER = {  # 3×3 corner area
    "x_lo": 3.0, "x_hi": 6.0,
    "y_lo": 3.0, "y_hi": 6.0,
    "x_center": 4.5, "y_center": 4.5,
}

# Wall clearance for obstacle placement
WALL_MARGIN = 0.5


def _which_segment(local_x: float, local_y: float) -> str:
    """Classify robot position into corridor segment."""
    in_corner = (
        CORNER["x_lo"] <= local_x <= CORNER["x_hi"] and
        CORNER["y_lo"] <= local_y <= CORNER["y_hi"]
    )
    if in_corner:
        return "corner"
    # Primary classification by which segment center is closer
    dist_h = abs(local_y - SEG_H["y_center"])  # distance from horizontal centerline
    dist_v = abs(local_x - SEG_V["x_center"])  # distance from vertical centerline
    return "horizontal" if dist_h < dist_v else "vertical"


class ObstacleManager:
    """
    Manages kinematic obstacle variants for L-shaped corridor.

    Obstacle allocation per level:
      L0: 0 obstacles
      L1: 1 per active segment = 2 total (horizontal + vertical)
      L2: 2 per active segment = 4 total
      L3: 4 total, speed increase
      L4: 2 per segment + 1 corner = 5 total
      L5: 5 total, speed increase

    Slots:
      [0]:   horizontal segment obstacle 1
      [1]:   horizontal segment obstacle 2
      [2]:   vertical segment obstacle 1
      [3]:   vertical segment obstacle 2
      [4]:   corner obstacle (level 4+)
    """

    # Slot assignments — fixed layout
    SLOT_H  = [0, 1, 2]   # horizontal segment slots
    SLOT_V  = [3, 4, 5]   # vertical segment slots
    SLOT_C  = [6]      # corner slot

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        assets: list[RigidObject],
        cfg: ObstacleSetCfg,
    ):
        assert len(assets) == NUM_VARIANTS
        self.env     = env
        self._assets = assets
        self.cfg     = cfg

        self.num_envs = env.num_envs
        self.device   = env.device

        # Curriculum params
        self._obstacle_count: int = 0   # per-segment count (0, 1, 2)
        self._max_speed: float    = 0.0
        self._corner_active: bool = False

        n = self.num_envs
        m = cfg.max_obstacles   # must be >= 5

        self._pos      = torch.zeros(n, m, 3, device=self.device)
        self._vel      = torch.zeros(n, m, 3, device=self.device)
        self._shape_id = torch.full((n, m), -1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------

    def set_curriculum_params(
        self,
        obstacle_count: int,
        max_speed: float,
        corner_active: bool = False,
    ) -> None:
        """
        Args:
            obstacle_count: obstacles PER SEGMENT (0, 1, 2, or 3)
            max_speed:      max translational speed m/s
            corner_active:  whether to place obstacle in corner zone
        """
        self._obstacle_count  = max(0, min(obstacle_count, 3))
        self._max_speed       = max(0.0, min(max_speed, self.cfg.max_speed))
        self._corner_active   = corner_active

    # ------------------------------------------------------------------
    # Environment hooks
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        
        # Pull the play_level directly from the environment configuration
        curr_cfg = getattr(self.env.cfg.curriculum, "obstacle_difficulty", None)
        if curr_cfg is not None:
            play_level = curr_cfg.params.get("cfg").play_level
            if play_level is not None:
                from ict_bot_nav.tasks.a_navigation.mdp import OBSTACLE_SCHEDULE
                sched = OBSTACLE_SCHEDULE[play_level]
                self.set_curriculum_params(
                    obstacle_count=sched.obstacle_count,
                    max_speed=sched.max_speed,
                    corner_active=sched.corner_active
                )

        cfg     = self.cfg
        n_reset = len(env_ids)
        m       = cfg.max_obstacles

        new_pos      = torch.zeros(n_reset, m, 3, device=self.device)
        new_vel      = torch.zeros(n_reset, m, 3, device=self.device)
        new_shape_id = torch.full(
            (n_reset, m), -1, dtype=torch.long, device=self.device
        )

        # Park all slots by default
        park = torch.tensor([PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device)
        new_pos[:] = park

        env_origins  = self.env.scene.env_origins
        robot_pos_w  = self.env.scene["robot"].data.root_pos_w

        for i, eid in enumerate(env_ids.tolist()):
            # Robot position in LOCAL corridor frame
            local_x = (robot_pos_w[eid, 0] - env_origins[eid, 0]).item()
            local_y = (robot_pos_w[eid, 1] - env_origins[eid, 1]).item()

            placed: list[tuple[float, float, int]] = []

            # --- Horizontal segment obstacles ---
            for slot_idx, slot in enumerate(self.SLOT_H):
                if slot_idx >= self._obstacle_count:
                    break   # park this slot

                shape_id = random.randint(0, NUM_VARIANTS - 1)
                new_shape_id[i, slot] = shape_id

                xy = self._sample_segment_xy(
                    "horizontal", local_x, local_y, shape_id, placed, cfg
                )
                if xy is not None:
                    placed.append((xy[0], xy[1], shape_id))
                    new_pos[i, slot] = torch.tensor(
                        [xy[0], xy[1], 0.35], device=self.device
                    )
                    new_vel[i, slot, :2] = self._random_velocity()
                else:
                    new_shape_id[i, slot] = -1  # park if no valid position found

            # --- Vertical segment obstacles ---
            for slot_idx, slot in enumerate(self.SLOT_V):
                if slot_idx >= self._obstacle_count:
                    break

                shape_id = random.randint(0, NUM_VARIANTS - 1)
                new_shape_id[i, slot] = shape_id

                xy = self._sample_segment_xy(
                    "vertical", local_x, local_y, shape_id, placed, cfg
                )
                if xy is not None:
                    placed.append((xy[0], xy[1], shape_id))
                    new_pos[i, slot] = torch.tensor(
                        [xy[0], xy[1], 0.35], device=self.device
                    )
                    new_vel[i, slot, :2] = self._random_velocity()
                else:
                    new_shape_id[i, slot] = -1

            # --- Corner obstacle ---
            if self._corner_active:
                slot     = self.SLOT_C[0]
                shape_id = random.randint(0, NUM_VARIANTS - 1)
                new_shape_id[i, slot] = shape_id

                xy = self._sample_corner_xy(shape_id, placed, cfg)
                if xy is not None:
                    placed.append((xy[0], xy[1], shape_id))
                    new_pos[i, slot] = torch.tensor(
                        [xy[0], xy[1], 0.35], device=self.device
                    )
                    # Corner obstacle moves slower — it is in a confined space
                    new_vel[i, slot, :2] = self._random_velocity(
                        speed_cap=min(self._max_speed, 0.3)
                    )

        self._pos[env_ids]      = new_pos
        self._vel[env_ids]      = new_vel
        self._shape_id[env_ids] = new_shape_id

        self._write_to_sim(env_ids)

    def step(self) -> None:
        """Integrate positions and bounce off L-corridor walls."""
        if self._obstacle_count == 0 and not self._corner_active:
            return
        if self._max_speed == 0.0:
            return

        dt  = self.env.physics_dt

        # All active slots
        active_slots = (
            self.SLOT_H[:self._obstacle_count] +
            self.SLOT_V[:self._obstacle_count] +
            (self.SLOT_C if self._corner_active else [])
        )

        if not active_slots:
            return

        for slot in active_slots:
            pos = self._pos[:, slot, :]   # [N, 3]
            vel = self._vel[:, slot, :]   # [N, 3]

            # Integrate
            pos[:, :2] += vel[:, :2] * dt

            x, y = pos[:, 0], pos[:, 1]
            vx, vy = vel[:, 0], vel[:, 1]

            if slot in self.SLOT_H:
                # Horizontal segment bounds
                flip_x = (x < SEG_H["x_lo"] + WALL_MARGIN) | \
                          (x > SEG_H["x_hi"] - WALL_MARGIN)
                flip_y = (y < SEG_H["y_lo"] + WALL_MARGIN) | \
                          (y > SEG_H["y_hi"] - WALL_MARGIN)

            elif slot in self.SLOT_V:
                # Vertical segment bounds
                flip_x = (x < SEG_V["x_lo"] + WALL_MARGIN) | \
                          (x > SEG_V["x_hi"] - WALL_MARGIN)
                flip_y = (y < SEG_V["y_lo"] + WALL_MARGIN) | \
                          (y > SEG_V["y_hi"] - WALL_MARGIN)

            else:
                # Corner bounds
                flip_x = (x < CORNER["x_lo"] + WALL_MARGIN) | \
                          (x > CORNER["x_hi"] - WALL_MARGIN)
                flip_y = (y < CORNER["y_lo"] + WALL_MARGIN) | \
                          (y > CORNER["y_hi"] - WALL_MARGIN)

            vel[:, 0] = torch.where(flip_x, -vx, vx)
            vel[:, 1] = torch.where(flip_y, -vy, vy)
            pos[:, 0] = torch.clamp(
                x,
                SEG_H["x_lo"] + WALL_MARGIN if slot in self.SLOT_H
                else SEG_V["x_lo"] + WALL_MARGIN if slot in self.SLOT_V
                else CORNER["x_lo"] + WALL_MARGIN,
                SEG_H["x_hi"] - WALL_MARGIN if slot in self.SLOT_H
                else SEG_V["x_hi"] - WALL_MARGIN if slot in self.SLOT_V
                else CORNER["x_hi"] - WALL_MARGIN,
            )
            pos[:, 1] = torch.clamp(
                y,
                SEG_H["y_lo"] + WALL_MARGIN if slot in self.SLOT_H
                else SEG_V["y_lo"] + WALL_MARGIN if slot in self.SLOT_V
                else CORNER["y_lo"] + WALL_MARGIN,
                SEG_H["y_hi"] - WALL_MARGIN if slot in self.SLOT_H
                else SEG_V["y_hi"] - WALL_MARGIN if slot in self.SLOT_V
                else CORNER["y_hi"] - WALL_MARGIN,
            )

            self._pos[:, slot, :] = pos
            self._vel[:, slot, :] = vel

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._write_to_sim(all_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_to_sim(self, env_ids: torch.Tensor) -> None:
        n           = len(env_ids)
        env_origins = self.env.scene.env_origins

        world_pos = self._pos[env_ids].clone()
        world_pos[:, :, 0] += env_origins[env_ids, 0].unsqueeze(1)
        world_pos[:, :, 1] += env_origins[env_ids, 1].unsqueeze(1)
        world_pos[:, :, 2] += env_origins[env_ids, 2].unsqueeze(1)

        park    = torch.tensor([PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device)
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        for variant_id in range(NUM_VARIANTS):
            asset = self._assets[variant_id]

            pos_list = []
            vel_list = []

            for i in range(n):
                eid       = env_ids[i]
                shape_row = self._shape_id[eid]
                matches   = (shape_row == variant_id).nonzero(as_tuple=False)

                if len(matches) > 0:
                    slot = int(matches[0, 0])
                    pos_list.append(world_pos[i, slot])
                    vel_list.append(self._vel[eid, slot])
                else:
                    pos_list.append(park)
                    vel_list.append(torch.zeros(3, device=self.device))

            pos_t  = torch.stack(pos_list)
            vel_t  = torch.stack(vel_list)
            pose   = torch.cat([pos_t, quat_id.unsqueeze(0).expand(n, -1)], dim=-1)
            twist  = torch.cat([vel_t, torch.zeros(n, 3, device=self.device)], dim=-1)

            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(twist, env_ids=env_ids)

    def _random_velocity(self, speed_cap: float | None = None) -> torch.Tensor:
        """Random 2D velocity vector within max_speed."""
        cap   = speed_cap if speed_cap is not None else self._max_speed
        if cap == 0.0:
            return torch.zeros(2, device=self.device)
        speed = random.uniform(0.1, cap)
        angle = random.uniform(0.0, 2.0 * math.pi)
        return torch.tensor(
            [speed * math.cos(angle), speed * math.sin(angle)],
            device=self.device
        )

    def _sample_segment_xy(
        self,
        segment: str,
        robot_local_x: float,
        robot_local_y: float,
        shape_id: int,
        placed: list[tuple[float, float, int]],
        cfg: ObstacleSetCfg,
        max_attempts: int = 200,
    ) -> tuple[float, float] | None:
        own_r = SHAPE_RADIUS[shape_id]

        if segment == "horizontal":
            # Robot moves in +X — obstacles ahead in X
            x_lo = max(robot_local_x + 0.5, SEG_H["x_lo"] + WALL_MARGIN)
            x_hi = min(robot_local_x + cfg.spawn_radius, SEG_H["x_hi"] - WALL_MARGIN)
            y_lo = SEG_H["y_lo"] + WALL_MARGIN
            y_hi = SEG_H["y_hi"] - WALL_MARGIN

        else:  # vertical
            # Robot moves in +Y toward corner
            # If robot is in horizontal segment (y ≈ 4.5), spawn throughout vertical
            # If robot is already in vertical segment, spawn ahead in +Y
            robot_in_vertical = (
                SEG_V["x_lo"] <= robot_local_x <= SEG_V["x_hi"] and
                robot_local_y < SEG_V["y_hi"] - WALL_MARGIN
            )
            if robot_in_vertical:
                y_lo = max(robot_local_y + 0.5, SEG_V["y_lo"] + WALL_MARGIN)
            else:
                # Robot in horizontal — spawn obstacles throughout vertical segment
                y_lo = SEG_V["y_lo"] + WALL_MARGIN
            y_hi = SEG_V["y_hi"] - WALL_MARGIN
            x_lo = SEG_V["x_lo"] + WALL_MARGIN
            x_hi = SEG_V["x_hi"] - WALL_MARGIN

        if x_lo >= x_hi or y_lo >= y_hi:
            return None

        for _ in range(max_attempts):
            x = random.uniform(x_lo, x_hi)
            y = random.uniform(y_lo, y_hi)

            valid = True
            for px, py, pid in placed:
                other_r  = SHAPE_RADIUS[pid]
                min_dist = own_r + other_r + cfg.min_obstacle_spacing
                if math.hypot(x - px, y - py) < min_dist:
                    valid = False
                    break

            if valid:
                return (x, y)

        return None

    def _sample_corner_xy(
        self,
        shape_id: int,
        placed: list[tuple[float, float, int]],
        cfg: ObstacleSetCfg,
        max_attempts: int = 60,
    ) -> tuple[float, float] | None:
        """Sample valid XY in corner zone."""
        own_r = SHAPE_RADIUS[shape_id]
        x_lo  = CORNER["x_lo"] + WALL_MARGIN
        x_hi  = CORNER["x_hi"] - WALL_MARGIN
        y_lo  = CORNER["y_lo"] + WALL_MARGIN
        y_hi  = CORNER["y_hi"] - WALL_MARGIN

        for _ in range(max_attempts):
            x = random.uniform(x_lo, x_hi)
            y = random.uniform(y_lo, y_hi)

            valid = True
            for px, py, pid in placed:
                other_r  = SHAPE_RADIUS[pid]
                min_dist = own_r + other_r + cfg.min_obstacle_spacing
                if math.hypot(x - px, y - py) < min_dist:
                    valid = False
                    break

            if valid:
                return (x, y)

        return None