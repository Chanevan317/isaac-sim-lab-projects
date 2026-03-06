from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.sensors import TiledCamera
from isaaclab.assets import RigidObject
from isaaclab.utils.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def get_tag_pixel_coords(env: ManagerBasedRLEnv, tag_cfg: SceneEntityCfg, sensor_name: str = "tiled_camera") -> torch.Tensor:
    # 1. Access camera and tag
    camera = env.scene[sensor_name]
    tag = env.scene[tag_cfg.name]

    # 2. Get world positions
    # tag_pos_w: [num_envs, 3]
    # cam_pos_w: [num_envs, 3], cam_quat_w: [num_envs, 4] (world-to-camera)
    tag_pos_w = tag.data.root_pos_w
    cam_pos_w = camera.data.pos_w
    cam_quat_w = camera.data.quat_w_world 

    # 3. Transform Tag position to Camera Local Frame
    # We subtract camera position and rotate by the inverse of camera orientation
    rel_pos = tag_pos_w - cam_pos_w
    # Isaac Lab cameras use ROS convention (+Z forward) for projection
    tag_pos_cam = quat_apply(quat_inv(cam_quat_w), rel_pos)

    # 4. Project using Intrinsic Matrix
    # Formula: p_pixel = K * p_camera / z
    K = camera.data.intrinsic_matrices # [num_envs, 3, 3]
    
    # We need to normalize by the depth (z-coordinate in camera frame)
    # Adding a small epsilon to avoid division by zero
    z = tag_pos_cam[:, 2:3].clamp(min=1e-6)
    
    # Standard perspective projection
    # u = (K[:, 0, 0] * tag_pos_cam[:, 0:1] / z) + K[:, 0, 2]
    # v = (K[:, 1, 1] * tag_pos_cam[:, 1:2] / z) + K[:, 1, 2]
    u = (K[:, 0, 0].unsqueeze(-1) * tag_pos_cam[:, 0:1] / z) + K[:, 0, 2].unsqueeze(-1)
    v = (K[:, 1, 1].unsqueeze(-1) * tag_pos_cam[:, 1:2] / z) + K[:, 1, 2].unsqueeze(-1)
    
    # 5. Normalize to [-1, 1] range for RL
    # (Optional: helpful for neural networks to have normalized inputs)
    img_h, img_w = camera.image_shape
    u_norm = (u / img_w) * 2.0 - 1.0
    v_norm = (v / img_h) * 2.0 - 1.0
    
    pixel_coords_2d = torch.cat([u_norm, v_norm], dim=-1)

    # 6. Visibility check
    # Tag is visible if it's in front of camera (z > 0) and within image bounds
    is_visible = (z > 0).float() * \
                 (u_norm.abs() <= 1.0).float() * \
                 (v_norm.abs() <= 1.0).float()

    return torch.cat([pixel_coords_2d, is_visible], dim=-1)


def get_imu_data(env: ManagerBasedRLEnv, sensor_name: str = "imu") -> torch.Tensor:
    # Just grab the IMU sensor from the scene
    imu = env.scene[sensor_name]

    # Return Angular Velocity (Gyro) and Linear Acceleration (Accel)
    # This gives the AI the 'feel' of the 1m pole's stability.
    return torch.cat([imu.data.ang_vel_b, imu.data.lin_acc_b], dim=-1)


def get_target_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns the forward velocity command assigned to the robot."""
    # This extracts the 'x' velocity (forward) from the command manager
    # We assume your command term is named 'base_velocity'
    return env.command_manager.get_command("base_velocity")[:, 0:1]