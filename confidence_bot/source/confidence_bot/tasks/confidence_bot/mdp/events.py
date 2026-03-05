from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def reset_camera_posture_uniform(env: ManagerBasedRLEnv, env_ids: torch.Tensor, z_range: tuple[float, float], pitch_range: tuple[float, float], sensor_name: str = "tiled_camera"):
    """Randomizes the camera's height and tilt (pitch) relative to its parent link."""
    # 1. Extract the sensor object
    tiled_camera = env.scene[sensor_name]
    num_resets = len(env_ids)

    # 2. Generate random values
    rand_z = (z_range[1] - z_range[0]) * torch.rand(num_resets, device=env.device) + z_range[0]
    rand_pitch = (pitch_range[1] - pitch_range[0]) * torch.rand(num_resets, device=env.device) + pitch_range[0]

    # 3. Apply Height (Z-axis)
    # We modify the world position's Z component for the resetting envs
    tiled_camera.data.pos_w[env_ids, 2] += rand_z

    # 4. Apply Pitch (Tilt)
    # Create a quaternion for the random pitch (rotation around Y)
    # Order is (roll, pitch, yaw)
    pitch_quat = quat_from_euler_xyz(
        torch.zeros(num_resets, device=env.device),
        rand_pitch,
        torch.zeros(num_resets, device=env.device)
    )

    # Multiply the current world orientation by the new random pitch
    # quat_mul handles the vectorized multiplication for all env_ids
    tiled_camera.data.quat_w_world[env_ids] = quat_mul(tiled_camera.data.quat_w_world[env_ids], pitch_quat)


def update_camera_fov_uniform(env: ManagerBasedRLEnv, env_ids: torch.Tensor, fov_range: tuple[float, float], sensor_name: str = "tiled_camera"):
    """Randomizes the Horizontal FOV by constructing new intrinsic matrices."""
    tiled_camera = env.scene[sensor_name]
    num_resets = len(env_ids)
    height, width = tiled_camera.image_shape
    
    # 1. Generate new FOV values (in degrees) and convert to radians
    new_fov_deg = (fov_range[1] - fov_range[0]) * torch.rand(num_resets, device=env.device) + fov_range[0]
    new_fov_rad = torch.deg2rad(new_fov_deg)
    
    # 2. Calculate Focal Length in pixels
    # Formula: f_pixels = (width / 2) / tan(fov_rad / 2)
    f_pixels = (width / 2.0) / torch.tan(new_fov_rad / 2.0)
    
    # 3. Construct the Intrinsic Matrices (K) [N, 3, 3]
    # K = [[fx,  0, cx],
    #      [ 0, fy, cy],
    #      [ 0,  0,  1]]
    K = torch.zeros((num_resets, 3, 3), device=env.device)
    
    # Set focal lengths (assuming square pixels, so fx = fy)
    K[:, 0, 0] = f_pixels
    K[:, 1, 1] = f_pixels
    
    # Set principal point (center of the image)
    K[:, 0, 2] = width / 2.0
    K[:, 1, 2] = height / 2.0
    
    # Homogeneous coordinate
    K[:, 2, 2] = 1.0
    
    # 4. Apply the change
    # Note: We pass the K matrices. The function will update the USD apertures automatically.
    tiled_camera.set_intrinsic_matrices(matrices=K, env_ids=env_ids)