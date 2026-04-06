# assets.py
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# A generic Box
CUBE_OBSTACLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(1.0, 1.0, 0.7),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
)

# A generic Cylinder
CYLINDER_OBSTACLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cylinder",
    spawn=sim_utils.CylinderCfg(
        radius=0.5, height=0.7,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
    ),
)