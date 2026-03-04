import os
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

## Path to your AprilTag texture or USD
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# You can use a simple plane USD with the tag texture applied
APRILTAG_USD_PATH = os.path.join(ROOT_DIR, "apriltag.usd") 

APRILTAG_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=APRILTAG_USD_PATH,
        # We make it kinematic so it doesn't fall through the floor 
        # or move when the robot bumps it.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, 
            disable_gravity=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(2.0, 0.0, 0.01), # Spawn 2 meters in front of the robot
        rot=(1.0, 0.0, 0.0, 0.0), # Flat on the floor
    ),
)