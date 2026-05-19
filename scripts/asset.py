from pathlib import Path

# Unified root directory for local Isaac/S2R asset files.
ASSET_ROOT = Path("D:/project/S2R/Asset")

# Scene asset keys (env.scene[...] / SceneEntityCfg(...)).
ROBOT_ASSET_NAME = "AUBObot"
ROBOT_ASSET_NAME_1 = ROBOT_ASSET_NAME
ROBOT_ASSET_NAME_2 = "AUBObot_2"
TARGET_ASSET_NAME = "target"
EE_BODY_NAME = "Flange"
# TEST_ASSET_NAME = ""

# AUBO robot local positions relative to the workstation prim.
AUBO_STATION_ROBOT_POS_1 = (0.034, -0.013, 0.816)
AUBO_STATION_ROBOT_POS_2 = (-0.81, 0.21, 0.816)


# Common USD paths.
AUBO_ROBOT_USD = ASSET_ROOT / "AUBO_E5" / "AUBO_E5.usd"
ENV_ASSET_USD  = ASSET_ROOT / "Laboratory" / "M_Laboratory.usd"
STATION_ASSET_USD  = ASSET_ROOT / "QKL-HX-300-II-00" / "WorkStation_All.usd"
