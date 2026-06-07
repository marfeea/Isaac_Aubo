from pathlib import Path

# Global asset root. All local USD paths are derived from this directory.
ASSET_ROOT = Path("D:/project/S2R/Asset")

# Scene asset keys and commonly referenced USD body names.
ROBOT_ASSET_NAME = "AUBObot"
ROBOT_ASSET_NAME_1 = ROBOT_ASSET_NAME
ROBOT_ASSET_NAME_2 = "AUBObot_2"
TARGET_ASSET_NAME = "target"
EE_BODY_NAME = "Flange"

# Common USD asset paths.
AUBO_ROBOT_USD = ASSET_ROOT / "AUBO_E5" / "AUBO_E5.usd"
ENV_ASSET_USD = ASSET_ROOT / "Laboratory" / "M_Laboratory.usd"
STATION_ASSET_USD = ASSET_ROOT / "QKL-HX-300-II-00" / "WorkStation_All.usd"
