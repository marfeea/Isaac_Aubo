from pathlib import Path

# Unified root directory for local Isaac/S2R asset files.
ASSET_ROOT = Path("D:/project/S2R/Asset")

# Scene asset keys (env.scene[...] / SceneEntityCfg(...)).
ROBOT_ASSET_NAME = "AUBObot"
TARGET_ASSET_NAME = "target"
EE_BODY_NAME = "Flange"


# Common USD paths.
AUBO_ROBOT_USD = ASSET_ROOT / "AUBO_E5" / "AUBO_E5.usd"
