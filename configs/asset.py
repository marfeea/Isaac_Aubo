from pathlib import Path

# Global asset root. All local USD paths are derived from this directory.
ASSET_ROOT = Path("D:/project/S2R/Asset")

# Scene asset keys and commonly referenced USD body names.
ROBOT_ASSET_NAME = "AUBObot"
ROBOT_ASSET_NAME_1 = ROBOT_ASSET_NAME
ROBOT_ASSET_NAME_2 = "AUBObot_2"
ROBOT_ARTICULATION_PRIM_NAME = "AUBO_E5"
TARGET_ASSET_NAME = "target"
EE_BODY_NAME = "Flange"
GRIPPER_PRIM_NAME = "ClawTool_C"

# 夹爪末端指向法兰的近似平移偏置，按 (X, Y, Z) 排列，单位为米。
# 旋转关系尚未标定，因此该配置不表示完整的刚体变换。
GRIPPER_TIP_TO_FLANGE_TRANSLATION = (0.0, 0.12, -0.102)

# Common USD asset paths.
AUBO_ROBOT_USD = ASSET_ROOT / "AUBO_E5" / "AUBO_E5_Withclaw.usd"
ENV_ASSET_USD = ASSET_ROOT / "Laboratory" / "M_Laboratory.usd"
STATION_ASSET_USD = ASSET_ROOT / "QKL-HX-300-II-00" / "WorkStation_All.usd"
