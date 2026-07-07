from __future__ import annotations

from math import cos, radians, sqrt
from pathlib import Path
from typing import NamedTuple


TASK_NAME = "WithClaw"
ROBOT_ASSET_NAME = "AUBObot"
SECOND_ROBOT_ASSET_NAME = "AUBObot_2"
EE_BODY_NAME = "Flange"
DEFAULT_TARGET_ASSET_NAME = "ws_interactive_reagent_01_sample_bottle"
WITH_CLAW_ROBOT_USD = Path("D:/project/S2R/Asset/AUBO_E5/AUBO_E5_Withclaw.usd")

# 用户确认继续后，按同一 Flange 局部坐标系内的反向向量冻结第一版偏置。
# USD 几何方向冲突保留在设计文档中，后续实测标定可只修改此单一事实源。
RECORDED_TCP_TO_FLANGE_TRANSLATION = (0.0, 0.12, -0.102)
DIRECT_REVERSE_TRANSLATION = (0.0, -0.12, 0.102)
FLANGE_TO_TCP_TRANSLATION_F: tuple[float, float, float] = DIRECT_REVERSE_TRANSLATION

# wxyz；工具 +Z 经该固定旋转映射到 Flange -X。
FLANGE_TO_TOOL_ROTATION_F = (sqrt(0.5), 0.0, -sqrt(0.5), 0.0)
# wxyz；工具 +Z 映射到目标 -Y，同时固定绕停靠轴的剩余自由度。
TARGET_TO_TOOL_ROTATION_T = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)
TOOL_FORWARD_AXIS_T = (0.0, 0.0, 1.0)
TARGET_DOCKING_AXIS_T = (0.0, -1.0, 0.0)

RL_SIM_DT = 1.0 / 120.0
RL_DECIMATION = 30
RL_EPISODE_LENGTH_S = 40.0
RL_MAX_EPISODE_STEPS = round(RL_EPISODE_LENGTH_S / (RL_DECIMATION * RL_SIM_DT))
RL_WORKSPACE = {
    "x": [-0.75, 0.75],
    "y": [-0.75, 0.75],
    "z": [0.20, 1.10],
}

TCP_PARKING_ENTER_DISTANCE = 0.03
TCP_PARKING_EXIT_DISTANCE = 0.045
TCP_PARKING_SPEED_THRESHOLD = 0.02
TCP_PARKING_REQUIRED_STEPS = 3
TCP_PROXIMITY_SIGMA = 0.20
TCP_PARKING_SPEED_SIGMA = 0.02
TOOL_ORIENTATION_REWARD_START_DISTANCE = 0.10
TOOL_ORIENTATION_LOCK_DISTANCE = 0.03
TOOL_ORIENTATION_SCORE_SIGMA_RAD = radians(30.0)
TOOL_ORIENTATION_MATCH_THRESHOLD_RAD = radians(5.0)
TOOL_ORIENTATION_MATCH_COS = cos(TOOL_ORIENTATION_MATCH_THRESHOLD_RAD)
TARGET_MAX_DISPLACEMENT = 0.03
TARGET_MAX_LINEAR_SPEED = 0.05
ILLEGAL_CONTACT_FORCE_THRESHOLD = 50.0
ROBOT_IGNORED_CONTACT_BODY_NAMES = ("Base_Link",)

REWARD_WEIGHTS = {
    "tcp_progress": 80.0,
    "tcp_proximity": 0.10,
    "tcp_parking": 0.10,
    "tool_axis_progress": 20.0,
    "parking_success": 100.0,
    "step_penalty": -0.25,
    "action_l2": -0.025,
    "action_rate_l2": -0.10,
    "out_of_workspace_penalty": -160.0,
    "illegal_collision_penalty": -200.0,
    "target_displaced_penalty": -160.0,
    "target_too_fast_penalty": -160.0,
    "time_out_penalty": -40.0,
}


class TargetInitialStateCfg(NamedTuple):
    """目标离散初始状态；pos/preposition 为 env 局部坐标，rot 为 wxyz。"""

    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    preposition: tuple[float, float, float]


TARGET_INITIAL_STATES = (
    TargetInitialStateCfg(
        "sample_bottle_state_01",
        (1.537, 0.203, 0.94),
        (0.0, 0.0, 0.0, 1.0),
        (1.537, 0.083, 0.94),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_02",
        (0.91167, 0.1753, 0.96789),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.1753, 0.96789),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_03",
        (0.91167, 0.03036, 0.96676),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.03036, 0.96676),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_04",
        (0.91235, -0.18557, 0.99091),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03235, -0.18557, 0.99091),
    ),
)
