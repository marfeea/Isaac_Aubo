from __future__ import annotations

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
TARGET_MAX_DISPLACEMENT = 0.03
TARGET_MAX_LINEAR_SPEED = 0.05
ILLEGAL_CONTACT_FORCE_THRESHOLD = 1.0e-6
ROBOT_IGNORED_CONTACT_BODY_NAMES = ("Base_Link",)

REWARD_WEIGHTS = {
    "tcp_progress": 80.0,
    "tcp_proximity": 1.0,
    "tcp_parking": 4.0,
    "tcp_dwell": 8.0,
    "step_penalty": -0.25,
    "action_l2": -0.025,
    "action_rate_l2": -0.10,
    "out_of_workspace_penalty": -100.0,
    "illegal_collision_penalty": -140.0,
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
    TargetInitialStateCfg(
        "sample_bottle_state_05",
        (0.90264, -0.50461, 1.0915),
        (1.0, 0.0, 0.0, 0.0),
        (0.90264, -0.38461, 1.0915),
    ),
)
