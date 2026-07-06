from __future__ import annotations

from pathlib import Path


TASK_NAME = "WithoutClaw"
ROBOT_ASSET_NAME = "AUBObot"
SECOND_ROBOT_ASSET_NAME = "AUBObot_2"
EE_BODY_NAME = "Flange"
DEFAULT_TARGET_ASSET_NAME = "ws_interactive_reagent_01_sample_bottle"
WITHOUT_CLAW_ROBOT_USD = Path("D:/project/S2R/Asset/AUBO_E5/AUBO_E5.usd")

RL_SIM_DT = 1.0 / 120.0
RL_DECIMATION = 30
RL_EPISODE_LENGTH_S = 40.0
RL_MAX_EPISODE_STEPS = round(RL_EPISODE_LENGTH_S / (RL_DECIMATION * RL_SIM_DT))
RL_WORKSPACE = {
    "x": [-0.75, 0.75],
    "y": [-0.75, 0.75],
    "z": [0.20, 1.10],
}
