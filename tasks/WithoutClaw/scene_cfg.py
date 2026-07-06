from __future__ import annotations

from configs.place_cfg import WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS
from configs.scene_cfg import (
    AUBO_ROBOT_PLACEMENT_CFG,
    TRAINING_ENV_SPACING,
    TRAINING_REPLICATE_PHYSICS,
    AuboTrainingSceneCfg,
)
from tasks.WithoutClaw.asset_cfg import make_without_claw_aubo_cfg
from tasks.WithoutClaw.collision_cfg import ROBOT_CONTACT_SENSOR_CFG, make_target_contact_sensor_cfg
from tasks.WithoutClaw.task_cfg import ROBOT_ASSET_NAME, SECOND_ROBOT_ASSET_NAME


class WithoutClawSceneCfg(AuboTrainingSceneCfg):
    """复用工作站背景，但显式替换为无夹爪机器人和旧 contact 路径。"""

    AUBObot = make_without_claw_aubo_cfg(ROBOT_ASSET_NAME, AUBO_ROBOT_PLACEMENT_CFG.world_pos_1)
    AUBObot_2 = make_without_claw_aubo_cfg(
        SECOND_ROBOT_ASSET_NAME,
        AUBO_ROBOT_PLACEMENT_CFG.world_pos_2,
    )
    robot_contact_sensor = ROBOT_CONTACT_SENSOR_CFG
    target_contact_sensor = make_target_contact_sensor_cfg(
        f"{{ENV_REGEX_NS}}/station/interactive/{WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS[0]['name']}"
    )


__all__ = ("TRAINING_ENV_SPACING", "TRAINING_REPLICATE_PHYSICS", "WithoutClawSceneCfg")
