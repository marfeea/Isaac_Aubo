from __future__ import annotations

from configs.scene_cfg import (
    AUBO_ROBOT_PLACEMENT_CFG,
    TRAINING_ENV_SPACING,
    TRAINING_REPLICATE_PHYSICS,
    AuboTrainingSceneCfg,
)
from tasks.WithClaw.asset_cfg import make_with_claw_aubo_cfg
from tasks.WithClaw.collision_cfg import ROBOT_CONTACT_SENSOR_CFG
from tasks.WithClaw.task_cfg import ROBOT_ASSET_NAME, SECOND_ROBOT_ASSET_NAME


class WithClawSceneCfg(AuboTrainingSceneCfg):
    """复用工作站和相机，只替换为任务内冻结的带夹爪机器人配置。"""

    AUBObot = make_with_claw_aubo_cfg(ROBOT_ASSET_NAME, AUBO_ROBOT_PLACEMENT_CFG.world_pos_1)
    AUBObot_2 = make_with_claw_aubo_cfg(
        SECOND_ROBOT_ASSET_NAME,
        AUBO_ROBOT_PLACEMENT_CFG.world_pos_2,
    )
    robot_contact_sensor = ROBOT_CONTACT_SENSOR_CFG
    target_contact_sensor = None


__all__ = ("TRAINING_ENV_SPACING", "TRAINING_REPLICATE_PHYSICS", "WithClawSceneCfg")
