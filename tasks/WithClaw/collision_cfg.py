from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

from tasks.WithClaw.task_cfg import ILLEGAL_CONTACT_FORCE_THRESHOLD, ROBOT_ASSET_NAME


ROBOT_CONTACT_SENSOR_NAME = "robot_contact_sensor"
ROBOT_ARTICULATION_PRIM_NAME = "AUBO_E5"
ROBOT_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path=f"{{ENV_REGEX_NS}}/{ROBOT_ASSET_NAME}/{ROBOT_ARTICULATION_PRIM_NAME}/.*",
    update_period=0.0,
    history_length=3,
    debug_vis=False,
    track_pose=True,
    track_air_time=False,
    force_threshold=ILLEGAL_CONTACT_FORCE_THRESHOLD,
)
