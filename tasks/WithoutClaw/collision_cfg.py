from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

from tasks.WithoutClaw.task_cfg import EE_BODY_NAME, ROBOT_ASSET_NAME


ROBOT_CONTACT_SENSOR_NAME = "robot_contact_sensor"
TARGET_CONTACT_SENSOR_NAME = "target_contact_sensor"
ROBOT_CONTACT_FORCE_THRESHOLD = 50.0
ROBOT_IGNORED_CONTACT_BODY_NAMES = ("Base_Link",)

ROBOT_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path=f"{{ENV_REGEX_NS}}/{ROBOT_ASSET_NAME}/.*",
    update_period=0.0,
    history_length=3,
    debug_vis=False,
    track_pose=True,
    track_air_time=False,
    force_threshold=ROBOT_CONTACT_FORCE_THRESHOLD,
)


def make_target_contact_sensor_cfg(target_prim_path: str) -> ContactSensorCfg:
    kwargs = {
        "prim_path": f"{{ENV_REGEX_NS}}/{ROBOT_ASSET_NAME}/{EE_BODY_NAME}",
        "update_period": 0.0,
        "history_length": 3,
        "debug_vis": False,
        "track_pose": True,
        "track_air_time": False,
        "force_threshold": 1.0e-6,
    }
    try:
        return ContactSensorCfg(**kwargs, filter_prim_paths_expr=[target_prim_path])
    except TypeError:
        cfg = ContactSensorCfg(**kwargs)
        cfg.filter_prim_paths_expr = [target_prim_path]
        return cfg
