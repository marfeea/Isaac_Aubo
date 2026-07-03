from __future__ import annotations

from dataclasses import dataclass

from configs.place_cfg import Quat, Vec3, WORKSTATION_POSE_CFG


CAMERA_SENSOR_SCENE_NAMES = ("camera_cfg", "camera_cfg_2", "camera_cfg_3")
CAMERA_CAPTURE_INTERVAL_S = 10.0
CAMERA_CAPTURE_OUTPUT_DIR = "data/test"


@dataclass(frozen=True)
class CameraSensorPoseCfg:
    """CameraSensor pose defaults relative to the workstation placement."""

    workstation_offset: Vec3 = (1.4, 0.0, 1.3)
    initial_rot: Quat = (0.5, 0.5, 0.5, 0.5)
    pose_convention: str = "opengl"

    @property
    def initial_pos(self) -> Vec3:
        return (
            WORKSTATION_POSE_CFG.pos[0] + self.workstation_offset[0],
            WORKSTATION_POSE_CFG.pos[1] + self.workstation_offset[1],
            WORKSTATION_POSE_CFG.pos[2] + self.workstation_offset[2],
        )


CAMERA_SENSOR_POSE_CFG = CameraSensorPoseCfg()
CAMERA_SENSOR_2_POSE_CFG = CameraSensorPoseCfg(
    workstation_offset=(0.0, -0.8, 2.0),
    initial_rot=(0.86603, 0.5, 0.0, 0.0),
)
CAMERA_SENSOR_3_POSE_CFG = CAMERA_SENSOR_POSE_CFG
CAMERA_SENSOR_POSE_CFGS = {
    "camera_cfg": CAMERA_SENSOR_POSE_CFG,
    "camera_cfg_2": CAMERA_SENSOR_2_POSE_CFG,
    "camera_cfg_3": CAMERA_SENSOR_3_POSE_CFG,
}
