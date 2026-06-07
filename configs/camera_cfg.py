from __future__ import annotations

from dataclasses import dataclass

from configs.place_cfg import Quat, Vec3, WORKSTATION_POSE_CFG


@dataclass(frozen=True)
class CameraSensorPoseCfg:
    """CameraSensor pose defaults relative to the workstation placement."""

    workstation_offset: Vec3 = (-0.5, 0.0, 0.9)
    initial_rot: Quat = (0.70711, 0.0, 0.70711, 0.0)
    pose_convention: str = "opengl"

    @property
    def initial_pos(self) -> Vec3:
        return (
            WORKSTATION_POSE_CFG.pos[0] + self.workstation_offset[0],
            WORKSTATION_POSE_CFG.pos[1] + self.workstation_offset[1],
            WORKSTATION_POSE_CFG.pos[2] + self.workstation_offset[2],
        )


CAMERA_SENSOR_POSE_CFG = CameraSensorPoseCfg()
