from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from tools.scene import AuboToolFns


class AuboContactToolFns:
    """接触传感器 schema 兼容工具。"""

    @staticmethod
    def get_optional_sensor(env, sensor_cfg: SceneEntityCfg | str):
        """从 scene 读取可选传感器；不存在时返回 None。"""
        sensor_name = AuboToolFns.resolve_asset_name(sensor_cfg)
        try:
            return env.scene[sensor_name]
        except Exception:
            return None

    @staticmethod
    def extract_contact_magnitude(sensor) -> torch.Tensor | None:
        """从不同 Isaac Lab 版本的传感器字段中提取接触力模长。"""
        if sensor is None or not hasattr(sensor, "data"):
            return None

        for attr in (
            "net_forces_w",
            "body_net_forces_w",
            "contact_forces_w",
            "force_matrix_w",
            "contact_force_matrix_w",
        ):
            tensor = getattr(sensor.data, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.shape[-1] >= 3:
                return torch.norm(tensor[..., :3], dim=-1)
        return None
