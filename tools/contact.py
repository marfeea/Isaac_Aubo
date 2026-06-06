from __future__ import annotations

from collections.abc import Callable, Sequence

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

    @staticmethod
    def extract_filtered_contact_magnitude(sensor) -> torch.Tensor | None:
        """Return magnitudes from target-filtered contact matrices, if present."""
        if sensor is None or not hasattr(sensor, "data"):
            return None

        for attr in ("force_matrix_w", "contact_force_matrix_w"):
            tensor = getattr(sensor.data, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.shape[-1] >= 3:
                return torch.norm(tensor[..., :3], dim=-1)
        return None

    @staticmethod
    def body_names(sensor) -> list[str]:
        """Return body names exposed by the contact sensor, if available."""
        for owner in (sensor, getattr(sensor, "data", None)):
            names = getattr(owner, "body_names", None)
            if names:
                return list(names)
        return []

    @staticmethod
    def extract_env_contact_magnitude(sensor, num_envs: int) -> torch.Tensor | None:
        """Return contact magnitudes reshaped as (num_envs, contact_channels)."""
        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            return None
        return AuboContactToolFns.reshape_contact_magnitude(magnitude, num_envs)

    @staticmethod
    def extract_env_filtered_contact_magnitude(sensor, num_envs: int) -> torch.Tensor | None:
        """Return filtered contact magnitudes reshaped as (num_envs, contact_channels)."""
        magnitude = AuboContactToolFns.extract_filtered_contact_magnitude(sensor)
        if magnitude is None:
            return None
        return AuboContactToolFns.reshape_contact_magnitude(magnitude, num_envs)

    @staticmethod
    def reshape_contact_magnitude(magnitude: torch.Tensor, num_envs: int) -> torch.Tensor:
        """Reshape a contact magnitude tensor to (num_envs, contact_channels)."""
        if magnitude.ndim == 0:
            return magnitude.reshape(1, 1)
        if magnitude.shape[0] == num_envs:
            return magnitude.reshape(num_envs, -1)
        if magnitude.shape[0] % num_envs == 0:
            return magnitude.reshape(num_envs, -1)
        return magnitude.reshape(1, -1)

    @staticmethod
    def extract_body_contact_magnitude(env, sensor_cfg: SceneEntityCfg | str, body_names: Sequence[str]) -> torch.Tensor | None:
        """Return per-env max contact magnitude for selected sensor body names."""
        sensor = AuboContactToolFns.get_optional_sensor(env, sensor_cfg)
        env_magnitude = AuboContactToolFns.extract_env_contact_magnitude(sensor, env.num_envs)
        if env_magnitude is None:
            return None

        sensor_body_names = AuboContactToolFns.body_names(sensor)
        if not sensor_body_names:
            return torch.amax(env_magnitude, dim=1)

        wanted = set(body_names)
        body_ids = [idx for idx, name in enumerate(sensor_body_names) if name in wanted and idx < env_magnitude.shape[1]]
        if not body_ids:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.amax(env_magnitude[:, body_ids], dim=1)

    @staticmethod
    def extract_non_body_contact_magnitude(env, sensor_cfg: SceneEntityCfg | str, body_names: Sequence[str]) -> torch.Tensor | None:
        """Return per-env max contact magnitude for all sensor bodies except selected names."""
        sensor = AuboContactToolFns.get_optional_sensor(env, sensor_cfg)
        env_magnitude = AuboContactToolFns.extract_env_contact_magnitude(sensor, env.num_envs)
        if env_magnitude is None:
            return None

        sensor_body_names = AuboContactToolFns.body_names(sensor)
        if not sensor_body_names:
            return torch.zeros(env.num_envs, device=env.device)

        excluded = set(body_names)
        body_ids = [idx for idx, name in enumerate(sensor_body_names) if name not in excluded and idx < env_magnitude.shape[1]]
        if not body_ids:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.amax(env_magnitude[:, body_ids], dim=1)


def enable_physx_contact_reports(scene_or_stage, *, threshold: float = 0.0) -> int:
    """Enable PhysX contact report callbacks for physics prims in the stage."""
    from pxr import PhysxSchema, UsdPhysics

    stage = scene_or_stage.stage if hasattr(scene_or_stage, "stage") else scene_or_stage
    enabled_count = 0
    for prim in stage.TraverseAll():
        if not prim.IsValid() or not prim.IsActive():
            continue
        if not (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            continue

        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contact_report_api.CreateThresholdAttr().Set(float(threshold))
        enabled_count += 1
    return enabled_count


class PhysxContactPairPrinter:
    """Print collider prim pairs when PhysX reports a new contact."""

    def __init__(
        self,
        scene,
        *,
        path_filter: Callable[[Sequence[str]], bool] | None = None,
        prefix: str = "[TRAIN][collision]",
    ):
        from omni.physx import get_physx_simulation_interface
        from pxr import UsdUtils

        self._stage_id = UsdUtils.StageCache.Get().GetId(scene.stage).ToLongInt()
        self._path_filter = path_filter
        self._prefix = prefix
        self._active_pairs: set[tuple[str, str]] = set()
        self._current_step = 0
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report_event
        )

    def set_step(self, step: int) -> None:
        self._current_step = int(step)

    def close(self) -> None:
        self._subscription = None

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        from omni.physx.bindings._physx import ContactEventType

        del contact_data
        for contact_header in contact_headers:
            if int(contact_header.stage_id) != self._stage_id:
                continue

            collider0 = self._id_to_path(contact_header.collider0)
            collider1 = self._id_to_path(contact_header.collider1)
            if not collider0 or not collider1:
                continue

            pair = (collider0, collider1) if collider0 <= collider1 else (collider1, collider0)
            if contact_header.type in (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST):
                if pair in self._active_pairs:
                    continue
                self._active_pairs.add(pair)
                actor0 = self._id_to_path(contact_header.actor0)
                actor1 = self._id_to_path(contact_header.actor1)
                if self._path_filter is not None and not self._path_filter((collider0, collider1, actor0, actor1)):
                    continue
                print(
                    f"{self._prefix} "
                    f"timestep={self._current_step} "
                    f"collider0={collider0} "
                    f"collider1={collider1} "
                    f"actor0={actor0} "
                    f"actor1={actor1}",
                    flush=True,
                )
            elif contact_header.type == ContactEventType.CONTACT_LOST:
                self._active_pairs.discard(pair)

    @staticmethod
    def _id_to_path(path_id: int) -> str:
        from pxr import PhysicsSchemaTools

        if int(path_id) == 0:
            return ""
        return str(PhysicsSchemaTools.intToSdfPath(path_id))
