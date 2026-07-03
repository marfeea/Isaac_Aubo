from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from isaaclab.managers import RecorderTermCfg
from isaaclab.managers.recorder_manager import (
    DatasetExportMode,
    RecorderManagerBaseCfg,
    RecorderTerm,
)
from isaaclab.utils import configclass

from configs.collision_cfg import ROBOT_CONTACT_SENSOR_NAME
from configs.dataset_cfg import DatasetCollectionCfg
from tools.dataset.builder import EpisodeBuilder
from tools.dataset.extractors import (
    complete_action_after_step,
    extract_action_pre_step,
    extract_camera_sample,
    extract_contact_events,
    extract_flags,
    extract_frame,
    extract_proprio_sample,
    extract_reward,
)
from tools.dataset.validator import validate_episode
from tools.dataset.writer import AuboDatasetWriter


class AuboDatasetRecorderTerm(RecorderTerm):
    """在 IsaacLab 自动 reset 之前完成 AUBO-RobotTraj 状态转移。"""

    cfg: AuboDatasetRecorderTermCfg

    def __init__(self, cfg: AuboDatasetRecorderTermCfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("首版正式采集器要求 num_envs=1，以保证传感器分片和 Episode 边界严格一致。")
        if cfg.collection is None:
            raise ValueError("RecorderTerm 缺少 DatasetCollectionCfg。")

        self.collection = cfg.collection
        self.target_asset_name = cfg.target_asset_name
        self.reward_config_hash = cfg.reward_config_hash
        self.teacher_id = cfg.teacher_id
        self._writer = AuboDatasetWriter(
            self.collection,
            dataset_metadata=copy.deepcopy(cfg.dataset_metadata),
            task_metadata=copy.deepcopy(cfg.task_metadata),
            sensor_metadata=self._sensor_metadata(),
            artifact_paths=[Path(path) for path in cfg.artifact_paths],
        )
        self._episode: EpisodeBuilder | None = None
        self._pending_action: dict[str, Any] | None = None
        self._camera_refs: dict[str, dict[str, str | None]] = {}
        self._last_stream_timestamp: dict[str, int] = {}
        self._stream_frame_numbers: dict[str, int] = {}
        self._last_contact_timestamp_ns: int | None = None
        self.completed_episode_count = 0
        self._closed = False

    def record_pre_reset(self, env_ids):
        if self._episode is not None and self._episode.transitions:
            self._finalize_episode(invalid=False)
        return None, None

    def record_post_reset(self, env_ids):
        if self.completed_episode_count >= self.collection.max_episodes:
            return None, None
        timestamp_ns = self._timestamp_ns()
        self._start_episode(timestamp_ns)
        self._sample_streams(timestamp_ns=timestamp_ns, sim_step=self._sim_step(), capture_camera=True)
        self._episode = EpisodeBuilder(
            episode_id=self._episode_id(self.completed_episode_count),
            task_id=self.collection.task_id,
            scene_id=self.collection.scene_id,
            seed=self.collection.seed,
            teacher_id=self.teacher_id,
            split=self.collection.split,
            start_timestamp_ns=timestamp_ns,
            initial_frame=self._frame(timestamp_ns),
        )
        return None, None

    def record_pre_step(self):
        if self._episode is not None:
            self._pending_action = extract_action_pre_step(self._env)
        return None, None

    def record_post_physics_decimation_step(self):
        if self._episode is None:
            return None, None
        # Recorder 回调发生在 scene.update 之前，当前 buffer 对应前一物理步。
        state_step = max(self._sim_step() - 1, 0)
        timestamp_ns = self._timestamp_ns(state_step)
        capture_camera = state_step % int(self._env.cfg.sim.render_interval) == 0
        self._sample_streams(timestamp_ns=timestamp_ns, sim_step=state_step, capture_camera=capture_camera)
        return None, None

    def record_post_step(self):
        if self._episode is None or self._pending_action is None:
            return None, None
        sim_step = self._sim_step()
        timestamp_ns = self._timestamp_ns(sim_step)
        capture_camera = sim_step % int(self._env.cfg.sim.render_interval) == 0
        self._sample_streams(timestamp_ns=timestamp_ns, sim_step=sim_step, capture_camera=capture_camera)

        source_joint = self._episode.current_frame["observation"]["deployable"]["robots"]["arm0"]["joint_position"]
        target_frame = self._frame(timestamp_ns)
        target_joint = target_frame["observation"]["deployable"]["robots"]["arm0"]["joint_position"]
        action = complete_action_after_step(self._env, self._pending_action)
        action["executed"]["joint_delta"] = [
            float(target) - float(source) for source, target in zip(source_joint, target_joint)
        ]
        self._episode.append_transition(
            target_frame=target_frame,
            action=action,
            reward=extract_reward(self._env, self.reward_config_hash, self.collection.discount),
            flags=extract_flags(self._env),
        )
        self._pending_action = None
        return None, None

    def _start_episode(self, timestamp_ns: int) -> None:
        del timestamp_ns
        self._episode = None
        self._pending_action = None
        self._camera_refs = {
            stream.sensor_id: {"rgb": None, "depth": None} for stream in self.collection.camera_streams
        }
        self._last_stream_timestamp.clear()
        self._stream_frame_numbers.clear()
        self._last_contact_timestamp_ns = None

    def _sample_streams(self, *, timestamp_ns: int, sim_step: int, capture_camera: bool) -> None:
        if self.completed_episode_count >= self.collection.max_episodes:
            return
        episode_id = self._episode_id(self.completed_episode_count)
        for arm_name, asset_name in self.collection.robot_assets:
            stream_id = f"{arm_name}_proprio"
            self._append_sample_once(
                stream_id=stream_id,
                episode_id=episode_id,
                timestamp_ns=timestamp_ns,
                fields=extract_proprio_sample(self._env, asset_name),
            )

        if capture_camera:
            for stream in self.collection.camera_streams:
                camera = self._env.scene[stream.scene_name]
                for output_name, ref_name, suffix in (
                    ("rgb", "rgb", "rgb"),
                    ("distance_to_image_plane", "depth", "depth"),
                ):
                    stream_id = f"{stream.sensor_id}_{suffix}"
                    reference = self._append_sample_once(
                        stream_id=stream_id,
                        episode_id=episode_id,
                        timestamp_ns=timestamp_ns,
                        fields=extract_camera_sample(camera, output_name),
                    )
                    if reference is not None:
                        self._camera_refs[stream.sensor_id][ref_name] = reference

        if self._episode is not None and (
            self._last_contact_timestamp_ns is None or timestamp_ns > self._last_contact_timestamp_ns
        ):
            for event in extract_contact_events(
                self._env,
                sensor_name=ROBOT_CONTACT_SENSOR_NAME,
                timestamp_ns=timestamp_ns,
                force_threshold=self.collection.contact_force_threshold,
            ):
                self._episode.append_event(event)
            self._last_contact_timestamp_ns = timestamp_ns

    def _append_sample_once(
        self,
        *,
        stream_id: str,
        episode_id: str,
        timestamp_ns: int,
        fields,
    ) -> str | None:
        previous = self._last_stream_timestamp.get(stream_id)
        if previous is not None and timestamp_ns <= previous:
            return None
        frame_number = self._stream_frame_numbers.get(stream_id, 0)
        reference = self._writer.append_sensor_sample(
            stream_id=stream_id,
            episode_id=episode_id,
            timestamp_ns=timestamp_ns,
            frame_number=frame_number,
            fields=fields,
        )
        self._last_stream_timestamp[stream_id] = timestamp_ns
        self._stream_frame_numbers[stream_id] = frame_number + 1
        return reference

    def _frame(self, timestamp_ns: int) -> dict[str, Any]:
        return extract_frame(
            self._env,
            self.collection,
            timestamp_ns=timestamp_ns,
            target_asset_name=self.target_asset_name,
            camera_refs=self._camera_refs,
        )

    def _finalize_episode(self, *, invalid: bool) -> None:
        if self._episode is None:
            return
        if invalid:
            flags = self._episode.transitions[-1]["flags"]
            flags["invalid"] = True
            flags["termination_reason"] = flags.get("termination_reason") or "human_abort"
        episode, frames, transitions, events = self._episode.finalize(invalid=invalid)
        episode["reset_sequence_index"] = self.completed_episode_count
        validate_episode(episode, frames, transitions, events)
        self._writer.append_episode(episode, frames, transitions, events)
        self.completed_episode_count += 1
        self._episode = None
        self._pending_action = None

    def _sensor_metadata(self) -> list[dict[str, Any]]:
        metadata = []
        for arm_name, asset_name in self.collection.robot_assets:
            robot = self._env.scene[asset_name]
            available_fields = ["joint_position", "joint_velocity"]
            if any(
                getattr(robot.data, name, None) is not None
                for name in ("applied_torque", "computed_torque", "joint_effort")
            ):
                available_fields.append("joint_effort")
            metadata.append(
                {
                    "stream_id": f"{arm_name}_proprio",
                    "sensor_id": f"{arm_name}/joint_state",
                    "modality": "proprioception",
                    "fields": available_fields,
                    "units": {"joint_position": "rad", "joint_velocity": "rad/s", "joint_effort": "N*m"},
                    "joint_names": list(getattr(robot, "joint_names", [])),
                    "nominal_hz": self.collection.simulation_hz,
                    "timestamp_unit": "ns",
                }
            )
        for stream in self.collection.camera_streams:
            camera_cfg = getattr(self._env.cfg.scene, stream.scene_name)
            camera = self._env.scene[stream.scene_name]
            intrinsics = getattr(camera.data, "intrinsic_matrices", None)
            if intrinsics is not None:
                if hasattr(intrinsics, "detach"):
                    intrinsics = intrinsics.detach().cpu()
                if getattr(intrinsics, "ndim", 0) >= 3 and intrinsics.shape[0] == self._env.num_envs:
                    intrinsics = intrinsics[0]
                intrinsics = intrinsics.tolist()
            camera_position = getattr(camera.data, "pos_w", None)
            camera_orientation = getattr(camera.data, "quat_w_world", None)
            if camera_position is not None:
                if hasattr(camera_position, "detach"):
                    camera_position = camera_position.detach().cpu()
                camera_position = camera_position[0] if getattr(camera_position, "ndim", 0) > 1 else camera_position
                camera_position = camera_position.tolist()
            if camera_orientation is not None:
                if hasattr(camera_orientation, "detach"):
                    camera_orientation = camera_orientation.detach().cpu()
                if getattr(camera_orientation, "ndim", 0) > 1:
                    camera_orientation = camera_orientation[0]
                camera_orientation = camera_orientation.tolist()
            for output_name, modality, dtype, unit in (
                ("rgb", "rgb", "uint8", None),
                ("distance_to_image_plane", "depth", "float32", "meter"),
            ):
                output = camera.data.output.get(output_name)
                shape = list(output.shape) if output is not None else [int(camera_cfg.height), int(camera_cfg.width)]
                if shape and shape[0] == self._env.num_envs:
                    shape = shape[1:]
                metadata.append(
                    {
                        "stream_id": f"{stream.sensor_id}_{modality}",
                        "sensor_id": stream.sensor_id,
                        "modality": modality,
                        "dtype": dtype,
                        "shape": shape,
                        "unit": unit,
                        "calibration_id": stream.calibration_id,
                        "intrinsic_matrix": intrinsics,
                        "world_T_camera": {
                            "position": camera_position,
                            "orientation_wxyz": camera_orientation,
                        },
                        "nominal_hz": self.collection.camera_hz,
                        "timestamp_unit": "ns",
                        "sampling_phase": "读取最近一次完成 render 和 scene.update 的原始输出",
                        "channel_order": (
                            "RGBA"
                            if modality == "rgb" and shape and shape[-1] == 4
                            else "RGB"
                            if modality == "rgb"
                            else None
                        ),
                        "validity_mask_field": "validity_mask" if modality == "depth" else None,
                    }
                )
        return metadata

    def _episode_id(self, index: int) -> str:
        return f"{self.collection.dataset_id}_ep_{index:08d}"

    def _sim_step(self) -> int:
        return int(getattr(self._env, "_sim_step_counter", 0))

    def _timestamp_ns(self, sim_step: int | None = None) -> int:
        step = self._sim_step() if sim_step is None else int(sim_step)
        return int(round(step * float(self._env.physics_dt) * 1.0e9))

    def close(self, file_path: str = "") -> None:
        del file_path
        if self._closed:
            return
        if self._episode is not None and self._episode.transitions:
            self._finalize_episode(invalid=True)
        self._writer.close()
        self._closed = True


@configclass
class AuboDatasetRecorderTermCfg(RecorderTermCfg):
    """正式轨迹 RecorderTerm 配置。"""

    class_type: type[RecorderTerm] = AuboDatasetRecorderTerm
    collection: DatasetCollectionCfg | None = None
    target_asset_name: str = ""
    reward_config_hash: str = ""
    teacher_id: str = ""
    dataset_metadata: dict = {}
    task_metadata: dict = {}
    artifact_paths: tuple[str, ...] = ()


@configclass
class AuboDatasetRecorderManagerCfg(RecorderManagerBaseCfg):
    """关闭 IsaacLab 默认 HDF5 导出，仅启用规范写入器。"""

    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False
    trajectory: AuboDatasetRecorderTermCfg = AuboDatasetRecorderTermCfg()


def make_dataset_recorder_cfg(
    collection: DatasetCollectionCfg,
    *,
    target_asset_name: str,
    reward_config_hash: str,
    teacher_id: str,
    dataset_metadata: dict[str, Any],
    task_metadata: dict[str, Any],
    artifact_paths: list[Path],
) -> AuboDatasetRecorderManagerCfg:
    cfg = AuboDatasetRecorderManagerCfg()
    cfg.trajectory.collection = collection
    cfg.trajectory.target_asset_name = target_asset_name
    cfg.trajectory.reward_config_hash = reward_config_hash
    cfg.trajectory.teacher_id = teacher_id
    cfg.trajectory.dataset_metadata = dataset_metadata
    cfg.trajectory.task_metadata = task_metadata
    cfg.trajectory.artifact_paths = tuple(str(path) for path in artifact_paths)
    return cfg


def get_dataset_recorder_term(env) -> AuboDatasetRecorderTerm:
    """从 RecorderManager 解析正式采集项。"""
    term = env.recorder_manager._terms.get("trajectory")
    if not isinstance(term, AuboDatasetRecorderTerm):
        raise RuntimeError("AUBO 数据采集 RecorderTerm 未正确加载。")
    return term
