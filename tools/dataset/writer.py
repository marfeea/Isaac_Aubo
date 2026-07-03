from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from configs.dataset_cfg import DatasetCollectionCfg
from tools.dataset.schema import schema_document


class DatasetDependencyError(RuntimeError):
    """正式数据写入依赖缺失。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.inprogress")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _parquet_safe(value: Any) -> Any:
    """将 PyArrow 不支持的零字段 struct 规范化为 null。"""
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _parquet_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parquet_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_parquet_safe(item) for item in value]
    return value


class AuboDatasetWriter:
    """以 Parquet 和 Zarr 分片写入无损主数据。"""

    def __init__(
        self,
        cfg: DatasetCollectionCfg,
        *,
        dataset_metadata: dict[str, Any],
        task_metadata: dict[str, Any],
        sensor_metadata: list[dict[str, Any]],
        artifact_paths: list[Path],
    ) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise DatasetDependencyError(
                "正式数据写入需要 pyarrow、zarr<3 和 numcodecs；请在 Isaac Python 环境安装项目依赖。"
            ) from exc

        self._pa = pa
        self._pq = pq
        self._zarr = zarr
        self._compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        self.cfg = cfg
        self.root = cfg.dataset_root.resolve()
        self._shard_index = 0
        self._episodes_in_shard = 0
        self._frames: list[dict[str, Any]] = []
        self._transitions: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._all_episode_rows: list[dict[str, Any]] = []
        self._shard_manifest: list[dict[str, Any]] = []
        self._sensor_groups: dict[str, Any] = {}
        self._sensor_counts: dict[str, int] = {}
        self._sensor_field_names: dict[str, tuple[str, ...]] = {}

        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"数据集目录已存在且非空，不允许原地改写：{self.root}")
        self._create_layout()
        self._write_initial_metadata(dataset_metadata, task_metadata, sensor_metadata, artifact_paths)

    def _create_layout(self) -> None:
        for relative in (
            "meta",
            "data/frames",
            "data/transitions",
            "data/events",
            "sensors",
            "annotations",
            "artifacts/configs",
            "manifests",
            "exports",
        ):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def _write_initial_metadata(
        self,
        dataset_metadata: dict[str, Any],
        task_metadata: dict[str, Any],
        sensor_metadata: list[dict[str, Any]],
        artifact_paths: list[Path],
    ) -> None:
        _write_json_atomic(self.root / "meta/dataset.json", dataset_metadata)
        _write_json_atomic(self.root / "meta/schema.json", schema_document())
        _write_json_atomic(self.root / "meta/sensors.json", sensor_metadata)
        self._write_parquet_atomic(self.root / "meta/tasks.parquet", [task_metadata])
        _write_json_atomic(self.root / "artifacts/asset_hashes.json", dataset_metadata.get("asset_hashes", []))
        _write_json_atomic(
            self.root / "artifacts/teacher_checkpoints.json",
            [dataset_metadata["teacher"]] if dataset_metadata.get("teacher") else [],
        )

        copied = []
        for source in artifact_paths:
            source = Path(source).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"追溯 artifact 不存在：{source}")
            destination = self.root / "artifacts/configs" / source.name
            shutil.copy2(source, destination)
            copied.append(
                {
                    "source": str(source),
                    "stored": str(destination.relative_to(self.root)),
                    "sha256": sha256_file(source),
                }
            )
        _write_json_atomic(self.root / "artifacts/config_hashes.json", copied)

    def append_sensor_sample(
        self,
        *,
        stream_id: str,
        episode_id: str,
        timestamp_ns: int,
        frame_number: int,
        fields: dict[str, np.ndarray],
        valid: bool = True,
    ) -> str:
        """向当前临时 Zarr shard 追加一个多字段传感器样本。"""
        if not fields:
            raise ValueError(f"传感器流 {stream_id} 没有数据字段。")
        field_names = tuple(sorted(fields))
        expected_fields = self._sensor_field_names.setdefault(stream_id, field_names)
        if field_names != expected_fields:
            raise ValueError(
                f"传感器流 {stream_id} 字段集合发生变化：已有 {expected_fields}，收到 {field_names}。"
            )
        group = self._sensor_group(stream_id)
        index = self._sensor_counts.get(stream_id, 0)
        self._append_zarr_value(group, "episode_id", np.asarray(episode_id, dtype="<U96"), index)
        self._append_zarr_value(group, "timestamp_ns", np.asarray(timestamp_ns, dtype=np.int64), index)
        self._append_zarr_value(group, "frame_number", np.asarray(frame_number, dtype=np.int64), index)
        self._append_zarr_value(group, "valid", np.asarray(valid, dtype=np.bool_), index)
        for name, value in fields.items():
            self._append_zarr_value(group, name, np.asarray(value), index)
        self._sensor_counts[stream_id] = index + 1
        return f"sensors/{stream_id}/chunk-{self._shard_index:06d}.zarr#{index}"

    def _sensor_group(self, stream_id: str):
        group = self._sensor_groups.get(stream_id)
        if group is not None:
            return group
        path = self.root / "sensors" / stream_id / f"chunk-{self._shard_index:06d}.zarr.inprogress"
        path.parent.mkdir(parents=True, exist_ok=True)
        group = self._zarr.open_group(str(path), mode="w")
        group.attrs.update(
            {
                "stream_id": stream_id,
                "schema_name": self.cfg.schema_name,
                "schema_version": self.cfg.schema_version,
                "timestamp_unit": "ns",
            }
        )
        self._sensor_groups[stream_id] = group
        return group

    def _append_zarr_value(self, group, name: str, value: np.ndarray, index: int) -> None:
        if name not in group:
            if index != 0:
                raise ValueError(f"Zarr 字段 {name} 在 stream 写入中途出现，拒绝静默补零。")
            shape = (0, *value.shape)
            chunk_leading = max(1, int(self.cfg.sensor_chunk_frames))
            chunks = (chunk_leading, *value.shape)
            group.create_dataset(
                name,
                shape=shape,
                chunks=chunks,
                dtype=value.dtype,
                compressor=self._compressor,
                overwrite=False,
            )
        array = group[name]
        if tuple(array.shape[1:]) != tuple(value.shape) or array.dtype != value.dtype:
            raise ValueError(
                f"Zarr 字段 {name} 的 dtype/shape 发生变化："
                f"已有 {array.dtype}/{array.shape[1:]}，收到 {value.dtype}/{value.shape}。"
            )
        array.resize((index + 1, *value.shape))
        array[index] = value

    def append_episode(
        self,
        episode: dict[str, Any],
        frames: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> None:
        self._episodes.append(episode)
        self._frames.extend(frames)
        self._transitions.extend(transitions)
        self._events.extend(events)
        self._episodes_in_shard += 1
        if self._episodes_in_shard >= self.cfg.episodes_per_shard:
            self.commit_shard()

    def commit_shard(self) -> None:
        if not self._episodes:
            return
        shard = self._shard_index
        committed_files = []
        for directory, stem, rows in (
            ("data/frames", "chunk", self._frames),
            ("data/transitions", "chunk", self._transitions),
            ("data/events", "contact", self._events),
        ):
            if not rows:
                continue
            path = self.root / directory / f"{stem}-{shard:06d}.parquet"
            self._write_parquet_atomic(path, rows)
            committed_files.append(str(path.relative_to(self.root)).replace("\\", "/"))

        for stream_id in sorted(self._sensor_groups):
            temporary = self.root / "sensors" / stream_id / f"chunk-{shard:06d}.zarr.inprogress"
            final = temporary.with_name(f"chunk-{shard:06d}.zarr")
            os.replace(temporary, final)
            committed_files.append(str(final.relative_to(self.root)).replace("\\", "/"))

        self._all_episode_rows.extend(self._episodes)
        self._write_parquet_atomic(self.root / "meta/episodes.parquet", self._all_episode_rows)
        self._shard_manifest.append(
            {
                "shard_index": shard,
                "episode_ids": [row["episode_id"] for row in self._episodes],
                "files": committed_files,
            }
        )
        _write_json_atomic(self.root / "manifests/shards.json", self._shard_manifest)
        self._write_checksums()

        self._shard_index += 1
        self._episodes_in_shard = 0
        self._frames.clear()
        self._transitions.clear()
        self._events.clear()
        self._episodes.clear()
        self._sensor_groups.clear()
        self._sensor_counts.clear()
        self._sensor_field_names.clear()

    def _write_parquet_atomic(self, path: Path, rows: list[dict[str, Any]]) -> None:
        temporary = path.with_name(f"{path.name}.inprogress")
        table = self._pa.Table.from_pylist([_parquet_safe(row) for row in rows])
        self._pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)

    def _write_checksums(self) -> None:
        checksum_path = self.root / "manifests/checksums.sha256"
        rows = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path == checksum_path or path.name.endswith(".inprogress"):
                continue
            relative = str(path.relative_to(self.root)).replace("\\", "/")
            rows.append(f"{sha256_file(path)}  {relative}")
        temporary = checksum_path.with_name(f"{checksum_path.name}.inprogress")
        temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
        os.replace(temporary, checksum_path)

    def close(self) -> None:
        self.commit_shard()
