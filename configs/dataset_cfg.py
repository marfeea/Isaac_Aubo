from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from configs.asset import ROBOT_ASSET_NAME, ROBOT_ASSET_NAME_2
from configs.camera_cfg import CAMERA_SENSOR_SCENE_NAMES


@dataclass(frozen=True)
class CameraDatasetStreamCfg:
    """一台场景相机在主数据中的稳定命名。"""

    scene_name: str
    sensor_id: str
    calibration_id: str


def _default_camera_streams() -> tuple[CameraDatasetStreamCfg, ...]:
    return tuple(
        CameraDatasetStreamCfg(
            scene_name=scene_name,
            sensor_id=scene_name,
            calibration_id=f"{scene_name}_v1",
        )
        for scene_name in CAMERA_SENSOR_SCENE_NAMES
    )


@dataclass(frozen=True)
class DatasetCollectionCfg:
    """AUBO-RobotTraj 主数据采集配置。"""

    dataset_id: str
    output_root: Path = Path("data/datasets")
    schema_name: str = "AUBO-RobotTraj"
    schema_version: str = "1.0.0"
    task_id: str = "reach_target_v1"
    task_suite: str = "aubo_reaching"
    task_name: str = "末端到达目标"
    instruction: str = "将末端移动到指定目标位置"
    scene_id: str = "aubo_workstation_v1"
    split: str = "train"
    seed: int = 0
    max_episodes: int = 1
    episodes_per_shard: int = 8
    sensor_chunk_frames: int = 16
    simulation_hz: float = 120.0
    policy_hz: float = 4.0
    camera_hz: float = 30.0
    discount: float = 0.99
    contact_force_threshold: float = 1.0e-6
    robot_assets: tuple[tuple[str, str], ...] = (
        ("arm0", ROBOT_ASSET_NAME),
        ("arm1", ROBOT_ASSET_NAME_2),
    )
    camera_streams: tuple[CameraDatasetStreamCfg, ...] = field(default_factory=_default_camera_streams)

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id 不能为空。")
        if self.schema_version != "1.0.0":
            raise ValueError(f"当前采集器只支持 schema 1.0.0，收到 {self.schema_version}。")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes 必须为正整数。")
        if self.episodes_per_shard <= 0 or self.sensor_chunk_frames <= 0:
            raise ValueError("分片 Episode 数和传感器 chunk 帧数必须为正整数。")
        if min(self.simulation_hz, self.policy_hz, self.camera_hz) <= 0.0:
            raise ValueError("采集频率必须为正数。")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount 必须位于 [0, 1]。")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"split 必须为 train/validation/test，收到 {self.split}。")

    @property
    def dataset_root(self) -> Path:
        return Path(self.output_root) / self.dataset_id
