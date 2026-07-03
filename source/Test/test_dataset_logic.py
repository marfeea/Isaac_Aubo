from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.dataset.builder import EpisodeBuilder
from tools.dataset.validator import DatasetValidationError, validate_episode


def make_frame(timestamp_ns: int) -> dict:
    return {
        "timestamp_ns": timestamp_ns,
        "observation": {
            "deployable": {
                "robots": {
                    "arm0": {
                        "joint_position": [0.0, 0.0],
                        "joint_velocity": [0.0, 0.0],
                        "ee_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                }
            }
        },
        "privileged": {"target_pose": {"position": [0.1, 0.2, 0.3]}},
    }


def make_action() -> dict:
    return {
        "policy_raw": [0.1, 0.0, 0.0],
        "command": {
            "representation": "delta_position",
            "frame": "arm0/base",
            "unit": "meter",
            "normalization": {"normalized": False},
            "ee_delta_position": [0.001, 0.0, 0.0],
        },
        "controller_target": {},
        "executed": {},
        "diagnostics": {},
    }


def make_reward() -> dict:
    return {
        "terms": {
            "progress": {"raw_value": 0.5, "weight": 2.0, "time_scale": 0.25, "contribution": 0.25}
        },
        "total": 0.25,
        "discount": 1.0,
    }


def make_episode_rows():
    builder = EpisodeBuilder(
        episode_id="fixture_ep_00000000",
        task_id="reach_target_v1",
        scene_id="fixture",
        seed=1,
        teacher_id="fixture_teacher",
        split="train",
        start_timestamp_ns=0,
        initial_frame=make_frame(0),
    )
    builder.append_transition(
        target_frame=make_frame(250_000_000),
        action=make_action(),
        reward=make_reward(),
        flags={
            "success": True,
            "terminated": True,
            "truncated": False,
            "invalid": False,
            "termination_reason": "goal_reached",
        },
    )
    return builder.finalize()


def test_episode_builder_preserves_transition_invariant() -> None:
    episode, frames, transitions, events = make_episode_rows()
    validate_episode(episode, frames, transitions, events)
    assert len(frames) == len(transitions) + 1
    assert transitions[0]["delta_t_s"] == pytest.approx(0.25)
    assert episode["undiscounted_return"] == pytest.approx(0.25)


def test_validator_rejects_reward_total_mismatch() -> None:
    episode, frames, transitions, events = make_episode_rows()
    broken = copy.deepcopy(transitions)
    broken[0]["reward"]["total"] = 1.0
    with pytest.raises(DatasetValidationError, match="reward contribution"):
        validate_episode(episode, frames, broken, events)


def test_validator_rejects_missing_terminal_frame() -> None:
    episode, frames, transitions, events = make_episode_rows()
    with pytest.raises(DatasetValidationError, match="num_frames"):
        validate_episode(episode, frames[:-1], transitions, events)


def test_writer_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("pyarrow")
    import numpy as np
    import pyarrow.parquet as pq

    from configs.dataset_cfg import DatasetCollectionCfg
    from tools.dataset.writer import AuboDatasetWriter

    cfg = DatasetCollectionCfg(
        dataset_id="round_trip_fixture",
        output_root=tmp_path,
        camera_streams=(),
        max_episodes=1,
        episodes_per_shard=1,
    )
    writer = AuboDatasetWriter(
        cfg,
        dataset_metadata={
            "dataset_id": cfg.dataset_id,
            "schema_name": cfg.schema_name,
            "schema_version": cfg.schema_version,
            "asset_hashes": [],
            "teacher": {},
        },
        task_metadata={"task_id": cfg.task_id},
        sensor_metadata=[],
        artifact_paths=[],
    )
    reference = writer.append_sensor_sample(
        stream_id="arm0_proprio",
        episode_id="fixture_ep_00000000",
        timestamp_ns=0,
        frame_number=0,
        fields={"joint_position": np.asarray([0.0, 1.0], dtype=np.float32)},
    )
    assert reference.endswith("chunk-000000.zarr#0")
    episode, frames, transitions, events = make_episode_rows()
    writer.append_episode(episode, frames, transitions, events)
    writer.close()

    root = cfg.dataset_root
    frame_table = pq.read_table(root / "data/frames/chunk-000000.parquet")
    transition_table = pq.read_table(root / "data/transitions/chunk-000000.parquet")
    assert frame_table.num_rows == 2
    assert transition_table.num_rows == 1
    assert (root / "sensors/arm0_proprio/chunk-000000.zarr").is_dir()
    assert (root / "manifests/checksums.sha256").is_file()
