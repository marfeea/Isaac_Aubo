from __future__ import annotations

from typing import Any

SCHEMA_NAME = "AUBO-RobotTraj"
SCHEMA_VERSION = "1.0.0"


def schema_document() -> dict[str, Any]:
    """返回供数据消费者解析的机器可读 schema 摘要。"""
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "time": {"timestamp_dtype": "int64", "timestamp_unit": "ns", "delta_t_unit": "s"},
        "coordinate_convention": "right_handed",
        "quaternion_order": "wxyz",
        "default_float_dtype": "float32",
        "frame": {
            "required": ["episode_id", "frame_index", "timestamp_ns", "observation", "privileged"],
            "robot_pose_frame": "world",
        },
        "transition": {
            "required": [
                "episode_id",
                "transition_index",
                "source_frame_index",
                "target_frame_index",
                "start_timestamp_ns",
                "end_timestamp_ns",
                "delta_t_s",
                "action",
                "reward",
                "flags",
            ],
            "action_command": {
                "representation": "delta_position",
                "frame": "arm0/base",
                "unit": "meter",
                "normalized": False,
            },
            "policy_raw": {"normalized": True, "range": [-1.0, 1.0]},
            "reward_discount": "teacher_policy_gamma",
        },
        "sensor_stream": {
            "required_index_fields": ["episode_id", "timestamp_ns", "frame_number", "valid"],
            "depth": {"dtype": "float32", "unit": "meter", "requires_validity_mask": True},
            "rgb": {"dtype": "uint8", "storage": "lossless_zarr"},
        },
        "termination_flags": ["terminated", "truncated", "success", "invalid"],
    }
