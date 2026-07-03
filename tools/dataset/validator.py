from __future__ import annotations

import math
from typing import Any


class DatasetValidationError(ValueError):
    """主数据不满足 AUBO-RobotTraj 不变量。"""


def _iter_numbers(value: Any, path: str = ""):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int | float):
        yield path, float(value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_numbers(item, f"{path}.{key}" if path else str(key))
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _iter_numbers(item, f"{path}[{index}]")


def _validate_finite(value: Any, *, allow_paths: tuple[str, ...] = ()) -> None:
    for path, number in _iter_numbers(value):
        if any(path.startswith(prefix) for prefix in allow_paths):
            continue
        if not math.isfinite(number):
            raise DatasetValidationError(f"字段 {path} 包含 NaN/Inf。")


def _validate_quaternions(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = f"{path}.{key}" if path else str(key)
            if "orientation_wxyz" in key and item is not None:
                if len(item) != 4:
                    raise DatasetValidationError(f"四元数 {item_path} 长度不是 4。")
                norm = math.sqrt(sum(float(component) ** 2 for component in item))
                if abs(norm - 1.0) > 1.0e-3:
                    raise DatasetValidationError(f"四元数 {item_path} 未归一化，norm={norm}。")
            _validate_quaternions(item, item_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_quaternions(item, f"{path}[{index}]")


def validate_episode(
    episode: dict[str, Any],
    frames: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
) -> None:
    """校验单个 Episode 的结构、时间、数值和终止语义。"""
    required_episode = {
        "episode_id",
        "task_id",
        "scene_id",
        "initial_state_id",
        "seed",
        "teacher_id",
        "num_transitions",
        "start_timestamp_ns",
        "end_timestamp_ns",
        "invalid",
        "split",
    }
    missing = required_episode - episode.keys()
    if missing:
        raise DatasetValidationError(f"Episode 缺少字段：{sorted(missing)}。")
    if len(frames) != len(transitions) + 1:
        raise DatasetValidationError("必须满足 num_frames == num_transitions + 1。")
    if int(episode["num_transitions"]) != len(transitions):
        raise DatasetValidationError("Episode.num_transitions 与 Transition 数量不一致。")

    timestamps = []
    for index, frame in enumerate(frames):
        if frame.get("episode_id") != episode["episode_id"] or frame.get("frame_index") != index:
            raise DatasetValidationError(f"Frame[{index}] 的 ID 或索引不一致。")
        if not {"timestamp_ns", "observation", "privileged"} <= frame.keys():
            raise DatasetValidationError(f"Frame[{index}] 缺少必填字段。")
        timestamps.append(int(frame["timestamp_ns"]))
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise DatasetValidationError("Frame 时间戳不是严格单调递增。")

    for index, transition in enumerate(transitions):
        if transition.get("transition_index") != index:
            raise DatasetValidationError(f"Transition[{index}] 索引不连续。")
        if transition.get("source_frame_index") != index or transition.get("target_frame_index") != index + 1:
            raise DatasetValidationError(f"Transition[{index}] Frame 引用错误。")
        expected_dt = (timestamps[index + 1] - timestamps[index]) / 1.0e9
        if expected_dt <= 0.0 or abs(float(transition["delta_t_s"]) - expected_dt) > 1.0e-9:
            raise DatasetValidationError(f"Transition[{index}] delta_t 与时间戳不一致。")
        action = transition.get("action", {})
        command = action.get("command", {})
        if not {"representation", "frame", "unit", "normalization"} <= command.keys():
            raise DatasetValidationError(f"Transition[{index}] 动作语义不完整。")
        controller_target = action.get("controller_target", {})
        if controller_target and not {
            "ee_representation",
            "ee_frame",
            "ee_position_unit",
            "ee_orientation",
            "joint_representation",
            "joint_unit",
        } <= controller_target.keys():
            raise DatasetValidationError(f"Transition[{index}] 控制器目标语义不完整。")
        executed = action.get("executed", {})
        if executed and not {
            "ee_representation",
            "ee_frame",
            "ee_unit",
            "joint_representation",
            "joint_unit",
        } <= executed.keys():
            raise DatasetValidationError(f"Transition[{index}] 执行动作语义不完整。")
        terms = transition.get("reward", {}).get("terms", {})
        contribution_sum = sum(float(term["contribution"]) for term in terms.values())
        total = float(transition["reward"]["total"])
        if abs(contribution_sum - total) > 1.0e-5:
            raise DatasetValidationError(f"Transition[{index}] reward contribution 求和不等于 total。")
        flags = transition.get("flags", {})
        if not {"terminated", "truncated", "success", "invalid"} <= flags.keys():
            raise DatasetValidationError(f"Transition[{index}] 终止标记不完整。")

    _validate_finite({"episode": episode, "frames": frames, "transitions": transitions, "events": events or []})
    _validate_quaternions(frames)
