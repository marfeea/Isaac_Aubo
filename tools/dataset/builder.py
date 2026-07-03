from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class EpisodeBuilder:
    """在内存中组装一个低维 Episode；大容量传感器由 writer 流式写入。"""

    episode_id: str
    task_id: str
    scene_id: str
    seed: int
    teacher_id: str
    split: str
    start_timestamp_ns: int
    initial_frame: dict[str, Any]
    frames: list[dict[str, Any]] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        frame = dict(self.initial_frame)
        frame["episode_id"] = self.episode_id
        frame["frame_index"] = 0
        self.frames.append(frame)

    @property
    def current_frame(self) -> dict[str, Any]:
        return self.frames[-1]

    def append_transition(
        self,
        *,
        target_frame: dict[str, Any],
        action: dict[str, Any],
        reward: dict[str, Any],
        flags: dict[str, Any],
    ) -> None:
        source = self.frames[-1]
        target_index = len(self.frames)
        target = dict(target_frame)
        target["episode_id"] = self.episode_id
        target["frame_index"] = target_index
        start_ns = int(source["timestamp_ns"])
        end_ns = int(target["timestamp_ns"])
        if end_ns <= start_ns:
            raise ValueError(f"Transition 时间戳必须递增，收到 {start_ns} -> {end_ns}。")

        transition_index = len(self.transitions)
        self.transitions.append(
            {
                "episode_id": self.episode_id,
                "transition_index": transition_index,
                "source_frame_index": transition_index,
                "target_frame_index": target_index,
                "start_timestamp_ns": start_ns,
                "end_timestamp_ns": end_ns,
                "delta_t_s": (end_ns - start_ns) / 1.0e9,
                "action": action,
                "reward": reward,
                "flags": flags,
            }
        )
        self.frames.append(target)

    def append_event(self, event: dict[str, Any]) -> None:
        row = dict(event)
        row["episode_id"] = self.episode_id
        self.events.append(row)

    def finalize(self, *, invalid: bool = False) -> tuple[dict[str, Any], list[dict], list[dict], list[dict]]:
        if not self.transitions:
            raise ValueError("空 Episode 不进入主数据集。")
        last_flags = self.transitions[-1]["flags"]
        reward_terms: dict[str, float] = {}
        undiscounted_return = 0.0
        discounted_return = 0.0
        discount_product = 1.0
        for transition in self.transitions:
            reward = transition["reward"]
            total = float(reward["total"])
            undiscounted_return += total
            discounted_return += discount_product * total
            discount_product *= float(reward.get("discount", 1.0))
            for name, term in reward["terms"].items():
                reward_terms[name] = reward_terms.get(name, 0.0) + float(term["contribution"])

        status = "invalid" if invalid else ("success" if last_flags["success"] else "failure")
        episode = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "scene_id": self.scene_id,
            "initial_state_id": _stable_hash(self.frames[0]["privileged"]),
            "seed": int(self.seed),
            "teacher_id": self.teacher_id,
            "num_transitions": len(self.transitions),
            "start_timestamp_ns": int(self.start_timestamp_ns),
            "end_timestamp_ns": int(self.frames[-1]["timestamp_ns"]),
            "invalid": bool(invalid),
            "split": self.split,
            "status": status,
            "termination_reason": last_flags.get("termination_reason"),
            "undiscounted_return": undiscounted_return,
            "discounted_return": discounted_return,
            "reward_term_returns": reward_terms,
        }
        return episode, self.frames, self.transitions, self.events
