from __future__ import annotations

from collections.abc import Sequence

import torch
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeTerminationRateCallback(BaseCallback):
    """按 PPO rollout 记录各终结条件在已结束 episode 中的占比。"""

    def __init__(self, termination_terms: Sequence[str]):
        super().__init__()
        self.termination_terms = tuple(str(term) for term in termination_terms)
        self._reset_rollout_stats()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is not None:
            self._collect_from_payload(infos)

        base_env = self._get_base_env()
        extras = getattr(base_env, "extras", None)
        if extras is not None:
            self._collect_from_payload(extras)
        return True

    def _on_rollout_end(self) -> None:
        for term, values in self._termination_buckets.items():
            if values:
                self.logger.record(f"custom/termination_{term}_rate", sum(values) / len(values))
        self._reset_rollout_stats()

    def _reset_rollout_stats(self) -> None:
        self._termination_buckets: dict[str, list[float]] = {
            term: [] for term in self.termination_terms
        }

    def _collect_from_payload(self, payload) -> None:
        if isinstance(payload, dict):
            self._collect_metric_keys(payload)
            for nested_key in ("episode", "log", "extras", "final_info"):
                nested = payload.get(nested_key)
                if nested is not None and nested is not payload:
                    self._collect_from_payload(nested)
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                self._collect_from_payload(item)

    def _collect_metric_keys(self, payload: dict) -> None:
        for term in self.termination_terms:
            key = f"Episode_Termination/{term}"
            if key not in payload:
                continue
            value = self._as_float(payload[key])
            if value is not None:
                self._termination_buckets[term].append(value)

    def _get_base_env(self):
        env = self.training_env
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "extras"):
                return env
            next_env = getattr(env, "env", None)
            if next_env is None:
                next_env = getattr(env, "unwrapped", None)
            if next_env is env:
                break
            env = next_env
        return self.training_env

    @staticmethod
    def _as_float(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu())
        if isinstance(value, (int, float, bool)):
            return float(value)
        if isinstance(value, (list, tuple)):
            numbers = [EpisodeTerminationRateCallback._as_float(item) for item in value]
            numbers = [item for item in numbers if item is not None]
            if numbers:
                return sum(numbers) / len(numbers)
        return None
