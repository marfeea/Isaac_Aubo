from __future__ import annotations

import torch


class AuboActionToolFns:
    """动作管理器相关的容错读取工具。"""

    @staticmethod
    def get_action(env) -> torch.Tensor:
        """读取当前原始 action；不存在时返回零张量。"""
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            return env.action_manager.action
        return torch.zeros((env.num_envs, 1), device=env.device)

    @staticmethod
    def get_action_rate(env, action: torch.Tensor | None = None) -> torch.Tensor:
        """读取 action delta；缺少 prev_action 时返回零张量。"""
        action = AuboActionToolFns.get_action(env) if action is None else action
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "prev_action"):
            return action - env.action_manager.prev_action
        return torch.zeros_like(action)

    @staticmethod
    def norm(env) -> torch.Tensor:
        """返回当前 action 的 L2 norm。"""
        return torch.norm(AuboActionToolFns.get_action(env), dim=-1)

    @staticmethod
    def rate_norm(env) -> torch.Tensor:
        """返回 action delta 的 L2 norm。"""
        action = AuboActionToolFns.get_action(env)
        return torch.norm(AuboActionToolFns.get_action_rate(env, action), dim=-1)


class AuboBufferToolFns:
    """环境临时缓存读写工具，用于奖励/终止项跨 step 记录状态。"""

    @staticmethod
    def get_or_create(env, name: str, value: torch.Tensor) -> torch.Tensor:
        """读取 env.name；不存在或环境数量不一致时用 value.clone() 初始化。"""
        current = getattr(env, name, None)
        if not isinstance(current, torch.Tensor) or current.shape[0] != env.num_envs:
            setattr(env, name, value.clone())
        return getattr(env, name)

    @staticmethod
    def sync_just_reset(env, buffer: torch.Tensor, value: torch.Tensor) -> None:
        """reset 首步同步缓存，避免 progress 奖励产生虚假增量。"""
        if hasattr(env, "episode_length_buf"):
            just_reset = env.episode_length_buf == 0
            buffer[just_reset] = value[just_reset]
