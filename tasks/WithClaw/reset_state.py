from __future__ import annotations

from collections.abc import Sequence

import torch

from tasks.WithClaw.task_cfg import TargetInitialStateCfg


def validate_target_states(states: Sequence[TargetInitialStateCfg]) -> None:
    if not states:
        raise ValueError("目标离散状态不能为空。")
    names = [state.name for state in states]
    if len(set(names)) != len(names):
        raise ValueError("目标离散状态名称必须唯一。")


def select_state_ids(
    states: Sequence[TargetInitialStateCfg],
    count: int,
    *,
    device: torch.device | str,
    fixed_state_name: str | None = None,
) -> torch.Tensor:
    validate_target_states(states)
    if count < 0:
        raise ValueError("count 不能为负数。")
    if fixed_state_name is None:
        return torch.randint(len(states), (count,), dtype=torch.long, device=device)
    names = tuple(state.name for state in states)
    try:
        fixed_state_id = names.index(fixed_state_name)
    except ValueError as exc:
        raise ValueError(f"未知目标状态：{fixed_state_name!r}。") from exc
    return torch.full((count,), fixed_state_id, dtype=torch.long, device=device)


def gather_reset_values(
    states: Sequence[TargetInitialStateCfg],
    state_ids: torch.Tensor,
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_target_states(states)
    if state_ids.ndim != 1 or env_origins.shape != (state_ids.shape[0], 3):
        raise ValueError("state_ids 必须为 (N,)，env_origins 必须为 (N, 3)。")
    position_table = torch.tensor([state.pos for state in states], dtype=env_origins.dtype, device=env_origins.device)
    rotation_table = torch.tensor([state.rot for state in states], dtype=env_origins.dtype, device=env_origins.device)
    preposition_table = torch.tensor(
        [state.preposition for state in states],
        dtype=env_origins.dtype,
        device=env_origins.device,
    )
    return (
        position_table[state_ids] + env_origins,
        rotation_table[state_ids],
        preposition_table[state_ids] + env_origins,
    )


def update_selected_state_names(
    current_names: list[str] | None,
    *,
    num_envs: int,
    env_ids: torch.Tensor,
    state_ids: torch.Tensor,
    states: Sequence[TargetInitialStateCfg],
) -> list[str]:
    names = list(current_names) if isinstance(current_names, list) and len(current_names) == num_envs else [""] * num_envs
    state_names = tuple(state.name for state in states)
    for env_id, state_id in zip(env_ids.detach().cpu().tolist(), state_ids.detach().cpu().tolist()):
        names[int(env_id)] = state_names[int(state_id)]
    return names
