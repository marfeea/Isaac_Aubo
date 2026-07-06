from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import SceneEntityCfg

from tools.scene import AuboToolFns


def reset_asset_to_discrete_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    state_names: Sequence[str],
    positions: Sequence[tuple[float, float, float]],
    orientations: Sequence[tuple[float, float, float, float]],
    fixed_state_name: str | None = None,
    state_index_buffer_name: str = "target_initial_state_ids",
) -> None:
    """从具名离散状态中选择资产位姿，并清零刚体速度。"""
    if not state_names or len(state_names) != len(positions) or len(state_names) != len(orientations):
        raise ValueError("Discrete pose names, positions, and orientations must have the same non-zero length.")
    if len(set(state_names)) != len(state_names):
        raise ValueError("Discrete pose state names must be unique.")

    env_ids = AuboToolFns.normalize_env_ids(env, env_ids)
    if fixed_state_name is None:
        state_ids = torch.randint(len(state_names), (len(env_ids),), device=env.device)
    else:
        try:
            fixed_state_id = tuple(state_names).index(fixed_state_name)
        except ValueError as exc:
            raise ValueError(f"Unknown discrete pose state name: {fixed_state_name!r}.") from exc
        state_ids = torch.full((len(env_ids),), fixed_state_id, dtype=torch.long, device=env.device)

    position_table = torch.tensor(positions, dtype=torch.float32, device=env.device)
    orientation_table = torch.tensor(orientations, dtype=torch.float32, device=env.device)
    AuboToolFns.set_object_pose(
        env,
        asset_cfg,
        position_table[state_ids],
        orientation_table[state_ids],
        env_ids=env_ids,
        relative_to_env_origins=True,
        zero_velocity=True,
    )

    state_id_buffer = getattr(env, state_index_buffer_name, None)
    if not isinstance(state_id_buffer, torch.Tensor) or state_id_buffer.shape != (env.num_envs,):
        state_id_buffer = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    state_id_buffer[env_ids] = state_ids
    setattr(env, state_index_buffer_name, state_id_buffer)
    state_names_buffer_name = f"{state_index_buffer_name.removesuffix('_ids')}_names"
    setattr(env, state_names_buffer_name, tuple(state_names))
