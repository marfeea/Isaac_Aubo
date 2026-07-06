from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from tasks.WithClaw.reset_state import gather_reset_values, select_state_ids, update_selected_state_names
from tasks.WithClaw.task_cfg import DEFAULT_TARGET_ASSET_NAME, ROBOT_ASSET_NAME, TARGET_INITIAL_STATES, TargetInitialStateCfg
from tools.scene import AuboToolFns


def _update_tensor_cache(
    env,
    name: str,
    env_ids: torch.Tensor,
    values: torch.Tensor,
    *,
    fill_value: float | int,
) -> torch.Tensor:
    expected_shape = (env.num_envs, *values.shape[1:])
    cache = getattr(env, name, None)
    if not isinstance(cache, torch.Tensor) or cache.shape != expected_shape:
        cache = torch.full(expected_shape, fill_value, dtype=values.dtype, device=values.device)
    cache[env_ids] = values
    setattr(env, name, cache)
    return cache


def _clear_episode_cache(env, name: str, env_ids: torch.Tensor, fill_value: float | int) -> None:
    cache = getattr(env, name, None)
    if isinstance(cache, torch.Tensor) and cache.shape[0] == env.num_envs:
        cache[env_ids] = fill_value


def reset_target_to_named_state(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    states: Sequence[TargetInitialStateCfg] = TARGET_INITIAL_STATES,
    fixed_state_name: str | None = None,
) -> None:
    """设置 target 离散位姿，并原子更新任务所需的逐环境 reset 缓存。"""
    env_ids = AuboToolFns.normalize_env_ids(env, env_ids)
    state_ids = select_state_ids(
        states,
        len(env_ids),
        device=env.device,
        fixed_state_name=fixed_state_name,
    )
    env_origins = env.scene.env_origins[env_ids]
    target_position_w, target_rotation_w, preposition_w = gather_reset_values(states, state_ids, env_origins)
    AuboToolFns.set_object_pose(
        env,
        asset_cfg,
        target_position_w,
        target_rotation_w,
        env_ids=env_ids,
        relative_to_env_origins=False,
        zero_velocity=True,
    )
    target = AuboToolFns.get_asset(env, asset_cfg)
    actual_target_position_w = target.data.root_pos_w[env_ids, :3].clone()

    _update_tensor_cache(env, "selected_state_ids", env_ids, state_ids, fill_value=-1)
    env.selected_state_names = update_selected_state_names(
        getattr(env, "selected_state_names", None),
        num_envs=env.num_envs,
        env_ids=env_ids,
        state_ids=state_ids,
        states=states,
    )
    _update_tensor_cache(env, "preposition_w", env_ids, preposition_w, fill_value=float("nan"))
    _update_tensor_cache(
        env,
        "target_initial_pos_w",
        env_ids,
        actual_target_position_w,
        fill_value=float("nan"),
    )
    for name, fill_value in (
        ("_prev_tcp_preposition_dist", float("nan")),
        ("_tcp_parking_zone", False),
        ("_tcp_dwell_steps", 0),
        ("_tcp_parking_eval_step", -1),
    ):
        _clear_episode_cache(env, name, env_ids, fill_value)


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_target_pose = EventTerm(
        func=reset_target_to_named_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg(DEFAULT_TARGET_ASSET_NAME)},
    )
