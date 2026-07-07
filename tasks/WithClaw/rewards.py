from __future__ import annotations

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.utils import configclass

from tasks.WithClaw.collision_cfg import ROBOT_CONTACT_SENSOR_NAME
from tasks.WithClaw.mdp_logic import (
    axis_aligned_workspace_violation,
    illegal_contact_failure,
    target_displacement_failure,
    target_speed_failure,
    tcp_parking_reward,
    tcp_progress,
    tcp_proximity,
)
from tasks.WithClaw.orientation import latched_orientation_progress
from tasks.WithClaw.runtime import (
    max_illegal_contact_force,
    parking_state,
    tcp_distance_and_speed,
    tcp_kinematics,
    tool_orientation_alignment,
)
from tasks.WithClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    ILLEGAL_CONTACT_FORCE_THRESHOLD,
    REWARD_WEIGHTS,
    RL_WORKSPACE,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TCP_PARKING_REQUIRED_STEPS,
    TCP_PARKING_SPEED_SIGMA,
    TCP_PROXIMITY_SIGMA,
    TARGET_MAX_DISPLACEMENT,
    TARGET_MAX_LINEAR_SPEED,
    TOOL_ORIENTATION_REWARD_START_DISTANCE,
)
from tools.logic import AuboRewardFns
from tools.scene import AuboToolFns


def reward_tcp_progress(env):
    distance, _ = tcp_distance_and_speed(env)
    previous = getattr(env, "_prev_tcp_preposition_dist", None)
    if previous is None or previous.shape != distance.shape:
        previous = distance.new_full(distance.shape, float("nan"))
    reward = tcp_progress(previous, distance)
    env._prev_tcp_preposition_dist = distance.clone()
    return reward


def reward_tcp_proximity(env):
    distance, _ = tcp_distance_and_speed(env)
    return tcp_proximity(distance, TCP_PROXIMITY_SIGMA)


def reward_tcp_parking(env):
    state = parking_state(env)
    _, speed = tcp_distance_and_speed(env)
    return tcp_parking_reward(state.in_zone, speed, TCP_PARKING_SPEED_SIGMA)


def reward_tool_axis_progress(env):
    distance, _ = tcp_distance_and_speed(env)
    score = tool_orientation_alignment(env).score
    previous_score = getattr(env, "_prev_tool_orientation_score", None)
    if not isinstance(previous_score, torch.Tensor) or previous_score.shape != score.shape:
        previous_score = score.new_full(score.shape, float("nan"))
    previous_active = getattr(env, "_tool_orientation_reward_active", None)
    if not isinstance(previous_active, torch.Tensor) or previous_active.shape != score.shape:
        previous_active = torch.zeros_like(score, dtype=torch.bool)

    state = latched_orientation_progress(
        distance,
        score,
        previous_score,
        previous_active,
        TOOL_ORIENTATION_REWARD_START_DISTANCE,
    )
    env._tool_orientation_reward_active = state.active
    env._prev_tool_orientation_score = score.clone()
    return state.progress


def reward_parking_success(env):
    return (parking_state(env).dwell_steps >= TCP_PARKING_REQUIRED_STEPS).float()


def penalty_tcp_out_of_workspace(env):
    return axis_aligned_workspace_violation(tcp_kinematics(env).root.position_b, RL_WORKSPACE).float()


def penalty_illegal_collision(env):
    force = max_illegal_contact_force(env, ROBOT_CONTACT_SENSOR_NAME, ROBOT_IGNORED_CONTACT_BODY_NAMES)
    return illegal_contact_failure(force, ILLEGAL_CONTACT_FORCE_THRESHOLD).float()


def penalty_target_displaced(env):
    initial = getattr(env, "target_initial_pos_w", None)
    if not isinstance(initial, torch.Tensor):
        raise RuntimeError("target_initial_pos_w 尚未由 WithClaw reset 事件初始化。")
    target = AuboToolFns.get_asset(env, DEFAULT_TARGET_ASSET_NAME)
    return target_displacement_failure(
        target.data.root_pos_w[:, :3],
        initial,
        TARGET_MAX_DISPLACEMENT,
    ).float()


def penalty_target_too_fast(env):
    target = AuboToolFns.get_asset(env, DEFAULT_TARGET_ASSET_NAME)
    return target_speed_failure(target.data.root_lin_vel_w[:, :3], TARGET_MAX_LINEAR_SPEED).float()


def penalty_time_out(env):
    return mdp.time_out(env).float()


@configclass
class RewardsCfg:
    tcp_progress = RewardTerm(func=reward_tcp_progress, weight=REWARD_WEIGHTS["tcp_progress"])
    tcp_proximity = RewardTerm(func=reward_tcp_proximity, weight=REWARD_WEIGHTS["tcp_proximity"])
    tcp_parking = RewardTerm(func=reward_tcp_parking, weight=REWARD_WEIGHTS["tcp_parking"])
    tool_axis_progress = RewardTerm(func=reward_tool_axis_progress, weight=REWARD_WEIGHTS["tool_axis_progress"])
    parking_success = RewardTerm(func=reward_parking_success, weight=REWARD_WEIGHTS["parking_success"])
    action_l2 = RewardTerm(func=AuboRewardFns.penalty_action_l2, weight=REWARD_WEIGHTS["action_l2"])
    action_rate_l2 = RewardTerm(
        func=AuboRewardFns.penalty_action_rate_l2,
        weight=REWARD_WEIGHTS["action_rate_l2"],
    )
    step_penalty = RewardTerm(func=AuboRewardFns.penalty_step, weight=REWARD_WEIGHTS["step_penalty"])
    out_of_workspace_penalty = RewardTerm(
        func=penalty_tcp_out_of_workspace,
        weight=REWARD_WEIGHTS["out_of_workspace_penalty"],
    )
    illegal_collision_penalty = RewardTerm(
        func=penalty_illegal_collision,
        weight=REWARD_WEIGHTS["illegal_collision_penalty"],
    )
    target_displaced_penalty = RewardTerm(
        func=penalty_target_displaced,
        weight=REWARD_WEIGHTS["target_displaced_penalty"],
    )
    target_too_fast_penalty = RewardTerm(
        func=penalty_target_too_fast,
        weight=REWARD_WEIGHTS["target_too_fast_penalty"],
    )
    time_out_penalty = RewardTerm(func=penalty_time_out, weight=REWARD_WEIGHTS["time_out_penalty"])
