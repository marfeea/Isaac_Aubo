from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.utils import configclass

from tasks.WithClaw.collision_cfg import ROBOT_CONTACT_SENSOR_NAME
from tasks.WithClaw.mdp_logic import (
    axis_aligned_workspace_violation,
    illegal_contact_failure,
    tcp_dwell_reward,
    tcp_parking_reward,
    tcp_progress,
    tcp_proximity,
)
from tasks.WithClaw.runtime import max_illegal_contact_force, parking_state, tcp_distance_and_speed, tcp_kinematics
from tasks.WithClaw.task_cfg import (
    ILLEGAL_CONTACT_FORCE_THRESHOLD,
    REWARD_WEIGHTS,
    RL_WORKSPACE,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TCP_PARKING_REQUIRED_STEPS,
    TCP_PARKING_SPEED_SIGMA,
    TCP_PROXIMITY_SIGMA,
)
from tools.logic import AuboRewardFns


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


def reward_tcp_dwell(env):
    return tcp_dwell_reward(parking_state(env).dwell_steps, TCP_PARKING_REQUIRED_STEPS)


def penalty_tcp_out_of_workspace(env):
    return axis_aligned_workspace_violation(tcp_kinematics(env).root.position_b, RL_WORKSPACE).float()


def penalty_illegal_collision(env):
    force = max_illegal_contact_force(env, ROBOT_CONTACT_SENSOR_NAME, ROBOT_IGNORED_CONTACT_BODY_NAMES)
    return illegal_contact_failure(force, ILLEGAL_CONTACT_FORCE_THRESHOLD).float()


@configclass
class RewardsCfg:
    tcp_progress = RewardTerm(func=reward_tcp_progress, weight=REWARD_WEIGHTS["tcp_progress"])
    tcp_proximity = RewardTerm(func=reward_tcp_proximity, weight=REWARD_WEIGHTS["tcp_proximity"])
    tcp_parking = RewardTerm(func=reward_tcp_parking, weight=REWARD_WEIGHTS["tcp_parking"])
    tcp_dwell = RewardTerm(func=reward_tcp_dwell, weight=REWARD_WEIGHTS["tcp_dwell"])
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
