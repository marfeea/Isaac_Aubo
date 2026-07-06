from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tasks.WithClaw.collision_cfg import ROBOT_CONTACT_SENSOR_NAME
from tasks.WithClaw.mdp_logic import (
    axis_aligned_workspace_violation,
    illegal_contact_failure,
    target_displacement_failure,
    target_speed_failure,
)
from tasks.WithClaw.runtime import max_illegal_contact_force, parking_state, tcp_kinematics
from tasks.WithClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    ILLEGAL_CONTACT_FORCE_THRESHOLD,
    RL_WORKSPACE,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TARGET_MAX_DISPLACEMENT,
    TARGET_MAX_LINEAR_SPEED,
    TCP_PARKING_REQUIRED_STEPS,
)
from tools.scene import AuboToolFns


def parking_success(env):
    return parking_state(env).dwell_steps >= TCP_PARKING_REQUIRED_STEPS


def illegal_collision(env):
    force = max_illegal_contact_force(env, ROBOT_CONTACT_SENSOR_NAME, ROBOT_IGNORED_CONTACT_BODY_NAMES)
    return illegal_contact_failure(force, ILLEGAL_CONTACT_FORCE_THRESHOLD)


def tcp_out_of_workspace(env):
    return axis_aligned_workspace_violation(tcp_kinematics(env).root.position_b, RL_WORKSPACE)


def target_displaced(env):
    initial = getattr(env, "target_initial_pos_w", None)
    if initial is None:
        raise RuntimeError("target_initial_pos_w 尚未由 WithClaw reset 事件初始化。")
    target = AuboToolFns.get_asset(env, DEFAULT_TARGET_ASSET_NAME)
    return target_displacement_failure(target.data.root_pos_w[:, :3], initial, TARGET_MAX_DISPLACEMENT)


def target_too_fast(env):
    target = AuboToolFns.get_asset(env, DEFAULT_TARGET_ASSET_NAME)
    return target_speed_failure(target.data.root_lin_vel_w[:, :3], TARGET_MAX_LINEAR_SPEED)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    parking_success = DoneTerm(func=parking_success)
    illegal_collision = DoneTerm(func=illegal_collision)
    tcp_out_of_workspace = DoneTerm(func=tcp_out_of_workspace)
    target_displaced = DoneTerm(func=target_displaced)
    target_too_fast = DoneTerm(func=target_too_fast)
