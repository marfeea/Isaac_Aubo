from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.utils import configclass

from tasks.WithoutClaw.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TARGET_CONTACT_SENSOR_NAME,
)
from tasks.WithoutClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    EE_BODY_NAME,
    RL_MAX_EPISODE_STEPS,
    RL_WORKSPACE,
    ROBOT_ASSET_NAME,
)
from tools.logic import AuboRewardFns


@configclass
class RewardsCfg:
    """旧版 Flange-to-target 奖励配置。"""

    ee_progress = RewardTerm(
        func=AuboRewardFns.reward_ee_progress,
        weight=80.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
        },
    )
    ee_distance_exp = RewardTerm(
        func=AuboRewardFns.reward_ee_distance_exp,
        weight=1.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
            "std": 0.20,
        },
    )
    success = RewardTerm(
        func=AuboRewardFns.reward_success,
        weight=25.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
            "threshold": 0.15,
            "progress_ref": 0.015,
            "action_norm_max": 0.75,
            "action_norm_std": 0.50,
            "action_rate_std": 0.75,
            "w_progress": 0.45,
            "w_action_mag": 0.30,
            "w_action_smooth": 0.25,
            "min_quality_score": 0.35,
        },
    )
    action_far_near = RewardTerm(
        func=AuboRewardFns.reward_action_far_near,
        weight=1.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
            "far_eps": 0.50,
            "close_eps": 0.20,
            "w_far_move": 0.10,
            "w_near_ineff": 0.60,
            "delta_min_far": 0.020,
            "delta_min_near": 0.008,
            "near_action_norm_max": 0.45,
        },
    )
    step_penalty = RewardTerm(func=AuboRewardFns.penalty_step, weight=-0.25, params={})
    action_l2 = RewardTerm(func=AuboRewardFns.penalty_action_l2, weight=-0.025, params={})
    action_rate_l2 = RewardTerm(func=AuboRewardFns.penalty_action_rate_l2, weight=-0.10, params={})
    out_of_workspace_penalty = RewardTerm(
        func=AuboRewardFns.penalty_ee_out_of_workspace,
        weight=-100.0,
        params={
            "asset_cfg": ROBOT_ASSET_NAME,
            "ee_frame_name": EE_BODY_NAME,
            "workspace": RL_WORKSPACE,
            "max_episode_steps": RL_MAX_EPISODE_STEPS,
            "early_failure_scale": 0.48,
        },
    )
    target_contact_penalty = RewardTerm(
        func=AuboRewardFns.penalty_target_contact,
        weight=-8.0,
        params={
            "sensor_cfg": ROBOT_CONTACT_SENSOR_NAME,
            "force_threshold": ROBOT_CONTACT_FORCE_THRESHOLD,
            "target_sensor_cfg": TARGET_CONTACT_SENSOR_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "allowed_body_names": (EE_BODY_NAME,),
            "ignored_body_names": ROBOT_IGNORED_CONTACT_BODY_NAMES,
            "target_contact_distance": 0.18,
            "target_contact_hard_force_threshold": 75.0,
        },
    )
    collision_penalty = RewardTerm(
        func=AuboRewardFns.penalty_collision,
        weight=-140.0,
        params={
            "sensor_cfg": ROBOT_CONTACT_SENSOR_NAME,
            "force_threshold": ROBOT_CONTACT_FORCE_THRESHOLD,
            "max_episode_steps": RL_MAX_EPISODE_STEPS,
            "early_failure_scale": 0.342857,
            "target_sensor_cfg": TARGET_CONTACT_SENSOR_NAME,
            "target_asset_name": DEFAULT_TARGET_ASSET_NAME,
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "allowed_body_names": (EE_BODY_NAME,),
            "ignored_body_names": ROBOT_IGNORED_CONTACT_BODY_NAMES,
            "target_contact_distance": 0.18,
            "target_contact_hard_force_threshold": 75.0,
        },
    )
