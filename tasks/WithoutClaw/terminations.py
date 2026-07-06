from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from tasks.WithoutClaw.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TARGET_CONTACT_SENSOR_NAME,
)
from tasks.WithoutClaw.task_cfg import DEFAULT_TARGET_ASSET_NAME, EE_BODY_NAME, RL_WORKSPACE, ROBOT_ASSET_NAME
from tools.logic import AuboTerminationFns


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=AuboTerminationFns.times_out, time_out=True)
    goal_reached = DoneTerm(
        func=AuboTerminationFns.goal_reached,
        params={
            "asset_cfg": ROBOT_ASSET_NAME,
            "goal_pos_name": DEFAULT_TARGET_ASSET_NAME,
            "ee_frame_name": EE_BODY_NAME,
            "pos_threshold": 0.15,
            "required_consecutive_steps": 3,
        },
    )
    obstacle_collision = DoneTerm(
        func=AuboTerminationFns.is_terminated_by_illegal_collision,
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
    self_collision = None
    ee_out_of_workspace = DoneTerm(
        func=AuboTerminationFns.ee_out_of_workspace,
        params={"asset_cfg": ROBOT_ASSET_NAME, "ee_frame_name": EE_BODY_NAME, "workspace": RL_WORKSPACE},
    )
