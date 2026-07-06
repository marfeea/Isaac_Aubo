from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from configs.place_cfg import WORKSTATION_INTERACTIVE_PLACEMENT_CFG
from tasks.WithoutClaw.task_cfg import ROBOT_ASSET_NAME
from tools.logic import reset_planning_obstacle_pose


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
    reset_obstacle_pose = None


def make_reset_target_pose_event(target_asset_name: str) -> EventTerm:
    """保留旧任务可选的椭球面 target 随机化。"""
    return EventTerm(
        func=reset_planning_obstacle_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(target_asset_name),
            "xy_radius": 0.45,
            "z_radius": 0.18,
            "z_range": (0.82, 1.06),
            "center": WORKSTATION_INTERACTIVE_PLACEMENT_CFG.base_pos,
        },
    )
