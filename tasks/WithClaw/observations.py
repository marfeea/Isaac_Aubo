from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from configs.lula_cfg import AuboLulaIKControllerCfg
from tasks.WithClaw.runtime import preposition_b, tcp_kinematics
from tasks.WithClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    FLANGE_TO_TCP_TRANSLATION_F,
    ROBOT_ASSET_NAME,
    TOOL_ORIENTATION_LOCK_DISTANCE,
    TOOL_ORIENTATION_REWARD_START_DISTANCE,
)
from tools.ik import AuboTaskSpaceIKAction


def tcp_pos_b(env):
    return tcp_kinematics(env).root.position_b


def tcp_vel_b(env):
    return tcp_kinematics(env).root.velocity_b


def tcp_to_preposition_b(env):
    return preposition_b(env) - tcp_kinematics(env).root.position_b


@configclass
class PolicyObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)})
    tcp_pos_b = ObsTerm(func=tcp_pos_b)
    preposition_b = ObsTerm(func=preposition_b)
    tcp_to_preposition_b = ObsTerm(func=tcp_to_preposition_b)
    tcp_vel_b = ObsTerm(func=tcp_vel_b)

    def __post_init__(self) -> None:
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: PolicyObsCfg = PolicyObsCfg()


@configclass
class AuboTaskSpaceIKActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AuboTaskSpaceIKAction
    asset_name: str = ROBOT_ASSET_NAME
    target_asset_name: str = DEFAULT_TARGET_ASSET_NAME
    joint_names: list[str] = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"]
    body_name: str = "Flange"
    action_dim: int = 3
    pos_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_position_delta: float = 0.01
    orientation_blend_start_distance: float = TOOL_ORIENTATION_REWARD_START_DISTANCE
    orientation_lock_distance: float = TOOL_ORIENTATION_LOCK_DISTANCE
    orientation_blend_tolerance: float = 1.00
    max_orientation_step: float = 0.10
    normalize_quat: bool = True
    orientation_goal_position_attr: str = "preposition_w"
    orientation_target_quaternion_attr: str = "desired_flange_quat_w"
    orientation_distance_offset_body: tuple[float, float, float] = FLANGE_TO_TCP_TRANSLATION_F
    latch_orientation_after_activation: bool = True
    controller: AuboLulaIKControllerCfg = AuboLulaIKControllerCfg()


@configclass
class ActionsCfg:
    task_space_ik: AuboTaskSpaceIKActionCfg = AuboTaskSpaceIKActionCfg()
