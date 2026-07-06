from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from configs.lula_cfg import AuboLulaIKControllerCfg
from tasks.WithoutClaw.task_cfg import DEFAULT_TARGET_ASSET_NAME, EE_BODY_NAME, ROBOT_ASSET_NAME
from tools.ik import AuboTaskSpaceIKAction
from tools.scene import AuboToolFns


@configclass
class StateOnlyObsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)})
    ee_pos = ObsTerm(
        func=AuboToolFns.get_body_pos_w,
        params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME), "body_name": EE_BODY_NAME},
    )
    ee_lin_vel = ObsTerm(
        func=AuboToolFns.get_body_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME), "body_name": EE_BODY_NAME},
    )
    target_pos = ObsTerm(
        func=AuboToolFns.get_root_pos_w,
        params={"asset_cfg": SceneEntityCfg(DEFAULT_TARGET_ASSET_NAME)},
    )
    to_target = ObsTerm(
        func=AuboToolFns.ee_to_target_vec_w,
        params={
            "robot_asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "ee_body_name": EE_BODY_NAME,
            "target_asset_cfg": SceneEntityCfg(DEFAULT_TARGET_ASSET_NAME),
        },
    )

    def __post_init__(self) -> None:
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: StateOnlyObsCfg = StateOnlyObsCfg()


@configclass
class AuboTaskSpaceIKActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = AuboTaskSpaceIKAction
    asset_name: str = ROBOT_ASSET_NAME
    target_asset_name: str = DEFAULT_TARGET_ASSET_NAME
    joint_names: list[str] = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"]
    body_name: str = EE_BODY_NAME
    action_dim: int = 3
    pos_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_position_delta: float = 0.01
    orientation_blend_start_distance: float = 0.40
    orientation_lock_distance: float = 0.20
    orientation_blend_tolerance: float = 1.00
    max_orientation_step: float = 0.10
    normalize_quat: bool = True
    controller: AuboLulaIKControllerCfg = AuboLulaIKControllerCfg()


@configclass
class ActionsCfg:
    task_space_ik: AuboTaskSpaceIKActionCfg = AuboTaskSpaceIKActionCfg()
