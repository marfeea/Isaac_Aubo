from typing import NamedTuple

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from configs.asset import (
    EE_BODY_NAME,
    ROBOT_ASSET_NAME,
)
from configs.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
    TARGET_CONTACT_SENSOR_NAME,
    make_target_contact_sensor_cfg,
)
from configs.lula_cfg import AuboLulaIKControllerCfg
from configs.place_cfg import WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS
from configs.scene_cfg import TRAINING_ENV_SPACING, TRAINING_REPLICATE_PHYSICS, AuboTrainingSceneCfg
from tools.ik import AuboTaskSpaceIKAction
from tools.logic import (
    AuboRewardFns,
    AuboTerminationFns,
)
from tools.randomization import reset_asset_to_discrete_pose
from tools.scene import AuboToolFns

DEFAULT_RL_TARGET_ASSET_NAME = WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS[0]["scene_key"]
RL_SIM_DT = 1.0 / 120.0
RL_DECIMATION = 30
RL_EPISODE_LENGTH_S = 40.0
RL_MAX_EPISODE_STEPS = round(RL_EPISODE_LENGTH_S / (RL_DECIMATION * RL_SIM_DT))
# AUBObot root-frame bounds for Flange.
RL_WORKSPACE = {
    "x": [-0.75, 0.75],
    "y": [-0.75, 0.75],
    "z": [0.20, 1.1],
}


class TargetInitialStateCfg(NamedTuple):
    """目标离散初始状态；pos 为 env 局部坐标，rot 为 wxyz 四元数。"""

    name: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    preposition: tuple[float, float, float]


TARGET_INITIAL_STATES = (
    TargetInitialStateCfg(
        "sample_bottle_state_01", (1.537, 0.203, 0.94), (0.0, 0.0, 0.0, 1.0), (1.537, 0.083, 0.94)
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_02",
        (0.91167, 0.1753, 0.96789),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.1753, 0.96789),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_03",
        (0.91167, 0.03036, 0.96676),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.03036, 0.96676),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_04",
        (0.91235, -0.18557, 0.99091),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03235, -0.18557, 0.99091),
    ),
    TargetInitialStateCfg(
        "sample_bottle_state_05",
        (0.90264, -0.50461, 1.0915),
        (1.0, 0.0, 0.0, 0.0),
        (0.90264, -0.38461, 1.0915),
    ),
)


def make_reset_target_pose_event(
    target_asset_name: str = DEFAULT_RL_TARGET_ASSET_NAME,
    fixed_state_name: str | None = None,
) -> EventTerm:
    return EventTerm(
        func=reset_asset_to_discrete_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(target_asset_name),
            "state_names": tuple(state.name for state in TARGET_INITIAL_STATES),
            "positions": tuple(state.pos for state in TARGET_INITIAL_STATES),
            "orientations": tuple(state.rot for state in TARGET_INITIAL_STATES),
            "fixed_state_name": fixed_state_name,
        },
    )


# 事件定义类，挂钩子用
@configclass
class EventCfg:
    # on startup
    # 暂时先保留，之后如果要做一次性配置，就放在这里

    # on reset
    # reset场景
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # reset机械臂
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "position_range": (0.0, 0.0),   # 第一版先严格回默认位姿
            "velocity_range": (0.0, 0.0),
        },
    )

    # 默认使用场景中已有的命名物体作为目标，不随机改写其位姿。
    reset_obstacle_pose = None


# 观测定义类A，仅状态训练
@configclass
class StateOnlyObsCfg(ObsGroup):
    """仅状态训练观测组"""

    # 1. 本体状态
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)},
    )

    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME)},
    )

    # 2. 末端状态
    ee_pos = ObsTerm(
        func=AuboToolFns.get_body_pos_w,
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "body_name": EE_BODY_NAME,
        },
    )

    ee_lin_vel = ObsTerm(
        func=AuboToolFns.get_body_lin_vel_w,
        params={
            "asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "body_name": EE_BODY_NAME,
        },
    )

    # 3. 任务状态
    target_pos = ObsTerm(
        func=AuboToolFns.get_root_pos_w,
        params={"asset_cfg": SceneEntityCfg(DEFAULT_RL_TARGET_ASSET_NAME)},
    )

    to_target = ObsTerm(
        func=AuboToolFns.ee_to_target_vec_w,
        params={
            "robot_asset_cfg": SceneEntityCfg(ROBOT_ASSET_NAME),
            "ee_body_name": EE_BODY_NAME,
            "target_asset_cfg": SceneEntityCfg(DEFAULT_RL_TARGET_ASSET_NAME),
        },
    )

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False

# todo 观测定义类B，仅视觉训练

# todo 观测定义类C，视觉+低维状态输入联合训练

# 观测总配置
@configclass
class ObservationsCfg:
    """环境总观测配置"""

    policy: StateOnlyObsCfg = StateOnlyObsCfg()

# 动作配置类
@configclass
class AuboTaskSpaceIKActionCfg(ActionTermCfg):
    """Aubo末端任务空间动作项配置."""

    class_type: type[ActionTerm] = AuboTaskSpaceIKAction

    asset_name: str = ROBOT_ASSET_NAME
    target_asset_name: str = DEFAULT_RL_TARGET_ASSET_NAME
    joint_names: list[str] = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"]
    body_name: str = EE_BODY_NAME

    # 暂且更改为三维 dx dy dz
    action_dim: int = 3

    # 位移缩放与每个控制周期的三维位移模长上限（米）。
    pos_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_position_delta: float = 0.01

    # 末端在 40 cm 内逐渐面对目标，在 20 cm 内完全面对目标。
    orientation_blend_start_distance: float = 0.40
    orientation_lock_distance: float = 0.20

    # 进入渐进区时先使用宽松姿态容差，再收紧到 Lula 的正常容差。
    orientation_blend_tolerance: float = 1.00

    # 每个策略控制周期允许下发的最大姿态变化（弧度）。
    max_orientation_step: float = 0.10

    # 是否将后4维归一化为单位四元数
    normalize_quat: bool = True

    # Lula IK 控制器配置；输入仍为根坐标系绝对目标位姿，输出仍为关节位置目标。
    controller: AuboLulaIKControllerCfg = AuboLulaIKControllerCfg()

# 动作注册类
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    task_space_ik = AuboTaskSpaceIKActionCfg(
        asset_name=ROBOT_ASSET_NAME,
        target_asset_name=DEFAULT_RL_TARGET_ASSET_NAME,
        joint_names=["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"],
        body_name=EE_BODY_NAME,
        normalize_quat=True,
    )

# 场景配置类：训练与测试都复用 configs.scene_cfg.AuboTrainingSceneCfg。
class AuboRLSceneCfg(AuboTrainingSceneCfg):
    pass

# 奖励配置类
@configclass
class RewardsCfg:
    """教师策略奖励配置."""

    # 1) progress：鼓励末端每步靠近目标。
    ee_progress = RewardTerm(
        func=AuboRewardFns.reward_ee_progress,
        weight=80.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
        },
    )

    # 2) dense distance：提供稠密收敛信号。
    ee_distance_exp = RewardTerm(
        func=AuboRewardFns.reward_ee_distance_exp,
        weight=1.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
            "std": 0.20,
        },
    )

    # 3) success：命中阈值后按动作质量调制奖励。
    success = RewardTerm(
        func=AuboRewardFns.reward_success,
        weight=25.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
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

    # 4) action_far_near：远区鼓励有效大动作，近区惩罚低效大动作。
    action_far_near = RewardTerm(
        func=AuboRewardFns.reward_action_far_near,
        weight=1.0,
        params={
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
            "far_eps": 0.50,
            "close_eps": 0.20,
            "w_far_move": 0.10,
            "w_near_ineff": 0.60,
            "delta_min_far": 0.020,
            "delta_min_near": 0.008,
            "near_action_norm_max": 0.45,
        },
    )

    # 5) step penalty：轻量但持续的时间代价。
    # IsaacLab 会按 step_dt=decimation*sim.dt=0.25 缩放 reward；
    # weight=-0.25 对应每次决策约 -0.0625，160 步 timeout 约 -10。
    step_penalty = RewardTerm(
        func=AuboRewardFns.penalty_step,
        weight=-0.25,
        params={},
    )

    # 6) action magnitude penalty：抑制过猛动作。
    action_l2 = RewardTerm(
        func=AuboRewardFns.penalty_action_l2,
        weight=-0.025,
        params={},
    )

    # 7) action rate penalty：抑制高频抖动。
    action_rate_l2 = RewardTerm(
        func=AuboRewardFns.penalty_action_rate_l2,
        weight=-0.10,
        params={},
    )

    # 8) out-of-workspace penalty：一次性终止惩罚不因回合步数翻倍而缩放。
    # Earlier failures get up to 48% extra penalty to avoid short bad rollouts.
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

    # 9) target contact penalty：Flange may lightly touch the active target,
    # but repeated contact is still mildly discouraged. 回合步数翻倍后逐步权重减半。
    target_contact_penalty = RewardTerm(
        func=AuboRewardFns.penalty_target_contact,
        weight=-8.0,
        params={
            "sensor_cfg": ROBOT_CONTACT_SENSOR_NAME,
            "force_threshold": ROBOT_CONTACT_FORCE_THRESHOLD,
            "target_sensor_cfg": TARGET_CONTACT_SENSOR_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "allowed_body_names": (EE_BODY_NAME,),
            "ignored_body_names": ROBOT_IGNORED_CONTACT_BODY_NAMES,
            "target_contact_distance": 0.18,
            "target_contact_hard_force_threshold": 75.0,
        },
    )

    # 10) collision penalty：一次性终止惩罚不因回合步数翻倍而缩放。
    # Earlier collisions get about -12 extra actual reward penalty.
    collision_penalty = RewardTerm(
        func=AuboRewardFns.penalty_collision,
        weight=-140.0,
        params={
            "sensor_cfg": ROBOT_CONTACT_SENSOR_NAME,
            "force_threshold": ROBOT_CONTACT_FORCE_THRESHOLD,
            "max_episode_steps": RL_MAX_EPISODE_STEPS,
            "early_failure_scale": 0.342857,
            "target_sensor_cfg": TARGET_CONTACT_SENSOR_NAME,
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
            "robot_asset_name": ROBOT_ASSET_NAME,
            "ee_body_name": EE_BODY_NAME,
            "allowed_body_names": (EE_BODY_NAME,),
            "ignored_body_names": ROBOT_IGNORED_CONTACT_BODY_NAMES,
            "target_contact_distance": 0.18,
            "target_contact_hard_force_threshold": 75.0,
        },
    )

    # 11) obstacle safety penalty：鼓励绕障留余量，不贴边走
    # 暂且没有加入障碍部分
    # obstacle_safe = RewardTerm(
    #     func=AuboRewardFns.penalty_ee_obstacle_safe,
    #     weight=-150.0,
    #     params={
    #         "robot_asset_name": ROBOT_ASSET_NAME,
    #         "ee_body_name": EE_BODY_NAME,
    #         "obstacle_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
    #         "safe_margin": 0.08,
    #     },
    # )

# 终结配置类
@configclass
class TerminationsCfg:
    """Minimal termination config for reaching / obstacle-aware reaching."""

    time_out = DoneTerm(
        func=AuboTerminationFns.times_out,
        time_out=True,
    )

    goal_reached = DoneTerm(
        func=AuboTerminationFns.goal_reached,
        params={
            "asset_cfg": ROBOT_ASSET_NAME,
            "goal_pos_name": DEFAULT_RL_TARGET_ASSET_NAME,
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
            "target_asset_name": DEFAULT_RL_TARGET_ASSET_NAME,
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
        params={
            "asset_cfg": ROBOT_ASSET_NAME,
            "ee_frame_name": EE_BODY_NAME,
            "workspace": RL_WORKSPACE,
        },
    )

# 训练环境类
@configclass
class AuboRLEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene : AuboRLSceneCfg = AuboRLSceneCfg(
        num_envs=1,
        env_spacing=TRAINING_ENV_SPACING,
        replicate_physics=TRAINING_REPLICATE_PHYSICS,
    )
    # 基础设置 动作空间，观测空间
    observations : ObservationsCfg = ObservationsCfg()
    actions : ActionsCfg = ActionsCfg()
    events : EventCfg = EventCfg()
    rewards : RewardsCfg = RewardsCfg()
    terminations : TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = RL_DECIMATION
        self.episode_length_s = RL_EPISODE_LENGTH_S
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = RL_SIM_DT
        self.sim.render_interval = 4


def configure_task_target(
    env_cfg: AuboRLEnvCfg,
    target_asset_name: str,
    *,
    randomize_target_pose: bool = False,
) -> None:
    """Route all RL task terms to the selected scene object key."""
    env_cfg.observations.policy.target_pos.params["asset_cfg"] = SceneEntityCfg(target_asset_name)
    env_cfg.observations.policy.to_target.params["target_asset_cfg"] = SceneEntityCfg(target_asset_name)

    env_cfg.actions.task_space_ik.target_asset_name = target_asset_name

    env_cfg.rewards.ee_progress.params["target_asset_name"] = target_asset_name
    env_cfg.rewards.ee_distance_exp.params["target_asset_name"] = target_asset_name
    env_cfg.rewards.success.params["target_asset_name"] = target_asset_name
    env_cfg.rewards.action_far_near.params["target_asset_name"] = target_asset_name
    env_cfg.rewards.target_contact_penalty.params["target_asset_name"] = target_asset_name
    env_cfg.rewards.collision_penalty.params["target_asset_name"] = target_asset_name

    env_cfg.terminations.goal_reached.params["goal_pos_name"] = target_asset_name
    env_cfg.terminations.obstacle_collision.params["target_asset_name"] = target_asset_name

    target_cfg = getattr(env_cfg.scene, target_asset_name)
    env_cfg.scene.target_contact_sensor = make_target_contact_sensor_cfg(target_cfg.prim_path)

    if randomize_target_pose:
        env_cfg.events.reset_obstacle_pose = make_reset_target_pose_event(target_asset_name)
    else:
        env_cfg.events.reset_obstacle_pose = None
