import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix, subtract_frame_transforms

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnvCfg


from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from tools.logic import (
    reset_planning_obstacle_pose,
    AuboRewardFns,
    AuboTerminationFns,
)
from tools.scene import AuboToolFns
from configs.asset import (
    EE_BODY_NAME,
    ROBOT_ASSET_NAME,
    WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS,
    WORKSTATION_INTERACTIVE_BASE_POS,
)
from configs.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
)
from configs.scene_cfg import AuboTrainingSceneCfg, TRAINING_ENV_SPACING, TRAINING_REPLICATE_PHYSICS

import isaaclab.envs.mdp as mdp


RL_TARGET_CENTER = WORKSTATION_INTERACTIVE_BASE_POS
DEFAULT_RL_TARGET_ASSET_NAME = WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS[0]["scene_key"]
RL_WORKSPACE = {
    "x": [0.45, 2.10],
    "y": [-0.85, 1.20],
    "z": [0.45, 1.60],
}


def make_reset_target_pose_event(target_asset_name: str = DEFAULT_RL_TARGET_ASSET_NAME) -> EventTerm:
    return EventTerm(
        func=reset_planning_obstacle_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(target_asset_name),
            "xy_radius": 0.45,
            "z_radius": 0.18,
            "z_range": (0.82, 1.06),
            "center": RL_TARGET_CENTER,
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

# 动作项逻辑
class AuboTaskSpaceIKAction(ActionTerm):
    """将3维任务空间位移动作映射为关节位置目标，姿态自动朝向目标点。
    action = [dx, dy, dz]
    """

    cfg: "AuboTaskSpaceIKActionCfg"

    def __init__(self, cfg: "AuboTaskSpaceIKActionCfg", env):
        
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]
        self.target_asset_name = cfg.target_asset_name

        self.entity_cfg = SceneEntityCfg(
            cfg.asset_name,
            joint_names=cfg.joint_names,
            body_names=[cfg.body_name],
        )
        self.entity_cfg.resolve(env.scene)

        self.joint_ids = self.entity_cfg.joint_ids
        self.body_id = self.entity_cfg.body_ids[0]

        # 对应官方tutorial里末端body jacobian索引常用 body_id - 1 处理
        self.ee_jacobi_idx = self.body_id - 1

        self._ik_controller = DifferentialIKController(
            cfg.controller,
            num_envs=env.num_envs,
            device=self.robot.device,
        )

        self._raw_actions = torch.zeros((env.num_envs, cfg.action_dim), device=self.robot.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """解析策略输出."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # 缩放位置增量
        self._processed_actions[:, 0] *= self.cfg.pos_scale[0]
        self._processed_actions[:, 1] *= self.cfg.pos_scale[1]
        self._processed_actions[:, 2] *= self.cfg.pos_scale[2]

        # 归一化四元数
        # if self.cfg.normalize_quat:
        #     quat = self._processed_actions[:, 3:7]
        #     quat = quat / torch.clamp(torch.norm(quat, dim=-1, keepdim=True), min=1e-8)
        #     self._processed_actions[:, 3:7] = quat

    def compute_target_facing_quat_b(
        self,
        ee_target_pos_b: torch.Tensor,
        goal_pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """根据“目标末端位置->目标点”方向生成朝向四元数 (w, x, y, z).

        AUBO USD 中 Flange 的局部 +X 轴指向法兰背面，因此这里让局部 -X
        对准目标点，视觉上才是法兰正面朝向目标。
        """
        eps = 1e-6
        device = ee_target_pos_b.device
        dtype = ee_target_pos_b.dtype

        face_dir = goal_pos_b - ee_target_pos_b
        face_dir_norm = torch.norm(face_dir, dim=-1, keepdim=True)
        default_face_dir = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3)
        face_dir = torch.where(
            face_dir_norm > eps,
            face_dir / torch.clamp(face_dir_norm, min=eps),
            default_face_dir,
        )

        # Rotation matrix columns are the target frame axes expressed in base frame.
        # Since Flange local +X points backward, local +X must point away from target.
        x_axis = -face_dir

        world_up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).repeat(x_axis.shape[0], 1)
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).repeat(x_axis.shape[0], 1)
        near_parallel = torch.abs((x_axis * world_up).sum(dim=-1, keepdim=True)) > 0.999
        up = torch.where(near_parallel, alt_up, world_up)

        y_axis = torch.cross(up, x_axis, dim=-1)
        y_axis = y_axis / torch.clamp(torch.norm(y_axis, dim=-1, keepdim=True), min=eps)

        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = z_axis / torch.clamp(torch.norm(z_axis, dim=-1, keepdim=True), min=eps)

        rot_mats = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        target_quat_b = quat_from_matrix(rot_mats)

        # 统一符号，减小跨步抖动
        sign = torch.where(target_quat_b[:, :1] < 0.0, -1.0, 1.0)
        return target_quat_b * sign

    def apply_actions(self):
        """执行IK并写入关节目标."""
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.body_id]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # 当前末端位置 + 位移增量
        target_pos_b = ee_pos_b + self._processed_actions[:, 0:3]

        goal_pos_w = AuboToolFns.get_root_pos_w(self._env, self.target_asset_name)
        identity_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=goal_pos_w.dtype)
        identity_quat_w[:, 0] = 1.0
        goal_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            goal_pos_w,
            identity_quat_w,
        )

        target_quat_b = self.compute_target_facing_quat_b(target_pos_b, goal_pos_b)

        ik_commands = torch.cat([target_pos_b, target_quat_b], dim=-1)
        self._ik_controller.set_command(ik_commands)

        joint_pos_des = self._ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )

        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)

# 动作配置类
@configclass
class AuboTaskSpaceIKActionCfg(ActionTermCfg):
    """Aubo末端任务空间动作项配置."""

    class_type: type[ActionTerm] = AuboTaskSpaceIKAction

    asset_name: str = ROBOT_ASSET_NAME
    target_asset_name: str = DEFAULT_RL_TARGET_ASSET_NAME
    joint_names: list[str] = ["Joint.*", "Flange"]
    body_name: str = EE_BODY_NAME

    # 暂且更改为三维 dx dy dz
    action_dim: int = 3

    # 位移缩放（米）
    pos_scale: tuple[float, float, float] = (0.05, 0.05, 0.05)

    # 是否将后4维归一化为单位四元数
    normalize_quat: bool = True

    # IK控制器配置
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )

# 动作注册类
@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    task_space_ik = AuboTaskSpaceIKActionCfg(
        asset_name=ROBOT_ASSET_NAME,
        target_asset_name=DEFAULT_RL_TARGET_ASSET_NAME,
        joint_names=["Joint.*", "Flange"],
        body_name=EE_BODY_NAME,
        pos_scale=(0.05, 0.05, 0.05),
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
            "w_near_ineff": 1.20,
            "delta_min_far": 0.020,
            "delta_min_near": 0.008,
            "near_action_norm_max": 0.45,
        },
    )

    # 5) step penalty：防止拖延。
    step_penalty = RewardTerm(
        func=AuboRewardFns.penalty_step,
        weight=-0.03,
        params={},
    )

    # 6) action magnitude penalty：抑制过猛动作。
    action_l2 = RewardTerm(
        func=AuboRewardFns.penalty_action_l2,
        weight=-0.02,
        params={},
    )

    # 7) action rate penalty：抑制高频抖动。
    action_rate_l2 = RewardTerm(
        func=AuboRewardFns.penalty_action_rate_l2,
        weight=-0.10,
        params={},
    )

    # 8) obstacle safety penalty：鼓励绕障留余量，不贴边走
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
        self.decimation = 30
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1/ 120
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

    env_cfg.terminations.goal_reached.params["goal_pos_name"] = target_asset_name

    if randomize_target_pose:
        env_cfg.events.reset_obstacle_pose = make_reset_target_pose_event(target_asset_name)
    else:
        env_cfg.events.reset_obstacle_pose = None
