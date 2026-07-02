from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import ActionTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_matrix, subtract_frame_transforms

from tools.lula_ik import AuboLulaIKController
from tools.scene import AuboToolFns

if TYPE_CHECKING:
    from configs.RLcfg import AuboTaskSpaceIKActionCfg


class AuboTaskSpaceIKAction(ActionTerm):
    """将4 Hz任务空间增量转换为固定目标，并通过 Lula 输出关节目标。"""

    cfg: AuboTaskSpaceIKActionCfg

    def __init__(self, cfg: AuboTaskSpaceIKActionCfg, env):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]
        self.target_asset_name = cfg.target_asset_name

        self.entity_cfg = SceneEntityCfg(
            cfg.asset_name,
            joint_names=cfg.joint_names,
            body_names=[cfg.body_name],
            preserve_order=True,
        )
        self.entity_cfg.resolve(env.scene)

        self.joint_ids = self.entity_cfg.joint_ids
        self.body_id = self.entity_cfg.body_ids[0]

        self._ik_controller = AuboLulaIKController(
            cfg.controller,
            num_envs=env.num_envs,
            device=self.robot.device,
        )

        self._raw_actions = torch.zeros((env.num_envs, cfg.action_dim), device=self.robot.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._target_pose_b = torch.zeros((env.num_envs, 7), device=self.robot.device)
        self._target_pose_b[:, 3] = 1.0

        self._blend_reference_quat_b = torch.zeros((env.num_envs, 4), device=self.robot.device)
        self._blend_reference_quat_b[:, 0] = 1.0
        self._last_command_quat_b = self._blend_reference_quat_b.clone()
        self._orientation_was_enabled = torch.zeros(env.num_envs, dtype=torch.bool, device=self.robot.device)
        self._orientation_blend = torch.zeros(env.num_envs, device=self.robot.device)
        self._ee_goal_distance = torch.full((env.num_envs,), float("inf"), device=self.robot.device)
        self._action_start_ee_pos_b = torch.zeros((env.num_envs, 3), device=self.robot.device)
        self._planned_position_delta_b = torch.zeros((env.num_envs, 3), device=self.robot.device)
        self._step_diagnostics_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=self.robot.device)

        if not (0.0 < cfg.orientation_lock_distance < cfg.orientation_blend_start_distance):
            raise ValueError(
                "orientation distances must satisfy 0 < lock_distance < blend_start_distance, got "
                f"lock={cfg.orientation_lock_distance}, start={cfg.orientation_blend_start_distance}."
            )
        if cfg.max_orientation_step <= 0.0:
            raise ValueError(f"max_orientation_step must be positive, got {cfg.max_orientation_step}.")
        if cfg.max_position_delta <= 0.0:
            raise ValueError(f"max_position_delta must be positive, got {cfg.max_position_delta}.")
        if cfg.orientation_blend_tolerance < cfg.controller.orientation_tolerance:
            raise ValueError(
                "orientation_blend_tolerance must be no smaller than the controller orientation tolerance, got "
                f"blend={cfg.orientation_blend_tolerance}, controller={cfg.controller.orientation_tolerance}."
            )

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def target_pose_b(self) -> torch.Tensor:
        """返回当前策略步缓存的机器人根坐标系目标位姿。"""
        return self._target_pose_b

    @property
    def ik_success(self) -> torch.Tensor:
        """返回每个环境最近一次 Lula IK 是否收敛。"""
        return self._ik_controller.last_success

    @property
    def orientation_blend(self) -> torch.Tensor:
        """返回当前朝向目标约束的插值权重。"""
        return self._orientation_blend

    @property
    def ee_goal_distance(self) -> torch.Tensor:
        """返回实际末端到目标点的距离。"""
        return self._ee_goal_distance

    @property
    def orientation_enabled(self) -> torch.Tensor:
        """返回当前是否向 Lula 启用姿态约束。"""
        return self._ik_controller.orientation_enabled

    @property
    def orientation_tolerance(self) -> torch.Tensor:
        """返回当前下发给 Lula 的姿态容差。"""
        return self._ik_controller.orientation_tolerance

    def get_step_execution_diagnostics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回计划位移、实际位移、沿计划方向的完成百分比和数据有效标记。"""
        ee_pos_b = self._get_ee_pos_b()
        planned_delta = self._planned_position_delta_b
        actual_delta = ee_pos_b - self._action_start_ee_pos_b
        planned_norm_sq = torch.sum(torch.square(planned_delta), dim=-1)
        projected_completion = (
            torch.sum(actual_delta * planned_delta, dim=-1)
            / torch.clamp(planned_norm_sq, min=1.0e-12)
            * 100.0
        )
        zero_command_completion = torch.where(
            torch.linalg.vector_norm(actual_delta, dim=-1) <= 1.0e-6,
            torch.full_like(projected_completion, 100.0),
            torch.zeros_like(projected_completion),
        )
        completion = torch.where(planned_norm_sq > 1.0e-12, projected_completion, zero_command_completion)
        return planned_delta, actual_delta, completion, self._step_diagnostics_valid

    def process_actions(self, actions: torch.Tensor) -> None:
        """每个环境步处理一次动作，并缓存该控制周期内不变的末端目标。"""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._processed_actions[:, 0] *= self.cfg.pos_scale[0]
        self._processed_actions[:, 1] *= self.cfg.pos_scale[1]
        self._processed_actions[:, 2] *= self.cfg.pos_scale[2]
        position_delta = self._processed_actions[:, 0:3]
        position_delta_norm = torch.linalg.vector_norm(position_delta, dim=-1, keepdim=True)
        position_delta_scale = torch.clamp(
            float(self.cfg.max_position_delta) / torch.clamp(position_delta_norm, min=1.0e-12),
            max=1.0,
        )
        position_delta.mul_(position_delta_scale)

        ee_pos_b = self._get_ee_pos_b()
        root_pose_w = self.robot.data.root_pose_w

        self._action_start_ee_pos_b.copy_(ee_pos_b)
        self._planned_position_delta_b.copy_(self._processed_actions[:, 0:3])
        self._step_diagnostics_valid[:] = True

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
        self._ee_goal_distance[:] = torch.norm(goal_pos_b - ee_pos_b, dim=-1)
        blend_range = self.cfg.orientation_blend_start_distance - self.cfg.orientation_lock_distance
        blend_t = torch.clamp(
            (self.cfg.orientation_blend_start_distance - self._ee_goal_distance) / blend_range,
            0.0,
            1.0,
        )
        self._orientation_blend[:] = blend_t * blend_t * (3.0 - 2.0 * blend_t)

        orientation_enabled = self._ee_goal_distance < self.cfg.orientation_blend_start_distance
        newly_enabled_ids = torch.nonzero(
            orientation_enabled & ~self._orientation_was_enabled,
            as_tuple=False,
        ).flatten()
        if newly_enabled_ids.numel() > 0:
            joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
            reference_rot_b = self._ik_controller.compute_forward_rotations(joint_pos[newly_enabled_ids])
            reference_quat_b = quat_from_matrix(reference_rot_b)
            self._blend_reference_quat_b[newly_enabled_ids] = reference_quat_b
            self._last_command_quat_b[newly_enabled_ids] = reference_quat_b

        # 朝向只由实际末端位置决定，避免策略的单步位置命令直接引入姿态抖动。
        facing_quat_b = self._compute_target_facing_quat_b(ee_pos_b, goal_pos_b)
        desired_quat_b = self._quat_slerp(
            self._blend_reference_quat_b,
            facing_quat_b,
            self._orientation_blend.unsqueeze(-1),
        )
        target_quat_b = self._limit_quat_step(
            self._last_command_quat_b,
            desired_quat_b,
            self.cfg.max_orientation_step,
        )
        self._last_command_quat_b[:] = target_quat_b
        self._orientation_was_enabled[:] = orientation_enabled

        orientation_tolerance = (
            self.cfg.orientation_blend_tolerance
            + self._orientation_blend
            * (self.cfg.controller.orientation_tolerance - self.cfg.orientation_blend_tolerance)
        )

        self._target_pose_b[:, 0:3] = target_pos_b
        self._target_pose_b[:, 3:7] = target_quat_b
        self._ik_controller.set_command(
            self._target_pose_b,
            orientation_enabled=orientation_enabled,
            orientation_tolerance=orientation_tolerance,
        )

    def apply_actions(self) -> None:
        """每个物理步向缓存的 Lula 关节解推进，并写入关节位置目标。"""
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_pos_des = self._ik_controller.compute(joint_pos=joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)

    def reset(self, env_ids=None) -> None:
        """清理重置环境的动作缓存，避免诊断数据沿用上一回合。"""
        cache_env_ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[cache_env_ids] = 0.0
        self._processed_actions[cache_env_ids] = 0.0
        self._target_pose_b[cache_env_ids] = 0.0
        self._target_pose_b[cache_env_ids, 3] = 1.0
        self._blend_reference_quat_b[cache_env_ids] = 0.0
        self._blend_reference_quat_b[cache_env_ids, 0] = 1.0
        self._last_command_quat_b[cache_env_ids] = 0.0
        self._last_command_quat_b[cache_env_ids, 0] = 1.0
        self._orientation_was_enabled[cache_env_ids] = False
        self._orientation_blend[cache_env_ids] = 0.0
        self._ee_goal_distance[cache_env_ids] = float("inf")
        self._action_start_ee_pos_b[cache_env_ids] = 0.0
        self._planned_position_delta_b[cache_env_ids] = 0.0
        self._step_diagnostics_valid[cache_env_ids] = False
        self._ik_controller.reset(env_ids=env_ids)

    def _get_ee_pos_b(self) -> torch.Tensor:
        """读取机器人根坐标系中的当前末端位置。"""
        ee_pose_w = self.robot.data.body_pose_w[:, self.body_id]
        root_pose_w = self.robot.data.root_pose_w
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        return ee_pos_b

    @staticmethod
    def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
        """归一化 wxyz 四元数。"""
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        return quat / torch.clamp(norm, min=1e-8)

    @staticmethod
    def _quat_slerp(start: torch.Tensor, end: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        """沿最短旋转路径执行批量 wxyz 四元数 SLERP。"""
        start = AuboTaskSpaceIKAction._normalize_quat(start)
        end = AuboTaskSpaceIKAction._normalize_quat(end)
        fraction = torch.clamp(fraction, 0.0, 1.0)

        dot = torch.sum(start * end, dim=-1, keepdim=True)
        end = torch.where(dot < 0.0, -end, end)
        dot = torch.clamp(torch.abs(dot), 0.0, 1.0)

        angle = torch.acos(dot)
        sin_angle = torch.sin(angle)
        denominator = torch.clamp(sin_angle, min=1e-6)
        slerp = (
            torch.sin((1.0 - fraction) * angle) / denominator * start
            + torch.sin(fraction * angle) / denominator * end
        )
        lerp = (1.0 - fraction) * start + fraction * end
        result = torch.where(sin_angle > 1e-4, slerp, lerp)
        return AuboTaskSpaceIKAction._normalize_quat(result)

    @staticmethod
    def _limit_quat_step(start: torch.Tensor, end: torch.Tensor, max_angle: float) -> torch.Tensor:
        """限制每个控制周期的姿态目标变化，避免大姿态命令直接导致 IK 失败。"""
        start = AuboTaskSpaceIKAction._normalize_quat(start)
        end = AuboTaskSpaceIKAction._normalize_quat(end)
        dot = torch.clamp(torch.abs(torch.sum(start * end, dim=-1, keepdim=True)), 0.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        fraction = torch.clamp(float(max_angle) / torch.clamp(angle, min=1e-6), max=1.0)
        return AuboTaskSpaceIKAction._quat_slerp(start, end, fraction)

    @staticmethod
    def _compute_target_facing_quat_b(
        ee_target_pos_b: torch.Tensor,
        goal_pos_b: torch.Tensor,
    ) -> torch.Tensor:
        """生成让法兰局部负X轴朝向目标点的根坐标系四元数。"""
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

        # 旋转矩阵的列表示目标坐标轴在机器人根坐标系中的方向。
        x_axis = -face_dir
        world_up = (
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).repeat(x_axis.shape[0], 1)
        )
        alt_up = (
            torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).repeat(x_axis.shape[0], 1)
        )
        near_parallel = torch.abs((x_axis * world_up).sum(dim=-1, keepdim=True)) > 0.999
        up = torch.where(near_parallel, alt_up, world_up)

        y_axis = torch.cross(up, x_axis, dim=-1)
        y_axis = y_axis / torch.clamp(torch.norm(y_axis, dim=-1, keepdim=True), min=eps)
        z_axis = torch.cross(x_axis, y_axis, dim=-1)
        z_axis = z_axis / torch.clamp(torch.norm(z_axis, dim=-1, keepdim=True), min=eps)

        rot_mats = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        target_quat_b = quat_from_matrix(rot_mats)
        # 统一等价四元数的符号，降低跨控制步的数值跳变。
        sign = torch.where(target_quat_b[:, :1] < 0.0, -1.0, 1.0)
        return target_quat_b * sign
