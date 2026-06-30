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

    def process_actions(self, actions: torch.Tensor) -> None:
        """每个环境步处理一次动作，并缓存该控制周期内不变的末端目标。"""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions
        self._processed_actions[:, 0] *= self.cfg.pos_scale[0]
        self._processed_actions[:, 1] *= self.cfg.pos_scale[1]
        self._processed_actions[:, 2] *= self.cfg.pos_scale[2]

        ee_pose_w = self.robot.data.body_pose_w[:, self.body_id]
        root_pose_w = self.robot.data.root_pose_w
        ee_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

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
        target_quat_b = self._compute_target_facing_quat_b(target_pos_b, goal_pos_b)

        self._target_pose_b[:, 0:3] = target_pos_b
        self._target_pose_b[:, 3:7] = target_quat_b
        self._ik_controller.set_command(self._target_pose_b)

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
        self._ik_controller.reset(env_ids=env_ids)

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
