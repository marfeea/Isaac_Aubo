from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

if TYPE_CHECKING:
    from configs.lula_cfg import AuboLulaIKControllerCfg


class AuboLulaIKController:
    """将 Lula 单环境 CPU IK 适配为 IsaacLab 批量张量接口。"""

    action_dim = 7

    def __init__(
        self,
        cfg: AuboLulaIKControllerCfg,
        num_envs: int,
        device: str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device

        self._solver = LulaKinematicsSolver(
            robot_description_path=cfg.robot_description_path,
            urdf_path=cfg.urdf_path,
        )
        self._validate_model_contract()

        num_joints = len(cfg.lula_joint_names)
        self._command = torch.zeros((self.num_envs, self.action_dim), device=device)
        self._command[:, 3] = 1.0
        self._joint_solution = torch.zeros((self.num_envs, num_joints), device=device)
        self._command_dirty = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    @property
    def last_success(self) -> torch.Tensor:
        """返回每个环境最近一次 Lula 求解是否收敛。"""
        return self._last_success

    @property
    def joint_solution(self) -> torch.Tensor:
        """返回未经过物理步增量限幅的最近关节解。"""
        return self._joint_solution

    def set_command(self, command: torch.Tensor) -> None:
        """缓存根坐标系目标位姿，格式为 ``[x, y, z, qw, qx, qy, qz]``。"""
        expected_shape = (self.num_envs, self.action_dim)
        if tuple(command.shape) != expected_shape:
            raise ValueError(f"Lula IK command shape must be {expected_shape}, got {tuple(command.shape)}.")
        if not torch.isfinite(command).all():
            raise ValueError("Lula IK command contains non-finite values.")

        quat = command[:, 3:7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        if torch.any(quat_norm < 1e-8):
            raise ValueError("Lula IK command contains a zero-length quaternion.")

        self._command.copy_(command)
        self._command[:, 3:7] = quat / quat_norm
        self._command_dirty[:] = True

    def compute(
        self,
        ee_pos: torch.Tensor | None = None,
        ee_quat: torch.Tensor | None = None,
        jacobian: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """兼容 DifferentialIK 的参数顺序，并返回同形状的关节位置目标。"""
        del ee_pos, ee_quat, jacobian
        if joint_pos is None:
            raise ValueError("Lula IK compute requires joint_pos.")

        expected_shape = self._joint_solution.shape
        if tuple(joint_pos.shape) != tuple(expected_shape):
            raise ValueError(
                f"Lula IK joint position shape must be {tuple(expected_shape)}, got {tuple(joint_pos.shape)}."
            )

        dirty_env_ids = torch.nonzero(self._command_dirty, as_tuple=False).flatten()
        if dirty_env_ids.numel() > 0:
            self._solve_dirty_commands(dirty_env_ids, joint_pos)

        max_joint_delta = self.cfg.max_joint_delta
        if max_joint_delta is None:
            return self._joint_solution.clone()
        if max_joint_delta <= 0.0:
            raise ValueError(f"max_joint_delta must be positive or None, got {max_joint_delta}.")

        joint_delta = torch.clamp(
            self._joint_solution - joint_pos,
            min=-max_joint_delta,
            max=max_joint_delta,
        )
        return joint_pos + joint_delta

    def reset(self, env_ids=None) -> None:
        """清理指定环境的命令、关节解和收敛状态。"""
        resolved_ids = self._resolve_env_ids(env_ids)
        self._command[resolved_ids] = 0.0
        self._command[resolved_ids, 3] = 1.0
        self._joint_solution[resolved_ids] = 0.0
        self._command_dirty[resolved_ids] = False
        self._last_success[resolved_ids] = False

    def _solve_dirty_commands(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
        env_ids_cpu = env_ids.detach().cpu().tolist()
        command_cpu = self._command[env_ids].detach().cpu().numpy()
        joint_pos_cpu = joint_pos[env_ids].detach().cpu().numpy()

        for local_index, env_id in enumerate(env_ids_cpu):
            command = command_cpu[local_index]
            warm_start = joint_pos_cpu[local_index]
            solution, success = self._solver.compute_inverse_kinematics(
                frame_name=self.cfg.end_effector_frame_name,
                target_position=command[0:3],
                target_orientation=command[3:7],
                warm_start=warm_start,
                position_tolerance=self.cfg.position_tolerance,
                orientation_tolerance=self.cfg.orientation_tolerance,
            )
            solution_is_valid = bool(success) and bool(np.isfinite(solution).all())
            if solution_is_valid:
                self._joint_solution[env_id] = torch.tensor(
                    np.array(solution, copy=True),
                    device=joint_pos.device,
                    dtype=joint_pos.dtype,
                )
            else:
                # 不收敛时保持当前关节位，避免 NaN 或陈旧解进入执行链。
                self._joint_solution[env_id] = joint_pos[env_id]
            self._last_success[env_id] = solution_is_valid
            self._command_dirty[env_id] = False

    def _validate_model_contract(self) -> None:
        solver_joint_names = tuple(self._solver.get_joint_names())
        expected_joint_names = tuple(self.cfg.lula_joint_names)
        if solver_joint_names != expected_joint_names:
            raise ValueError(
                "Lula cspace joint order does not match controller config: "
                f"solver={solver_joint_names}, config={expected_joint_names}."
            )

        frame_names = set(self._solver.get_all_frame_names())
        if self.cfg.end_effector_frame_name not in frame_names:
            raise ValueError(
                f"Lula end-effector frame '{self.cfg.end_effector_frame_name}' is missing from the URDF."
            )

    def _resolve_env_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]

        resolved_ids = torch.as_tensor(env_ids, device=self.device)
        if resolved_ids.dtype == torch.bool:
            return torch.nonzero(resolved_ids, as_tuple=False).flatten()
        return resolved_ids.to(dtype=torch.long).flatten()
