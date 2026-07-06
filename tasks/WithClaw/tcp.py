from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TcpWorldState:
    position_w: torch.Tensor
    velocity_w: torch.Tensor


@dataclass(frozen=True)
class TcpRootState:
    position_b: torch.Tensor
    velocity_b: torch.Tensor


@dataclass(frozen=True)
class TcpKinematics:
    world: TcpWorldState
    root: TcpRootState


def _validate_vector_batch(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} 必须为 (N, 3)，实际为 {tuple(value.shape)}。")


def _validate_quaternion_batch(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[-1] != 4:
        raise ValueError(f"{name} 必须为 wxyz (N, 4)，实际为 {tuple(value.shape)}。")


def normalize_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    _validate_quaternion_batch("quaternion", quaternion)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= 1.0e-8):
        raise ValueError("四元数范数必须大于 1e-8。")
    return quaternion / norm


def rotate_vector_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """使用 wxyz 四元数批量旋转三维向量。"""
    _validate_vector_batch("vector", vector)
    quaternion = normalize_quaternion_wxyz(quaternion)
    if quaternion.shape[0] != vector.shape[0]:
        raise ValueError("四元数与向量的 batch 大小必须一致。")
    scalar = quaternion[:, :1]
    axis = quaternion[:, 1:4]
    axis_cross_vector = torch.cross(axis, vector, dim=-1)
    return vector + 2.0 * (scalar * axis_cross_vector + torch.cross(axis, axis_cross_vector, dim=-1))


def rotate_vector_inverse_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quaternion_wxyz(quaternion)
    inverse = torch.cat((quaternion[:, :1], -quaternion[:, 1:4]), dim=-1)
    return rotate_vector_wxyz(inverse, vector)


def expand_local_offset(
    local_offset_f: torch.Tensor | tuple[float, float, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    offset = torch.as_tensor(local_offset_f, dtype=reference.dtype, device=reference.device)
    if offset.shape == (3,):
        offset = offset.reshape(1, 3).expand(reference.shape[0], 3)
    _validate_vector_batch("local_offset_f", offset)
    if offset.shape[0] != reference.shape[0]:
        raise ValueError("局部偏置与 reference 的 batch 大小必须一致。")
    return offset


def compute_tcp_world_state(
    flange_position_w: torch.Tensor,
    flange_quaternion_w: torch.Tensor,
    flange_linear_velocity_w: torch.Tensor,
    flange_angular_velocity_w: torch.Tensor,
    flange_to_tcp_translation_f: torch.Tensor | tuple[float, float, float],
) -> TcpWorldState:
    """由 Flange 刚体状态计算 TCP 世界位置和包含切向项的线速度。"""
    for name, value in (
        ("flange_position_w", flange_position_w),
        ("flange_linear_velocity_w", flange_linear_velocity_w),
        ("flange_angular_velocity_w", flange_angular_velocity_w),
    ):
        _validate_vector_batch(name, value)
    offset_f = expand_local_offset(flange_to_tcp_translation_f, flange_position_w)
    offset_w = rotate_vector_wxyz(flange_quaternion_w, offset_f)
    position_w = flange_position_w + offset_w
    tangential_velocity_w = torch.cross(flange_angular_velocity_w, offset_w, dim=-1)
    return TcpWorldState(
        position_w=position_w,
        velocity_w=flange_linear_velocity_w + tangential_velocity_w,
    )


def express_tcp_state_in_root(
    tcp_state_w: TcpWorldState,
    root_position_w: torch.Tensor,
    root_quaternion_w: torch.Tensor,
    root_linear_velocity_w: torch.Tensor,
    root_angular_velocity_w: torch.Tensor,
) -> TcpRootState:
    """把 TCP 点的位置与完整相对速度表达在机器人根坐标系。"""
    for name, value in (
        ("root_position_w", root_position_w),
        ("root_linear_velocity_w", root_linear_velocity_w),
        ("root_angular_velocity_w", root_angular_velocity_w),
    ):
        _validate_vector_batch(name, value)
    root_to_tcp_w = tcp_state_w.position_w - root_position_w
    relative_velocity_w = (
        tcp_state_w.velocity_w
        - root_linear_velocity_w
        - torch.cross(root_angular_velocity_w, root_to_tcp_w, dim=-1)
    )
    return TcpRootState(
        position_b=rotate_vector_inverse_wxyz(root_quaternion_w, root_to_tcp_w),
        velocity_b=rotate_vector_inverse_wxyz(root_quaternion_w, relative_velocity_w),
    )


def read_tcp_kinematics(
    robot,
    flange_body_id: int,
    flange_to_tcp_translation_f: torch.Tensor | tuple[float, float, float],
) -> TcpKinematics:
    """从 Isaac Lab articulation data 读取 Flange 与根状态并计算 TCP。"""
    body_pose_w = robot.data.body_pose_w[:, int(flange_body_id)]
    world_state = compute_tcp_world_state(
        body_pose_w[:, :3],
        body_pose_w[:, 3:7],
        robot.data.body_lin_vel_w[:, int(flange_body_id), :3],
        robot.data.body_ang_vel_w[:, int(flange_body_id), :3],
        flange_to_tcp_translation_f,
    )
    root_state = express_tcp_state_in_root(
        world_state,
        robot.data.root_pose_w[:, :3],
        robot.data.root_pose_w[:, 3:7],
        robot.data.root_lin_vel_w[:, :3],
        robot.data.root_ang_vel_w[:, :3],
    )
    return TcpKinematics(world=world_state, root=root_state)
