from __future__ import annotations

from dataclasses import dataclass

import torch

from tasks.WithClaw.tcp import normalize_quaternion_wxyz, rotate_vector_wxyz


@dataclass(frozen=True)
class ToolAxisAlignment:
    cosine: torch.Tensor
    angle: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True)
class OrientationRewardState:
    progress: torch.Tensor
    active: torch.Tensor


def quaternion_conjugate_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = normalize_quaternion_wxyz(quaternion)
    return torch.cat((quaternion[:, :1], -quaternion[:, 1:4]), dim=-1)


def quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = normalize_quaternion_wxyz(left)
    right = normalize_quaternion_wxyz(right)
    if left.shape != right.shape:
        raise ValueError(f"四元数 batch 形状必须一致，实际为 {tuple(left.shape)} 与 {tuple(right.shape)}。")
    lw, lv = left[:, :1], left[:, 1:4]
    rw, rv = right[:, :1], right[:, 1:4]
    scalar = lw * rw - torch.sum(lv * rv, dim=-1, keepdim=True)
    vector = lw * rv + rw * lv + torch.cross(lv, rv, dim=-1)
    return normalize_quaternion_wxyz(torch.cat((scalar, vector), dim=-1))


def expand_quaternion_wxyz(
    value: torch.Tensor | tuple[float, float, float, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    quaternion = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if quaternion.shape == (4,):
        quaternion = quaternion.reshape(1, 4).expand(reference.shape[0], 4)
    if quaternion.shape != reference.shape:
        raise ValueError(
            f"固定四元数必须为 (4,) 或 {tuple(reference.shape)}，实际为 {tuple(quaternion.shape)}。"
        )
    return normalize_quaternion_wxyz(quaternion)


def desired_flange_quaternion_wxyz(
    target_quaternion_w: torch.Tensor,
    target_to_tool_rotation_t: torch.Tensor | tuple[float, float, float, float],
    flange_to_tool_rotation_f: torch.Tensor | tuple[float, float, float, float],
) -> torch.Tensor:
    """由目标初始姿态和两个固定安装变换计算期望 Flange 世界姿态。"""
    target_quaternion_w = normalize_quaternion_wxyz(target_quaternion_w)
    target_to_tool = expand_quaternion_wxyz(target_to_tool_rotation_t, target_quaternion_w)
    flange_to_tool = expand_quaternion_wxyz(flange_to_tool_rotation_f, target_quaternion_w)
    desired_tool_w = quaternion_multiply_wxyz(target_quaternion_w, target_to_tool)
    return quaternion_multiply_wxyz(desired_tool_w, quaternion_conjugate_wxyz(flange_to_tool))


def tool_axis_alignment(
    flange_quaternion_w: torch.Tensor,
    target_quaternion_w: torch.Tensor,
    flange_to_tool_rotation_f: torch.Tensor | tuple[float, float, float, float],
    tool_forward_axis_t: torch.Tensor | tuple[float, float, float],
    target_docking_axis_t: torch.Tensor | tuple[float, float, float],
    score_sigma_rad: float,
) -> ToolAxisAlignment:
    flange_quaternion_w = normalize_quaternion_wxyz(flange_quaternion_w)
    target_quaternion_w = normalize_quaternion_wxyz(target_quaternion_w)
    flange_to_tool = expand_quaternion_wxyz(flange_to_tool_rotation_f, flange_quaternion_w)
    tool_quaternion_w = quaternion_multiply_wxyz(flange_quaternion_w, flange_to_tool)

    tool_axis = torch.as_tensor(tool_forward_axis_t, dtype=flange_quaternion_w.dtype, device=flange_quaternion_w.device)
    target_axis = torch.as_tensor(target_docking_axis_t, dtype=target_quaternion_w.dtype, device=target_quaternion_w.device)
    tool_axis = tool_axis.reshape(1, 3).expand(flange_quaternion_w.shape[0], 3)
    target_axis = target_axis.reshape(1, 3).expand(target_quaternion_w.shape[0], 3)
    tool_axis_w = rotate_vector_wxyz(tool_quaternion_w, tool_axis)
    target_axis_w = rotate_vector_wxyz(target_quaternion_w, target_axis)
    cosine = torch.clamp(torch.sum(tool_axis_w * target_axis_w, dim=-1), -1.0, 1.0)
    angle = torch.acos(cosine)
    sigma = max(float(score_sigma_rad), 1.0e-8)
    score = torch.exp(-torch.square(angle / sigma))
    return ToolAxisAlignment(cosine=cosine, angle=angle, score=score)


def latched_orientation_progress(
    distance: torch.Tensor,
    score: torch.Tensor,
    previous_score: torch.Tensor,
    previous_active: torch.Tensor,
    activation_distance: float,
) -> OrientationRewardState:
    """首次进入阈值后锁存差分奖励；进入当步只建立基准，不产生跳变奖励。"""
    active = previous_active | (distance < float(activation_distance))
    progress = torch.where(
        previous_active & torch.isfinite(previous_score),
        score - previous_score,
        torch.zeros_like(score),
    )
    return OrientationRewardState(progress=progress, active=active)
