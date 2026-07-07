from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParkingState:
    in_zone: torch.Tensor
    dwell_steps: torch.Tensor


def update_parking_state(
    distance: torch.Tensor,
    speed: torch.Tensor,
    previous_in_zone: torch.Tensor,
    previous_dwell_steps: torch.Tensor,
    *,
    enter_distance: float,
    exit_distance: float,
    speed_threshold: float,
    orientation_matched: torch.Tensor | None = None,
) -> ParkingState:
    """执行带迟滞的单步停车状态转移。"""
    if not enter_distance < exit_distance:
        raise ValueError("停车区进入距离必须小于退出距离。")
    entered = previous_in_zone | (distance < float(enter_distance))
    in_zone = entered & ~(previous_in_zone & (distance > float(exit_distance)))
    low_speed = speed < float(speed_threshold)
    if orientation_matched is None:
        orientation_matched = torch.ones_like(low_speed)
    dwell_steps = torch.where(
        in_zone & low_speed & orientation_matched,
        previous_dwell_steps + 1,
        torch.zeros_like(previous_dwell_steps),
    )
    return ParkingState(in_zone=in_zone, dwell_steps=dwell_steps)


def update_parking_state_once(
    distance: torch.Tensor,
    speed: torch.Tensor,
    previous_in_zone: torch.Tensor,
    previous_dwell_steps: torch.Tensor,
    current_step: torch.Tensor,
    previous_eval_step: torch.Tensor,
    *,
    enter_distance: float,
    exit_distance: float,
    speed_threshold: float,
    orientation_matched: torch.Tensor | None = None,
) -> tuple[ParkingState, torch.Tensor]:
    """同一环境控制步内只提交一次状态转移。"""
    candidate = update_parking_state(
        distance,
        speed,
        previous_in_zone,
        previous_dwell_steps,
        enter_distance=enter_distance,
        exit_distance=exit_distance,
        speed_threshold=speed_threshold,
        orientation_matched=orientation_matched,
    )
    should_update = previous_eval_step != current_step
    state = ParkingState(
        in_zone=torch.where(should_update, candidate.in_zone, previous_in_zone),
        dwell_steps=torch.where(should_update, candidate.dwell_steps, previous_dwell_steps),
    )
    eval_step = torch.where(should_update, current_step, previous_eval_step)
    return state, eval_step


def tcp_progress(previous_distance: torch.Tensor, current_distance: torch.Tensor) -> torch.Tensor:
    return torch.where(
        torch.isfinite(previous_distance),
        previous_distance - current_distance,
        torch.zeros_like(current_distance),
    )


def tcp_proximity(distance: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.square(distance / max(float(sigma), 1.0e-8)))


def tcp_parking_reward(in_zone: torch.Tensor, speed: torch.Tensor, sigma: float) -> torch.Tensor:
    speed_score = torch.exp(-torch.square(speed / max(float(sigma), 1.0e-8)))
    return in_zone.to(dtype=speed.dtype) * speed_score


def tcp_dwell_reward(dwell_steps: torch.Tensor, required_steps: int) -> torch.Tensor:
    return torch.clamp(dwell_steps.to(dtype=torch.float32) / max(int(required_steps), 1), 0.0, 1.0)


def axis_aligned_workspace_violation(position: torch.Tensor, workspace: dict) -> torch.Tensor:
    return (
        (position[:, 0] < float(workspace["x"][0]))
        | (position[:, 0] > float(workspace["x"][1]))
        | (position[:, 1] < float(workspace["y"][0]))
        | (position[:, 1] > float(workspace["y"][1]))
        | (position[:, 2] < float(workspace["z"][0]))
        | (position[:, 2] > float(workspace["z"][1]))
    )


def target_displacement_failure(
    target_position_w: torch.Tensor,
    initial_position_w: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return torch.linalg.vector_norm(target_position_w - initial_position_w, dim=-1) > float(threshold)


def target_speed_failure(target_linear_velocity_w: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.linalg.vector_norm(target_linear_velocity_w, dim=-1) > float(threshold)


def illegal_contact_failure(max_force: torch.Tensor, threshold: float) -> torch.Tensor:
    return max_force > float(threshold)
