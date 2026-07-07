from __future__ import annotations

import torch

from tasks.WithClaw.mdp_logic import update_parking_state
from tasks.WithClaw.orientation import (
    desired_flange_quaternion_wxyz,
    expand_quaternion_wxyz,
    latched_orientation_progress,
    tool_axis_alignment,
)
from tasks.WithClaw.task_cfg import (
    FLANGE_TO_TOOL_ROTATION_F,
    REWARD_WEIGHTS,
    TARGET_DOCKING_AXIS_T,
    TARGET_INITIAL_STATES,
    TARGET_TO_TOOL_ROTATION_T,
    TOOL_FORWARD_AXIS_T,
    TOOL_ORIENTATION_REWARD_START_DISTANCE,
    TOOL_ORIENTATION_SCORE_SIGMA_RAD,
)
from tasks.WithClaw.tcp import rotate_vector_wxyz


def test_flange_to_tool_rotation_maps_tool_positive_z_to_flange_negative_x() -> None:
    reference = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rotation = expand_quaternion_wxyz(FLANGE_TO_TOOL_ROTATION_F, reference)
    tool_positive_z = torch.tensor([[0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        rotate_vector_wxyz(rotation, tool_positive_z),
        torch.tensor([[-1.0, 0.0, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_each_target_state_produces_an_exact_tool_axis_match() -> None:
    target_quaternion = torch.tensor([state.rot for state in TARGET_INITIAL_STATES])
    desired_flange = desired_flange_quaternion_wxyz(
        target_quaternion,
        TARGET_TO_TOOL_ROTATION_T,
        FLANGE_TO_TOOL_ROTATION_F,
    )
    alignment = tool_axis_alignment(
        desired_flange,
        target_quaternion,
        FLANGE_TO_TOOL_ROTATION_F,
        TOOL_FORWARD_AXIS_T,
        TARGET_DOCKING_AXIS_T,
        TOOL_ORIENTATION_SCORE_SIGMA_RAD,
    )
    torch.testing.assert_close(alignment.cosine, torch.ones_like(alignment.cosine), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(alignment.score, torch.ones_like(alignment.score), atol=1.0e-6, rtol=0.0)


def test_latched_progress_starts_without_entry_bonus_and_preserves_signed_deltas() -> None:
    inactive = torch.tensor([False])
    first = latched_orientation_progress(
        torch.tensor([0.09]),
        torch.tensor([0.40]),
        torch.tensor([0.20]),
        inactive,
        TOOL_ORIENTATION_REWARD_START_DISTANCE,
    )
    assert first.active.item()
    assert first.progress.item() == 0.0

    improved = latched_orientation_progress(
        torch.tensor([0.08]), torch.tensor([0.70]), torch.tensor([0.40]), first.active, 0.10
    )
    worsened = latched_orientation_progress(
        torch.tensor([0.20]), torch.tensor([0.50]), torch.tensor([0.70]), improved.active, 0.10
    )
    torch.testing.assert_close(improved.progress, torch.tensor([0.30]))
    torch.testing.assert_close(worsened.progress, torch.tensor([-0.20]))
    assert worsened.active.item()


def test_orientation_reward_has_five_point_net_cap_after_time_scaling() -> None:
    step_dt = 0.25
    maximum_net_progress = 1.0
    assert REWARD_WEIGHTS["tool_axis_progress"] * step_dt * maximum_net_progress == 5.0


def test_parking_dwell_requires_orientation_match() -> None:
    state = update_parking_state(
        torch.tensor([0.01, 0.01]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([True, True]),
        torch.tensor([1, 1]),
        enter_distance=0.03,
        exit_distance=0.045,
        speed_threshold=0.02,
        orientation_matched=torch.tensor([True, False]),
    )
    assert state.dwell_steps.tolist() == [2, 0]
