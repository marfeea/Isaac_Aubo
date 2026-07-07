from __future__ import annotations

import torch

from tasks.WithClaw.mdp_logic import ParkingState, update_parking_state_once
from tasks.WithClaw.orientation import ToolAxisAlignment, tool_axis_alignment
from tasks.WithClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    EE_BODY_NAME,
    FLANGE_TO_TOOL_ROTATION_F,
    FLANGE_TO_TCP_TRANSLATION_F,
    ROBOT_ASSET_NAME,
    TCP_PARKING_ENTER_DISTANCE,
    TCP_PARKING_EXIT_DISTANCE,
    TCP_PARKING_SPEED_THRESHOLD,
    TARGET_INITIAL_STATES,
    TARGET_DOCKING_AXIS_T,
    TOOL_FORWARD_AXIS_T,
    TOOL_ORIENTATION_MATCH_COS,
    TOOL_ORIENTATION_SCORE_SIGMA_RAD,
)
from tasks.WithClaw.tcp import TcpKinematics, read_tcp_kinematics, rotate_vector_inverse_wxyz
from tools.contact import AuboContactToolFns
from tools.scene import AuboToolFns


def tcp_kinematics(env) -> TcpKinematics:
    robot = AuboToolFns.get_asset(env, ROBOT_ASSET_NAME)
    body_id = AuboToolFns.get_first_body_id(robot, EE_BODY_NAME)
    return read_tcp_kinematics(robot, body_id, FLANGE_TO_TCP_TRANSLATION_F)


def preposition_b(env) -> torch.Tensor:
    preposition_w = getattr(env, "preposition_w", None)
    if not isinstance(preposition_w, torch.Tensor) or preposition_w.shape != (env.num_envs, 3):
        # ObservationManager 会在首次 reset 事件前调用一次函数以推断维度。
        local = torch.tensor(TARGET_INITIAL_STATES[0].preposition, dtype=torch.float32, device=env.device)
        preposition_w = local.reshape(1, 3).expand(env.num_envs, 3) + env.scene.env_origins
    robot = AuboToolFns.get_asset(env, ROBOT_ASSET_NAME)
    return rotate_vector_inverse_wxyz(
        robot.data.root_pose_w[:, 3:7],
        preposition_w - robot.data.root_pose_w[:, :3],
    )


def tcp_distance_and_speed(env) -> tuple[torch.Tensor, torch.Tensor]:
    kinematics = tcp_kinematics(env)
    distance = torch.linalg.vector_norm(preposition_b(env) - kinematics.root.position_b, dim=-1)
    speed = torch.linalg.vector_norm(kinematics.root.velocity_b, dim=-1)
    return distance, speed


def tool_orientation_alignment(env) -> ToolAxisAlignment:
    target_rotation_w = getattr(env, "target_initial_rot_w", None)
    if not isinstance(target_rotation_w, torch.Tensor) or target_rotation_w.shape != (env.num_envs, 4):
        target = AuboToolFns.get_asset(env, DEFAULT_TARGET_ASSET_NAME)
        target_rotation_w = target.data.root_pose_w[:, 3:7]
    robot = AuboToolFns.get_asset(env, ROBOT_ASSET_NAME)
    body_id = AuboToolFns.get_first_body_id(robot, EE_BODY_NAME)
    flange_rotation_w = robot.data.body_pose_w[:, body_id, 3:7]
    return tool_axis_alignment(
        flange_rotation_w,
        target_rotation_w,
        FLANGE_TO_TOOL_ROTATION_F,
        TOOL_FORWARD_AXIS_T,
        TARGET_DOCKING_AXIS_T,
        TOOL_ORIENTATION_SCORE_SIGMA_RAD,
    )


def parking_state(env) -> ParkingState:
    """每个控制步只推进一次状态，供多个奖励和终止项安全复用。"""
    distance, speed = tcp_distance_and_speed(env)
    step = getattr(env, "episode_length_buf", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    previous_eval_step = getattr(env, "_tcp_parking_eval_step", None)
    if not isinstance(previous_eval_step, torch.Tensor) or previous_eval_step.shape != step.shape:
        previous_eval_step = torch.full_like(step, -1)
    previous_in_zone = getattr(env, "_tcp_parking_zone", None)
    if not isinstance(previous_in_zone, torch.Tensor) or previous_in_zone.shape != step.shape:
        previous_in_zone = torch.zeros_like(step, dtype=torch.bool)
    previous_dwell = getattr(env, "_tcp_dwell_steps", None)
    if not isinstance(previous_dwell, torch.Tensor) or previous_dwell.shape != step.shape:
        previous_dwell = torch.zeros_like(step, dtype=torch.long)
    orientation_matched = tool_orientation_alignment(env).cosine > TOOL_ORIENTATION_MATCH_COS

    state, eval_step = update_parking_state_once(
        distance,
        speed,
        previous_in_zone,
        previous_dwell,
        step,
        previous_eval_step,
        enter_distance=TCP_PARKING_ENTER_DISTANCE,
        exit_distance=TCP_PARKING_EXIT_DISTANCE,
        speed_threshold=TCP_PARKING_SPEED_THRESHOLD,
        orientation_matched=orientation_matched,
    )
    env._tcp_parking_zone = state.in_zone
    env._tcp_dwell_steps = state.dwell_steps
    env._tcp_parking_eval_step = eval_step
    return state


def max_illegal_contact_force(env, sensor_name: str, ignored_body_names: tuple[str, ...]) -> torch.Tensor:
    sensor = AuboContactToolFns.get_optional_sensor(env, sensor_name)
    magnitude = AuboContactToolFns.extract_env_contact_magnitude(sensor, env.num_envs)
    if magnitude is None:
        return torch.zeros(env.num_envs, device=env.device)
    body_names = AuboContactToolFns.body_names(sensor)
    if not body_names:
        return torch.amax(magnitude, dim=1)
    ignored = set(ignored_body_names)
    body_ids = [index for index, name in enumerate(body_names) if name not in ignored and index < magnitude.shape[1]]
    if not body_ids:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.amax(magnitude[:, body_ids], dim=1)
