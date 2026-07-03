from __future__ import annotations

from typing import Any

import numpy as np
import torch

from configs.asset import EE_BODY_NAME
from configs.dataset_cfg import DatasetCollectionCfg
from tools.contact import AuboContactToolFns
from tools.scene import AuboToolFns


def _numpy(value: Any, env_id: int = 0) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] > env_id:
        array = array[env_id]
    return np.asarray(array)


def _list(value: Any, env_id: int = 0) -> list:
    return _numpy(value, env_id).tolist()


def _optional_tensor(data: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    for name in names:
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _object_pose_w(env, asset_name: str, env_id: int) -> dict[str, list | None]:
    asset = env.scene[asset_name]
    data = getattr(asset, "data", None)
    root_pose = _optional_tensor(data, ("root_pose_w", "root_state_w"))
    if root_pose is not None:
        return {
            "position": _list(root_pose[:, 0:3], env_id),
            "orientation_wxyz": _list(root_pose[:, 3:7], env_id),
        }

    for owner in (asset, getattr(asset, "_view", None)):
        get_world_poses = getattr(owner, "get_world_poses", None)
        if not callable(get_world_poses):
            continue
        poses = get_world_poses()
        if isinstance(poses, tuple) and len(poses) >= 2:
            return {"position": _list(poses[0], env_id), "orientation_wxyz": _list(poses[1], env_id)}

    target_position = AuboToolFns.get_root_pos_w(env, asset_name)
    return {"position": _list(target_position, env_id), "orientation_wxyz": None}


def extract_robot_observation(env, asset_name: str, env_id: int = 0) -> dict[str, Any]:
    """提取具名机器人测量值，末端位姿和速度使用世界坐标系。"""
    robot = env.scene[asset_name]
    data = robot.data
    body_id = AuboToolFns.get_first_body_id(robot, EE_BODY_NAME)
    body_pose = data.body_pose_w[:, body_id]
    body_velocity = _optional_tensor(data, ("body_vel_w",))
    joint_effort = _optional_tensor(data, ("applied_torque", "computed_torque", "joint_effort"))

    if body_velocity is not None:
        body_velocity = body_velocity[:, body_id]
        linear_velocity = _list(body_velocity[:, 0:3], env_id)
        angular_velocity = _list(body_velocity[:, 3:6], env_id)
    else:
        linear_velocity = _list(data.body_lin_vel_w[:, body_id, 0:3], env_id)
        angular = _optional_tensor(data, ("body_ang_vel_w",))
        angular_velocity = _list(angular[:, body_id, 0:3], env_id) if angular is not None else None

    return {
        "joint_names": list(getattr(robot, "joint_names", [])),
        "joint_position": _list(data.joint_pos, env_id),
        "joint_velocity": _list(data.joint_vel, env_id),
        "joint_effort": _list(joint_effort, env_id) if joint_effort is not None else None,
        "ee_position": _list(body_pose[:, 0:3], env_id),
        "ee_orientation_wxyz": _list(body_pose[:, 3:7], env_id),
        "ee_linear_velocity": linear_velocity,
        "ee_angular_velocity": angular_velocity,
    }


def extract_frame(
    env,
    cfg: DatasetCollectionCfg,
    *,
    timestamp_ns: int,
    target_asset_name: str,
    camera_refs: dict[str, dict[str, str | None]],
    env_id: int = 0,
) -> dict[str, Any]:
    robots = {
        arm_name: extract_robot_observation(env, asset_name, env_id)
        for arm_name, asset_name in cfg.robot_assets
    }
    target_pose_value = _object_pose_w(env, target_asset_name, env_id)

    cameras = {}
    for stream in cfg.camera_streams:
        references = camera_refs.get(stream.sensor_id, {})
        camera = env.scene[stream.scene_name]
        camera_data = camera.data
        position = _optional_tensor(camera_data, ("pos_w",))
        orientation = _optional_tensor(camera_data, ("quat_w_world", "quat_w_ros", "quat_w_opengl"))
        cameras[stream.sensor_id] = {
            "rgb_ref": references.get("rgb"),
            "depth_ref": references.get("depth"),
            "calibration_id": stream.calibration_id,
            "world_T_camera": {
                "position": _list(position, env_id) if position is not None else None,
                "orientation_wxyz": _list(orientation, env_id) if orientation is not None else None,
            },
        }

    return {
        "timestamp_ns": int(timestamp_ns),
        "observation": {
            "deployable": {"robots": robots, "cameras": cameras},
            "task": {"instruction_id": cfg.task_id},
        },
        "privileged": {
            "object_poses": {target_asset_name: target_pose_value},
            "object_velocities": {},
            "target_pose": target_pose_value,
            "simulator_state_ref": None,
            "semantic_segmentation_ref": None,
            "instance_segmentation_ref": None,
        },
    }


def extract_proprio_sample(env, asset_name: str, env_id: int = 0) -> dict[str, np.ndarray]:
    robot = env.scene[asset_name]
    effort = _optional_tensor(robot.data, ("applied_torque", "computed_torque", "joint_effort"))
    fields = {
        "joint_position": _numpy(robot.data.joint_pos, env_id).astype(np.float32, copy=False),
        "joint_velocity": _numpy(robot.data.joint_vel, env_id).astype(np.float32, copy=False),
    }
    if effort is not None:
        fields["joint_effort"] = _numpy(effort, env_id).astype(np.float32, copy=False)
    return fields


def extract_camera_sample(camera, modality: str, env_id: int = 0) -> dict[str, np.ndarray]:
    output = camera.data.output[modality]
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    array = np.asarray(output)
    num_envs = int(getattr(camera, "num_envs", 1))
    if array.ndim > 0 and array.shape[0] == num_envs:
        array = array[env_id]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if modality == "rgb":
        if array.dtype != np.uint8:
            raise ValueError(f"RGB 主数据必须为 uint8，收到 {array.dtype}。")
        return {"data": array}
    if modality == "distance_to_image_plane":
        depth = array.astype(np.float32, copy=False)
        return {"data": depth, "validity_mask": np.isfinite(depth) & (depth > 0.0)}
    raise ValueError(f"不支持的主数据相机 modality：{modality}")


def extract_action_pre_step(env, env_id: int = 0) -> dict[str, Any]:
    term = env.action_manager.get_term("task_space_ik")
    raw = _numpy(term.raw_actions, env_id).astype(np.float32, copy=False)
    processed = _numpy(term.processed_actions, env_id).astype(np.float32, copy=False)
    scale = np.asarray(term.cfg.pos_scale, dtype=np.float32)
    unclipped = raw[:3] * scale
    return {
        "policy_raw": raw.tolist(),
        "command": {
            "representation": "delta_position",
            "frame": "arm0/base",
            "unit": "meter",
            "normalization": {
                "normalized": False,
                "source_normalized": True,
                "source_range": [-1.0, 1.0],
                "scale": scale.tolist(),
            },
            "ee_delta_position": processed[:3].tolist(),
        },
        "controller_target": {
            "ee_representation": "absolute_pose",
            "ee_frame": "arm0/base",
            "ee_position_unit": "meter",
            "ee_orientation": "quaternion_wxyz",
            "ee_target_pose": _list(term.target_pose_b, env_id),
            "joint_representation": "absolute_position",
            "joint_unit": "rad",
            "joint_position_target": None,
        },
        "executed": {
            "ee_representation": "delta_position",
            "ee_frame": "arm0/base",
            "ee_unit": "meter",
            "ee_delta_position": None,
            "joint_representation": "delta_position",
            "joint_unit": "rad",
            "joint_delta": None,
        },
        "diagnostics": {
            "clipped": bool(not np.allclose(unclipped, processed[:3], atol=1.0e-7)),
            "ik_success": None,
            "control_latency_ms": None,
        },
    }


def complete_action_after_step(env, action: dict[str, Any], env_id: int = 0) -> dict[str, Any]:
    term = env.action_manager.get_term("task_space_ik")
    _, actual_delta, completion, valid = term.get_step_execution_diagnostics()
    action["controller_target"]["joint_position_target"] = _list(term.joint_position_target, env_id)
    action["executed"]["ee_delta_position"] = _list(actual_delta, env_id)
    action["diagnostics"]["ik_success"] = bool(_numpy(term.ik_success, env_id))
    action["diagnostics"]["execution_completion_percent"] = float(_numpy(completion, env_id))
    action["diagnostics"]["execution_valid"] = bool(_numpy(valid, env_id))
    return action


def extract_reward(env, reward_config_hash: str, discount: float, env_id: int = 0) -> dict[str, Any]:
    manager = env.reward_manager
    step_reward = getattr(manager, "_step_reward", None)
    if not isinstance(step_reward, torch.Tensor):
        raise RuntimeError("当前 IsaacLab RewardManager 未暴露逐项 step reward，无法无损记录奖励分解。")
    terms = {}
    contribution_total = 0.0
    for index, name in enumerate(manager.active_terms):
        term_cfg = manager.get_term_cfg(name)
        weight = float(term_cfg.weight)
        weighted_without_time = float(step_reward[env_id, index].detach().cpu())
        contribution = weighted_without_time * float(env.step_dt)
        raw_value = weighted_without_time / weight if weight != 0.0 else 0.0
        terms[name] = {
            "raw_value": raw_value,
            "weight": weight,
            "time_scale": float(env.step_dt),
            "contribution": contribution,
        }
        contribution_total += contribution
    actual_total = float(env.reward_buf[env_id].detach().cpu())
    if abs(contribution_total - actual_total) > 1.0e-5:
        raise RuntimeError(
            f"RewardManager 逐项贡献求和 {contribution_total} 与 reward_buf {actual_total} 不一致。"
        )
    return {
        "terms": terms,
        "total": actual_total,
        "discount": float(discount),
        "reward_config_hash": reward_config_hash,
    }


def extract_flags(env, env_id: int = 0) -> dict[str, Any]:
    manager = env.termination_manager
    active = []
    for name in manager.active_terms:
        if bool(manager.get_term(name)[env_id].detach().cpu()):
            active.append(name)
    success = "goal_reached" in active or "success" in active
    return {
        "success": success,
        "terminated": bool(env.reset_terminated[env_id].detach().cpu()),
        "truncated": bool(env.reset_time_outs[env_id].detach().cpu()),
        "invalid": False,
        "termination_reason": ",".join(active) if active else None,
    }


def extract_contact_events(
    env,
    *,
    sensor_name: str,
    timestamp_ns: int,
    force_threshold: float,
    env_id: int = 0,
) -> list[dict[str, Any]]:
    sensor = AuboContactToolFns.get_optional_sensor(env, sensor_name)
    magnitude = AuboContactToolFns.extract_env_contact_magnitude(sensor, env.num_envs)
    if magnitude is None:
        return []
    names = AuboContactToolFns.body_names(sensor)
    values = magnitude[env_id].detach().flatten().cpu().tolist()
    events = []
    for index, force in enumerate(values):
        if float(force) <= float(force_threshold):
            continue
        source = names[index] if index < len(names) else f"contact_channel_{index}"
        events.append(
            {
                "timestamp_ns": int(timestamp_ns),
                "event_type": "contact",
                "source_entity": source,
                "target_entity": "unknown",
                "payload": {"force_magnitude_n": float(force), "sensor_id": sensor_name},
            }
        )
    return events
