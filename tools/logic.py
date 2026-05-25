from __future__ import annotations

from typing import Sequence

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

from configs.asset import EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME
from tools.contact import AuboContactToolFns
from tools.rl import AuboActionToolFns, AuboBufferToolFns
from tools.scene import AuboToolFns


class AuboEventFns:
    """AUBO reset/startup 事件逻辑。"""

    @staticmethod
    def goal_pos_w(
        env,
        env_ids: torch.Tensor | None = None,
        target_asset_name: str = TARGET_ASSET_NAME,
        goal_pos_name: str = "goal_pos_w",
    ) -> torch.Tensor:
        """缓存目标世界坐标，并返回指定 env 的目标坐标。"""
        env_ids = AuboToolFns.normalize_env_ids(env, env_ids)
        goal_pos_w = AuboToolFns.get_root_pos_w(env, target_asset_name).clone()
        setattr(env, goal_pos_name, goal_pos_w)
        return goal_pos_w[env_ids]


def reset_planning_obstacle_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_radius: float = 0.70,
    z_radius: float = 0.20,
    z_range: tuple[float, float] = (0.30, 0.70),
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
) -> None:
    """把规划障碍物 reset 到各环境局部椭球高度带上的随机位置。"""
    env_ids = AuboToolFns.normalize_env_ids(env, env_ids)
    target = AuboToolFns.get_asset(env, asset_cfg)

    target_pos_local = AuboToolFns.sample_ellipsoid_surface_band(
        n=len(env_ids),
        device=env.device,
        xy_radius=xy_radius,
        z_radius=z_radius,
        z_range=z_range,
        center=center,
    )
    target_pos_w = AuboToolFns.local_pos_to_world(env, env_ids, target_pos_local)
    target.write_root_state_to_sim(AuboToolFns.make_root_state(target_pos_w), env_ids=env_ids)


class AuboRewardFns:
    """AUBO reaching / obstacle-aware reaching 奖励逻辑。"""

    @staticmethod
    def reward_ee_distance_exp(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        std: float = 0.20,
    ) -> torch.Tensor:
        """稠密位置奖励：exp(-distance / std)。"""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        return torch.exp(-dist / std)

    @staticmethod
    def reward_ee_progress(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """末端向目标靠近的进展奖励：previous_distance - current_distance。"""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        prev_dist = AuboBufferToolFns.get_or_create(env, "_prev_ee_target_dist", dist)
        AuboBufferToolFns.sync_just_reset(env, prev_dist, dist)

        reward = prev_dist - dist
        env._prev_ee_target_dist = dist.clone()
        return reward

    @staticmethod
    def reward_success(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        threshold: float = 0.15,
        progress_ref: float = 0.015,
        action_norm_max: float = 0.75,
        action_norm_std: float = 0.50,
        action_rate_std: float = 0.75,
        w_progress: float = 0.45,
        w_action_mag: float = 0.30,
        w_action_smooth: float = 0.25,
        min_quality_score: float = 0.35,
    ) -> torch.Tensor:
        """命中阈值后，根据末段推进、动作幅值和动作平滑性调制成功奖励。"""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        progress = AuboRewardFns._positive_progress(env, "_prev_success_quality_dist", dist)

        action_norm = AuboActionToolFns.norm(env)
        action_rate_norm = AuboActionToolFns.rate_norm(env)

        progress_score = torch.clamp(progress / max(progress_ref, 1e-6), 0.0, 1.0)
        action_excess = torch.clamp(action_norm - action_norm_max, min=0.0)
        action_mag_score = torch.exp(-action_excess / max(action_norm_std, 1e-6))
        action_smooth_score = torch.exp(-action_rate_norm / max(action_rate_std, 1e-6))

        quality = (
            w_progress * progress_score
            + w_action_mag * action_mag_score
            + w_action_smooth * action_smooth_score
        )
        quality = torch.clamp(quality, 0.0, 1.0)
        quality = min_quality_score + (1.0 - min_quality_score) * quality

        return (dist < threshold).float() * quality

    @staticmethod
    def reward_action_far_near(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        far_eps: float = 0.5,
        close_eps: float = 0.2,
        w_far_move: float = 0.10,
        w_near_ineff: float = 1.2,
        delta_min_far: float = 0.02,
        delta_min_near: float = 0.008,
        near_action_norm_max: float = 0.45,
    ) -> torch.Tensor:
        """远区奖励有效大动作，近区惩罚低效大动作。"""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        progress = AuboRewardFns._positive_progress(env, "_prev_action_far_near_dist", dist)

        action = AuboActionToolFns.get_action(env)
        action_norm = torch.norm(action, dim=-1)
        max_action_norm = torch.sqrt(torch.tensor(float(action.shape[-1]), device=env.device))

        effective_far = torch.clamp(progress / max(delta_min_far, 1e-6), 0.0, 2.0)
        far_reward = w_far_move * action_norm * effective_far * (dist > far_eps).float()

        large_action = torch.clamp(
            (action_norm - near_action_norm_max) / torch.clamp(max_action_norm - near_action_norm_max, min=1e-6),
            0.0,
            1.0,
        )
        low_progress = torch.clamp((delta_min_near - progress) / max(delta_min_near, 1e-6), 0.0, 1.0)
        near_penalty = w_near_ineff * large_action * low_progress * (dist < close_eps).float()

        return far_reward - near_penalty

    @staticmethod
    def penalty_ee_obstacle_safe(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        obstacle_asset_name: str = TARGET_ASSET_NAME,
        safe_margin: float = 0.08,
    ) -> torch.Tensor:
        """末端接近障碍物中心的安全距离惩罚。"""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, obstacle_asset_name)
        return torch.square(torch.clamp(safe_margin - dist, min=0.0))

    @staticmethod
    def penalty_action_l2(env) -> torch.Tensor:
        """动作幅值惩罚。"""
        action = AuboActionToolFns.get_action(env)
        return torch.sum(torch.square(action), dim=-1)

    @staticmethod
    def penalty_action_rate_l2(env) -> torch.Tensor:
        """动作变化惩罚。"""
        action = AuboActionToolFns.get_action(env)
        return torch.sum(torch.square(AuboActionToolFns.get_action_rate(env, action)), dim=-1)

    @staticmethod
    def penalty_step(env) -> torch.Tensor:
        """每步常数惩罚项，配合负权重使用。"""
        return torch.ones(env.num_envs, device=env.device)

    @staticmethod
    def _positive_progress(env, buffer_name: str, dist: torch.Tensor) -> torch.Tensor:
        prev_dist = AuboBufferToolFns.get_or_create(env, buffer_name, dist)
        AuboBufferToolFns.sync_just_reset(env, prev_dist, dist)
        progress = torch.clamp(prev_dist - dist, min=0.0)
        setattr(env, buffer_name, dist.clone())
        return progress


class AuboTerminationFns:
    """AUBO reaching / obstacle-aware reaching 终止逻辑。"""

    @staticmethod
    def goal_reached(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        goal_pos_name: str = TARGET_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        pos_threshold: float = 0.15,
        required_consecutive_steps: int = 3,
    ) -> torch.Tensor:
        """末端连续若干步保持在目标阈值内时终止。"""
        dist = AuboToolFns.ee_target_distance_w(env, asset_cfg, ee_frame_name, goal_pos_name)
        reached = dist < pos_threshold
        AuboTerminationFns._print_hit_envs("[Test] terminate env{} true", reached)

        counter = AuboBufferToolFns.get_or_create(
            env,
            "_goal_reached_consecutive_steps",
            torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
        )
        counter = torch.where(reached, counter + 1, torch.zeros_like(counter))
        if hasattr(env, "episode_length_buf"):
            counter[env.episode_length_buf == 0] = 0
        env._goal_reached_consecutive_steps = counter

        done = counter >= int(required_consecutive_steps)
        AuboTerminationFns._print_hit_envs("[Test] terminate env{} success", done)
        return done

    @staticmethod
    def ee_out_of_workspace(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        """末端离开各环境局部轴对齐工作空间时终止。"""
        if workspace is None:
            workspace = {"x": [-0.45, 0.45], "y": [-0.45, 0.45], "z": [0.05, 1.10]}

        ee_pos_w = AuboToolFns.get_body_pos_w(env, asset_cfg, ee_frame_name)
        ee_pos_local = ee_pos_w - env.scene.env_origins
        return AuboToolFns.axis_aligned_workspace_mask(ee_pos_local, workspace)

    @staticmethod
    def is_terminated_by_illegal_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        body_names: Sequence[str] | None = None,
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """不同传感器 schema 下的非法碰撞终止。"""
        del body_names
        return AuboTerminationFns._contact_done(env, sensor_cfg, force_threshold)

    @staticmethod
    def is_terminated_by_self_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """不同传感器 schema 下的自碰撞终止。"""
        return AuboTerminationFns._contact_done(env, sensor_cfg, force_threshold)

    @staticmethod
    def times_out(env) -> torch.Tensor:
        """复用 Isaac Lab 默认 timeout 终止项。"""
        return mdp.time_out(env)

    @staticmethod
    def _contact_done(env, sensor_cfg: SceneEntityCfg | str, force_threshold: float) -> torch.Tensor:
        sensor = AuboContactToolFns.get_optional_sensor(env, sensor_cfg)
        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if magnitude.ndim == 1:
            return magnitude > force_threshold
        return torch.any(magnitude > force_threshold, dim=tuple(range(1, magnitude.ndim)))

    @staticmethod
    def _print_hit_envs(template: str, flags: torch.Tensor) -> None:
        hit_env_ids = torch.nonzero(flags, as_tuple=False).squeeze(-1)
        for env_id in hit_env_ids.detach().cpu().tolist():
            print(template.format(env_id))
