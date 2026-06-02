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
    def penalty_ee_out_of_workspace(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        workspace: dict | None = None,
        max_episode_steps: int = 80,
        early_failure_scale: float = 0.48,
    ) -> torch.Tensor:
        """Return a larger penalty signal for earlier workspace failures."""
        if workspace is None:
            workspace = {"x": [-0.45, 0.45], "y": [-0.45, 0.45], "z": [0.05, 1.10]}

        ee_pos_b = AuboToolFns.get_body_pos_in_root_frame(env, asset_cfg, ee_frame_name)
        failed = AuboToolFns.axis_aligned_workspace_mask(ee_pos_b, workspace).float()
        return failed * AuboRewardFns._early_failure_multiplier(env, max_episode_steps, early_failure_scale)

    @staticmethod
    def penalty_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        force_threshold: float = 1e-6,
        max_episode_steps: int = 80,
        early_failure_scale: float = 0.342857,
    ) -> torch.Tensor:
        """Return a larger penalty signal for earlier collision failures."""
        failed = AuboTerminationFns._contact_done(env, sensor_cfg, force_threshold).float()
        return failed * AuboRewardFns._early_failure_multiplier(env, max_episode_steps, early_failure_scale)

    @staticmethod
    def _positive_progress(env, buffer_name: str, dist: torch.Tensor) -> torch.Tensor:
        prev_dist = AuboBufferToolFns.get_or_create(env, buffer_name, dist)
        AuboBufferToolFns.sync_just_reset(env, prev_dist, dist)
        progress = torch.clamp(prev_dist - dist, min=0.0)
        setattr(env, buffer_name, dist.clone())
        return progress

    @staticmethod
    def _early_failure_multiplier(
        env,
        max_episode_steps: int,
        early_failure_scale: float,
    ) -> torch.Tensor:
        """Scale terminal failure penalties so earlier failures are worse."""
        max_steps = max(int(max_episode_steps), 1)
        if hasattr(env, "episode_length_buf"):
            step = env.episode_length_buf.to(device=env.device, dtype=torch.float32)
        else:
            step = torch.zeros(env.num_envs, device=env.device)

        remaining_fraction = torch.clamp(1.0 - step / float(max_steps), 0.0, 1.0)
        return 1.0 + float(early_failure_scale) * remaining_fraction


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
        log_termination: bool = False,
        termination_reason: str = "goal_reached",
    ) -> torch.Tensor:
        """末端连续若干步保持在目标阈值内时终止。"""
        dist = AuboToolFns.ee_target_distance_w(env, asset_cfg, ee_frame_name, goal_pos_name)
        reached = dist < pos_threshold

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
        return AuboTerminationFns._print_done(env, done, termination_reason, log_termination)

    @staticmethod
    def ee_out_of_workspace(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        workspace: dict | None = None,
        log_termination: bool = False,
        termination_reason: str = "ee_out_of_workspace",
    ) -> torch.Tensor:
        """末端离开机器人根坐标系下的轴对齐工作空间时终止。"""
        if workspace is None:
            workspace = {"x": [-0.45, 0.45], "y": [-0.45, 0.45], "z": [0.05, 1.10]}

        ee_pos_b = AuboToolFns.get_body_pos_in_root_frame(env, asset_cfg, ee_frame_name)
        done = AuboToolFns.axis_aligned_workspace_mask(ee_pos_b, workspace)
        return AuboTerminationFns._print_done(env, done, termination_reason, log_termination)

    @staticmethod
    def is_terminated_by_illegal_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        body_names: Sequence[str] | None = None,
        force_threshold: float = 1e-6,
        log_termination: bool = False,
        termination_reason: str = "obstacle_collision",
    ) -> torch.Tensor:
        """不同传感器 schema 下的非法碰撞终止。"""
        del body_names
        done = AuboTerminationFns._contact_done(env, sensor_cfg, force_threshold)
        return AuboTerminationFns._print_done(env, done, termination_reason, log_termination)

    @staticmethod
    def is_terminated_by_self_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        force_threshold: float = 1e-6,
        log_termination: bool = False,
        termination_reason: str = "self_collision",
    ) -> torch.Tensor:
        """不同传感器 schema 下的自碰撞终止。"""
        done = AuboTerminationFns._contact_done(env, sensor_cfg, force_threshold)
        return AuboTerminationFns._print_done(env, done, termination_reason, log_termination)

    @staticmethod
    def times_out(
        env,
        log_termination: bool = False,
        termination_reason: str = "time_out",
    ) -> torch.Tensor:
        """复用 Isaac Lab 默认 timeout 终止项。"""
        done = mdp.time_out(env)
        return AuboTerminationFns._print_done(env, done, termination_reason, log_termination)

    @staticmethod
    def _print_done(
        env,
        done: torch.Tensor,
        reason: str,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled or done.numel() == 0:
            return done

        done_ids = torch.nonzero(done.detach(), as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return done

        step = AuboTerminationFns._global_step(env)
        printed = getattr(env, "_aubo_printed_termination_keys", None)
        if printed is None:
            printed = set()
            env._aubo_printed_termination_keys = printed

        episode_lengths = getattr(env, "episode_length_buf", None)
        for env_id in done_ids.detach().cpu().tolist():
            if step >= 0:
                key = (step, int(env_id), str(reason))
                if key in printed:
                    continue
                printed.add(key)
                if len(printed) > 10000:
                    printed.clear()
                    printed.add(key)

            episode_length_text = ""
            if episode_lengths is not None and int(env_id) < int(episode_lengths.numel()):
                episode_length = int(episode_lengths[int(env_id)].detach().cpu())
                episode_length_text = f" episode_length={episode_length}"

            print(
                "[TRAIN][termination] "
                f"timestep={step} "
                f"env={int(env_id)} "
                f"reason={reason}"
                f"{episode_length_text}",
                flush=True,
            )
        return done

    @staticmethod
    def _global_step(env) -> int:
        for name in ("common_step_counter", "_sim_step_counter"):
            value = getattr(env, name, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                return int(value.detach().flatten()[0].cpu())
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return -1

    @staticmethod
    def _contact_done(env, sensor_cfg: SceneEntityCfg | str, force_threshold: float) -> torch.Tensor:
        sensor = AuboContactToolFns.get_optional_sensor(env, sensor_cfg)
        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        flags = magnitude > force_threshold
        if flags.shape[0] == env.num_envs:
            if flags.ndim == 1:
                return flags
            return torch.any(flags, dim=tuple(range(1, flags.ndim)))

        if flags.shape[0] % env.num_envs == 0:
            grouped = flags.reshape(env.num_envs, -1, *flags.shape[1:])
            return torch.any(grouped, dim=tuple(range(1, grouped.ndim)))

        raise RuntimeError(
            "Contact termination returned an incompatible leading dimension: "
            f"shape={tuple(flags.shape)}, num_envs={env.num_envs}. "
            "Check the ContactSensor prim_path matches one logical robot group per environment."
        )

