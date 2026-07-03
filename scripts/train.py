# aubo机械臂教师策略训练脚本

import argparse
from pathlib import Path
import time
import traceback

import _bootstrap  # noqa: F401
from isaaclab.app import AppLauncher

# 1) 先解析命令行
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument(
    "--progress_bar",
    action="store_true",
    help="Enable SB3 rich/tqdm progress bar. Disabled by default to avoid optional dependency startup failures.",
)
parser.add_argument(
    "--target_asset_name",
    type=str,
    default=None,
    help="Scene key of the named object used as the RL reaching target.",
)
parser.add_argument(
    "--enable_camera_sensor",
    action="store_true",
    help="Keep the CameraSensor in the training scene. Disabled by default because training uses state observations.",
)
parser.add_argument(
    "--skip_reset_scene_event",
    action="store_true",
    help="Disable the broad mdp.reset_scene_to_default reset event; useful for isolating first-reset stalls.",
)
parser.add_argument(
    "--log_reward_breakdown",
    action="store_true",
    help="Print reward-component means at the end of each PPO rollout.",
)
parser.add_argument(
    "--log_collisions",
    action="store_true",
    help="Print collider prim paths when PhysX reports a new contact during training.",
)
parser.add_argument(
    "--log_terminations",
    action="store_true",
    help="Print env index and reason whenever a training environment terminates.",
)
parser.add_argument(
    "--log_ik_steps",
    action="store_true",
    help="逐动作打印 IK 计划位移、实际位移和执行完成度。",
)
parser.add_argument(
    "--n_steps",
    type=int,
    default=1024,
    help="PPO rollout steps per environment before each update.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="PPO minibatch size.",
)
parser.add_argument(
    "--interactive_viewer",
    action="store_true",
    help="Periodically pump the Isaac Sim UI during training. Slower, but useful when debugging with the viewport open.",
)
parser.add_argument(
    "--viewer_update_interval",
    type=int,
    default=10,
    help="When --interactive_viewer is enabled, pump the UI every N SB3 callback calls.",
)
parser.add_argument(
    "--viewer_yield_seconds",
    type=float,
    default=0.001,
    help="When --interactive_viewer is enabled, yield this many seconds after pumping the UI.",
)

# 让 AppLauncher 也能接管它需要的参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2) 先启动 SimulationApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3) 必须在 SimulationApp 启动后，再 import 这些
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from configs.RLcfg import (
    AuboRLEnvCfg,
    DEFAULT_RL_TARGET_ASSET_NAME,
    configure_task_target,
)
from configs.asset import ROBOT_ASSET_NAME
from configs.camera_cfg import CAMERA_SENSOR_SCENE_NAMES
from configs.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
)
from tools.contact import AuboContactToolFns, PhysxContactPairPrinter, enable_physx_contact_reports


PPO_HYPERPARAMS = {
    "learning_rate": 2e-4,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.003,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


class EpisodeRewardBreakdownCallback(BaseCallback):
    """Log and optionally print rollout reward composition from IsaacLab episode extras."""

    REWARD_TERMS = (
        "ee_progress",
        "ee_distance_exp",
        "success",
        "action_far_near",
        "step_penalty",
        "action_l2",
        "action_rate_l2",
        "out_of_workspace_penalty",
        "target_contact_penalty",
        "collision_penalty",
    )
    TERMINATION_TERMS = ("goal_reached", "time_out", "obstacle_collision", "self_collision", "ee_out_of_workspace")

    def __init__(self, print_enabled: bool = False):
        super().__init__()
        self.print_enabled = bool(print_enabled)
        self._rollout_index = 0
        self._reset_rollout_stats()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        self._collect_from_payload(infos, count_episode=True)
        base_env = self._get_base_env()
        extras = getattr(base_env, "extras", None)
        if extras is not None:
            self._collect_from_payload(extras, count_episode=False)

        return True

    def _on_rollout_end(self) -> None:
        self._rollout_index += 1
        reward_parts = {
            term: self._mean(values)
            for term, values in self._reward_buckets.items()
            if values
        }
        termination_parts = {
            term: self._mean(values)
            for term, values in self._termination_buckets.items()
            if values
        }
        for key, value in reward_parts.items():
            self.logger.record(f"custom/reward_{key}_mean", value)
        for key, value in termination_parts.items():
            self.logger.record(f"custom/termination_{key}_rate", value)

        if self.print_enabled:
            if reward_parts:
                reward_text = ", ".join(f"{key}={value:.4f}" for key, value in reward_parts.items())
            else:
                reward_text = "none"
            if termination_parts:
                termination_text = ", ".join(f"{key}={value:.3f}" for key, value in termination_parts.items())
            else:
                termination_text = "none"
            print(
                "[TRAIN][reward_breakdown] "
                f"rollout={self._rollout_index} "
                f"timestep={self.num_timesteps} "
                f"completed_episodes={self._completed_episodes} "
                f"rewards={{ {reward_text} }} "
                f"terminations={{ {termination_text} }}",
                flush=True,
            )
        self._reset_rollout_stats()

    def _reset_rollout_stats(self) -> None:
        self._completed_episodes = 0
        self._reward_buckets: dict[str, list[float]] = {term: [] for term in self.REWARD_TERMS}
        self._termination_buckets: dict[str, list[float]] = {term: [] for term in self.TERMINATION_TERMS}

    def _collect_from_payload(self, payload, *, count_episode: bool) -> None:
        if isinstance(payload, dict):
            self._collect_metric_keys(payload)
            if count_episode and isinstance(payload.get("episode"), dict):
                self._completed_episodes += 1
            for nested_key in ("episode", "log", "extras", "final_info"):
                nested = payload.get(nested_key)
                if nested is not None and nested is not payload:
                    self._collect_from_payload(nested, count_episode=count_episode and nested_key == "episode")
        elif isinstance(payload, (list, tuple)):
            for item in payload:
                self._collect_from_payload(item, count_episode=count_episode)

    def _collect_metric_keys(self, payload: dict) -> None:
        for term in self.REWARD_TERMS:
            self._append_metric(payload, f"Episode_Reward/{term}", self._reward_buckets[term])
        for term in self.TERMINATION_TERMS:
            self._append_metric(payload, f"Episode_Termination/{term}", self._termination_buckets[term])

    def _append_metric(self, payload: dict, key: str, bucket: list[float]) -> None:
        if key not in payload:
            return
        value = self._as_float(payload[key])
        if value is not None:
            bucket.append(value)

    def _get_base_env(self):
        env = self.training_env
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            if hasattr(env, "extras"):
                return env
            next_env = getattr(env, "env", None)
            if next_env is None:
                next_env = getattr(env, "unwrapped", None)
            if next_env is env:
                break
            env = next_env
        return self.training_env

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _as_float(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu())
        if isinstance(value, (int, float, bool)):
            return float(value)
        if isinstance(value, (list, tuple)):
            numbers = [EpisodeRewardBreakdownCallback._as_float(item) for item in value]
            numbers = [item for item in numbers if item is not None]
            if not numbers:
                return None
            return sum(numbers) / len(numbers)
        return None


class IKDiagnosticsCallback(BaseCallback):
    """按 rollout 记录 IK 收敛率、末端距离和分段朝向区间占比。"""

    def __init__(self, action_term, *, print_steps: bool = False):
        super().__init__()
        self.action_term = action_term
        self.print_steps = bool(print_steps)
        self._reset_stats()

    def _on_step(self) -> bool:
        if self.print_steps:
            self._print_step_execution()

        distance = self.action_term.ee_goal_distance.detach()
        finite = torch.isfinite(distance)
        if not torch.any(finite):
            return True

        distance = distance[finite]
        success = self.action_term.ik_success.detach()[finite]
        start = float(self.action_term.cfg.orientation_blend_start_distance)
        lock = float(self.action_term.cfg.orientation_lock_distance)

        self._sample_count += int(distance.numel())
        self._success_count += int(success.count_nonzero().cpu())
        self._distance_sum += float(distance.sum().cpu())
        self._distance_min = min(self._distance_min, float(distance.min().cpu()))
        self._far_count += int((distance >= start).count_nonzero().cpu())
        self._blend_count += int(((distance < start) & (distance > lock)).count_nonzero().cpu())
        self._near_count += int((distance <= lock).count_nonzero().cpu())
        return True

    def _on_rollout_end(self) -> None:
        if self._sample_count == 0:
            return
        denominator = float(self._sample_count)
        self.logger.record("custom/ik_success_rate", self._success_count / denominator)
        self.logger.record("custom/ee_goal_distance_mean", self._distance_sum / denominator)
        self.logger.record("custom/ee_goal_distance_min", self._distance_min)
        self.logger.record("custom/orientation_far_rate", self._far_count / denominator)
        self.logger.record("custom/orientation_blend_rate", self._blend_count / denominator)
        self.logger.record("custom/orientation_near_rate", self._near_count / denominator)
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._sample_count = 0
        self._success_count = 0
        self._distance_sum = 0.0
        self._distance_min = float("inf")
        self._far_count = 0
        self._blend_count = 0
        self._near_count = 0

    def _print_step_execution(self) -> None:
        planned, actual, completion, valid = self.action_term.get_step_execution_diagnostics()
        planned = planned.detach().cpu()
        actual = actual.detach().cpu()
        completion = completion.detach().cpu()
        valid = valid.detach().cpu()

        for env_id in torch.nonzero(valid, as_tuple=False).flatten().tolist():
            planned_text = self._format_vector(planned[env_id])
            actual_text = self._format_vector(actual[env_id])
            print(
                "[训练][IK动作] "
                f"动作步={self.n_calls} "
                f"全局步数={self.num_timesteps} "
                f"环境编号={env_id} "
                f"IK计划增量位移（米）={planned_text} "
                f"实际执行增量位移（米）={actual_text} "
                f"执行完成度={float(completion[env_id]):.2f}%",
                flush=True,
            )

    @staticmethod
    def _format_vector(vector: torch.Tensor) -> str:
        return f"({float(vector[0]):+.6f}, {float(vector[1]):+.6f}, {float(vector[2]):+.6f})"


class ContactReportStepCallback(BaseCallback):
    """Keep PhysX contact-report prints aligned with SB3 timesteps."""

    def __init__(self, contact_printer: PhysxContactPairPrinter):
        super().__init__()
        self.contact_printer = contact_printer

    def _on_step(self) -> bool:
        self.contact_printer.set_step(self.num_timesteps)
        return True


class ContactSensorCollisionCallback(BaseCallback):
    """Print robot contact sensor hits when PhysX pair reports are unavailable."""

    def __init__(
        self,
        isaac_env,
        *,
        sensor_name: str = ROBOT_CONTACT_SENSOR_NAME,
        force_threshold: float = ROBOT_CONTACT_FORCE_THRESHOLD,
        ignored_body_names: tuple[str, ...] = ROBOT_IGNORED_CONTACT_BODY_NAMES,
    ):
        super().__init__()
        self.isaac_env = isaac_env
        self.sensor_name = sensor_name
        self.force_threshold = float(force_threshold)
        self.ignored_body_names = tuple(ignored_body_names)
        self._active_envs: set[int] = set()
        self._missing_reported = False
        self._empty_reported = False

    def _on_step(self) -> bool:
        try:
            sensor = self.isaac_env.scene[self.sensor_name]
        except Exception:
            if not self._missing_reported:
                self._missing_reported = True
                print(f"[TRAIN][collision_sensor] sensor={self.sensor_name} missing", flush=True)
            return True

        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            if not self._empty_reported:
                self._empty_reported = True
                print(f"[TRAIN][collision_sensor] sensor={self.sensor_name} has no readable force tensor", flush=True)
            return True

        env_magnitude = self._reshape_contact_magnitude(magnitude, self.isaac_env.num_envs)
        if env_magnitude.numel() == 0:
            return True

        max_force, flat_ids = torch.max(env_magnitude, dim=1)
        current_envs = set(torch.nonzero(max_force > self.force_threshold, as_tuple=False).squeeze(-1).detach().cpu().tolist())
        episode_lengths = getattr(self.isaac_env, "episode_length_buf", None)
        reset_envs = self._reset_envs(episode_lengths, current_envs)
        new_envs = (current_envs - self._active_envs) | reset_envs
        self._active_envs = current_envs
        body_names = self._body_names(sensor)
        for env_id in sorted(new_envs):
            flat_index = int(flat_ids[env_id].detach().cpu())
            body_name = body_names[flat_index] if 0 <= flat_index < len(body_names) else f"flat_index={flat_index}"
            illegal_body, illegal_force = self._max_non_ignored_body(
                env_magnitude[env_id],
                body_names,
                self.ignored_body_names,
            )
            episode_length_text = self._episode_length_text(episode_lengths, env_id)
            print(
                "[TRAIN][collision_sensor] "
                f"timestep={self.num_timesteps} "
                f"env={env_id} "
                f"robot_body={body_name} "
                f"max_force={float(max_force[env_id].detach().cpu()):.6f} "
                f"non_ignored_body={illegal_body} "
                f"non_ignored_force={illegal_force:.6f} "
                f"top_bodies={self._top_body_text(env_magnitude[env_id], body_names)} "
                f"raw_shape={tuple(magnitude.shape)}"
                f"{episode_length_text}",
                flush=True,
            )
        return True

    @staticmethod
    def _reshape_contact_magnitude(magnitude: torch.Tensor, num_envs: int) -> torch.Tensor:
        if magnitude.ndim == 0:
            return magnitude.reshape(1, 1)
        if magnitude.shape[0] == num_envs:
            return magnitude.reshape(num_envs, -1)
        if magnitude.shape[0] % num_envs == 0:
            return magnitude.reshape(num_envs, -1)
        return magnitude.reshape(1, -1)

    @staticmethod
    def _reset_envs(episode_lengths, current_envs: set[int]) -> set[int]:
        if episode_lengths is None:
            return set()
        reset_envs: set[int] = set()
        for env_id in current_envs:
            if env_id >= int(episode_lengths.numel()):
                continue
            episode_length = int(episode_lengths[env_id].detach().cpu())
            if episode_length <= 1:
                reset_envs.add(env_id)
        return reset_envs

    @staticmethod
    def _episode_length_text(episode_lengths, env_id: int) -> str:
        if episode_lengths is None or env_id >= int(episode_lengths.numel()):
            return ""
        episode_length = int(episode_lengths[env_id].detach().cpu())
        return f" episode_length={episode_length}"

    @staticmethod
    def _max_non_ignored_body(
        env_magnitude: torch.Tensor,
        body_names: list[str],
        ignored_body_names: tuple[str, ...],
    ) -> tuple[str, float]:
        ignored = set(ignored_body_names)
        best_name = "none"
        best_force = 0.0
        for index, value in enumerate(env_magnitude.detach().flatten().cpu().tolist()):
            body_name = body_names[index] if 0 <= index < len(body_names) else f"flat_index={index}"
            if body_name in ignored:
                continue
            force = float(value)
            if force > best_force:
                best_name = body_name
                best_force = force
        return best_name, best_force

    @staticmethod
    def _top_body_text(env_magnitude: torch.Tensor, body_names: list[str], count: int = 3) -> str:
        values = env_magnitude.detach().flatten()
        if values.numel() == 0:
            return "none"
        top_count = min(int(count), int(values.numel()))
        forces, ids = torch.topk(values, k=top_count)
        parts = []
        for force, body_id in zip(forces.cpu().tolist(), ids.cpu().tolist()):
            body_name = body_names[int(body_id)] if 0 <= int(body_id) < len(body_names) else f"flat_index={int(body_id)}"
            parts.append(f"{body_name}:{float(force):.6f}")
        return ",".join(parts)

    @staticmethod
    def _body_names(sensor) -> list[str]:
        for owner in (sensor, getattr(sensor, "data", None)):
            names = getattr(owner, "body_names", None)
            if names:
                return list(names)
        return []


class ViewerPumpCallback(BaseCallback):
    """Give the Isaac Sim GUI extra chances to process viewport input while SB3 is learning."""

    def __init__(self, simulation_app, every_calls: int = 10, yield_seconds: float = 0.001):
        super().__init__()
        self.simulation_app = simulation_app
        self.every_calls = max(int(every_calls), 1)
        self.yield_seconds = max(float(yield_seconds), 0.0)

    def _on_step(self) -> bool:
        if self.n_calls % self.every_calls == 0:
            self.simulation_app.update()
            if self.yield_seconds > 0.0:
                time.sleep(self.yield_seconds)
        return True


def print_training_summary(env_cfg: AuboRLEnvCfg, target_asset_name: str) -> None:
    """Print the one startup log block required for a training run."""
    action_cfg = env_cfg.actions.task_space_ik
    hyperparams = {
        "target_asset_name": target_asset_name,
        "num_envs": env_cfg.scene.num_envs,
        "total_timesteps": args_cli.total_timesteps,
        "device": env_cfg.sim.device,
        "n_steps": args_cli.n_steps,
        "rollout_timesteps": args_cli.n_steps * env_cfg.scene.num_envs,
        "batch_size": args_cli.batch_size,
        "orientation_blend_start_distance": action_cfg.orientation_blend_start_distance,
        "orientation_lock_distance": action_cfg.orientation_lock_distance,
        "orientation_blend_tolerance": action_cfg.orientation_blend_tolerance,
        "max_orientation_step": action_cfg.max_orientation_step,
        "position_scale": action_cfg.pos_scale,
        "max_position_delta": action_cfg.max_position_delta,
        **PPO_HYPERPARAMS,
        "progress_bar": args_cli.progress_bar,
        "camera_sensor": args_cli.enable_camera_sensor,
        "skip_reset_scene_event": args_cli.skip_reset_scene_event,
        "interactive_viewer": args_cli.interactive_viewer and not getattr(args_cli, "headless", False),
        "log_reward_breakdown": args_cli.log_reward_breakdown,
        "log_collisions": args_cli.log_collisions,
        "log_terminations": args_cli.log_terminations,
        "逐动作IK日志": args_cli.log_ik_steps,
        "sb3_console_log": True,
        "sb3_log_interval": 1,
    }
    print("[TRAIN] Training configuration:", flush=True)
    for key, value in hyperparams.items():
        print(f"[TRAIN]   {key}: {value}", flush=True)


def robot_contact_filter(paths) -> bool:
    return any(ROBOT_ASSET_NAME in path.split("/") for path in paths if path)


def configure_termination_logging(env_cfg: AuboRLEnvCfg, enabled: bool) -> None:
    """Inject per-term print controls without changing the default task config."""
    for term_name in EpisodeRewardBreakdownCallback.TERMINATION_TERMS:
        term_cfg = getattr(env_cfg.terminations, term_name, None)
        if term_cfg is None:
            continue
        if getattr(term_cfg, "params", None) is None:
            term_cfg.params = {}
        term_cfg.params["log_termination"] = bool(enabled)
        term_cfg.params["termination_reason"] = term_name


def main():
    # 1) 创建环境配置
    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if not args_cli.enable_camera_sensor:
        for camera_name in CAMERA_SENSOR_SCENE_NAMES:
            setattr(env_cfg.scene, camera_name, None)
    target_asset_name = args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME
    configure_task_target(env_cfg, target_asset_name)
    configure_termination_logging(env_cfg, args_cli.log_terminations)
    if args_cli.skip_reset_scene_event:
        env_cfg.events.reset_scene = None
    print_training_summary(env_cfg, target_asset_name)

    # 训练阶段建议关闭 viewer 依赖；如果你想偶尔看画面，再单独做 eval 脚本
    # env_cfg.viewer.eye = (8.0, 0.0, 5.0)

    # 2) 创建 Isaac Lab 环境
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    contact_printer = None
    if args_cli.log_collisions:
        enable_physx_contact_reports(isaac_env.scene)
        contact_printer = PhysxContactPairPrinter(isaac_env.scene, path_filter=robot_contact_filter)

    # 3) 最后一步再包 SB3 wrapper
    # 需要记录 Episode_Reward/* 与 Episode_Termination/* 到 infos，因此关闭 fast_variant。
    env = Sb3VecEnvWrapper(isaac_env, fast_variant=False)

    # 4) checkpoint 回调
    log_dir = Path("./logs/sb3_aubo")
    ckpt_dir = Path("./checkpoints/sb3_aubo")
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,                 # 按 env.step 频率存
        save_path=str(ckpt_dir),
        name_prefix="ppo_aubo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    curve_callback = EpisodeRewardBreakdownCallback(print_enabled=args_cli.log_reward_breakdown)
    action_terms = getattr(isaac_env.action_manager, "_terms", {})
    ik_action_term = action_terms.get("task_space_ik")
    if ik_action_term is None:
        raise RuntimeError("Unable to resolve the task_space_ik action term for IK diagnostics.")
    callback_items = [
        checkpoint_callback,
        curve_callback,
        IKDiagnosticsCallback(ik_action_term, print_steps=args_cli.log_ik_steps),
    ]
    if contact_printer is not None:
        callback_items.append(ContactReportStepCallback(contact_printer))
        callback_items.append(ContactSensorCollisionCallback(isaac_env))
    if args_cli.interactive_viewer:
        if not getattr(args_cli, "headless", False):
            callback_items.append(
                ViewerPumpCallback(
                    simulation_app,
                    every_calls=args_cli.viewer_update_interval,
                    yield_seconds=args_cli.viewer_yield_seconds,
                )
            )
    callbacks = CallbackList(callback_items)

    # 5) PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_HYPERPARAMS["learning_rate"],
        n_steps=args_cli.n_steps,
        batch_size=args_cli.batch_size,
        n_epochs=PPO_HYPERPARAMS["n_epochs"],
        gamma=PPO_HYPERPARAMS["gamma"],
        gae_lambda=PPO_HYPERPARAMS["gae_lambda"],
        clip_range=PPO_HYPERPARAMS["clip_range"],
        ent_coef=PPO_HYPERPARAMS["ent_coef"],
        vf_coef=PPO_HYPERPARAMS["vf_coef"],
        max_grad_norm=PPO_HYPERPARAMS["max_grad_norm"],
        verbose=1,
        tensorboard_log=str(log_dir),
        device=args_cli.device,
    )

    # 6) 开始训练
    try:
        model.learn(
            total_timesteps=args_cli.total_timesteps,
            callback=callbacks,
            progress_bar=args_cli.progress_bar,
            log_interval=1,
        )

        # 7) 保存最终模型
        model.save(str(ckpt_dir / "ppo_aubo_final"))
    finally:
        if contact_printer is not None:
            contact_printer.close()
        # 8) 关闭环境
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[TRAIN][fatal] Unhandled exception during training:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
