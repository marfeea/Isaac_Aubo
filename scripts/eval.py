import argparse
from pathlib import Path
import time
import traceback

import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher


def resolve_checkpoint_path(weight_name: str, default_dir: Path) -> Path:
    """Resolve checkpoint path from a user-provided weight name."""
    raw = Path(weight_name)
    if raw.is_file():
        return raw

    candidate = default_dir / weight_name
    if candidate.is_file():
        return candidate

    if candidate.suffix != ".zip":
        candidate_zip = candidate.with_suffix(".zip")
        if candidate_zip.is_file():
            return candidate_zip

    raise FileNotFoundError(
        f"Checkpoint not found: '{weight_name}'. Tried '{candidate}' and '{candidate.with_suffix('.zip')}'."
    )


parser = argparse.ArgumentParser(description="Evaluate SB3 Aubo policy in the train-matched scene.")
parser.add_argument("--weight", type=str, required=True, help="Weight name or full checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for evaluation.")
parser.add_argument("--episodes", type=int, default=10, help="Number of completed episodes to collect.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Maximum eval environment steps. Disabled by default; useful when episodes do not terminate.",
)
parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
parser.add_argument(
    "--target_asset_name",
    type=str,
    default=None,
    help="Scene key of the named object used as the RL reaching target.",
)
parser.add_argument(
    "--enable_camera_sensor",
    action="store_true",
    help="Keep the CameraSensor in the eval scene. Disabled by default to match training.",
)
parser.add_argument(
    "--skip_reset_scene_event",
    action="store_true",
    help="Disable the broad mdp.reset_scene_to_default reset event, matching the train diagnostic switch.",
)
parser.add_argument(
    "--log_collisions",
    action="store_true",
    help="Print robot contact sensor and PhysX contact pairs during evaluation.",
)
parser.add_argument(
    "--log_terminations",
    action="store_true",
    help="Print env index and reason whenever an eval environment terminates.",
)
parser.add_argument(
    "--no_log_ee_actions",
    action="store_true",
    help="Disable per-step end-effector and action XYZ logging in world coordinates.",
)
parser.add_argument(
    "--interactive_viewer",
    action="store_true",
    help="Periodically pump the Isaac Sim UI during evaluation.",
)
parser.add_argument(
    "--viewer_update_interval",
    type=int,
    default=1,
    help="When --interactive_viewer is enabled, pump the UI every N eval steps.",
)
parser.add_argument(
    "--viewer_yield_seconds",
    type=float,
    default=0.001,
    help="When --interactive_viewer is enabled, yield this many seconds after pumping the UI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from configs.RLcfg import AuboRLEnvCfg, DEFAULT_RL_TARGET_ASSET_NAME, configure_task_target
from configs.asset import EE_BODY_NAME, ROBOT_ASSET_NAME
from configs.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    ROBOT_IGNORED_CONTACT_BODY_NAMES,
)
from tools.contact import AuboContactToolFns, PhysxContactPairPrinter, enable_physx_contact_reports


TERMINATION_TERMS = (
    "goal_reached",
    "time_out",
    "obstacle_collision",
    "self_collision",
    "ee_out_of_workspace",
)


def format_xyz(value, precision: int = 6) -> str:
    if isinstance(value, torch.Tensor):
        values = value.detach().flatten().cpu().tolist()
    else:
        values = list(value)
    return "(" + ", ".join(f"{float(item):.{precision}f}" for item in values[:3]) + ")"


def rotate_vec_by_quat_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion in Isaac Lab's (w, x, y, z) convention."""
    quat_vec = quat[:, 1:4]
    quat_w = quat[:, 0:1]
    uv = torch.cross(quat_vec, vec, dim=-1)
    uuv = torch.cross(quat_vec, uv, dim=-1)
    return vec + 2.0 * (quat_w * uv + uuv)


class EvalEEActionWorldLogger:
    """Convert the active task-space action into world XYZ and print per-step motion."""

    def __init__(
        self,
        isaac_env,
        *,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.isaac_env = isaac_env
        self.robot_asset_name = robot_asset_name
        self.ee_body_name = ee_body_name
        self.robot = isaac_env.scene[robot_asset_name]
        body_ids = self.robot.find_bodies(ee_body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(f"Body '{ee_body_name}' not found in robot asset '{robot_asset_name}'.")
        self.body_id = int(body_ids[0])
        self.pos_scale = torch.tensor(pos_scale, dtype=torch.float32, device=isaac_env.device).view(1, 3)

    def snapshot(self, actions) -> dict[str, torch.Tensor]:
        raw_action = self._actions_tensor(actions)
        scaled_action_b = raw_action[:, 0:3] * self.pos_scale.to(device=raw_action.device, dtype=raw_action.dtype)

        root_pose_w = self.robot.data.root_pose_w
        root_quat_w = root_pose_w[:, 3:7]
        ee_pos_w = self.ee_pos_w()
        action_delta_w = rotate_vec_by_quat_wxyz(root_quat_w, scaled_action_b)

        return {
            "raw_action": raw_action.detach().clone(),
            "scaled_action_b": scaled_action_b.detach().clone(),
            "ee_before_w": ee_pos_w.detach().clone(),
            "action_delta_w": action_delta_w.detach().clone(),
            "target_pos_w": (ee_pos_w + action_delta_w).detach().clone(),
        }

    def ee_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pose_w[:, self.body_id, 0:3]

    def print_step(self, step: int, snapshot: dict[str, torch.Tensor], dones=None) -> None:
        ee_after_w = self.ee_pos_w().detach().clone()
        actual_delta_w = ee_after_w - snapshot["ee_before_w"]
        delta_error_w = actual_delta_w - snapshot["action_delta_w"]
        dones_list = self._dones_list(dones)

        for env_id in range(self.isaac_env.num_envs):
            done_text = ""
            if dones_list is not None and env_id < len(dones_list):
                done_text = f" done={bool(dones_list[env_id])}"
            print(
                "[Eval][ee_action_w] "
                f"step={step} "
                f"env={env_id} "
                f"ee_w={format_xyz(snapshot['ee_before_w'][env_id])} "
                f"action_w_xyz={format_xyz(snapshot['action_delta_w'][env_id])} "
                f"target_w={format_xyz(snapshot['target_pos_w'][env_id])} "
                f"ee_after_w={format_xyz(ee_after_w[env_id])} "
                f"actual_delta_w={format_xyz(actual_delta_w[env_id])} "
                f"delta_error_w={format_xyz(delta_error_w[env_id])} "
                f"raw_action={format_xyz(snapshot['raw_action'][env_id])} "
                f"scaled_action_base={format_xyz(snapshot['scaled_action_b'][env_id])}"
                f"{done_text}",
                flush=True,
            )

    def _actions_tensor(self, actions) -> torch.Tensor:
        tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.isaac_env.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != self.isaac_env.num_envs:
            tensor = tensor.reshape(self.isaac_env.num_envs, -1)
        return tensor

    @staticmethod
    def _dones_list(dones):
        if dones is None:
            return None
        if isinstance(dones, torch.Tensor):
            return dones.detach().cpu().tolist()
        try:
            return list(dones)
        except TypeError:
            return [dones]


class EvalContactSensorLogger:
    """Print robot contact sensor hits during the manual eval loop."""

    def __init__(
        self,
        isaac_env,
        *,
        sensor_name: str = ROBOT_CONTACT_SENSOR_NAME,
        force_threshold: float = ROBOT_CONTACT_FORCE_THRESHOLD,
        ignored_body_names: tuple[str, ...] = ROBOT_IGNORED_CONTACT_BODY_NAMES,
    ):
        self.isaac_env = isaac_env
        self.sensor_name = sensor_name
        self.force_threshold = float(force_threshold)
        self.ignored_body_names = tuple(ignored_body_names)
        self._active_envs: set[int] = set()
        self._missing_reported = False
        self._empty_reported = False

    def step(self, step: int) -> None:
        try:
            sensor = self.isaac_env.scene[self.sensor_name]
        except Exception:
            if not self._missing_reported:
                self._missing_reported = True
                print(f"[Eval][collision_sensor] sensor={self.sensor_name} missing", flush=True)
            return

        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            if not self._empty_reported:
                self._empty_reported = True
                print(f"[Eval][collision_sensor] sensor={self.sensor_name} has no readable force tensor", flush=True)
            return

        env_magnitude = self._reshape_contact_magnitude(magnitude, self.isaac_env.num_envs)
        if env_magnitude.numel() == 0:
            return

        max_force, flat_ids = torch.max(env_magnitude, dim=1)
        current_envs = set(
            torch.nonzero(max_force > self.force_threshold, as_tuple=False).squeeze(-1).detach().cpu().tolist()
        )
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
            print(
                "[Eval][collision_sensor] "
                f"step={step} "
                f"env={env_id} "
                f"robot_body={body_name} "
                f"max_force={float(max_force[env_id].detach().cpu()):.6f} "
                f"non_ignored_body={illegal_body} "
                f"non_ignored_force={illegal_force:.6f} "
                f"top_bodies={self._top_body_text(env_magnitude[env_id], body_names)} "
                f"raw_shape={tuple(magnitude.shape)}"
                f"{self._episode_length_text(episode_lengths, env_id)}",
                flush=True,
            )

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
            if int(episode_lengths[env_id].detach().cpu()) <= 1:
                reset_envs.add(env_id)
        return reset_envs

    @staticmethod
    def _episode_length_text(episode_lengths, env_id: int) -> str:
        if episode_lengths is None or env_id >= int(episode_lengths.numel()):
            return ""
        return f" episode_length={int(episode_lengths[env_id].detach().cpu())}"

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


def robot_contact_filter(paths) -> bool:
    return any(ROBOT_ASSET_NAME in path.split("/") for path in paths if path)


def configure_termination_logging(env_cfg: AuboRLEnvCfg, enabled: bool) -> None:
    """Inject per-term print controls without changing the default task config."""
    for term_name in TERMINATION_TERMS:
        term_cfg = getattr(env_cfg.terminations, term_name, None)
        if term_cfg is None:
            continue
        if getattr(term_cfg, "params", None) is None:
            term_cfg.params = {}
        term_cfg.params["log_termination"] = bool(enabled)
        term_cfg.params["termination_reason"] = term_name


def print_eval_summary(env_cfg: AuboRLEnvCfg, model_path: Path, target_asset_name: str) -> None:
    print("[Eval] Evaluation configuration:", flush=True)
    for key, value in {
        "checkpoint": model_path,
        "target_asset_name": target_asset_name,
        "num_envs": env_cfg.scene.num_envs,
        "episodes": args_cli.episodes,
        "max_steps": args_cli.max_steps,
        "device": env_cfg.sim.device,
        "camera_sensor": args_cli.enable_camera_sensor,
        "skip_reset_scene_event": args_cli.skip_reset_scene_event,
        "deterministic": args_cli.deterministic,
        "log_collisions": args_cli.log_collisions,
        "log_terminations": args_cli.log_terminations,
        "log_ee_actions": not args_cli.no_log_ee_actions,
        "interactive_viewer": args_cli.interactive_viewer and not getattr(args_cli, "headless", False),
        "sb3_fast_variant": False,
    }.items():
        print(f"[Eval]   {key}: {value}", flush=True)


def as_float(value) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu())
    if isinstance(value, (int, float, bool)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def termination_reason_from_info(info) -> str:
    if not isinstance(info, dict):
        return "unknown"

    candidates = [info]
    for nested_key in ("episode", "final_info", "log", "extras"):
        nested = info.get(nested_key)
        if isinstance(nested, dict):
            candidates.append(nested)

    reasons = []
    for payload in candidates:
        for term in TERMINATION_TERMS:
            value = payload.get(f"Episode_Termination/{term}")
            value_float = as_float(value)
            if value_float is not None and value_float > 0.0:
                reasons.append(term)
    return ",".join(dict.fromkeys(reasons)) if reasons else "unknown"


def info_for_env(infos, env_id: int):
    if isinstance(infos, (list, tuple)):
        return infos[env_id] if env_id < len(infos) else None
    if isinstance(infos, dict):
        return infos
    return None


def pump_viewer(step: int) -> None:
    if not args_cli.interactive_viewer or getattr(args_cli, "headless", False):
        return
    interval = max(int(args_cli.viewer_update_interval), 1)
    if step % interval != 0:
        return
    simulation_app.update()
    yield_seconds = max(float(args_cli.viewer_yield_seconds), 0.0)
    if yield_seconds > 0.0:
        time.sleep(yield_seconds)


def main() -> None:
    ckpt_dir = Path("./checkpoints/sb3_aubo")
    model_path = resolve_checkpoint_path(args_cli.weight, ckpt_dir)

    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if not args_cli.enable_camera_sensor:
        env_cfg.scene.camera_cfg = None
    target_asset_name = args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME
    configure_task_target(env_cfg, target_asset_name)
    configure_termination_logging(env_cfg, args_cli.log_terminations)
    if args_cli.skip_reset_scene_event:
        env_cfg.events.reset_scene = None
    print_eval_summary(env_cfg, model_path, target_asset_name)

    env = None
    contact_printer = None
    try:
        isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
        ee_action_logger = None
        if not args_cli.no_log_ee_actions:
            ee_action_logger = EvalEEActionWorldLogger(
                isaac_env,
                robot_asset_name=env_cfg.actions.task_space_ik.asset_name,
                ee_body_name=env_cfg.actions.task_space_ik.body_name,
                pos_scale=env_cfg.actions.task_space_ik.pos_scale,
            )
        contact_sensor_logger = None
        if args_cli.log_collisions:
            enable_physx_contact_reports(isaac_env.scene)
            contact_printer = PhysxContactPairPrinter(isaac_env.scene, path_filter=robot_contact_filter)
            contact_sensor_logger = EvalContactSensorLogger(isaac_env)

        env = Sb3VecEnvWrapper(isaac_env, fast_variant=False)
        print(f"[Eval] Env observation_space: {env.observation_space}", flush=True)
        print(f"[Eval] Env action_space: {env.action_space}", flush=True)
        print(f"[Eval] Loading checkpoint from: {model_path}", flush=True)
        model = PPO.load(str(model_path), env=env, device=args_cli.device, print_system_info=True)

        print("[Eval] SB3 VecEnv reset() starting...", flush=True)
        obs = env.reset()
        print("[Eval] SB3 VecEnv reset() finished; policy rollout is starting.", flush=True)

        ep_returns = [0.0 for _ in range(args_cli.num_envs)]
        ep_lengths = [0 for _ in range(args_cli.num_envs)]

        finished_returns: list[float] = []
        finished_lengths: list[int] = []
        finished_reasons: list[str] = []
        step = 0

        while len(finished_returns) < args_cli.episodes and simulation_app.is_running():
            if args_cli.max_steps is not None and step >= int(args_cli.max_steps):
                print(f"[Eval] Reached max_steps={args_cli.max_steps}; stopping evaluation.", flush=True)
                break

            actions, _ = model.predict(obs, deterministic=args_cli.deterministic)
            ee_action_snapshot = ee_action_logger.snapshot(actions) if ee_action_logger is not None else None
            obs, rewards, dones, infos = env.step(actions)
            step += 1

            if ee_action_logger is not None and ee_action_snapshot is not None:
                ee_action_logger.print_step(step, ee_action_snapshot, dones)
            if contact_printer is not None:
                contact_printer.set_step(step)
            if contact_sensor_logger is not None:
                contact_sensor_logger.step(step)
            pump_viewer(step)

            for i in range(args_cli.num_envs):
                ep_returns[i] += float(rewards[i])
                ep_lengths[i] += 1

                if bool(dones[i]):
                    reason = termination_reason_from_info(info_for_env(infos, i))
                    finished_returns.append(ep_returns[i])
                    finished_lengths.append(ep_lengths[i])
                    finished_reasons.append(reason)
                    idx = len(finished_returns)
                    print(
                        "[Eval] "
                        f"Episode {idx:02d}: "
                        f"env={i} "
                        f"return={ep_returns[i]:.4f} "
                        f"length={ep_lengths[i]} "
                        f"reason={reason}",
                        flush=True,
                    )
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

                    if len(finished_returns) >= args_cli.episodes:
                        break

        if len(finished_returns) == 0:
            print("[Eval] No completed episodes collected.", flush=True)
        else:
            mean_return = sum(finished_returns) / len(finished_returns)
            mean_length = sum(finished_lengths) / len(finished_lengths)
            reason_counts = {reason: finished_reasons.count(reason) for reason in dict.fromkeys(finished_reasons)}
            print("-" * 80, flush=True)
            print(f"[Eval] Checkpoint: {model_path}", flush=True)
            print(f"[Eval] Episodes: {len(finished_returns)}", flush=True)
            print(f"[Eval] Mean Return: {mean_return:.4f}", flush=True)
            print(f"[Eval] Mean Length: {mean_length:.2f}", flush=True)
            print(f"[Eval] Terminations: {reason_counts}", flush=True)
    finally:
        if contact_printer is not None:
            contact_printer.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[Eval][fatal] Unhandled exception during evaluation:", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
