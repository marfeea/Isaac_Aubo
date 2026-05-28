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
    "--log_interval",
    type=int,
    default=1_000,
    help="Print a plain-text training heartbeat every N SB3 timesteps. Use 0 to disable.",
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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from configs.RLcfg import AuboRLEnvCfg, DEFAULT_RL_TARGET_ASSET_NAME, configure_task_target


class EpisodeCurveCallback(BaseCallback):
    """Log custom episodic curves from IsaacLab extras to TensorBoard."""

    KEY_MAP = {
        "Episode_Termination/goal_reached": "success_rate",
        "Episode_Termination/time_out": "timeout_rate",
        "Episode_Reward/success": "reward_success_mean",
        "Episode_Reward/ee_progress": "reward_ee_progress_mean",
        "Episode_Reward/action_rate_l2": "action_rate_l2_mean",
        "Episode_Reward/step_penalty": "step_penalty_mean",
    }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos is None:
            return True

        buckets: dict[str, list[float]] = {v: [] for v in self.KEY_MAP.values()}
        for info in infos:
            episode_info = info.get("episode")
            if not isinstance(episode_info, dict):
                continue
            for src_key, dst_key in self.KEY_MAP.items():
                if src_key in episode_info:
                    buckets[dst_key].append(float(episode_info[src_key]))

        for key, values in buckets.items():
            if values:
                self.logger.record(f"custom/{key}", sum(values) / len(values))
        return True


class TrainingHeartbeatCallback(BaseCallback):
    """Print a lightweight progress line even when rich/tqdm does not repaint well."""

    def __init__(self, every_timesteps: int = 1_000):
        super().__init__()
        self.every_timesteps = max(int(every_timesteps), 0)
        self._next_timestep = self.every_timesteps
        self._printed_first_step = False

    def _on_training_start(self) -> None:
        print("[TRAIN] SB3 reset completed; rollout collection is starting.", flush=True)

    def _on_step(self) -> bool:
        if not self._printed_first_step:
            print(f"[TRAIN] First env step completed at timestep={self.num_timesteps}.", flush=True)
            self._printed_first_step = True

        if self.every_timesteps > 0 and self.num_timesteps >= self._next_timestep:
            print(f"[TRAIN] heartbeat timestep={self.num_timesteps}", flush=True)
            while self._next_timestep <= self.num_timesteps:
                self._next_timestep += self.every_timesteps
        return True


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


def wrap_env_call_diagnostics(env):
    """Print entry/exit markers around SB3 VecEnv reset and first step."""

    original_reset = env.reset
    original_step_wait = getattr(env, "step_wait", None)
    state = {"step_wait_seen": False}

    def reset_with_diagnostics(*args, **kwargs):
        start = time.perf_counter()
        print("[TRAIN][diagnostic] SB3 VecEnv reset() starting...", flush=True)
        result = original_reset(*args, **kwargs)
        print(f"[TRAIN][diagnostic] SB3 VecEnv reset() finished in {time.perf_counter() - start:.3f}s.", flush=True)
        return result

    env.reset = reset_with_diagnostics

    if original_step_wait is not None:

        def step_wait_with_diagnostics(*args, **kwargs):
            should_print = not state["step_wait_seen"]
            if should_print:
                state["step_wait_seen"] = True
                start = time.perf_counter()
                print("[TRAIN][diagnostic] SB3 VecEnv first step_wait() starting...", flush=True)
            result = original_step_wait(*args, **kwargs)
            if should_print:
                print(
                    f"[TRAIN][diagnostic] SB3 VecEnv first step_wait() finished in "
                    f"{time.perf_counter() - start:.3f}s.",
                    flush=True,
                )
            return result

        env.step_wait = step_wait_with_diagnostics

    return env


def main():
    # 1) 创建环境配置
    print("[TRAIN] Building AuboRLEnvCfg...", flush=True)
    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if not args_cli.enable_camera_sensor:
        env_cfg.scene.camera_cfg = None
        print("[TRAIN] CameraSensor disabled for training.", flush=True)
    configure_task_target(env_cfg, args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME)
    if args_cli.skip_reset_scene_event:
        env_cfg.events.reset_scene = None
        print("[TRAIN] Broad reset_scene event disabled for diagnostics.", flush=True)

    # 训练阶段建议关闭 viewer 依赖；如果你想偶尔看画面，再单独做 eval 脚本
    # env_cfg.viewer.eye = (8.0, 0.0, 5.0)

    # 2) 创建 Isaac Lab 环境
    print(f"[TRAIN] Creating ManagerBasedRLEnv num_envs={env_cfg.scene.num_envs} device={env_cfg.sim.device}...", flush=True)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3) 最后一步再包 SB3 wrapper
    # 需要记录 Episode_Reward/* 与 Episode_Termination/* 到 infos，因此关闭 fast_variant。
    print("[TRAIN] Wrapping env with Sb3VecEnvWrapper...", flush=True)
    env = Sb3VecEnvWrapper(env, fast_variant=False)
    env = wrap_env_call_diagnostics(env)

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

    curve_callback = EpisodeCurveCallback()
    callback_items = [
        checkpoint_callback,
        curve_callback,
        TrainingHeartbeatCallback(args_cli.log_interval),
    ]
    if args_cli.interactive_viewer:
        if getattr(args_cli, "headless", False):
            print("[TRAIN] --interactive_viewer ignored because --headless is enabled.", flush=True)
        else:
            callback_items.append(
                ViewerPumpCallback(
                    simulation_app,
                    every_calls=args_cli.viewer_update_interval,
                    yield_seconds=args_cli.viewer_yield_seconds,
                )
            )
            print(
                "[TRAIN] Interactive viewer pump enabled "
                f"interval={args_cli.viewer_update_interval} yield={args_cli.viewer_yield_seconds}s.",
                flush=True,
            )
    callbacks = CallbackList(callback_items)

    # 5) PPO
    print("[TRAIN] Creating PPO model...", flush=True)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=args_cli.n_steps,
        batch_size=args_cli.batch_size,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(log_dir),
        device=args_cli.device,
    )

    # 6) 开始训练
    print(
        f"[TRAIN] Starting learn total_timesteps={args_cli.total_timesteps} "
        f"progress_bar={args_cli.progress_bar}...",
        flush=True,
    )
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=callbacks,
        progress_bar=args_cli.progress_bar,
        log_interval=1,
    )

    # 7) 保存最终模型
    model.save(str(ckpt_dir / "ppo_aubo_final"))

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
