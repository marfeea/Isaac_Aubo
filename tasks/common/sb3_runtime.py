from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from tasks.common.paths import TaskOutputPaths, checkpoint_stem
from tasks.common.training_callbacks import EpisodeTerminationRateCallback


PPO_HYPERPARAMS = {
    "learning_rate": 2.0e-4,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.003,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def _assert_finite_observation(observation) -> None:
    arrays = observation.values() if isinstance(observation, dict) else (observation,)
    for value in arrays:
        array = np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError("环境 reset 返回了非有限观测。")


def resolve_checkpoint_path(weight_name: str, default_dir: Path) -> Path:
    raw_path = Path(weight_name)
    candidates = (raw_path, default_dir / raw_path, (default_dir / raw_path).with_suffix(".zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 checkpoint：{weight_name!r}；默认目录为 {default_dir}。")


def train_sb3(
    env_cfg,
    *,
    task_name: str,
    device: str,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    progress_bar: bool,
    run_label: str | None = None,
) -> Path:
    """运行任务无关的 SB3 PPO 训练主循环。"""
    output_paths = TaskOutputPaths.for_task(task_name)
    output_paths.ensure_exists()
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(isaac_env, fast_variant=False)
    try:
        _assert_finite_observation(env.reset())
        rollout_size = max(int(n_steps), 2) * int(env_cfg.scene.num_envs)
        effective_batch_size = min(max(int(batch_size), 2), rollout_size)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(10_000 // max(int(env_cfg.scene.num_envs), 1), 1),
            save_path=str(output_paths.checkpoint_dir),
            name_prefix=f"ppo_{task_name}",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        termination_curve_callback = EpisodeTerminationRateCallback(
            isaac_env.termination_manager._term_names
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=max(int(n_steps), 2),
            batch_size=effective_batch_size,
            verbose=1,
            tensorboard_log=str(output_paths.log_dir),
            device=device,
            **PPO_HYPERPARAMS,
        )
        model.learn(
            total_timesteps=max(int(total_timesteps), rollout_size),
            callback=CallbackList([checkpoint_callback, termination_curve_callback]),
            progress_bar=bool(progress_bar),
            log_interval=1,
        )
        final_path = output_paths.checkpoint_dir / checkpoint_stem(task_name, run_label)
        model.save(str(final_path))
        return final_path.with_suffix(".zip")
    finally:
        env.close()


def evaluate_sb3(
    env_cfg,
    *,
    task_name: str,
    device: str,
    weight_name: str,
    episodes: int,
    max_steps: int | None,
    deterministic: bool,
    simulation_app,
) -> dict[str, object]:
    """运行任务无关的 SB3 checkpoint 回放。"""
    output_paths = TaskOutputPaths.for_task(task_name)
    model_path = resolve_checkpoint_path(weight_name, output_paths.checkpoint_dir)
    isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(isaac_env, fast_variant=False)
    try:
        model = PPO.load(str(model_path), env=env, device=device)
        observation = env.reset()
        _assert_finite_observation(observation)
        episode_returns = np.zeros(env_cfg.scene.num_envs, dtype=np.float64)
        completed_returns: list[float] = []
        terminated_count = 0
        truncated_count = 0
        steps = 0
        while len(completed_returns) < int(episodes) and simulation_app.is_running():
            if max_steps is not None and steps >= int(max_steps):
                break
            actions, _ = model.predict(observation, deterministic=deterministic)
            observation, rewards, dones, infos = env.step(actions)
            _assert_finite_observation(observation)
            if not np.isfinite(np.asarray(rewards)).all():
                raise RuntimeError("环境 step 返回了非有限奖励。")
            episode_returns += np.asarray(rewards, dtype=np.float64)
            steps += 1
            for env_id, done in enumerate(dones):
                if bool(done):
                    if bool(infos[env_id].get("TimeLimit.truncated", False)):
                        truncated_count += 1
                    else:
                        terminated_count += 1
                    completed_returns.append(float(episode_returns[env_id]))
                    episode_returns[env_id] = 0.0
                    if len(completed_returns) >= int(episodes):
                        break
        result = {
            "checkpoint": str(model_path),
            "episodes": len(completed_returns),
            "steps": steps,
            "terminated": terminated_count,
            "truncated": truncated_count,
            "reward_terms": tuple(isaac_env.reward_manager._term_names),
            "termination_terms": tuple(isaac_env.termination_manager._term_names),
            "mean_return": (
                float(np.mean(completed_returns)) if completed_returns else None
            ),
        }
        print(f"[EVAL][{task_name}] {result}", flush=True)
        return result
    finally:
        env.close()
