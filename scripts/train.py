# aubo机械臂教师策略训练脚本

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# 1) 先解析命令行
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)

# 让 AppLauncher 也能接管它需要的参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2) 先启动 SimulationApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3) 必须在 SimulationApp 启动后，再 import 这些
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from RLcfg import AuboRLEnvCfg


def main():
    # 1) 创建环境配置
    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # 训练阶段建议关闭 viewer 依赖；如果你想偶尔看画面，再单独做 eval 脚本
    # env_cfg.viewer.eye = (8.0, 0.0, 5.0)

    # 2) 创建 Isaac Lab 环境
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3) 最后一步再包 SB3 wrapper
    env = Sb3VecEnvWrapper(env)

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

    # 5) PPO
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=256,
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
    model.learn(
        total_timesteps=args_cli.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    # 7) 保存最终模型
    model.save(str(ckpt_dir / "ppo_aubo_final"))

    # 8) 关闭环境
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()