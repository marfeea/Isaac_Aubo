import argparse
from pathlib import Path

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


parser = argparse.ArgumentParser(description="Evaluate SB3 Aubo policy.")
parser.add_argument("--weight", type=str, required=True, help="Weight name or full checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for evaluation.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from configs.RLcfg import AuboRLEnvCfg


def main() -> None:
    ckpt_dir = Path("./checkpoints/sb3_aubo")
    model_path = resolve_checkpoint_path(args_cli.weight, ckpt_dir)

    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    model = PPO.load(str(model_path), env=env, device=args_cli.device, print_system_info=True)

    obs = env.reset()
    ep_returns = [0.0 for _ in range(args_cli.num_envs)]
    ep_lengths = [0 for _ in range(args_cli.num_envs)]

    finished_returns: list[float] = []
    finished_lengths: list[int] = []

    while len(finished_returns) < args_cli.episodes and simulation_app.is_running():
        actions, _ = model.predict(obs, deterministic=args_cli.deterministic)
        obs, rewards, dones, infos = env.step(actions)

        for i in range(args_cli.num_envs):
            ep_returns[i] += float(rewards[i])
            ep_lengths[i] += 1

            if bool(dones[i]):
                finished_returns.append(ep_returns[i])
                finished_lengths.append(ep_lengths[i])
                idx = len(finished_returns)
                print(
                    f"[Eval] Episode {idx:02d}: return={ep_returns[i]:.4f}, length={ep_lengths[i]}"
                )
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

                if len(finished_returns) >= args_cli.episodes:
                    break

    if len(finished_returns) == 0:
        print("[Eval] No completed episodes collected.")
    else:
        mean_return = sum(finished_returns) / len(finished_returns)
        mean_length = sum(finished_lengths) / len(finished_lengths)
        print("-" * 80)
        print(f"[Eval] Checkpoint: {model_path}")
        print(f"[Eval] Episodes: {len(finished_returns)}")
        print(f"[Eval] Mean Return: {mean_return:.4f}")
        print(f"[Eval] Mean Length: {mean_length:.2f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
