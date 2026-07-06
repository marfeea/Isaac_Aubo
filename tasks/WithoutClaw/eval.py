from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="验证无夹爪 AUBO Flange reach checkpoint。")
parser.add_argument("--weight", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--target_asset_name", type=str, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from tasks.WithoutClaw.env_cfg import (
        DEFAULT_RL_TARGET_ASSET_NAME,
        WithoutClawEnvCfg,
        configure_task_target,
        disable_camera_sensors,
    )
    from tasks.WithoutClaw.task_cfg import TASK_NAME
    from tasks.common.sb3_runtime import evaluate_sb3

    env_cfg = WithoutClawEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    disable_camera_sensors(env_cfg)
    target_name = args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME
    configure_task_target(env_cfg, target_name, randomize_target_pose=False)
    evaluate_sb3(
        env_cfg,
        task_name=TASK_NAME,
        device=args_cli.device,
        weight_name=args_cli.weight,
        episodes=args_cli.episodes,
        max_steps=args_cli.max_steps,
        deterministic=args_cli.deterministic,
        simulation_app=simulation_app,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
