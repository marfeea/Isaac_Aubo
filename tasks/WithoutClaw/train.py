from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="训练无夹爪 AUBO Flange reach 任务。")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--total_timesteps", type=int, default=1_000_000)
parser.add_argument("--n_steps", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--target_asset_name", type=str, default=None)
parser.add_argument("--randomize_target_pose", action="store_true")
parser.add_argument("--progress_bar", action="store_true")
parser.add_argument("--run_label", type=str, default=None)
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
    from tasks.common.sb3_runtime import train_sb3

    env_cfg = WithoutClawEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    disable_camera_sensors(env_cfg)
    target_name = args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME
    configure_task_target(env_cfg, target_name, randomize_target_pose=args_cli.randomize_target_pose)
    final_path = train_sb3(
        env_cfg,
        task_name=TASK_NAME,
        device=args_cli.device,
        total_timesteps=args_cli.total_timesteps,
        n_steps=args_cli.n_steps,
        batch_size=args_cli.batch_size,
        progress_bar=args_cli.progress_bar,
        run_label=args_cli.run_label,
    )
    print(f"[TRAIN][{TASK_NAME}] final_checkpoint={final_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
