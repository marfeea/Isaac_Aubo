from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="验证带夹爪 AUBO TCP 停靠 checkpoint。")
parser.add_argument("--weight", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--fixed_state_name", type=str, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from tasks.WithClaw.env_cfg import WithClawEnvCfg, configure_fixed_target_state, disable_camera_sensors
    from tasks.WithClaw.task_cfg import TASK_NAME
    from tasks.common.sb3_runtime import evaluate_sb3

    env_cfg = WithClawEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    disable_camera_sensors(env_cfg)
    configure_fixed_target_state(env_cfg, args_cli.fixed_state_name)
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
