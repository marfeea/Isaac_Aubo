from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="读取初始 Flange 姿态并核对已记录的 TCP 世界偏置。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    from configs.scene_cfg import AUBO_ROBOT_PLACEMENT_CFG, make_aubo_cfg
    from configs.asset import ROBOT_ASSET_NAME

    class RuntimePoseSceneCfg(InteractiveSceneCfg):
        AUBObot = make_aubo_cfg(ROBOT_ASSET_NAME, AUBO_ROBOT_PLACEMENT_CFG.world_pos_1)

    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene = InteractiveScene(RuntimePoseSceneCfg(num_envs=1, env_spacing=2.0))
    simulation.reset()
    scene.update(simulation.get_physics_dt())

    robot = scene[ROBOT_ASSET_NAME]
    body_ids = robot.find_bodies("Flange")[0]
    if len(body_ids) != 1:
        raise RuntimeError(f"期望唯一 Flange body，实际 body_ids={body_ids}")
    flange_pose_w = robot.data.body_pose_w[0, int(body_ids[0])]
    flange_quat_w = flange_pose_w[3:7].reshape(1, 4)
    recorded_flange_to_tcp_w = torch.tensor(
        [[0.0, -0.12, 0.102]],
        dtype=flange_pose_w.dtype,
        device=flange_pose_w.device,
    )
    inferred_flange_to_tcp_f = quat_apply_inverse(flange_quat_w, recorded_flange_to_tcp_w)
    reconstructed_w = quat_apply(flange_quat_w, inferred_flange_to_tcp_f)

    print(f"[TCP_CALIBRATION] flange_pos_w={flange_pose_w[:3].detach().cpu().tolist()}", flush=True)
    print(f"[TCP_CALIBRATION] flange_quat_w_wxyz={flange_quat_w[0].detach().cpu().tolist()}", flush=True)
    print(
        "[TCP_CALIBRATION] "
        f"recorded_flange_to_tcp_w={recorded_flange_to_tcp_w[0].cpu().tolist()}",
        flush=True,
    )
    print(
        "[TCP_CALIBRATION] "
        f"inferred_flange_to_tcp_f={inferred_flange_to_tcp_f[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        "[TCP_CALIBRATION] "
        f"reconstructed_world_vector={reconstructed_w[0].detach().cpu().tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
