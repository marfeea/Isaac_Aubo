"""逐一验证 RL 目标离散初始状态的短时物理稳定性。"""

import argparse
from types import SimpleNamespace

import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="验证五个目标初始状态是否发生跳变或飞出。")
parser.add_argument("--steps", type=int, default=120, help="每个状态仿真的物理步数。")
parser.add_argument("--max_initial_error", type=float, default=0.002, help="允许的初始化位置误差，单位米。")
parser.add_argument("--max_step_displacement", type=float, default=0.02, help="允许的单步最大位移，单位米。")
parser.add_argument("--max_drift", type=float, default=0.05, help="允许的最大累计漂移，单位米。")
parser.add_argument("--max_speed", type=float, default=1.0, help="允许的最大线速度，单位米每秒。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from configs.place_cfg import (
    WORKSTATION_TABLETOP_ASSET_SPECS,
    CollisionBodyKind,
    WorkstationTabletopLoadCfg,
    install_workstation_tabletop_scene_cfgs,
)
from configs.RLcfg import DEFAULT_RL_TARGET_ASSET_NAME, TARGET_INITIAL_STATES
from tools.randomization import reset_asset_to_discrete_pose

TARGET_TEST_ASSET_SPECS = tuple(
    spec
    for spec in WORKSTATION_TABLETOP_ASSET_SPECS
    if spec.kind == CollisionBodyKind.STATIC or spec.scene_key == DEFAULT_RL_TARGET_ASSET_NAME
)


class TargetStateTestSceneCfg(InteractiveSceneCfg):
    """只加载工作站支撑件与目标，隔离机械臂和渲染资源对测试的影响。"""

    install_workstation_tabletop_scene_cfgs(
        locals(),
        WorkstationTabletopLoadCfg(specs=TARGET_TEST_ASSET_SPECS),
    )


def main() -> bool:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    scene_cfg = TargetStateTestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    env = SimpleNamespace(scene=scene, device=scene.device, num_envs=scene.num_envs)
    env_ids = torch.tensor([0], dtype=torch.long, device=scene.device)
    state_names = tuple(state.name for state in TARGET_INITIAL_STATES)
    positions = tuple(state.pos for state in TARGET_INITIAL_STATES)
    orientations = tuple(state.rot for state in TARGET_INITIAL_STATES)
    target_cfg = SceneEntityCfg(DEFAULT_RL_TARGET_ASSET_NAME)
    target = scene[DEFAULT_RL_TARGET_ASSET_NAME]
    all_stable = True

    for state in TARGET_INITIAL_STATES:
        reset_asset_to_discrete_pose(
            env,
            env_ids,
            target_cfg,
            state_names,
            positions,
            orientations,
            fixed_state_name=state.name,
        )
        expected_pos_w = torch.tensor(state.pos, device=scene.device) + scene.env_origins[0]
        initial_pos_w = target.data.root_pos_w[0, :3].clone()
        initial_error = torch.linalg.vector_norm(initial_pos_w - expected_pos_w).item()
        previous_pos_w = initial_pos_w
        max_step_displacement = 0.0
        max_drift = 0.0
        max_speed = 0.0
        finite = True

        for _ in range(args_cli.steps):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            pos_w = target.data.root_pos_w[0, :3].clone()
            speed = torch.linalg.vector_norm(target.data.root_lin_vel_w[0, :3]).item()
            finite = (
                finite
                and bool(torch.isfinite(pos_w).all())
                and bool(torch.isfinite(target.data.root_lin_vel_w).all())
            )
            max_step_displacement = max(max_step_displacement, torch.linalg.vector_norm(pos_w - previous_pos_w).item())
            max_drift = max(max_drift, torch.linalg.vector_norm(pos_w - initial_pos_w).item())
            max_speed = max(max_speed, speed)
            previous_pos_w = pos_w

        stable = (
            finite
            and initial_error <= args_cli.max_initial_error
            and max_step_displacement <= args_cli.max_step_displacement
            and max_drift <= args_cli.max_drift
            and max_speed <= args_cli.max_speed
        )
        all_stable = all_stable and stable
        print(
            f"[{'PASS' if stable else 'FAIL'}] {state.name}: "
            f"initial_error={initial_error:.6f}m, "
            f"max_step={max_step_displacement:.6f}m, "
            f"max_drift={max_drift:.6f}m, max_speed={max_speed:.6f}m/s, finite={finite}",
            flush=True,
        )

    return all_stable


if __name__ == "__main__":
    passed = main()
    simulation_app.close()
    raise SystemExit(0 if passed else 1)
