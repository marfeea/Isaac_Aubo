# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Open Isaac Sim to inspect the local AUBO asset and scene placement.

This script is intentionally lightweight:
- loads the same AUBO scene used by RLcfg.py
- prints robot/target placement, joint names, body names and flange info
- keeps a looping joint playback so the asset can be visually inspected
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect AUBO asset configuration in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene copies to spawn.")
parser.add_argument("--cycle_steps", type=int, default=240, help="Simulation steps per playback waypoint.")
parser.add_argument("--print_every", type=int, default=120, help="Print live pose info every N steps. Use 0 to disable.")
parser.add_argument("--ee_body_name", type=str, default=None, help="End-effector/flange body name to resolve.")
parser.add_argument("--list_all", action="store_true", help="Print all joint/body names without truncation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from asset import AUBO_ROBOT_USD, EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME
from RLcfg import AuboRLSceneCfg
from Testcfg import TestSceneCfg


def _to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return list(value)


def _get_named_list(asset, attr_name: str) -> list[str]:
    """Read names from either asset.data or the asset object across IsaacLab versions."""
    for owner in (getattr(asset, "data", None), asset):
        if owner is None:
            continue
        value = getattr(owner, attr_name, None)
        if value is not None:
            return [str(item) for item in _to_list(value)]
    return []


def _format_vec(tensor: torch.Tensor, precision: int = 4) -> str:
    values = tensor.detach().cpu().flatten().tolist()
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _print_names(title: str, names: list[str], list_all: bool, limit: int = 40) -> None:
    print(f"\n[INFO] {title} ({len(names)}):")
    shown = names if list_all else names[:limit]
    for index, name in enumerate(shown):
        print(f"  {index:02d}: {name}")
    if not list_all and len(names) > limit:
        print(f"  ... {len(names) - limit} more. Re-run with --list_all to print all.")


def _resolve_robot_entity(scene: InteractiveScene, ee_body_name: str) -> SceneEntityCfg:
    entity_cfg = SceneEntityCfg(
        ROBOT_ASSET_NAME,
        joint_names=["Joint.*"],
        body_names=[ee_body_name],
    )
    entity_cfg.resolve(scene)
    return entity_cfg


def print_asset_report(scene: InteractiveScene, ee_body_name: str) -> SceneEntityCfg:
    """Print one-time asset and placement diagnostics."""
    robot = scene[ROBOT_ASSET_NAME]
    target = scene[TARGET_ASSET_NAME]

    joint_names = _get_named_list(robot, "joint_names")
    body_names = _get_named_list(robot, "body_names")

    print("\n========== Isaac Asset Inspection ==========")
    print(f"[INFO] USD path           : {AUBO_ROBOT_USD}")
    print(f"[INFO] Robot scene key    : {ROBOT_ASSET_NAME}")
    print(f"[INFO] Target scene key   : {TARGET_ASSET_NAME}")
    print(f"[INFO] Requested EE body  : {ee_body_name}")
    print(f"[INFO] Num envs           : {scene.num_envs}")
    print(f"[INFO] Env origin[0]      : {_format_vec(scene.env_origins[0])}")

    print(f"[INFO] Robot root pos[0]  : {_format_vec(robot.data.root_pos_w[0, :3])}")
    print(f"[INFO] Robot root quat[0] : {_format_vec(robot.data.root_pose_w[0, 3:7])}")
    print(f"[INFO] Target pos[0]      : {_format_vec(target.data.root_pos_w[0, :3])}")

    _print_names("Joint names", joint_names, args_cli.list_all)
    _print_names("Body names", body_names, args_cli.list_all)

    try:
        entity_cfg = _resolve_robot_entity(scene, ee_body_name)
    except Exception:
        candidates = [
            name
            for name in body_names
            if any(token in name.lower() for token in ("flange", "tool", "ee", "end", "tcp"))
        ]
        if candidates:
            print("\n[WARN] Possible flange/end-effector body names:")
            for name in candidates:
                print(f"  - {name}")
        raise
    ee_body_id = int(entity_cfg.body_ids[0])
    ee_pos_w = robot.data.body_pose_w[0, ee_body_id, :3]
    ee_quat_w = robot.data.body_pose_w[0, ee_body_id, 3:7]

    print("\n[INFO] Resolved flange/end-effector:")
    print(f"  body name : {ee_body_name}")
    print(f"  body id   : {ee_body_id}")
    print(f"  joint ids : {entity_cfg.joint_ids}")
    print(f"  pos_w[0]  : {_format_vec(ee_pos_w)}")
    print(f"  quat_w[0] : {_format_vec(ee_quat_w)}")
    print("===========================================\n")

    return entity_cfg


def reset_robot_to_default(scene: InteractiveScene) -> None:
    robot = scene[ROBOT_ASSET_NAME]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()


def make_joint_waypoints(device: str, num_envs: int) -> torch.Tensor:
    """Joint playback waypoints in degrees for visual inspection."""
    waypoints_deg = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 20.0, -30.0, 40.0, 10.0],
        [25.0, 10.0, -45.0, 25.0, -20.0],
        [-25.0, 30.0, -20.0, 55.0, 20.0],
    ]
    waypoints_rad = [[math.radians(value) for value in row] for row in waypoints_deg]
    return torch.tensor(waypoints_rad, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, num_envs, 1)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, entity_cfg: SceneEntityCfg) -> None:
    robot = scene[ROBOT_ASSET_NAME]
    target = scene[TARGET_ASSET_NAME]
    sim_dt = sim.get_physics_dt()

    waypoints = make_joint_waypoints(robot.device, scene.num_envs)
    current_idx = 0
    next_idx = 1
    step_in_cycle = 0
    total_steps = 0

    q_start = waypoints[current_idx].clone()
    q_goal = waypoints[next_idx].clone()
    joint_pos_des = q_start.clone()

    reset_robot_to_default(scene)
    scene.write_data_to_sim()

    while simulation_app.is_running():
        alpha = step_in_cycle / max(args_cli.cycle_steps - 1, 1)
        smooth_alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
        joint_pos_des[:] = (1.0 - smooth_alpha) * q_start + smooth_alpha * q_goal

        robot.set_joint_position_target(joint_pos_des, joint_ids=entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if args_cli.print_every > 0 and total_steps % args_cli.print_every == 0:
            ee_body_id = int(entity_cfg.body_ids[0])
            ee_pos_w = robot.data.body_pose_w[0, ee_body_id, :3]
            root_pos_w = robot.data.root_pos_w[0, :3]
            target_pos_w = target.data.root_pos_w[0, :3]
            print(
                "[LIVE] "
                f"step={total_steps} "
                f"root={_format_vec(root_pos_w)} "
                f"ee={_format_vec(ee_pos_w)} "
                f"target={_format_vec(target_pos_w)}"
            )

        step_in_cycle += 1
        total_steps += 1

        if step_in_cycle >= args_cli.cycle_steps:
            step_in_cycle = 0
            current_idx = next_idx
            next_idx = (next_idx + 1) % len(waypoints)
            q_start = waypoints[current_idx].clone()
            q_goal = waypoints[next_idx].clone()
            print(f"[INFO] Switching playback waypoint: {current_idx} -> {next_idx}")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 1.8], [0.0, 0.0, 0.45])

    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(sim.get_physics_dt())

    ee_body_name = args_cli.ee_body_name or EE_BODY_NAME
    entity_cfg = print_asset_report(scene, ee_body_name)

    run_simulator(sim, scene, entity_cfg)


if __name__ == "__main__":
    main()
    simulation_app.close()
