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
from pathlib import Path

import _bootstrap  # noqa: F401
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect AUBO asset configuration in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene copies to spawn.")
parser.add_argument("--cycle_steps", type=int, default=240, help="Simulation steps per playback waypoint.")
parser.add_argument("--print_every", type=int, default=120, help="Print live pose info every N steps. Use 0 to disable.")
parser.add_argument("--ee_body_name", type=str, default=None, help="End-effector/flange body name to resolve.")
parser.add_argument("--list_all", action="store_true", help="Print all joint/body names without truncation.")
parser.add_argument("--camera_name", type=str, default="camera_cfg", help="Scene camera key used for test image capture.")
parser.add_argument("--picture_dir", type=str, default=None, help="Image output directory. Defaults to <project>/picture.")
parser.add_argument("--capture_after_seconds", type=float, default=5.0, help="Capture one camera image after this sim time.")
parser.add_argument("--capture_data_type", type=str, default="rgb", help="Camera data output type to save.")
parser.add_argument(
    "--contact_print_every",
    type=int,
    default=1,
    help="Print ContactSensor force hits every N sim steps. Use 0 to disable.",
)
parser.add_argument(
    "--contact_force_threshold",
    type=float,
    default=None,
    help="Minimum ContactSensor force magnitude to print. Defaults to collision_cfg.py.",
)
parser.add_argument(
    "--apply_workstation_collision_config",
    action="store_true",
    help="Apply the workstation collision scan and temporary overrides from collision_cfg.py.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics, UsdUtils

from configs.asset import (
    AUBO_ROBOT_USD,
    EE_BODY_NAME,
    ROBOT_ASSET_NAME,
    TARGET_ASSET_NAME,
)
from configs.camera_cfg import CAMERA_SENSOR_POSE_CFG
from configs.place_cfg import WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS
from configs.RenderCfg import TEST_RENDER_CFG
from configs.collision_cfg import (
    ROBOT_CONTACT_FORCE_THRESHOLD,
    ROBOT_CONTACT_SENSOR_NAME,
    apply_workstation_collision_config,
    disable_workstation_collision_prims,
)
from tools.camera import AuboCameraFns
from tools.contact import AuboContactToolFns
from configs.Testcfg import TestSceneCfg
from configs.scene_cfg import TRAINING_ENV_SPACING, TRAINING_REPLICATE_PHYSICS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _format_tuple(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _first_pose_from_asset(asset) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first world pose from either RigidObject data or an XformPrimView."""
    data = getattr(asset, "data", None)
    if data is not None and hasattr(data, "root_pos_w") and hasattr(data, "root_pose_w"):
        return data.root_pos_w[0, :3], data.root_pose_w[0, 3:7]

    pose_owner = asset if hasattr(asset, "get_world_poses") else getattr(asset, "_view", None)
    if pose_owner is None or not hasattr(pose_owner, "get_world_poses"):
        raise AttributeError(f"Scene asset of type '{type(asset).__name__}' does not expose world pose data.")

    positions, orientations = pose_owner.get_world_poses()
    positions = torch.as_tensor(positions)
    orientations = torch.as_tensor(orientations)

    if positions.ndim == 3:
        return positions[0, 0, :3], orientations[0, 0, :4]
    return positions[0, :3], orientations[0, :4]


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
    # target = scene[TARGET_ASSET_NAME]

    joint_names = _get_named_list(robot, "joint_names")
    body_names = _get_named_list(robot, "body_names")

    print("\n========== Isaac Asset Inspection ==========")
    print(f"[INFO] USD path           : {AUBO_ROBOT_USD}")
    print(f"[INFO] Robot scene key    : {ROBOT_ASSET_NAME}")
    # print(f"[INFO] Target scene key   : {TARGET_ASSET_NAME}")
    print(f"[INFO] Requested EE body  : {ee_body_name}")
    print(f"[INFO] Num envs           : {scene.num_envs}")
    print(f"[INFO] Env origin[0]      : {_format_vec(scene.env_origins[0])}")

    print(f"[INFO] Robot root pos[0]  : {_format_vec(robot.data.root_pos_w[0, :3])}")
    print(f"[INFO] Robot root quat[0] : {_format_vec(robot.data.root_pose_w[0, 3:7])}")
    # print(f"[INFO] Target pos[0]      : {_format_vec(target.data.root_pos_w[0, :3])}")

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


def print_workstation_interactive_report(scene: InteractiveScene) -> None:
    """Print expected and actual root poses for split workstation interactive assets."""
    print("\n========== Workstation Interactive Assets ==========")
    for placement in WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS:
        scene_key = str(placement["scene_key"])
        try:
            asset = scene[scene_key]
        except KeyError:
            print(f"[WARN] {scene_key}: missing from scene")
            continue

        actual_pos, actual_quat = _first_pose_from_asset(asset)
        print(
            f"[INFO] {scene_key} "
            f"source={placement['source_name']} "
            f"configured_pos={_format_tuple(placement['pos'])} "
            f"actual_root_pos_w={_format_vec(actual_pos)} "
            f"configured_rot={_format_tuple(placement['rot'])} "
            f"actual_root_rot_w={_format_vec(actual_quat)} "
            f"configured_scale={_format_tuple(placement.get('scale', (1.0, 1.0, 1.0)))}"
        )
    print("====================================================\n")


def print_camera_pose_report(scene: InteractiveScene, camera_name: str) -> None:
    """Print the configured and actual pose for the scene camera sensor."""
    camera = AuboCameraFns.get_camera(scene=scene, camera_name=camera_name)
    print("\n========== Camera Sensor Pose ==========")
    print(f"[INFO] Camera scene key       : {camera_name}")
    print(f"[INFO] Configured pos         : {_format_tuple(CAMERA_SENSOR_POSE_CFG.initial_pos)}")
    print(f"[INFO] Configured rot         : {_format_tuple(CAMERA_SENSOR_POSE_CFG.initial_rot)}")
    try:
        actual_pos, actual_quat = _first_pose_from_asset(camera)
    except Exception as exc:
        print(f"[WARN] Failed to read camera world pose: {exc}")
    else:
        print(f"[INFO] Actual root pos_w[0]   : {_format_vec(actual_pos)}")
        print(f"[INFO] Actual root rot_w[0]   : {_format_vec(actual_quat)}")
    print("========================================\n")


def reset_robot_to_default(scene: InteractiveScene) -> None:
    robot = scene[ROBOT_ASSET_NAME]
    root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:])
    robot.write_root_velocity_to_sim(root_velocity)
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


def disable_station_collision_test_prims(scene: InteractiveScene) -> None:
    """TODO: 临时禁用 station 中会导致 AUBObot1 初始化碰撞的两个物体。

    后续更换 station 模型后删除这段测试逻辑。
    当前禁用对象位于 split-USD station 路径下：
    - M_SupportTray_07
    - M_Reagent_05
    """
    disable_workstation_collision_prims(scene)


def enable_contact_reports(scene: InteractiveScene) -> int:
    """Enable PhysX contact reports on collision, rigid-body and articulation prims."""
    enabled_count = 0
    for prim in scene.stage.TraverseAll():
        if not prim.IsValid() or not prim.IsActive():
            continue
        if not (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            continue

        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        contact_report_api.CreateThresholdAttr().Set(0)
        enabled_count += 1

    print(f"[INFO] Enabled contact reports on {enabled_count} physics prims.")
    return enabled_count


class ContactReportPrinter:
    """Print collider prim pairs when PhysX reports a new contact."""

    def __init__(self, scene: InteractiveScene):
        self._stage_id = UsdUtils.StageCache.Get().GetId(scene.stage).ToLongInt()
        self._active_pairs: set[tuple[str, str]] = set()
        self._current_step = 0
        self._subscription = get_physx_simulation_interface().subscribe_contact_report_events(
            self._on_contact_report_event
        )

    def set_step(self, step: int) -> None:
        self._current_step = step

    def close(self) -> None:
        self._subscription = None

    def _on_contact_report_event(self, contact_headers, contact_data) -> None:
        del contact_data
        for contact_header in contact_headers:
            if int(contact_header.stage_id) != self._stage_id:
                continue

            collider0 = self._id_to_path(contact_header.collider0)
            collider1 = self._id_to_path(contact_header.collider1)
            if not collider0 or not collider1:
                continue

            pair = (collider0, collider1) if collider0 <= collider1 else (collider1, collider0)
            if contact_header.type in (ContactEventType.CONTACT_FOUND, ContactEventType.CONTACT_PERSIST):
                if pair not in self._active_pairs:
                    self._active_pairs.add(pair)
                    actor0 = self._id_to_path(contact_header.actor0)
                    actor1 = self._id_to_path(contact_header.actor1)
                    print(
                        "[CONTACT] "
                        f"step={self._current_step} "
                        f"collider0={collider0} "
                        f"collider1={collider1} "
                        f"actor0={actor0} "
                        f"actor1={actor1}"
                    )
            elif contact_header.type == ContactEventType.CONTACT_LOST:
                self._active_pairs.discard(pair)

    @staticmethod
    def _id_to_path(path_id: int) -> str:
        if int(path_id) == 0:
            return ""
        return str(PhysicsSchemaTools.intToSdfPath(path_id))


class ContactSensorPrinter:
    """Print ContactSensor force magnitudes above a threshold."""

    def __init__(
        self,
        scene: InteractiveScene,
        sensor_name: str = ROBOT_CONTACT_SENSOR_NAME,
        force_threshold: float = ROBOT_CONTACT_FORCE_THRESHOLD,
        print_every: int = 1,
    ):
        self._scene = scene
        self._sensor_name = sensor_name
        self._force_threshold = float(force_threshold)
        self._print_every = int(print_every)
        self._missing_reported = False
        self._empty_reported = False

    def update(self, step: int) -> None:
        if self._print_every <= 0 or step % self._print_every != 0:
            return

        try:
            sensor = self._scene[self._sensor_name]
        except Exception:
            if not self._missing_reported:
                self._missing_reported = True
                print(f"[WARN] ContactSensor '{self._sensor_name}' is not registered in the scene.")
            return

        magnitude = AuboContactToolFns.extract_contact_magnitude(sensor)
        if magnitude is None:
            if not self._empty_reported:
                self._empty_reported = True
                print(f"[WARN] ContactSensor '{self._sensor_name}' has no readable force tensor yet.")
            return

        env_magnitude = self._reshape_contact_magnitude(magnitude)
        if env_magnitude.numel() == 0:
            return

        max_force, flat_ids = torch.max(env_magnitude, dim=1)
        hit_env_ids = torch.nonzero(max_force > self._force_threshold, as_tuple=False).squeeze(-1)
        for env_id in hit_env_ids.detach().cpu().tolist():
            print(
                "[CONTACT_SENSOR] "
                f"step={step} "
                f"env={env_id} "
                f"max_force={float(max_force[env_id].detach().cpu()):.6f} "
                f"flat_index={int(flat_ids[env_id].detach().cpu())} "
                f"shape={tuple(magnitude.shape)}"
            )

    def _reshape_contact_magnitude(self, magnitude: torch.Tensor) -> torch.Tensor:
        if magnitude.ndim == 0:
            return magnitude.reshape(1, 1)
        if magnitude.shape[0] == self._scene.num_envs:
            return magnitude.reshape(self._scene.num_envs, -1)
        return magnitude.reshape(1, -1)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, entity_cfg: SceneEntityCfg) -> None:
    robot = scene[ROBOT_ASSET_NAME]
    # target = scene[TARGET_ASSET_NAME]
    sim_dt = sim.get_physics_dt()
    contact_printer = ContactReportPrinter(scene)
    contact_sensor_printer = ContactSensorPrinter(
        scene,
        force_threshold=(
            ROBOT_CONTACT_FORCE_THRESHOLD
            if args_cli.contact_force_threshold is None
            else args_cli.contact_force_threshold
        ),
        print_every=args_cli.contact_print_every,
    )

    waypoints = make_joint_waypoints(robot.device, scene.num_envs)
    current_idx = 0
    next_idx = 1
    step_in_cycle = 0
    total_steps = 0
    capture_done = False

    q_start = waypoints[current_idx].clone()
    q_goal = waypoints[next_idx].clone()
    joint_pos_des = q_start.clone()

    reset_robot_to_default(scene)
    scene.write_data_to_sim()

    while simulation_app.is_running():
        # 机械臂播放逻辑：原本用于在若干关节路点之间做平滑插值，并把目标关节角写入控制器。
        alpha = step_in_cycle / max(args_cli.cycle_steps - 1, 1)
        smooth_alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
        joint_pos_des[:] = (1.0 - smooth_alpha) * q_start + smooth_alpha * q_goal
        robot.set_joint_position_target(joint_pos_des, joint_ids=entity_cfg.joint_ids)

        # 将当前场景缓存写入仿真，随后推进一个物理步。
        scene.write_data_to_sim()
        contact_printer.set_step(total_steps + 1)
        sim.step()

        # 从仿真中读取最新状态，更新 scene 内各资产和传感器数据。
        scene.update(sim_dt)
        contact_sensor_printer.update(total_steps + 1)

        # 按仿真时间触发一次相机保存；env_ids=None 表示保存所有并行环境。
        sim_time = (total_steps + 1) * sim_dt
        if not capture_done and sim_time >= args_cli.capture_after_seconds:
            capture_done = True
            try:
                image_paths = AuboCameraFns.save_camera_images(
                    scene=scene,
                    camera_name=args_cli.camera_name,
                    output_dir=args_cli.picture_dir,
                    root_dir=PROJECT_ROOT,
                    data_type=args_cli.capture_data_type,
                    env_ids=None,
                    step=total_steps + 1,
                )
                print(f"[INFO] Saved {len(image_paths)} camera images:")
                for image_path in image_paths:
                    print(f"  {image_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save camera image: {exc}")

        # 定期打印第一套环境中的 robot root、末端和 target 位置，便于观察场景是否稳定。
        if args_cli.print_every > 0 and total_steps % args_cli.print_every == 0:
            ee_body_id = int(entity_cfg.body_ids[0])
            ee_pos_w = robot.data.body_pose_w[0, ee_body_id, :3]
            root_pos_w = robot.data.root_pos_w[0, :3]
            # target_pos_w = target.data.root_pos_w[0, :3]
            print(
                "[LIVE] "
                f"step={total_steps} "
                f"root={_format_vec(root_pos_w)} "
                f"ee={_format_vec(ee_pos_w)} "
                # f"target={_format_vec(target_pos_w)}"
            )

        step_in_cycle += 1
        total_steps += 1

        # 关节播放路点切换逻辑。机械臂运动关闭时仍保留计数，方便后续恢复播放代码。
        if step_in_cycle >= args_cli.cycle_steps:
            step_in_cycle = 0
            current_idx = next_idx
            next_idx = (next_idx + 1) % len(waypoints)
            q_start = waypoints[current_idx].clone()
            q_goal = waypoints[next_idx].clone()
            print(f"[INFO] Switching playback waypoint: {current_idx} -> {next_idx}")

    contact_printer.close()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, render=TEST_RENDER_CFG.to_isaaclab())
    sim = sim_utils.SimulationContext(sim_cfg)
    TEST_RENDER_CFG.apply_runtime_settings()
    sim.set_camera_view(TEST_RENDER_CFG.viewport_camera_eye, TEST_RENDER_CFG.viewport_camera_target)

    # 测试场景设置
    scene_cfg = TestSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=TRAINING_ENV_SPACING,
        replicate_physics=TRAINING_REPLICATE_PHYSICS,
    )
    scene = InteractiveScene(scene_cfg)
    if args_cli.apply_workstation_collision_config:
        apply_workstation_collision_config(scene)
    else:
        disable_station_collision_test_prims(scene)
    enable_contact_reports(scene)

    sim.reset()
    scene.update(sim.get_physics_dt())
    AuboCameraFns.set_camera_pose(
        scene=scene,
        camera_name=args_cli.camera_name,
    )
    scene.update(sim.get_physics_dt())

    ee_body_name = args_cli.ee_body_name or EE_BODY_NAME
    print_camera_pose_report(scene, args_cli.camera_name)
    print_workstation_interactive_report(scene)
    entity_cfg = print_asset_report(scene, ee_body_name)

    run_simulator(sim, scene, entity_cfg)


if __name__ == "__main__":
    main()
    simulation_app.close()
