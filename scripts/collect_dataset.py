from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="使用 SB3 教师策略采集 AUBO-RobotTraj 1.0.0 主数据。")
parser.add_argument("--weight", type=str, required=True, help="教师 PPO checkpoint 名称或完整路径。")
parser.add_argument("--dataset_id", type=str, required=True, help="新数据集的唯一 ID。")
parser.add_argument("--episodes", type=int, default=1, help="需要完成并提交的 Episode 数。")
parser.add_argument("--output_root", type=Path, default=Path("data/datasets"), help="数据集父目录。")
parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--record_cameras", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--episodes_per_shard", type=int, default=8)
parser.add_argument("--sensor_chunk_frames", type=int, default=16)
parser.add_argument("--target_asset_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.record_cameras)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from configs.asset import AUBO_ROBOT_USD, ENV_ASSET_USD
from configs.camera_cfg import CAMERA_SENSOR_SCENE_NAMES
from configs.dataset_cfg import DatasetCollectionCfg
from configs.place_cfg import WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS, WORKSTATION_TABLETOP_ASSET_SPECS
from configs.RLcfg import DEFAULT_RL_TARGET_ASSET_NAME, AuboRLEnvCfg, configure_task_target
from tools.dataset.isaac_recorder import get_dataset_recorder_term, make_dataset_recorder_cfg
from tools.dataset.writer import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_checkpoint_path(value: str) -> Path:
    raw = Path(value)
    candidates = [raw, PROJECT_ROOT / "checkpoints/sb3_aubo" / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.suffix != ".zip" and candidate.with_suffix(".zip").is_file():
            return candidate.with_suffix(".zip").resolve()
    raise FileNotFoundError(f"找不到教师 checkpoint：{value}")


def git_trace() -> dict:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain=v1")
        diff = run("diff", "--binary", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty": None,
            "working_tree_status_sha256": None,
            "tracked_diff_sha256": None,
        }
    return {
        "commit": commit,
        "dirty": bool(status),
        "working_tree_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def hash_assets() -> list[dict]:
    assets = [
        ("aubo_robot_usd", AUBO_ROBOT_USD),
        ("laboratory_usd", ENV_ASSET_USD),
    ]
    assets.extend((f"workstation/{spec.name}", spec.usd_path) for spec in WORKSTATION_TABLETOP_ASSET_SPECS)
    assets.extend(
        (f"workstation/interactive/{placement['scene_key']}", placement["usd_path"])
        for placement in WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS
    )
    rows = []
    hashed_paths: dict[Path, str] = {}
    for name, path in assets:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"必须追溯的资产不存在：{resolved}")
        digest = hashed_paths.get(resolved)
        if digest is None:
            digest = sha256_file(resolved)
            hashed_paths[resolved] = digest
        rows.append({"name": name, "uri": str(resolved), "sha256": digest})
    return rows


def hash_collection_code() -> list[dict]:
    paths = [
        PROJECT_ROOT / "scripts/collect_dataset.py",
        PROJECT_ROOT / "tools/ik.py",
        PROJECT_ROOT / "tools/contact.py",
        PROJECT_ROOT / "tools/scene.py",
        PROJECT_ROOT / "tools/logic.py",
        *sorted((PROJECT_ROOT / "tools/dataset").glob("*.py")),
    ]
    return [
        {"uri": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": sha256_file(path)}
        for path in paths
    ]


def build_metadata(collection: DatasetCollectionCfg, checkpoint: Path, deterministic: bool) -> tuple[dict, dict]:
    trace = git_trace()
    try:
        isaaclab_version = importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        isaaclab_version = "unknown"
    checkpoint_hash = sha256_file(checkpoint)
    teacher_id = f"sb3_ppo:{checkpoint.stem}:{checkpoint_hash[:12]}"
    dataset = {
        "dataset_id": collection.dataset_id,
        "schema_name": collection.schema_name,
        "schema_version": collection.schema_version,
        "domain": "simulation",
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": trace["commit"],
        "git_dirty": trace["dirty"],
        "working_tree_status_sha256": trace["working_tree_status_sha256"],
        "tracked_diff_sha256": trace["tracked_diff_sha256"],
        "isaaclab_version": isaaclab_version,
        "robot_family": "AUBO",
        "coordinate_convention": "right_handed",
        "quaternion_order": "wxyz",
        "default_float_dtype": "float32",
        "clocks": {
            "simulation_hz": collection.simulation_hz,
            "policy_hz": collection.policy_hz,
            "camera_hz": collection.camera_hz,
            "source": "Isaac simulation step counter",
            "timestamp_unit": "ns",
        },
        "teacher": {
            "teacher_id": teacher_id,
            "algorithm": "stable_baselines3.PPO",
            "checkpoint_uri": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "deterministic": bool(deterministic),
            "discount": collection.discount,
        },
        "asset_hashes": hash_assets(),
        "code_hashes": hash_collection_code(),
    }
    task_definition = PROJECT_ROOT / "configs/RLcfg.py"
    task = {
        "task_id": collection.task_id,
        "task_suite": collection.task_suite,
        "name": collection.task_name,
        "instruction": collection.instruction,
        "definition": {
            "type": "isaaclab_config",
            "uri": "configs/RLcfg.py",
            "sha256": sha256_file(task_definition),
        },
        "success_criteria_version": 1,
    }
    return dataset, task


def main() -> None:
    checkpoint = resolve_checkpoint_path(args_cli.weight)
    checkpoint_model = PPO.load(str(checkpoint), device=args_cli.device)
    output_root = args_cli.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    collection = DatasetCollectionCfg(
        dataset_id=args_cli.dataset_id,
        output_root=output_root,
        split=args_cli.split,
        seed=args_cli.seed,
        max_episodes=args_cli.episodes,
        episodes_per_shard=args_cli.episodes_per_shard,
        sensor_chunk_frames=args_cli.sensor_chunk_frames,
        discount=float(checkpoint_model.gamma),
    )
    if not args_cli.record_cameras:
        collection = replace(collection, camera_streams=())
    dataset_metadata, task_metadata = build_metadata(collection, checkpoint, args_cli.deterministic)
    del checkpoint_model
    target_asset_name = args_cli.target_asset_name or DEFAULT_RL_TARGET_ASSET_NAME
    reward_cfg_path = PROJECT_ROOT / "configs/RLcfg.py"
    artifact_paths = [
        reward_cfg_path,
        PROJECT_ROOT / "configs/asset.py",
        PROJECT_ROOT / "configs/scene_cfg.py",
        PROJECT_ROOT / "configs/camera_cfg.py",
        PROJECT_ROOT / "configs/collision_cfg.py",
        PROJECT_ROOT / "configs/place_cfg.py",
        PROJECT_ROOT / "configs/lula_cfg.py",
        PROJECT_ROOT / "configs/lula/aubo_e5.urdf",
        PROJECT_ROOT / "configs/lula/aubo_e5_robot_description.yaml",
    ]

    env_cfg = AuboRLEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    if not args_cli.record_cameras:
        for camera_name in CAMERA_SENSOR_SCENE_NAMES:
            setattr(env_cfg.scene, camera_name, None)
    configure_task_target(env_cfg, target_asset_name)
    env_cfg.recorders = make_dataset_recorder_cfg(
        collection,
        target_asset_name=target_asset_name,
        reward_config_hash=sha256_file(reward_cfg_path),
        teacher_id=dataset_metadata["teacher"]["teacher_id"],
        dataset_metadata=dataset_metadata,
        task_metadata=task_metadata,
        artifact_paths=artifact_paths,
    )

    env = None
    recorder = None
    try:
        isaac_env = ManagerBasedRLEnv(cfg=env_cfg)
        isaac_env.seed(args_cli.seed)
        recorder = get_dataset_recorder_term(isaac_env)
        env = Sb3VecEnvWrapper(isaac_env, fast_variant=False)
        model = PPO.load(str(checkpoint), env=env, device=args_cli.device)
        observation = env.reset()
        reported = 0
        while recorder.completed_episode_count < collection.max_episodes and simulation_app.is_running():
            actions, _ = model.predict(observation, deterministic=args_cli.deterministic)
            observation, _, _, _ = env.step(actions)
            if recorder.completed_episode_count > reported:
                reported = recorder.completed_episode_count
                print(f"[Dataset] 已完成 {reported}/{collection.max_episodes} 个 Episode。", flush=True)
        if recorder.completed_episode_count < collection.max_episodes:
            raise RuntimeError("Isaac 应用在完成目标 Episode 数前停止。")
        recorder.close()
        env.close()
        env = None
        print(f"[Dataset] 数据集已写入：{collection.dataset_root.resolve()}", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[Dataset][fatal] 数据采集失败：", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
