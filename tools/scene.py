from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.managers import SceneEntityCfg

from configs.asset import EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME


class AuboToolFns:
    """与业务解耦的 AUBO 场景查询和几何计算工具。"""

    @staticmethod
    def project_root() -> Path:
        """返回当前项目根目录。"""
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def resolve_asset_name(asset_cfg: SceneEntityCfg | str) -> str:
        """从 SceneEntityCfg 或字符串中解析 env.scene 的实体名称。"""
        if isinstance(asset_cfg, str):
            return asset_cfg
        name = getattr(asset_cfg, "name", None)
        if isinstance(name, str):
            return name
        raise TypeError(
            f"Unsupported asset_cfg type '{type(asset_cfg)}'. "
            "Expected str or SceneEntityCfg (or object with .name)."
        )

    @staticmethod
    def get_asset(env, asset_cfg: SceneEntityCfg | str):
        """根据实体名称从 env.scene 获取资产实例。"""
        return AuboToolFns.get_scene(env)[AuboToolFns.resolve_asset_name(asset_cfg)]

    @staticmethod
    def get_scene(scene_or_env):
        """从 InteractiveScene 或带有 .scene 的环境对象中解析出 scene。"""
        return scene_or_env.scene if hasattr(scene_or_env, "scene") else scene_or_env

    @staticmethod
    def normalize_env_ids(env, env_ids=None) -> torch.Tensor:
        """把 None、int、list 或 Tensor 统一为位于 env.device 的 long Tensor。"""
        if env_ids is None:
            scene = AuboToolFns.get_scene(env)
            device = AuboToolFns._scene_device(env)
            num_envs = int(getattr(scene, "num_envs", getattr(env, "num_envs", 1)))
            return torch.arange(num_envs, dtype=torch.long, device=device)
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=AuboToolFns._scene_device(env))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=AuboToolFns._scene_device(env), dtype=torch.long).flatten()
        return torch.tensor(list(env_ids), dtype=torch.long, device=AuboToolFns._scene_device(env))

    @staticmethod
    def set_object_pose(
        scene_or_env,
        object_name: SceneEntityCfg | str,
        pos,
        rot=(1.0, 0.0, 0.0, 0.0),
        env_ids=None,
        relative_to_env_origins: bool = True,
        zero_velocity: bool = True,
    ) -> None:
        """按名称设置场景物体在一个或多个并行环境中的位置和旋转。

        参数:
            scene_or_env: InteractiveScene，或包含 .scene 的环境对象。
            object_name: 场景实体名称，如 "target"、"AUBObot"；也支持 SceneEntityCfg。
            pos: 位置，支持单个 (x, y, z)，也支持与 env_ids 数量一致的 (N, 3)。
            rot: 四元数，顺序为 (w, x, y, z)，支持单个 (4,) 或 (N, 4)。
            env_ids: 要控制的环境 id；None 表示控制全部环境。
            relative_to_env_origins: True 时 pos 视为各环境局部坐标，会自动加 env origin。
            zero_velocity: 对 RigidObject/Articulation 生效，设置 pose 后同步清零 root 速度。
        """
        scene = AuboToolFns.get_scene(scene_or_env)
        asset_name = AuboToolFns.resolve_asset_name(object_name)
        asset = scene[asset_name]
        env_ids = AuboToolFns.normalize_env_ids(scene_or_env, env_ids)
        device = AuboToolFns._scene_device(scene_or_env)

        pos_w = AuboToolFns._expand_pose_value(pos, len(env_ids), 3, device)
        if relative_to_env_origins and hasattr(scene, "env_origins"):
            origin_env_ids = env_ids.to(scene.env_origins.device)
            pos_w = pos_w.to(scene.env_origins.device) + scene.env_origins[origin_env_ids]

        quat_w = AuboToolFns._expand_pose_value(rot, len(env_ids), 4, pos_w.device)
        env_ids = env_ids.to(pos_w.device)
        root_pose = torch.cat([pos_w, quat_w], dim=-1)

        if AuboToolFns._try_write_root_pose(asset, root_pose, env_ids, zero_velocity):
            return
        if AuboToolFns._try_set_world_poses(asset, pos_w, quat_w, env_ids):
            return

        view = getattr(asset, "_view", None)
        if view is not None and AuboToolFns._try_set_world_poses(view, pos_w, quat_w, env_ids):
            return

        raise AttributeError(f"Scene object '{asset_name}' does not support root/world pose updates.")

    @staticmethod
    def get_first_body_id(asset, body_name: str) -> int:
        """在资产上按 body_name 匹配并返回第一个 body id。"""
        body_ids = asset.find_bodies(body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(f"Body '{body_name}' not found in asset '{asset}'.")
        return int(body_ids[0])

    @staticmethod
    def get_body_pos_w(env, asset_cfg: SceneEntityCfg | str, body_name: str) -> torch.Tensor:
        """返回指定 body 的世界坐标位置，形状为 (num_envs, 3)。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        body_id = AuboToolFns.get_first_body_id(asset, body_name)
        return asset.data.body_pose_w[:, body_id, :3]

    @staticmethod
    def get_body_lin_vel_w(env, asset_cfg: SceneEntityCfg | str, body_name: str) -> torch.Tensor:
        """返回指定 body 的世界坐标线速度，形状为 (num_envs, 3)。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        body_id = AuboToolFns.get_first_body_id(asset, body_name)
        return asset.data.body_lin_vel_w[:, body_id, :3]

    @staticmethod
    def get_root_pos_w(env, asset_cfg: SceneEntityCfg | str) -> torch.Tensor:
        """返回资产根节点世界坐标位置，形状为 (num_envs, 3)。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        data = getattr(asset, "data", None)
        root_pos_w = getattr(data, "root_pos_w", None)
        if isinstance(root_pos_w, torch.Tensor):
            return AuboToolFns._expand_single_env_pose(env, root_pos_w[:, :3])

        poses = AuboToolFns._try_get_world_poses(asset)
        if poses is None:
            view = getattr(asset, "_view", None)
            poses = AuboToolFns._try_get_world_poses(view)
        if poses is not None:
            positions, _ = poses
            if positions.ndim == 1:
                return AuboToolFns._expand_single_env_pose(env, positions[:3].reshape(1, 3))
            return AuboToolFns._expand_single_env_pose(env, positions[:, :3])

        asset_name = AuboToolFns.resolve_asset_name(asset_cfg)
        raise AttributeError(f"Scene object '{asset_name}' does not expose a readable world/root pose.")

    @staticmethod
    def ee_to_target_vec_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_cfg: SceneEntityCfg | str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到目标的世界坐标位移向量，形状为 (num_envs, 3)。"""
        ee_pos_w = AuboToolFns.get_body_pos_w(env, robot_asset_cfg, ee_body_name)
        target_pos_w = AuboToolFns.get_root_pos_w(env, target_asset_cfg)
        return target_pos_w - ee_pos_w

    @staticmethod
    def ee_target_distance_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_cfg: SceneEntityCfg | str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到目标的欧氏距离，形状为 (num_envs,)。"""
        return torch.norm(
            AuboToolFns.ee_to_target_vec_w(env, robot_asset_cfg, ee_body_name, target_asset_cfg),
            dim=-1,
        )

    @staticmethod
    def ee_goal_distance_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        goal_pos_name: str = "goal_pos_w",
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到目标点缓存或目标资产的欧氏距离，形状为 (num_envs,)。"""
        ee_pos_w = AuboToolFns.get_body_pos_w(env, robot_asset_cfg, ee_body_name)
        goal_pos_w = AuboToolFns.resolve_goal_pos_w(env, goal_pos_name, target_asset_name)
        return torch.norm(ee_pos_w - goal_pos_w, dim=-1)

    @staticmethod
    def resolve_goal_pos_w(
        env,
        goal_pos_name: str = "goal_pos_w",
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """按 env 缓存、scene key、默认 goal_pos_w、目标资产的顺序解析目标点坐标。"""
        goal_pos = getattr(env, goal_pos_name, None)
        if isinstance(goal_pos, torch.Tensor) and goal_pos.ndim == 2 and goal_pos.shape[-1] >= 3:
            return goal_pos[:, :3]

        scene = getattr(env, "scene", None)
        if scene is not None:
            try:
                asset = scene[goal_pos_name]
            except KeyError:
                asset = None
            if asset is not None:
                data = getattr(asset, "data", None)
                root_pos_w = getattr(data, "root_pos_w", None)
                if isinstance(root_pos_w, torch.Tensor) and root_pos_w.ndim == 2 and root_pos_w.shape[-1] >= 3:
                    return root_pos_w[:, :3]

        fallback = getattr(env, "goal_pos_w", None)
        if (
            goal_pos_name != "goal_pos_w"
            and isinstance(fallback, torch.Tensor)
            and fallback.ndim == 2
            and fallback.shape[-1] >= 3
        ):
            return fallback[:, :3]

        return AuboToolFns.get_root_pos_w(env, target_asset_name)

    @staticmethod
    def local_pos_to_world(env, env_ids: torch.Tensor, local_pos: torch.Tensor) -> torch.Tensor:
        """把各并行环境局部坐标转换为世界坐标。"""
        return env.scene.env_origins[env_ids] + local_pos

    @staticmethod
    def axis_aligned_workspace_mask(local_pos: torch.Tensor, workspace: dict) -> torch.Tensor:
        """返回局部坐标是否超出 xyz 轴对齐工作空间。"""
        x_min, x_max = map(float, workspace["x"])
        y_min, y_max = map(float, workspace["y"])
        z_min, z_max = map(float, workspace["z"])

        out_x = (local_pos[:, 0] < x_min) | (local_pos[:, 0] > x_max)
        out_y = (local_pos[:, 1] < y_min) | (local_pos[:, 1] > y_max)
        out_z = (local_pos[:, 2] < z_min) | (local_pos[:, 2] > z_max)
        return out_x | out_y | out_z

    @staticmethod
    def sample_ellipsoid_surface_band(
        n: int,
        device,
        xy_radius: float,
        z_radius: float,
        z_range: tuple[float, float],
        center: tuple[float, float, float],
    ) -> torch.Tensor:
        """在椭球指定 z 高度带的表面圆周上采样局部坐标。"""
        z_min, z_max = z_range
        z = torch.rand(n, 1, device=device) * (z_max - z_min) + z_min

        cz = float(center[2])
        inside = 1.0 - ((z - cz) ** 2) / (float(z_radius) ** 2)
        radius = float(xy_radius) * torch.sqrt(torch.clamp(inside, min=0.0))

        theta = 2.0 * torch.pi * torch.rand(n, 1, device=device)
        x = radius * torch.cos(theta) + float(center[0])
        y = radius * torch.sin(theta) + float(center[1])
        return torch.cat([x, y, z], dim=-1)

    @staticmethod
    def make_root_state(
        pos_w: torch.Tensor,
        quat_w: torch.Tensor | None = None,
        lin_vel_w: torch.Tensor | None = None,
        ang_vel_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """组装 Isaac Lab root state 张量，形状为 (N, 13)。"""
        n = pos_w.shape[0]
        device = pos_w.device
        dtype = pos_w.dtype

        if quat_w is None:
            quat_w = torch.zeros((n, 4), device=device, dtype=dtype)
            quat_w[:, 0] = 1.0
        if lin_vel_w is None:
            lin_vel_w = torch.zeros((n, 3), device=device, dtype=dtype)
        if ang_vel_w is None:
            ang_vel_w = torch.zeros((n, 3), device=device, dtype=dtype)

        return torch.cat([pos_w, quat_w, lin_vel_w, ang_vel_w], dim=-1)

    @staticmethod
    def _scene_device(scene_or_env):
        """推断 scene/env 当前使用的 torch device。"""
        scene = AuboToolFns.get_scene(scene_or_env)
        if hasattr(scene_or_env, "device"):
            return scene_or_env.device
        if hasattr(scene, "device"):
            return scene.device
        if hasattr(scene, "env_origins") and isinstance(scene.env_origins, torch.Tensor):
            return scene.env_origins.device
        return "cpu"

    @staticmethod
    def _expand_pose_value(value, count: int, width: int, device) -> torch.Tensor:
        """把单个 pose 值或每环境 pose 值统一成 (count, width) 张量。"""
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=torch.float32)
        else:
            tensor = torch.tensor(value, dtype=torch.float32, device=device)

        if tensor.ndim == 1:
            if tensor.shape[0] != width:
                raise ValueError(f"Expected pose value width {width}, got {tuple(tensor.shape)}.")
            return tensor.unsqueeze(0).repeat(count, 1)
        if tensor.ndim == 2 and tensor.shape == (count, width):
            return tensor
        raise ValueError(f"Expected pose value shape ({width},) or ({count}, {width}), got {tuple(tensor.shape)}.")

    @staticmethod
    def _try_write_root_pose(asset, root_pose: torch.Tensor, env_ids: torch.Tensor, zero_velocity: bool) -> bool:
        """尝试通过 RigidObject/Articulation 的 root pose 接口写入位姿。"""
        if not hasattr(asset, "write_root_pose_to_sim"):
            return False
        try:
            asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        except TypeError:
            asset.write_root_pose_to_sim(root_pose)

        if zero_velocity and hasattr(asset, "write_root_velocity_to_sim"):
            root_velocity = torch.zeros((root_pose.shape[0], 6), dtype=root_pose.dtype, device=root_pose.device)
            try:
                asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            except TypeError:
                asset.write_root_velocity_to_sim(root_velocity)
        return True

    @staticmethod
    def _try_set_world_poses(target, positions: torch.Tensor, orientations: torch.Tensor, env_ids: torch.Tensor) -> bool:
        """尝试通过 set_world_poses 接口写入静态资产或 XForm view 的位姿。"""
        if not hasattr(target, "set_world_poses"):
            return False

        call_variants = (
            {"env_ids": env_ids},
            {"indices": env_ids},
            "positional_indices",
            {},
        )
        for kwargs in call_variants:
            try:
                if kwargs == "positional_indices":
                    target.set_world_poses(positions, orientations, env_ids)
                else:
                    target.set_world_poses(positions, orientations, **kwargs)
                return True
            except TypeError:
                continue
        return False

    @staticmethod
    def _try_get_world_poses(target):
        """尝试通过 get_world_poses 接口读取静态资产或 XForm view 的位姿。"""
        if target is None or not hasattr(target, "get_world_poses"):
            return None
        try:
            positions, orientations = target.get_world_poses()
        except TypeError:
            return None
        if isinstance(positions, torch.Tensor) and isinstance(orientations, torch.Tensor):
            return positions, orientations
        return None

    @staticmethod
    def _expand_single_env_pose(env, positions: torch.Tensor) -> torch.Tensor:
        """Expand a single env_0 world pose to all env origins when XForm views return one row."""
        scene = getattr(env, "scene", None)
        env_origins = getattr(scene, "env_origins", None)
        num_envs = int(getattr(env, "num_envs", positions.shape[0]))
        if not isinstance(env_origins, torch.Tensor) or positions.shape[0] != 1 or num_envs <= 1:
            return positions

        env_origins = env_origins.to(device=positions.device, dtype=positions.dtype)
        local_pos = positions[0:1] - env_origins[0:1]
        return local_pos + env_origins[:num_envs]

    @staticmethod
    def _is_pos_tensor(value) -> bool:
        return isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[-1] >= 3
