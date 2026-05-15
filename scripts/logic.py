import torch

from datetime import datetime
from pathlib import Path
import binascii
import struct
import zlib

from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from typing import Sequence
from asset import EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME
from tool import AuboToolFns

class AuboCameraFns:
    """Aubo 相机相关工具函数集合.

    这个类只负责相机的通用操作：
    1. 获取场景中的相机对象；
    2. 设置多环境相机位姿；
    3. 将 Isaac Lab 相机输出转换并保存为 PNG 图片。
    """

    @staticmethod
    def _project_root() -> Path:
        """返回项目根目录路径.

        当前文件位于 ``scripts/logic.py``，所以向上两级可得到项目根目录。
        该路径用于在调用者没有指定输出目录时，默认保存到 ``picture`` 文件夹。
        """
        return Path(__file__).resolve().parents[1]

    @staticmethod
    def _get_camera(scene=None, camera=None, camera_name: str = "camera_cfg"):
        """统一获取相机对象.

        调用者可以直接传入 ``camera``，也可以传入 ``scene`` 和 ``camera_name``。
        如果直接传入相机，则优先使用该对象；否则从 ``scene[camera_name]`` 中取相机。
        """
        if camera is not None:
            return camera
        if scene is None:
            raise ValueError("Either scene or camera must be provided.")
        return scene[camera_name]

    @staticmethod
    def _infer_num_envs(scene=None, camera=None, output=None) -> int:
        """推断当前相机对应的环境数量.

        优先从 ``scene.num_envs`` 读取；如果没有 scene，则尝试从 camera 读取。
        最后根据相机输出张量的第一维推断，例如 ``[num_envs, H, W, C]``。
        推断失败时返回 1，兼容单环境或单张图片的调用方式。
        """
        if scene is not None and hasattr(scene, "num_envs"):
            return int(scene.num_envs)
        if camera is not None and hasattr(camera, "num_envs"):
            return int(camera.num_envs)
        if output is not None and hasattr(output, "ndim") and output.ndim == 4:
            return int(output.shape[0])
        return 1

    @staticmethod
    def _normalize_env_ids(env_ids, num_envs: int) -> list[int]:
        """将不同形式的环境 id 统一转换为 Python int 列表.

        ``env_ids=None`` 表示选择所有环境；也支持单个 int、Tensor、
        list/tuple 等形式，方便外部调用时不必手动做类型转换。
        """
        if env_ids is None:
            return list(range(num_envs))
        if isinstance(env_ids, torch.Tensor):
            return [int(env_id) for env_id in env_ids.detach().cpu().flatten().tolist()]
        if isinstance(env_ids, int):
            return [int(env_ids)]
        return [int(env_id) for env_id in env_ids]

    @staticmethod
    def set_camera_pose(
        scene=None,
        camera=None,
        camera_name: str = "camera_cfg",
        pos: tuple[float, float, float] = (1.0, 0.0, 0.6),
        # Quaternion order is w, x, y, z. This is +90 degrees around Y.
        rot: tuple[float, float, float, float] = (0.70711, 0.0, 0.70711, 0.0),
        env_ids=None,
        relative_to_env_origins: bool = True,
        convention: str = "opengl",
    ) -> None:
        """设置指定环境中的相机位姿.

        参数说明：
        - ``pos``：相机位置，默认按每个环境的局部坐标理解；
        - ``rot``：四元数，顺序为 ``w, x, y, z``；
        - ``env_ids``：要设置的环境 id，``None`` 表示全部环境；
        - ``relative_to_env_origins``：为 True 时，会将 ``pos`` 加上各环境
          的 ``scene.env_origins``，保证多环境下每个相机保持相同局部机位；
        - ``convention``：传给 Isaac Lab 相机接口的坐标约定。使用
          ``opengl`` 时，四元数更接近 USD/Isaac Sim Transform 面板显示。
        """
        camera = AuboCameraFns._get_camera(scene, camera, camera_name)
        num_envs = AuboCameraFns._infer_num_envs(scene, camera)
        env_id_list = AuboCameraFns._normalize_env_ids(env_ids, num_envs)

        # 相机和场景可能在 GPU 上；这里尽量沿用已有 device，避免跨设备张量错误。
        device = getattr(camera, "device", None)
        if device is None and scene is not None and hasattr(scene, "device"):
            device = scene.device
        device = device or "cpu"

        # 构造每个环境一份 position。多环境时，局部坐标需要加对应 env origin。
        positions = torch.tensor(pos, dtype=torch.float32, device=device).repeat(len(env_id_list), 1)
        if relative_to_env_origins and scene is not None and hasattr(scene, "env_origins"):
            env_id_tensor = torch.tensor(env_id_list, dtype=torch.long, device=scene.env_origins.device)
            positions = positions.to(scene.env_origins.device) + scene.env_origins[env_id_tensor]
        orientations = torch.tensor(rot, dtype=torch.float32, device=positions.device).repeat(len(env_id_list), 1)
        env_id_tensor = torch.tensor(env_id_list, dtype=torch.long, device=positions.device)

        # 优先使用 Isaac Lab Camera 的公开接口；不同版本参数名略有差异，所以做兼容。
        if hasattr(camera, "set_world_poses"):
            try:
                camera.set_world_poses(positions, orientations, convention=convention, env_ids=env_id_tensor)
                return
            except TypeError:
                try:
                    camera.set_world_poses(positions, orientations, env_ids=env_id_tensor)
                    return
                except TypeError:
                    camera.set_world_poses(positions, orientations)
                    return

        # 兼容部分版本中 camera 没暴露 set_world_poses、但内部 view 支持设置位姿的情况。
        view = getattr(camera, "_view", None)
        if view is not None and hasattr(view, "set_world_poses"):
            try:
                view.set_world_poses(positions, orientations, env_id_tensor)
            except TypeError:
                try:
                    view.set_world_poses(positions, orientations, indices=env_id_tensor)
                except TypeError:
                    view.set_world_poses(positions, orientations)
            return

        raise AttributeError(f"Camera '{camera_name}' does not support setting world poses.")

    @staticmethod
    def _write_png(path: Path, image_array) -> None:
        """将 uint8 图像数组写成 PNG 文件.

        为了避免额外依赖 Pillow/OpenCV，这里直接按 PNG 格式写入：
        生成 IHDR、IDAT、IEND 三类 chunk，并用 zlib 压缩像素行数据。
        支持灰度图、RGB 图和 RGBA 图。
        """

        def _chunk(chunk_type: bytes, data: bytes) -> bytes:
            """构造一个 PNG chunk，包括长度、类型、数据和 CRC 校验."""
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        # 根据数组形状判断 PNG 颜色类型：0=灰度，2=RGB，6=RGBA。
        height, width = image_array.shape[:2]
        if image_array.ndim == 2:
            color_type = 0
        elif image_array.shape[2] == 3:
            color_type = 2
        elif image_array.shape[2] == 4:
            color_type = 6
        else:
            raise ValueError(f"Unsupported image shape for PNG: {image_array.shape}")

        # PNG 每行前需要一个 filter byte。这里使用 0，表示不使用行滤波。
        raw_rows = b"".join(b"\x00" + image_array[row].tobytes() for row in range(height))
        header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw_rows))
            + _chunk(b"IEND", b"")
        )
        path.write_bytes(png_bytes)

    @staticmethod
    def _camera_output_to_uint8(output: torch.Tensor, env_id: int):
        """将相机输出转换为可保存的 uint8 图像数组.

        Isaac Lab 相机输出通常是 Tensor，形状可能是：
        - ``[num_envs, H, W, C]``：多环境批量输出；
        - ``[H, W, C]`` 或 ``[H, W]``：单张图片。

        本函数会先按 ``env_id`` 取出对应环境图片，再处理通道顺序、单通道、
        浮点归一化和 NaN/Inf，最终返回 PNG 写入函数可直接使用的 uint8 数组。
        """
        import numpy as np

        image = output
        # 多环境输出时，第一维是环境 id。
        if image.ndim == 4:
            image = image[int(env_id)]
        elif image.ndim != 3 and image.ndim != 2:
            raise ValueError(f"Unsupported camera output shape: {tuple(image.shape)}")

        array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)

        # 兼容 CHW 格式；保存 PNG 前统一为 HWC。
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]

        # RGB 通常可能是 0-1 浮点或 0-255 数值；统一裁剪并转换到 uint8。
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            finite_mask = np.isfinite(array)
            if not finite_mask.all():
                array = np.where(finite_mask, array, 0.0)
            if array.size > 0 and float(array.max()) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)

        return array

    @staticmethod
    def save_camera_image(
        scene=None,
        camera=None,
        camera_name: str = "camera_cfg",
        output_dir: str | Path | None = None,
        root_dir: str | Path | None = None,
        data_type: str = "rgb",
        env_id: int = 0,
        file_name: str | None = None,
        step: int | None = None,
    ) -> Path:
        """保存单个环境的一张相机图片，并返回保存路径.

        调用者可以通过参数解耦保存逻辑：
        - ``scene/camera/camera_name`` 决定从哪里取相机；
        - ``data_type`` 决定保存 RGB、深度或分割等哪一路输出；
        - ``env_id`` 决定保存哪个环境的图片；
        - ``output_dir/root_dir/file_name`` 决定保存位置和文件名。
        """
        camera = AuboCameraFns._get_camera(scene, camera, camera_name)

        # 检查相机输出是否存在，以及请求的数据类型是否已由 CameraCfg 配置。
        if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
            raise ValueError(f"Camera '{camera_name}' does not expose data.output.")
        if data_type not in camera.data.output:
            available = ", ".join(camera.data.output.keys())
            raise KeyError(f"Camera output '{data_type}' not found. Available: {available}")

        # 默认保存到项目根目录 picture/；如果 output_dir 是相对路径，则也相对项目根目录。
        base_dir = Path(root_dir).resolve() if root_dir is not None else AuboCameraFns._project_root()
        save_dir = Path(output_dir) if output_dir is not None else base_dir / "picture"
        if not save_dir.is_absolute():
            save_dir = base_dir / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # 默认文件名包含数据类型、环境 id、仿真步数和时间戳，避免多次保存互相覆盖。
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_part = f"_step{step}" if step is not None else ""
            file_name = f"{data_type}_env{int(env_id)}{step_part}_{timestamp}.png"
        save_path = save_dir / file_name

        # 将相机输出转换为 uint8 数组后写成 PNG。
        image_array = AuboCameraFns._camera_output_to_uint8(camera.data.output[data_type], env_id)
        AuboCameraFns._write_png(save_path, image_array)
        return save_path

    @staticmethod
    def save_camera_images(
        scene=None,
        camera=None,
        camera_name: str = "camera_cfg",
        output_dir: str | Path | None = None,
        root_dir: str | Path | None = None,
        data_type: str = "rgb",
        env_ids=None,
        file_name: str | None = None,
        step: int | None = None,
    ) -> list[Path]:
        """批量保存多个环境的相机图片，并返回所有保存路径.

        ``env_ids=None`` 表示保存所有环境。该函数内部复用
        ``save_camera_image``，因此单张图片的路径规则、数据类型检查和 PNG
        转换逻辑保持一致。
        """
        camera = AuboCameraFns._get_camera(scene, camera, camera_name)
        # 先做一次统一检查，避免循环中保存到一半才发现数据类型不存在。
        if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
            raise ValueError(f"Camera '{camera_name}' does not expose data.output.")
        if data_type not in camera.data.output:
            available = ", ".join(camera.data.output.keys())
            raise KeyError(f"Camera output '{data_type}' not found. Available: {available}")

        # 根据 scene/camera/output 推断环境数量，并将 env_ids 统一成列表。
        output = camera.data.output[data_type]
        num_envs = AuboCameraFns._infer_num_envs(scene, camera, output)
        env_id_list = AuboCameraFns._normalize_env_ids(env_ids, num_envs)

        saved_paths: list[Path] = []
        for env_id in env_id_list:
            env_file_name = file_name
            # 如果外部指定了文件名，批量保存时自动追加 env id，避免不同环境互相覆盖。
            if env_file_name is not None:
                path = Path(env_file_name)
                env_file_name = f"{path.stem}_env{env_id}{path.suffix or '.png'}"
            saved_paths.append(
                AuboCameraFns.save_camera_image(
                    scene=scene,
                    camera=camera,
                    camera_name=camera_name,
                    output_dir=output_dir,
                    root_dir=root_dir,
                    data_type=data_type,
                    env_id=env_id,
                    file_name=env_file_name,
                    step=step,
                )
            )
        return saved_paths

class AuboEventFns:
    """Aubo 事件相关函数集合."""

    @staticmethod
    def goal_pos_w(
        env,
        env_ids: torch.Tensor | None = None,
        target_asset_name: str = TARGET_ASSET_NAME,
        goal_pos_name: str = "goal_pos_w",
    ) -> torch.Tensor:
        """返回并缓存目标世界坐标 (num_envs, 3)."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        goal_pos_w = AuboToolFns.get_root_pos_w(env, target_asset_name).clone()
        setattr(env, goal_pos_name, goal_pos_w)
        return goal_pos_w[env_ids]

def reset_planning_obstacle_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_radius=0.70,          # 椭球在 x/y 方向半轴 a
    z_radius=0.20,           # 椭球在 z 方向半轴 c，需 >= 0.2
    z_range=(0.30, 0.70),    # 只取这一段
    center=(0.0, 0.0, 0.5),  # 椭球中心，相对 env origin
):
    device = env.device
    target = AuboToolFns.get_asset(env, asset_cfg)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    n = len(env_ids)

    # -----------------------------
    # 1) 采样 z
    # -----------------------------
    z_min, z_max = z_range
    z = torch.rand(n, 1, device=device) * (z_max - z_min) + z_min

    # -----------------------------
    # 2) 根据椭球面方程，计算该 z 截面的圆半径
    #    (x^2 + y^2)/a^2 + (z-cz)^2/c^2 = 1
    # => r(z) = a * sqrt(1 - ((z-cz)^2 / c^2))
    # -----------------------------
    cz = center[2]
    inside = 1.0 - ((z - cz) ** 2) / (z_radius ** 2)
    inside = torch.clamp(inside, min=0.0)
    r = xy_radius * torch.sqrt(inside)

    # -----------------------------
    # 3) 在该圆上采样角度 theta
    # -----------------------------
    theta = 2.0 * torch.pi * torch.rand(n, 1, device=device)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # -----------------------------
    # 4) 拼成相对 env origin 的局部坐标
    # -----------------------------
    target_pos_local = torch.cat(
        [
            x + center[0],
            y + center[1],
            z,
        ],
        dim=-1,
    )  # [N, 3]

    # -----------------------------
    # 5) 转成世界坐标
    #    多环境下建议加 env origin
    # -----------------------------
    env_origins = env.scene.env_origins[env_ids]   # [N, 3]
    target_pos_w = env_origins + target_pos_local

    # -----------------------------
    # 6) 固定朝向 + 零速度
    # -----------------------------
    target_quat_w = torch.zeros((n, 4), device=device)
    target_quat_w[:, 0] = 1.0  # w, x, y, z

    target_lin_vel_w = torch.zeros((n, 3), device=device)
    target_ang_vel_w = torch.zeros((n, 3), device=device)

    target_state = torch.cat(
        [target_pos_w, target_quat_w, target_lin_vel_w, target_ang_vel_w], dim=-1
    )  # [N, 13]

    # -----------------------------
    # 7) 写回世界系根状态
    # -----------------------------
    target.write_root_state_to_sim(target_state, env_ids=env_ids)

class AuboRewardFns:
    """Aubo reaching / obstacle-aware reaching 的奖励函数集合."""

    @staticmethod
    def _get_action(env) -> torch.Tensor:
        """Best-effort access to the current raw policy action."""
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            return env.action_manager.action
        return torch.zeros((env.num_envs, 1), device=env.device)

    @staticmethod
    def _get_action_rate(env, act: torch.Tensor) -> torch.Tensor:
        """Best-effort action delta; returns zeros when previous action is unavailable."""
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "prev_action"):
            return act - env.action_manager.prev_action
        return torch.zeros_like(act)

    @staticmethod
    def reward_ee_distance_exp(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        std: float = 0.20,
    ) -> torch.Tensor:
        """稠密位置奖励: exp(-d / std)."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        return torch.exp(-dist / std)

    @staticmethod
    def reward_ee_progress(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """progress reward: d_prev - d_curr."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)

        if not hasattr(env, "_prev_ee_target_dist"):
            env._prev_ee_target_dist = dist.clone()

        # reset后同步，避免首步虚假progress
        just_reset = env.episode_length_buf == 0
        env._prev_ee_target_dist[just_reset] = dist[just_reset]

        reward = env._prev_ee_target_dist - dist
        env._prev_ee_target_dist = dist.clone()
        return reward

    @staticmethod
    def reward_success(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        threshold: float = 0.15,
        progress_ref: float = 0.015,
        action_norm_max: float = 0.75,
        action_norm_std: float = 0.50,
        action_rate_std: float = 0.75,
        w_progress: float = 0.45,
        w_action_mag: float = 0.30,
        w_action_smooth: float = 0.25,
        min_quality_score: float = 0.35,
    ) -> torch.Tensor:
        """成功奖励: 命中阈值后按末段推进、动作幅值和平滑性调制."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)

        buf_name = "_prev_success_quality_dist"
        if (not hasattr(env, buf_name)) or (getattr(env, buf_name).shape[0] != env.num_envs):
            setattr(env, buf_name, dist.clone())
        prev_dist = getattr(env, buf_name)

        if hasattr(env, "episode_length_buf"):
            just_reset = env.episode_length_buf == 0
            prev_dist[just_reset] = dist[just_reset]

        progress = torch.clamp(prev_dist - dist, min=0.0)
        setattr(env, buf_name, dist.clone())

        act = AuboRewardFns._get_action(env)
        action_norm = torch.norm(act, dim=-1)
        action_rate_norm = torch.norm(AuboRewardFns._get_action_rate(env, act), dim=-1)

        progress_score = torch.clamp(progress / max(progress_ref, 1e-6), 0.0, 1.0)
        action_excess = torch.clamp(action_norm - action_norm_max, min=0.0)
        action_mag_score = torch.exp(-action_excess / max(action_norm_std, 1e-6))
        action_smooth_score = torch.exp(-action_rate_norm / max(action_rate_std, 1e-6))

        quality = (
            w_progress * progress_score
            + w_action_mag * action_mag_score
            + w_action_smooth * action_smooth_score
        )
        quality = torch.clamp(quality, 0.0, 1.0)
        quality = min_quality_score + (1.0 - min_quality_score) * quality

        return (dist < threshold).float() * quality

    @staticmethod
    def reward_action_far_near(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        far_eps: float = 0.5,
        close_eps: float = 0.2,
        w_far_move: float = 0.10,
        w_near_ineff: float = 1.2,
        delta_min_far: float = 0.02,
        delta_min_near: float = 0.008,
        near_action_norm_max: float = 0.45,
    ) -> torch.Tensor:
        """远近分区动作奖励: 远区奖励有效大动作，近区惩罚低效大动作."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)

        buf_name = "_prev_action_far_near_dist"
        if (not hasattr(env, buf_name)) or (getattr(env, buf_name).shape[0] != env.num_envs):
            setattr(env, buf_name, dist.clone())
        prev_dist = getattr(env, buf_name)

        if hasattr(env, "episode_length_buf"):
            just_reset = env.episode_length_buf == 0
            prev_dist[just_reset] = dist[just_reset]

        progress = torch.clamp(prev_dist - dist, min=0.0)
        setattr(env, buf_name, dist.clone())

        act = AuboRewardFns._get_action(env)
        action_norm = torch.norm(act, dim=-1)
        max_action_norm = torch.sqrt(torch.tensor(float(act.shape[-1]), device=env.device))

        far_mask = dist > far_eps
        close_mask = dist < close_eps

        effective_far = torch.clamp(progress / max(delta_min_far, 1e-6), 0.0, 2.0)
        far_reward = w_far_move * action_norm * effective_far * far_mask.float()

        large_action = torch.clamp(
            (action_norm - near_action_norm_max) / torch.clamp(max_action_norm - near_action_norm_max, min=1e-6),
            0.0,
            1.0,
        )
        low_progress = torch.clamp((delta_min_near - progress) / max(delta_min_near, 1e-6), 0.0, 1.0)
        near_penalty = w_near_ineff * large_action * low_progress * close_mask.float()

        return far_reward - near_penalty

    @staticmethod
    def penalty_ee_obstacle_safe(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        obstacle_asset_name: str = TARGET_ASSET_NAME,
        safe_margin: float = 0.08,
    ) -> torch.Tensor:
        """末端接近障碍物中心的安全距离惩罚.
        第一版近似项：用末端-障碍中心距离代替真实几何最小距离.
        """
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, obstacle_asset_name)
        return torch.square(torch.clamp(safe_margin - dist, min=0.0))

    @staticmethod
    def penalty_action_l2(env) -> torch.Tensor:
        """动作幅值惩罚."""
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            act = env.action_manager.action
            return torch.sum(torch.square(act), dim=-1)
        return torch.zeros(env.num_envs, device=env.device)

    @staticmethod
    def penalty_action_rate_l2(env) -> torch.Tensor:
        """动作变化惩罚."""
        if (
            hasattr(env, "action_manager")
            and hasattr(env.action_manager, "action")
            and hasattr(env.action_manager, "prev_action")
        ):
            act = env.action_manager.action
            prev_act = env.action_manager.prev_action
            return torch.sum(torch.square(act - prev_act), dim=-1)
        return torch.zeros(env.num_envs, device=env.device)

    @staticmethod
    def penalty_step(env) -> torch.Tensor:
        """每步常数惩罚项，配合负权重使用."""
        return torch.ones(env.num_envs, device=env.device)  

class AuboTerminationFns:
    """Aubo reaching / obstacle-aware reaching 的终止函数集合."""

    @staticmethod
    def _get_optional_contact_sensor(env, sensor_cfg: SceneEntityCfg | str):
        sensor_name = AuboToolFns.resolve_asset_name(sensor_cfg)
        try:
            return env.scene[sensor_name]
        except Exception:
            return None

    @staticmethod
    def _extract_contact_magnitude(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None

        data = sensor.data
        candidates = [
            "net_forces_w",
            "body_net_forces_w",
            "contact_forces_w",
            "force_matrix_w",
            "contact_force_matrix_w",
        ]
        for attr in candidates:
            if hasattr(data, attr):
                tensor = getattr(data, attr)
                if isinstance(tensor, torch.Tensor) and tensor.shape[-1] >= 3:
                    return torch.norm(tensor[..., :3], dim=-1)
        return None


    @staticmethod
    def goal_reached(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        goal_pos_name: str = TARGET_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        pos_threshold: float = 0.15,
        required_consecutive_steps: int = 3,
    ) -> torch.Tensor:
        """Terminate when EE stays near goal for consecutive steps."""
        dist = AuboToolFns.ee_target_distance_w(env, asset_cfg, ee_frame_name, goal_pos_name)
        reached = dist < pos_threshold

        # 打印日志
        flags = reached
        hit_env_ids = torch.nonzero(flags, as_tuple=False).squeeze(-1)
        for env_id in hit_env_ids.detach().cpu().tolist():
            print(f"[Test] terminate env{env_id} true")


        buf_name = "_goal_reached_consecutive_steps"
        if (not hasattr(env, buf_name)) or (getattr(env, buf_name).shape[0] != env.num_envs):
            setattr(env, buf_name, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        counter = getattr(env, buf_name)

        counter = torch.where(reached, counter + 1, torch.zeros_like(counter))
        if hasattr(env, "episode_length_buf"):
            just_reset = env.episode_length_buf == 0
            counter[just_reset] = 0
        setattr(env, buf_name, counter)
        done = counter >= int(required_consecutive_steps)

        hit_env_ids1 = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for env_id1 in hit_env_ids1.detach().cpu().tolist():
            print(f"[Test] terminate env{env_id1} success")
        return done

    @staticmethod
    def ee_out_of_workspace(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        """Terminate when EE leaves an axis-aligned workspace in world frame."""
        if workspace is None:
            workspace = {
                "x": [-0.45, 0.45],
                "y": [-0.45, 0.45],
                "z": [0.05, 1.10],
            }

        # EE absolute world position
        ee_pos_w = AuboToolFns.get_body_pos_w(env, asset_cfg, ee_frame_name)   # [N, 3]

        # 每个并行环境自己的 origin
        env_origins = env.scene.env_origins                 # [N, 3]

        # 转到各自 env 局部坐标
        ee_pos_e = ee_pos_w - env_origins                   # [N, 3]

        x_min, x_max = float(workspace["x"][0]), float(workspace["x"][1])
        y_min, y_max = float(workspace["y"][0]), float(workspace["y"][1])
        z_min, z_max = float(workspace["z"][0]), float(workspace["z"][1])

        out_x = (ee_pos_e[:, 0] < x_min) | (ee_pos_e[:, 0] > x_max)
        out_y = (ee_pos_e[:, 1] < y_min) | (ee_pos_e[:, 1] > y_max)
        out_z = (ee_pos_e[:, 2] < z_min) | (ee_pos_e[:, 2] > z_max)

        done = out_x | out_y | out_z

        return done

    @staticmethod
    def is_terminated_by_illegal_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        body_names: Sequence[str] | None = None,
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """Best-effort illegal-collision termination for different sensor schemas."""
        del body_names
        mag = AuboTerminationFns._extract_contact_magnitude(
            AuboTerminationFns._get_optional_contact_sensor(env, sensor_cfg)
        )
        if mag is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if mag.ndim == 1:
            return mag > force_threshold
        return torch.any(mag > force_threshold, dim=tuple(range(1, mag.ndim)))

    @staticmethod
    def is_terminated_by_self_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """Best-effort self-collision termination for different sensor schemas."""
        mag = AuboTerminationFns._extract_contact_magnitude(
            AuboTerminationFns._get_optional_contact_sensor(env, sensor_cfg)
        )
        if mag is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if mag.ndim == 1:
            return mag > force_threshold
        return torch.any(mag > force_threshold, dim=tuple(range(1, mag.ndim)))

    @staticmethod
    def times_out(env) -> torch.Tensor:
        return mdp.time_out(env)

