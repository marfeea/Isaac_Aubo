from __future__ import annotations

import binascii
import struct
import zlib
from datetime import datetime
from pathlib import Path

import torch

from configs.camera_cfg import CAMERA_SENSOR_POSE_CFG
from tools.scene import AuboToolFns


class AuboCameraFns:
    """相机位姿设置与相机输出保存工具。"""

    @staticmethod
    def get_camera(scene=None, camera=None, camera_name: str = "camera_cfg"):
        """优先使用直接传入的 camera，否则从 scene[camera_name] 读取。"""
        if camera is not None:
            return camera
        if scene is None:
            raise ValueError("Either scene or camera must be provided.")
        return scene[camera_name]

    @staticmethod
    def infer_num_envs(scene=None, camera=None, output=None) -> int:
        """从 scene、camera 或输出张量推断并行环境数量。"""
        if scene is not None and hasattr(scene, "num_envs"):
            return int(scene.num_envs)
        if camera is not None and hasattr(camera, "num_envs"):
            return int(camera.num_envs)
        if output is not None and hasattr(output, "ndim") and output.ndim == 4:
            return int(output.shape[0])
        return 1

    @staticmethod
    def normalize_env_ids(env_ids, num_envs: int) -> list[int]:
        """把 None、int、Tensor 或序列统一成 Python int 列表。"""
        if env_ids is None:
            return list(range(num_envs))
        if isinstance(env_ids, int):
            return [int(env_ids)]
        if isinstance(env_ids, torch.Tensor):
            return [int(env_id) for env_id in env_ids.detach().cpu().flatten().tolist()]
        return [int(env_id) for env_id in env_ids]

    @staticmethod
    def set_camera_pose(
        scene=None,
        camera=None,
        camera_name: str = "camera_cfg",
        pos: tuple[float, float, float] | None = None,
        rot: tuple[float, float, float, float] | None = None,
        env_ids=None,
        relative_to_env_origins: bool = True,
        convention: str = CAMERA_SENSOR_POSE_CFG.pose_convention,
    ) -> None:
        """设置一个或多个并行环境中的相机世界位姿。"""
        camera = AuboCameraFns.get_camera(scene, camera, camera_name)
        num_envs = AuboCameraFns.infer_num_envs(scene, camera)
        env_id_list = AuboCameraFns.normalize_env_ids(env_ids, num_envs)
        pos = CAMERA_SENSOR_POSE_CFG.initial_pos if pos is None else pos
        rot = CAMERA_SENSOR_POSE_CFG.initial_rot if rot is None else rot

        device = getattr(camera, "device", None)
        if device is None and scene is not None and hasattr(scene, "device"):
            device = scene.device
        device = device or "cpu"

        positions = torch.tensor(pos, dtype=torch.float32, device=device).repeat(len(env_id_list), 1)
        if relative_to_env_origins and scene is not None and hasattr(scene, "env_origins"):
            env_id_tensor = torch.tensor(env_id_list, dtype=torch.long, device=scene.env_origins.device)
            positions = positions.to(scene.env_origins.device) + scene.env_origins[env_id_tensor]

        orientations = torch.tensor(rot, dtype=torch.float32, device=positions.device).repeat(len(env_id_list), 1)
        env_id_tensor = torch.tensor(env_id_list, dtype=torch.long, device=positions.device)

        if AuboCameraFns._try_set_world_poses(camera, positions, orientations, env_id_tensor, convention):
            return

        view = getattr(camera, "_view", None)
        if view is not None and AuboCameraFns._try_set_world_poses(view, positions, orientations, env_id_tensor, None):
            return

        raise AttributeError(f"Camera '{camera_name}' does not support setting world poses.")

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
        """保存单个环境的一张相机图片，并返回保存路径。"""
        camera = AuboCameraFns.get_camera(scene, camera, camera_name)
        AuboCameraFns._validate_camera_output(camera, data_type, camera_name)

        save_dir = AuboCameraFns._resolve_output_dir(output_dir, root_dir)
        save_path = save_dir / AuboCameraFns._resolve_file_name(file_name, data_type, env_id, step)

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
        """批量保存多个环境的相机图片，并返回所有保存路径。"""
        camera = AuboCameraFns.get_camera(scene, camera, camera_name)
        AuboCameraFns._validate_camera_output(camera, data_type, camera_name)

        output = camera.data.output[data_type]
        num_envs = AuboCameraFns.infer_num_envs(scene, camera, output)
        env_id_list = AuboCameraFns.normalize_env_ids(env_ids, num_envs)

        return [
            AuboCameraFns.save_camera_image(
                scene=scene,
                camera=camera,
                camera_name=camera_name,
                output_dir=output_dir,
                root_dir=root_dir,
                data_type=data_type,
                env_id=env_id,
                file_name=AuboCameraFns._file_name_for_env(file_name, env_id),
                step=step,
            )
            for env_id in env_id_list
        ]

    @staticmethod
    def _try_set_world_poses(target, positions, orientations, env_id_tensor, convention: str | None) -> bool:
        if not hasattr(target, "set_world_poses"):
            return False
        call_variants = (
            {"convention": convention, "env_ids": env_id_tensor} if convention is not None else None,
            {"env_ids": env_id_tensor},
            {"indices": env_id_tensor},
            "positional_indices",
            {},
        )
        for kwargs in call_variants:
            if kwargs is None:
                continue
            try:
                if kwargs == "positional_indices":
                    target.set_world_poses(positions, orientations, env_id_tensor)
                else:
                    target.set_world_poses(positions, orientations, **kwargs)
                return True
            except TypeError:
                continue
        return False

    @staticmethod
    def _validate_camera_output(camera, data_type: str, camera_name: str) -> None:
        if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
            raise ValueError(f"Camera '{camera_name}' does not expose data.output.")
        if data_type not in camera.data.output:
            available = ", ".join(camera.data.output.keys())
            raise KeyError(f"Camera output '{data_type}' not found. Available: {available}")

    @staticmethod
    def _resolve_output_dir(output_dir: str | Path | None, root_dir: str | Path | None) -> Path:
        base_dir = Path(root_dir).resolve() if root_dir is not None else AuboToolFns.project_root()
        save_dir = Path(output_dir) if output_dir is not None else base_dir / "picture"
        if not save_dir.is_absolute():
            save_dir = base_dir / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    @staticmethod
    def _resolve_file_name(file_name: str | None, data_type: str, env_id: int, step: int | None) -> str:
        if file_name is not None:
            return file_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        step_part = f"_step{step}" if step is not None else ""
        return f"{data_type}_env{int(env_id)}{step_part}_{timestamp}.png"

    @staticmethod
    def _file_name_for_env(file_name: str | None, env_id: int) -> str | None:
        if file_name is None:
            return None
        path = Path(file_name)
        return f"{path.stem}_env{env_id}{path.suffix or '.png'}"

    @staticmethod
    def _camera_output_to_uint8(output, env_id: int):
        import numpy as np

        image = output
        if image.ndim == 4:
            image = image[int(env_id)]
        elif image.ndim not in (2, 3):
            raise ValueError(f"Unsupported camera output shape: {tuple(image.shape)}")

        array = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)

        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]

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
    def _write_png(path: Path, image_array) -> None:
        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", binascii.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        height, width = image_array.shape[:2]
        if image_array.ndim == 2:
            color_type = 0
        elif image_array.shape[2] == 3:
            color_type = 2
        elif image_array.shape[2] == 4:
            color_type = 6
        else:
            raise ValueError(f"Unsupported image shape for PNG: {image_array.shape}")

        raw_rows = b"".join(b"\x00" + image_array[row].tobytes() for row in range(height))
        header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw_rows))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png_bytes)
