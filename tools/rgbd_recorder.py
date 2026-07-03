from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from configs.camera_cfg import CAMERA_CAPTURE_INTERVAL_S, CAMERA_CAPTURE_OUTPUT_DIR
from tools.camera import AuboCameraFns


class PeriodicRgbdRecorder:
    """按仿真时间周期保存场景相机的 RGB 与原始米制深度。"""

    def __init__(
        self,
        scene,
        camera_names: tuple[str, ...],
        script_name: str,
        interval_s: float = CAMERA_CAPTURE_INTERVAL_S,
        output_dir: str | Path = CAMERA_CAPTURE_OUTPUT_DIR,
    ) -> None:
        if interval_s <= 0.0:
            raise ValueError(f"Camera capture interval must be positive, got {interval_s}.")

        self.scene = scene
        self.camera_names = tuple(camera_names)
        self.script_name = str(script_name)
        self.interval_s = float(interval_s)
        self.next_capture_time_s = self.interval_s
        self.output_dir = self._resolve_output_dir(output_dir)

    def maybe_capture(self, step: int, sim_time_s: float) -> list[Path]:
        """到达下一个采集时刻时保存一组同步 RGB-D 数据。"""
        if float(sim_time_s) + 1.0e-9 < self.next_capture_time_s:
            return []

        paths = self.capture(step)
        while self.next_capture_time_s <= float(sim_time_s) + 1.0e-9:
            self.next_capture_time_s += self.interval_s
        return paths

    def capture(self, step: int) -> list[Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_paths: list[Path] = []

        for camera_name in self.camera_names:
            camera = AuboCameraFns.get_camera(scene=self.scene, camera_name=camera_name)
            AuboCameraFns._validate_camera_output(camera, "rgb", camera_name)
            AuboCameraFns._validate_camera_output(camera, "distance_to_image_plane", camera_name)

            num_envs = AuboCameraFns.infer_num_envs(
                scene=self.scene,
                camera=camera,
                output=camera.data.output["rgb"],
            )
            for env_id in range(num_envs):
                env_part = f"_env{env_id}" if num_envs > 1 else ""
                file_stem = (
                    f"{camera_name}_{self.script_name}{env_part}"
                    f"_step{int(step):08d}_{timestamp}"
                )
                rgb_path = AuboCameraFns.save_camera_image(
                    camera=camera,
                    output_dir=self.output_dir,
                    data_type="rgb",
                    env_id=env_id,
                    file_name=f"{file_stem}_rgb.png",
                )
                depth_path = self.output_dir / f"{file_stem}_depth.npy"
                depth = self._output_to_numpy(
                    camera.data.output["distance_to_image_plane"],
                    env_id,
                ).astype(np.float32, copy=False)
                np.save(depth_path, depth, allow_pickle=False)
                saved_paths.extend((rgb_path, depth_path))

        print(
            f"[CameraCapture] step={int(step)} saved={len(saved_paths)} files to {self.output_dir}",
            flush=True,
        )
        return saved_paths

    @staticmethod
    def _resolve_output_dir(output_dir: str | Path) -> Path:
        path = Path(output_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _output_to_numpy(output, env_id: int) -> np.ndarray:
        value = output[int(env_id)] if getattr(output, "ndim", 0) == 4 else output
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Unsupported depth output shape: {tuple(array.shape)}")
        return array
