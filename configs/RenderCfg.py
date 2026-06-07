from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import isaaclab.sim as sim_utils

from configs.place_cfg import Vec3, WORKSTATION_POSE_CFG


@dataclass(frozen=True)
class RenderCfg:
    """Project render settings matching the test scene viewport preset."""

    viewport_camera_workstation_offset: Vec3 = (-1.5, 0.2, 1.0)
    viewport_camera_forward_w: Vec3 = (1.0, 0.0, 0.0)
    rendering_mode: str = "quality"
    renderer: str = "RaytracedLighting"
    enable_translucency: bool = True
    enable_reflections: bool = True
    enable_global_illumination: bool = False
    enable_direct_lighting: bool = True
    direct_lighting_samples_per_pixel: int = 2
    enable_shadows: bool = True
    enable_dl_denoiser: bool = True
    dlss_mode: int = 1
    runtime_settings: dict[str, Any] = field(
        default_factory=lambda: {
            # Renderer: Real-Time.
            "/rtx/rendermode": "RaytracedLighting",
            # Eco Mode.
            "/rtx/raytracing/cached/enabled": True,
            # Direct Lighting.
            "/rtx/directLighting/enabled": True,
            "/rtx/directLighting/sampledLighting/enabled": True,
            "/rtx/directLighting/sampledLighting/samplesPerPixel": 2,
            "/rtx/directLighting/maxRayIntensity": 6400.0,
            "/rtx/directLighting/meshLightSampling/enabled": True,
            "/rtx/shadows/enabled": True,
            # Indirect Diffuse Lighting.
            "/rtx/indirectDiffuse/enabled": False,
            # Reflections.
            "/rtx/reflections/enabled": True,
            "/rtx/reflections/samplesPerPixel": 1,
            "/rtx/reflections/maxRayIntensity": 19200.0,
            "/rtx/reflections/maxBounces": 1,
            "/rtx/reflections/roughnessCacheThreshold": 0.3,
            # Translucency.
            "/rtx/translucency/enabled": True,
            "/rtx/translucency/maxRefractionBounces": 6,
            "/rtx/translucency/reflectionSeenThroughRefraction": False,
            "/rtx/translucency/fractionalCutoutOpacity": True,
            "/rtx/translucency/depthCorrectionForDoF": True,
            "/rtx/translucency/motionVectorCorrection": True,
            "/rtx/translucency/worldEpsilonThreshold": 0.35,
            "/rtx/translucency/roughnessSampling": False,
            "/rtx/translucency/maxRayIntensity": 19200.0,
            "/rtx/translucency/invisibleLightBehindTranslucencyInReflections": False,
            # NVIDIA DLSS.
            "/rtx-transient/dldenoiser/enabled": True,
            "/rtx/post/dlss/execMode": 1,
        }
    )

    @property
    def viewport_camera_eye(self) -> Vec3:
        return (
            WORKSTATION_POSE_CFG.pos[0] + self.viewport_camera_workstation_offset[0],
            WORKSTATION_POSE_CFG.pos[1] + self.viewport_camera_workstation_offset[1],
            WORKSTATION_POSE_CFG.pos[2] + self.viewport_camera_workstation_offset[2],
        )

    @property
    def viewport_camera_target(self) -> Vec3:
        eye = self.viewport_camera_eye
        return (
            eye[0] + self.viewport_camera_forward_w[0],
            eye[1] + self.viewport_camera_forward_w[1],
            eye[2] + self.viewport_camera_forward_w[2],
        )

    def to_isaaclab(self) -> sim_utils.RenderCfg:
        """Build the IsaacLab RenderCfg used by SimulationCfg."""
        return sim_utils.RenderCfg(
            rendering_mode=self.rendering_mode,
            enable_translucency=self.enable_translucency,
            enable_reflections=self.enable_reflections,
            enable_global_illumination=self.enable_global_illumination,
            enable_direct_lighting=self.enable_direct_lighting,
            samples_per_pixel=self.direct_lighting_samples_per_pixel,
            enable_shadows=self.enable_shadows,
            enable_dl_denoiser=self.enable_dl_denoiser,
            dlss_mode=self.dlss_mode,
            carb_settings={
                "rtx.rendermode": self.renderer,
                "rtx.raytracing.cached.enabled": True,
                "rtx.directLighting.sampledLighting.enabled": True,
            },
        )

    def apply_runtime_settings(self) -> None:
        """Apply detailed RTX settings after SimulationContext has initialized."""
        import carb

        settings = carb.settings.get_settings()
        for key, value in self.runtime_settings.items():
            if isinstance(value, bool):
                settings.set_bool(key, value)
            elif isinstance(value, int):
                settings.set_int(key, value)
            elif isinstance(value, float):
                settings.set_float(key, value)
            elif isinstance(value, str):
                settings.set_string(key, value)
            else:
                settings.set(key, value)


TEST_RENDER_CFG = RenderCfg()
