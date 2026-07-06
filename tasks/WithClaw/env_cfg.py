from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from configs.camera_cfg import CAMERA_SENSOR_SCENE_NAMES
from tasks.WithClaw.events import EventCfg
from tasks.WithClaw.observations import ActionsCfg, ObservationsCfg
from tasks.WithClaw.rewards import RewardsCfg
from tasks.WithClaw.scene_cfg import TRAINING_ENV_SPACING, TRAINING_REPLICATE_PHYSICS, WithClawSceneCfg
from tasks.WithClaw.task_cfg import RL_DECIMATION, RL_EPISODE_LENGTH_S, RL_SIM_DT
from tasks.WithClaw.terminations import TerminationsCfg


@configclass
class WithClawEnvCfg(ManagerBasedRLEnvCfg):
    scene: WithClawSceneCfg = WithClawSceneCfg(
        num_envs=1,
        env_spacing=TRAINING_ENV_SPACING,
        replicate_physics=TRAINING_REPLICATE_PHYSICS,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = RL_DECIMATION
        self.episode_length_s = RL_EPISODE_LENGTH_S
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = RL_SIM_DT
        self.sim.render_interval = 4


def configure_fixed_target_state(env_cfg: WithClawEnvCfg, state_name: str | None) -> None:
    env_cfg.events.reset_target_pose.params["fixed_state_name"] = state_name


def disable_camera_sensors(env_cfg: WithClawEnvCfg) -> None:
    for camera_name in CAMERA_SENSOR_SCENE_NAMES:
        setattr(env_cfg.scene, camera_name, None)
