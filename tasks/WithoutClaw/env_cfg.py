from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from configs.camera_cfg import CAMERA_SENSOR_SCENE_NAMES
from tasks.WithoutClaw.collision_cfg import make_target_contact_sensor_cfg
from tasks.WithoutClaw.events import EventCfg, make_reset_target_pose_event
from tasks.WithoutClaw.observations import ActionsCfg, ObservationsCfg
from tasks.WithoutClaw.rewards import RewardsCfg
from tasks.WithoutClaw.scene_cfg import TRAINING_ENV_SPACING, TRAINING_REPLICATE_PHYSICS, WithoutClawSceneCfg
from tasks.WithoutClaw.task_cfg import (
    DEFAULT_TARGET_ASSET_NAME,
    RL_DECIMATION,
    RL_EPISODE_LENGTH_S,
    RL_SIM_DT,
)
from tasks.WithoutClaw.terminations import TerminationsCfg


DEFAULT_RL_TARGET_ASSET_NAME = DEFAULT_TARGET_ASSET_NAME


@configclass
class WithoutClawEnvCfg(ManagerBasedRLEnvCfg):
    scene: WithoutClawSceneCfg = WithoutClawSceneCfg(
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


def configure_task_target(
    env_cfg: WithoutClawEnvCfg,
    target_asset_name: str,
    *,
    randomize_target_pose: bool = False,
) -> None:
    """把旧任务的全部观测、奖励和终止项路由到同一 target。"""
    env_cfg.observations.policy.target_pos.params["asset_cfg"] = SceneEntityCfg(target_asset_name)
    env_cfg.observations.policy.to_target.params["target_asset_cfg"] = SceneEntityCfg(target_asset_name)
    env_cfg.actions.task_space_ik.target_asset_name = target_asset_name
    for term_name in ("ee_progress", "ee_distance_exp", "success", "action_far_near", "target_contact_penalty", "collision_penalty"):
        getattr(env_cfg.rewards, term_name).params["target_asset_name"] = target_asset_name
    env_cfg.terminations.goal_reached.params["goal_pos_name"] = target_asset_name
    env_cfg.terminations.obstacle_collision.params["target_asset_name"] = target_asset_name
    target_cfg = getattr(env_cfg.scene, target_asset_name)
    env_cfg.scene.target_contact_sensor = make_target_contact_sensor_cfg(target_cfg.prim_path)
    env_cfg.events.reset_obstacle_pose = (
        make_reset_target_pose_event(target_asset_name) if randomize_target_pose else None
    )


def disable_camera_sensors(env_cfg: WithoutClawEnvCfg) -> None:
    for camera_name in CAMERA_SENSOR_SCENE_NAMES:
        setattr(env_cfg.scene, camera_name, None)
