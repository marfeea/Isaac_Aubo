from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg

from configs.asset import (
    AUBO_ROBOT_USD,
    AUBO_WORLD_ROBOT_POS_1,
    AUBO_WORLD_ROBOT_POS_2,
    AUBO_WORLD_ROBOT_ROT,
    CAMERA_INITIAL_POS,
    CAMERA_INITIAL_ROT,
    CAMERA_POSE_CONVENTION,
    ENV_ASSET_USD,
    ROBOT_ASSET_NAME,
    ROBOT_ASSET_NAME_2,
    WORKSTATION_POS,
    WORKSTATION_ROT,
    WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS,
)
from configs.collision_cfg import ROBOT_CONTACT_SENSOR_CFG, make_target_contact_sensor_cfg
from configs.place_cfg import WorkstationTabletopLoadCfg, install_workstation_tabletop_scene_cfgs

import isaaclab.sim as sim_utils


TRAINING_ENV_SPACING = 25
TRAINING_REPLICATE_PHYSICS = True


AUBO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AUBO_ROBOT_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=AUBO_WORLD_ROBOT_POS_1,
        rot=AUBO_WORLD_ROBOT_ROT,
        joint_pos={
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.0,
            "Joint4": 0.0,
            "Joint5": 0.0,
            "Flange": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint.*", "Flange"],
            effort_limit_sim=2400.0,
            velocity_limit_sim=3.14,
            stiffness={
                "Joint1": 6000.0,
                "Joint2": 6000.0,
                "Joint3": 6000.0,
                "Joint4": 8000.0,
                "Joint5": 6000.0,
                "Flange": 6000.0,
            },
            damping={
                "Joint1": 4000.0,
                "Joint2": 4000.0,
                "Joint3": 4000.0,
                "Joint4": 4000.0,
                "Joint5": 4000.0,
                "Flange": 4000.0,
            },
        ),
    },
)


def make_aubo_cfg(robot_name: str, world_pos: tuple[float, float, float]) -> ArticulationCfg:
    return AUBO_CONFIG.replace(
        prim_path=f"{{ENV_REGEX_NS}}/{robot_name}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=world_pos,
            rot=AUBO_WORLD_ROBOT_ROT,
            joint_pos={
                "Joint1": 0.0,
                "Joint2": 0.0,
                "Joint3": 0.0,
                "Joint4": 0.0,
                "Joint5": 0.0,
                "Flange": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )


class AuboTrainingSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    my_env = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Laboratory",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ENV_ASSET_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=False,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    install_workstation_tabletop_scene_cfgs(
        locals(),
        WorkstationTabletopLoadCfg(
            prim_root="{ENV_REGEX_NS}/station",
            station_pos=WORKSTATION_POS,
            station_rot=WORKSTATION_ROT,
            include_missing_assets=False,
            strict_assets=False,
            create_parent_xforms=True,
        ),
    )

    AUBObot = make_aubo_cfg(ROBOT_ASSET_NAME, AUBO_WORLD_ROBOT_POS_1)
    AUBObot_2 = make_aubo_cfg(ROBOT_ASSET_NAME_2, AUBO_WORLD_ROBOT_POS_2)

    robot_contact_sensor = ROBOT_CONTACT_SENSOR_CFG
    target_contact_sensor = make_target_contact_sensor_cfg(
        f"{{ENV_REGEX_NS}}/station/interactive/{WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS[0]['name']}"
    )

    camera_cfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_INITIAL_POS,
            rot=CAMERA_INITIAL_ROT,
            convention=CAMERA_POSE_CONVENTION,
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )


def make_training_scene_cfg(num_envs: int = 1) -> AuboTrainingSceneCfg:
    return AuboTrainingSceneCfg(
        num_envs=num_envs,
        env_spacing=TRAINING_ENV_SPACING,
        replicate_physics=TRAINING_REPLICATE_PHYSICS,
    )
