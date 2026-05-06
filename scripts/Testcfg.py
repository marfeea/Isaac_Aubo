import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix, subtract_frame_transforms

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg


from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg
from logic import (
    reset_planning_obstacle_pose,
    AuboRewardFns,
    AuboTerminationFns,
)
from tool import AuboToolFns
from asset import AUBO_ROBOT_USD, EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME, ENV_ASSET_USD

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils


# AUBOcfg 遨博机械臂实体配置
posa=(0,0,0)
AUBO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(AUBO_ROBOT_USD),
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
        pos=posa,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.0,
            "Joint4": 0.0,
            "Joint5": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint[1-5]"],
            effort_limit_sim=2400.0,
            velocity_limit_sim=3.14,
            stiffness={
                "Joint1": 6000.0,
                "Joint2": 6000.0,
                "Joint3": 6000.0,
                "Joint4": 8000.0,
                "Joint5": 6000.0,
            },
            damping={
                "Joint1": 4000.0,
                "Joint2": 4000.0,
                "Joint3": 4000.0,
                "Joint4": 4000.0,
                "Joint5": 4000.0,
            },
        ),
    },
)


# 场景配置类
class TestSceneCfg(InteractiveSceneCfg):
    # # Ground-plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane", 
    #     spawn=sim_utils.GroundPlaneCfg(),)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0, 
            color=(0.75, 0.75, 0.75),
        ),
    )

    # 背景环境资产：作为静态 USD 场景加载，不注册为可控 articulation。
    my_env = AssetBaseCfg(
        prim_path="/World/Laboratory",
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


    # 配置需要的环境交互物
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/target",
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,   # 推荐：静态障碍物
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.6),   # 高度取半高，保证落在地面上
        ),
    )

    # robot
    AUBObot = AUBO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/AUBObot")
