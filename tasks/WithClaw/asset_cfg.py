from __future__ import annotations

from math import radians

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from configs.scene_cfg import AUBO_ROBOT_PLACEMENT_CFG
from tasks.WithClaw.task_cfg import WITH_CLAW_ROBOT_USD


INITIAL_JOINT_POS = {
    "Joint1": radians(0.0),
    "Joint2": radians(-30.0),
    "Joint3": radians(70.0),
    "Joint4": radians(45.0),
    "Joint5": radians(90.0),
    "Flange": radians(0.0),
    "UpperFinger": 0.0115,
    "DownFinger": 0.0,
}

WITH_CLAW_AUBO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(WITH_CLAW_ROBOT_USD),
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
        pos=AUBO_ROBOT_PLACEMENT_CFG.world_pos_1,
        rot=AUBO_ROBOT_PLACEMENT_CFG.world_rot,
        joint_pos=dict(INITIAL_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint.*", "Flange"],
            effort_limit_sim=2400.0,
            velocity_limit_sim=3.14,
            stiffness=6000.0,
            damping=600.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["UpperFinger", "DownFinger"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)


def make_with_claw_aubo_cfg(robot_name: str, world_pos: tuple[float, float, float]) -> ArticulationCfg:
    return WITH_CLAW_AUBO_CFG.replace(
        prim_path=f"{{ENV_REGEX_NS}}/{robot_name}",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=world_pos,
            rot=AUBO_ROBOT_PLACEMENT_CFG.world_rot,
            joint_pos=dict(INITIAL_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
    )
