# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# 自用测试脚本，用于测试IsaacSim的调用能力，理解IsaacSim

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from asset import ASSET_ROOT

# 移动小车配置
JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

# 摆臂机械臂配置
DOFBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Yahboom/Dofbot/dofbot.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
        },
        pos=(0.25, -0.25, 0.0),
    ),
    actuators={
        "front_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint3_act": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "joint4_act": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)


# 新增代码 openarm cfg, 创建两个openarm cfg

pos1=(0.5, 0.5, 0)
pos2=(0.5, -0.5, 0)
OPENARM_CONFIG1 = ArticulationCfg(
    # sim_utils，需要引入的工具库
    spawn=sim_utils.UsdFileCfg(
        # 引入usd文件
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd",
        #定义刚体信息
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),

    # 初始姿态
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            "openarm_finger_joint.*": 0.02,
        },
        pos=pos1,
    ),


    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_joint[1-7]"],
            velocity_limit_sim={
                "openarm_joint[1-2]": 2.175,
                "openarm_joint[3-4]": 2.175,
                "openarm_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_joint[1-2]": 40.0,
                "openarm_joint[3-4]": 27.0,
                "openarm_joint[5-7]": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_finger_joint.*"],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

OPENARM_CONFIG2= ArticulationCfg(
    # sim_utils，需引入的工具库
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/OpenArm/openarm_unimanual/openarm_unimanual.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_joint1": 1.57,
            "openarm_joint2": 0.0,
            "openarm_joint3": -1.57,
            "openarm_joint4": 1.57,
            "openarm_joint5": 0.0,
            "openarm_joint6": 0.0,
            "openarm_joint7": 0.0,
            "openarm_finger_joint.*": 0.044,
        },
        pos=pos2,
    ),
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=["openarm_joint[1-7]"],
            velocity_limit_sim={
                "openarm_joint[1-2]": 2.175,
                "openarm_joint[3-4]": 2.175,
                "openarm_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_joint[1-2]": 40.0,
                "openarm_joint[3-4]": 27.0,
                "openarm_joint[5-7]": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["openarm_finger_joint.*"],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# 新增代码 AUBOcfg 创建一个遨博机械臂
posa=(1,1,0)
AUBO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_ROOT}/AUBO_E5/AUBO_E5.usd",
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

# 可交互场景的配置
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Armbot1 = OPENARM_CONFIG1.replace(prim_path="{ENV_REGEX_NS}/Robot1")
    Armbot2 = OPENARM_CONFIG2.replace(prim_path="{ENV_REGEX_NS}/Robot2")
    AUBObot = AUBO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/AUBObot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # 遨博测试代码
    joint_deg = [0, 20, -30, 40, 10]
    joint_rad = torch.tensor(
        [[x * math.pi / 180.0 for x in joint_deg]],
        dtype=torch.float32,
        device=scene["AUBObot"].device,
    )

    scene["AUBObot"].set_joint_position_target(joint_rad, joint_ids=[0, 1, 2, 3, 4])
    scene["AUBObot"].write_data_to_sim()

    # IK测试代码，创建IKcontroller
    aubo_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    aubo_ik_con = DifferentialIKController(aubo_ik_cfg, num_envs=1, device=scene["AUBObot"].device)

    


    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_jetbot_state = scene["Jetbot"].data.default_root_state.clone()
            root_jetbot_state[:, :3] += scene.env_origins
            root_dofbot_state = scene["Dofbot"].data.default_root_state.clone()
            root_dofbot_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Jetbot"].write_root_pose_to_sim(root_jetbot_state[:, :7])
            scene["Jetbot"].write_root_velocity_to_sim(root_jetbot_state[:, 7:])
            scene["Dofbot"].write_root_pose_to_sim(root_dofbot_state[:, :7])
            scene["Dofbot"].write_root_velocity_to_sim(root_dofbot_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Jetbot"].data.default_joint_pos.clone(),
                scene["Jetbot"].data.default_joint_vel.clone(),
            )
            scene["Jetbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            joint_pos, joint_vel = (
                scene["Dofbot"].data.default_joint_pos.clone(),
                scene["Dofbot"].data.default_joint_vel.clone(),
            )
            scene["Dofbot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Jetbot and Dofbot state...")

        # drive around
        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[10.0, 10.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, 5.0]])

        scene["Jetbot"].set_joint_velocity_target(action)

        # wave
        wave_action = scene["Dofbot"].data.default_joint_pos
        wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["Dofbot"].set_joint_position_target(wave_action)


        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    # print("[INFO]: Setup complete...")
    # for jointname in scene["Armbot1"].data.joint_names.count:
    #     print("[TEST]:" + jointname)
    # print("[TEST]:" + scene["Armbot1"].data.default_joint_pos.shape)
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
