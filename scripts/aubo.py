# aubo机械臂教师策略训练脚本

import argparse

import _bootstrap  # noqa: F401
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

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms


from configs.asset import EE_BODY_NAME, VIEWPORT_CAMERA_EYE, VIEWPORT_CAMERA_TARGET
from configs.RLcfg import StateOnlyObsCfg, ActionsCfg, EventCfg, AuboRLSceneCfg




# 训练环境类
class AuboRLEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置 todo 搞懂env_spacing的作用
    scene = AuboRLSceneCfg(num_envs = 512, env_spacing=2.5)
    # 基础设置 动作空间，观测空间
    observation = StateOnlyObsCfg()
    actions = ActionsCfg()
    events = EventCfg()


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    

    # 遨博测试代码
    AUBObot = scene["AUBObot"]

    # joint_deg = [0, 20, -30, 40, 10]
    # joint_rad = torch.tensor(
    #     [[x * math.pi / 180.0 for x in joint_deg]],
    #     dtype=torch.float32,
    #     device=AUBObot.device,
    # )
    # AUBObot.set_joint_position_target(joint_rad, joint_ids=[0, 1, 2, 3, 4])
    

    # IK测试代码，创建IKcontroller
    aubo_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    aubo_ik_con = DifferentialIKController(aubo_ik_cfg, num_envs=1, device=AUBObot.device)

    # 引用机械臂实例化后内容
    aubo_entity = SceneEntityCfg("AUBObot",joint_names=["Joint.*"], body_names=[EE_BODY_NAME])
    aubo_entity.resolve(scene)
    ee_jacobi_idx = aubo_entity.body_ids[0] - 1

    # 计算IK使用
    jacobian = AUBObot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, aubo_entity.joint_ids]
    ee_pose_w = AUBObot.data.body_pose_w[:,aubo_entity.body_ids[0]]
    root_pose_w = AUBObot.data.root_pose_w
    joint_pos = AUBObot.data.joint_pos[:,aubo_entity.joint_ids]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:,0:3],
        root_pose_w[:,3:7],
        ee_pose_w[:,0:3],
        ee_pose_w[:,3:7],
    )
    joint_pos_des = aubo_ik_con.compute(
        ee_pos_b,
        ee_quat_b,
        jacobian,
        joint_pos,
    )

    # 设定朝向位姿
    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # 初始化逻辑变量
    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, aubo_ik_con.action_dim, device=AUBObot.device)
    ik_commands[:] = ee_goals[current_goal_idx]
    ik_commands[:] = ee_goals[current_goal_idx]
    aubo_ik_con.set_command(ik_commands)

    
    


    while simulation_app.is_running():
        # reset
        if count % 400 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = AUBObot.data.default_joint_pos.clone()
            joint_vel = AUBObot.data.default_joint_vel.clone()
            AUBObot.write_joint_state_to_sim(joint_pos, joint_vel)
            AUBObot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, aubo_entity.joint_ids].clone()
            # reset controller
            aubo_ik_con.reset()
            aubo_ik_con.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = AUBObot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, aubo_entity.joint_ids]
            ee_pose_w = AUBObot.data.body_pose_w[:, aubo_entity.body_ids[0]]
            root_pose_w = AUBObot.data.root_pose_w
            joint_pos = AUBObot.data.joint_pos[:, aubo_entity.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = aubo_ik_con.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # 写入
        AUBObot.set_joint_position_target(joint_pos_des, joint_ids=aubo_entity.joint_ids)
        AUBObot.write_data_to_sim()

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
    sim.set_camera_view(VIEWPORT_CAMERA_EYE, VIEWPORT_CAMERA_TARGET)
    # Design scene
    scene_cfg = AuboRLSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    # Run the simulator
    run_simulator(sim, scene)



if __name__ == "__main__":
    main()
    simulation_app.close()
