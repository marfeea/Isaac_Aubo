# AUBO 机械臂 Lula IK 交互测试脚本

import argparse

import _bootstrap  # noqa: F401

from isaaclab.app import AppLauncher

# 添加命令行参数。
parser = argparse.ArgumentParser(description="在 IsaacLab 场景中测试 AUBO Lula IK 控制链。")
parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量。")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="最大仿真步数；0 表示持续运行直到应用关闭。",
)
# 添加 AppLauncher 参数。
AppLauncher.add_app_launcher_args(parser)
# 解析参数。
args_cli = parser.parse_args()

# 启动 Omniverse 应用。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene

from configs.asset import EE_BODY_NAME, ROBOT_ASSET_NAME
from configs.lula_cfg import AuboLulaIKControllerCfg
from configs.RenderCfg import TEST_RENDER_CFG
from configs.RLcfg import AuboRLSceneCfg
from tools.lula_ik import AuboLulaIKController


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    sim_dt = sim.get_physics_dt()
    count = 0
    total_steps = 0

    robot = scene[ROBOT_ASSET_NAME]
    aubo_entity = SceneEntityCfg(
        ROBOT_ASSET_NAME,
        joint_names=["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"],
        body_names=[EE_BODY_NAME],
        preserve_order=True,
    )
    aubo_entity.resolve(scene)
    aubo_ik_controller = AuboLulaIKController(
        AuboLulaIKControllerCfg(),
        num_envs=scene.num_envs,
        device=robot.device,
    )

    # 目标位姿位于机器人根坐标系，四元数顺序为 wxyz。
    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=robot.device)

    current_goal_idx = 0
    ik_commands = torch.zeros(scene.num_envs, aubo_ik_controller.action_dim, device=robot.device)
    joint_pos_des = robot.data.joint_pos[:, aubo_entity.joint_ids].clone()

    while simulation_app.is_running() and (args_cli.max_steps <= 0 or total_steps < args_cli.max_steps):
        if count % 400 == 0:
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            aubo_ik_controller.reset()
            ik_commands[:] = ee_goals[current_goal_idx]
            aubo_ik_controller.set_command(ik_commands)
            joint_pos_des = aubo_ik_controller.compute(joint_pos=joint_pos[:, aubo_entity.joint_ids])
            print(
                f"[Lula IK] goal={current_goal_idx} success={aubo_ik_controller.last_success.detach().cpu().tolist()}"
            )
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            joint_pos = robot.data.joint_pos[:, aubo_entity.joint_ids]
            joint_pos_des = aubo_ik_controller.compute(joint_pos=joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=aubo_entity.joint_ids)
        robot.write_data_to_sim()

        scene.write_data_to_sim()
        sim.step()
        count += 1
        total_steps += 1
        scene.update(sim_dt)


def main():
    """创建场景并运行 Lula IK 交互测试。"""
    # 初始化仿真上下文。
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(TEST_RENDER_CFG.viewport_camera_eye, TEST_RENDER_CFG.viewport_camera_target)
    # 创建场景。
    scene_cfg = AuboRLSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # 启动仿真。
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
