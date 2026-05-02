import torch

from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp
from typing import Sequence
from asset import EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME
from tool import AuboToolFns

class AuboEventFns:
    """Aubo 事件相关函数集合."""

    @staticmethod
    def goal_pos_w(
        env,
        env_ids: torch.Tensor | None = None,
        target_asset_name: str = TARGET_ASSET_NAME,
        goal_pos_name: str = "goal_pos_w",
    ) -> torch.Tensor:
        """返回并缓存目标世界坐标 (num_envs, 3)."""
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        goal_pos_w = AuboToolFns.get_root_pos_w(env, target_asset_name).clone()
        setattr(env, goal_pos_name, goal_pos_w)
        return goal_pos_w[env_ids]

def reset_planning_obstacle_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    xy_radius=0.70,          # 椭球在 x/y 方向半轴 a
    z_radius=0.20,           # 椭球在 z 方向半轴 c，需 >= 0.2
    z_range=(0.30, 0.70),    # 只取这一段
    center=(0.0, 0.0, 0.5),  # 椭球中心，相对 env origin
):
    device = env.device
    target = AuboToolFns.get_asset(env, asset_cfg)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    n = len(env_ids)

    # -----------------------------
    # 1) 采样 z
    # -----------------------------
    z_min, z_max = z_range
    z = torch.rand(n, 1, device=device) * (z_max - z_min) + z_min

    # -----------------------------
    # 2) 根据椭球面方程，计算该 z 截面的圆半径
    #    (x^2 + y^2)/a^2 + (z-cz)^2/c^2 = 1
    # => r(z) = a * sqrt(1 - ((z-cz)^2 / c^2))
    # -----------------------------
    cz = center[2]
    inside = 1.0 - ((z - cz) ** 2) / (z_radius ** 2)
    inside = torch.clamp(inside, min=0.0)
    r = xy_radius * torch.sqrt(inside)

    # -----------------------------
    # 3) 在该圆上采样角度 theta
    # -----------------------------
    theta = 2.0 * torch.pi * torch.rand(n, 1, device=device)

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    # -----------------------------
    # 4) 拼成相对 env origin 的局部坐标
    # -----------------------------
    target_pos_local = torch.cat(
        [
            x + center[0],
            y + center[1],
            z,
        ],
        dim=-1,
    )  # [N, 3]

    # -----------------------------
    # 5) 转成世界坐标
    #    多环境下建议加 env origin
    # -----------------------------
    env_origins = env.scene.env_origins[env_ids]   # [N, 3]
    target_pos_w = env_origins + target_pos_local

    # -----------------------------
    # 6) 固定朝向 + 零速度
    # -----------------------------
    target_quat_w = torch.zeros((n, 4), device=device)
    target_quat_w[:, 0] = 1.0  # w, x, y, z

    target_lin_vel_w = torch.zeros((n, 3), device=device)
    target_ang_vel_w = torch.zeros((n, 3), device=device)

    target_state = torch.cat(
        [target_pos_w, target_quat_w, target_lin_vel_w, target_ang_vel_w], dim=-1
    )  # [N, 13]

    # -----------------------------
    # 7) 写回世界系根状态
    # -----------------------------
    target.write_root_state_to_sim(target_state, env_ids=env_ids)

class AuboRewardFns:
    """Aubo reaching / obstacle-aware reaching 的奖励函数集合."""

    @staticmethod
    def reward_ee_distance_exp(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        std: float = 0.20,
    ) -> torch.Tensor:
        """稠密位置奖励: exp(-d / std)."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)
        return torch.exp(-dist / std)

    @staticmethod
    def reward_ee_progress(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """progress reward: d_prev - d_curr."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)

        if not hasattr(env, "_prev_ee_target_dist"):
            env._prev_ee_target_dist = dist.clone()

        # reset后同步，避免首步虚假progress
        just_reset = env.episode_length_buf == 0
        env._prev_ee_target_dist[just_reset] = dist[just_reset]

        reward = env._prev_ee_target_dist - dist
        env._prev_ee_target_dist = dist.clone()
        return reward

    @staticmethod
    def reward_success(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_name: str = TARGET_ASSET_NAME,
        threshold: float = 0.15,
    ) -> torch.Tensor:
        """成功判定项，命中阈值返回1，否则0."""
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, target_asset_name)

        # 打印日志
        flags = (dist < threshold)
        hit_env_ids = torch.nonzero(flags, as_tuple=False).squeeze(-1)

        for env_id in hit_env_ids.detach().cpu().tolist():
            print(f"[Test] reward compute env{env_id} true")
        return (dist < threshold).float()

    @staticmethod
    def penalty_ee_obstacle_safe(
        env,
        robot_asset_name: str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        obstacle_asset_name: str = TARGET_ASSET_NAME,
        safe_margin: float = 0.08,
    ) -> torch.Tensor:
        """末端接近障碍物中心的安全距离惩罚.
        第一版近似项：用末端-障碍中心距离代替真实几何最小距离.
        """
        dist = AuboToolFns.ee_target_distance_w(env, robot_asset_name, ee_body_name, obstacle_asset_name)
        return torch.square(torch.clamp(safe_margin - dist, min=0.0))

    @staticmethod
    def penalty_action_l2(env) -> torch.Tensor:
        """动作幅值惩罚."""
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            act = env.action_manager.action
            return torch.sum(torch.square(act), dim=-1)
        return torch.zeros(env.num_envs, device=env.device)

    @staticmethod
    def penalty_action_rate_l2(env) -> torch.Tensor:
        """动作变化惩罚."""
        if (
            hasattr(env, "action_manager")
            and hasattr(env.action_manager, "action")
            and hasattr(env.action_manager, "prev_action")
        ):
            act = env.action_manager.action
            prev_act = env.action_manager.prev_action
            return torch.sum(torch.square(act - prev_act), dim=-1)
        return torch.zeros(env.num_envs, device=env.device)

    @staticmethod
    def penalty_step(env) -> torch.Tensor:
        """每步常数惩罚项，配合负权重使用."""
        return torch.ones(env.num_envs, device=env.device)  

class AuboTerminationFns:
    """Aubo reaching / obstacle-aware reaching 的终止函数集合."""

    @staticmethod
    def _get_optional_contact_sensor(env, sensor_cfg: SceneEntityCfg | str):
        sensor_name = AuboToolFns.resolve_asset_name(sensor_cfg)
        try:
            return env.scene[sensor_name]
        except Exception:
            return None

    @staticmethod
    def _extract_contact_magnitude(sensor) -> torch.Tensor | None:
        if sensor is None or not hasattr(sensor, "data"):
            return None

        data = sensor.data
        candidates = [
            "net_forces_w",
            "body_net_forces_w",
            "contact_forces_w",
            "force_matrix_w",
            "contact_force_matrix_w",
        ]
        for attr in candidates:
            if hasattr(data, attr):
                tensor = getattr(data, attr)
                if isinstance(tensor, torch.Tensor) and tensor.shape[-1] >= 3:
                    return torch.norm(tensor[..., :3], dim=-1)
        return None


    @staticmethod
    def goal_reached(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        goal_pos_name: str = TARGET_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        pos_threshold: float = 0.15,
        required_consecutive_steps: int = 3,
    ) -> torch.Tensor:
        """Terminate when EE stays near goal for consecutive steps."""
        dist = AuboToolFns.ee_target_distance_w(env, asset_cfg, ee_frame_name, goal_pos_name)
        reached = dist < pos_threshold

        # 打印日志
        flags = reached
        hit_env_ids = torch.nonzero(flags, as_tuple=False).squeeze(-1)
        for env_id in hit_env_ids.detach().cpu().tolist():
            print(f"[Test] terminate env{env_id} true")


        buf_name = "_goal_reached_consecutive_steps"
        if (not hasattr(env, buf_name)) or (getattr(env, buf_name).shape[0] != env.num_envs):
            setattr(env, buf_name, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        counter = getattr(env, buf_name)

        counter = torch.where(reached, counter + 1, torch.zeros_like(counter))
        if hasattr(env, "episode_length_buf"):
            just_reset = env.episode_length_buf == 0
            counter[just_reset] = 0
        setattr(env, buf_name, counter)
        done = counter >= int(required_consecutive_steps)

        hit_env_ids1 = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for env_id1 in hit_env_ids1.detach().cpu().tolist():
            print(f"[Test] terminate env{env_id1} success")
        return done

    @staticmethod
    def ee_out_of_workspace(
        env,
        asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_frame_name: str = EE_BODY_NAME,
        workspace: dict | None = None,
    ) -> torch.Tensor:
        """Terminate when EE leaves an axis-aligned workspace in world frame."""
        if workspace is None:
            workspace = {
                "x": [-0.45, 0.45],
                "y": [-0.45, 0.45],
                "z": [0.05, 1.10],
            }

        # EE absolute world position
        ee_pos_w = AuboToolFns.get_body_pos_w(env, asset_cfg, ee_frame_name)   # [N, 3]

        # 每个并行环境自己的 origin
        env_origins = env.scene.env_origins                 # [N, 3]

        # 转到各自 env 局部坐标
        ee_pos_e = ee_pos_w - env_origins                   # [N, 3]

        x_min, x_max = float(workspace["x"][0]), float(workspace["x"][1])
        y_min, y_max = float(workspace["y"][0]), float(workspace["y"][1])
        z_min, z_max = float(workspace["z"][0]), float(workspace["z"][1])

        out_x = (ee_pos_e[:, 0] < x_min) | (ee_pos_e[:, 0] > x_max)
        out_y = (ee_pos_e[:, 1] < y_min) | (ee_pos_e[:, 1] > y_max)
        out_z = (ee_pos_e[:, 2] < z_min) | (ee_pos_e[:, 2] > z_max)

        done = out_x | out_y | out_z

        return done

    @staticmethod
    def is_terminated_by_illegal_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        body_names: Sequence[str] | None = None,
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """Best-effort illegal-collision termination for different sensor schemas."""
        del body_names
        mag = AuboTerminationFns._extract_contact_magnitude(
            AuboTerminationFns._get_optional_contact_sensor(env, sensor_cfg)
        )
        if mag is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if mag.ndim == 1:
            return mag > force_threshold
        return torch.any(mag > force_threshold, dim=tuple(range(1, mag.ndim)))

    @staticmethod
    def is_terminated_by_self_collision(
        env,
        sensor_cfg: SceneEntityCfg | str = "contact_sensor",
        force_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """Best-effort self-collision termination for different sensor schemas."""
        mag = AuboTerminationFns._extract_contact_magnitude(
            AuboTerminationFns._get_optional_contact_sensor(env, sensor_cfg)
        )
        if mag is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if mag.ndim == 1:
            return mag > force_threshold
        return torch.any(mag > force_threshold, dim=tuple(range(1, mag.ndim)))

    @staticmethod
    def times_out(env) -> torch.Tensor:
        return mdp.time_out(env)

