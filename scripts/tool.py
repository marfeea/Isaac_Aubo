import torch

from isaaclab.managers import SceneEntityCfg

from asset import EE_BODY_NAME, ROBOT_ASSET_NAME, TARGET_ASSET_NAME


class AuboToolFns:
    """Aubo 环境逻辑复用工具函数集合。"""

    @staticmethod
    def resolve_asset_name(asset_cfg: SceneEntityCfg | str) -> str:
        """从 `SceneEntityCfg` 或字符串中解析场景实体名称。"""
        if isinstance(asset_cfg, str):
            return asset_cfg
        name = getattr(asset_cfg, "name", None)
        if isinstance(name, str):
            return name
        raise TypeError(
            f"Unsupported asset_cfg type '{type(asset_cfg)}'. "
            "Expected str or SceneEntityCfg (or object with .name)."
        )

    @staticmethod
    def get_asset(env, asset_cfg: SceneEntityCfg | str):
        """根据解析后的实体名称，从 `env.scene` 获取资产实例。"""
        return env.scene[AuboToolFns.resolve_asset_name(asset_cfg)]

    @staticmethod
    def get_first_body_id(asset, body_name: str) -> int:
        """在资产上按 `body_name` 匹配并返回第一个 body 的 id。"""
        body_ids = asset.find_bodies(body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(f"Body '{body_name}' not found in asset '{asset}'.")
        return int(body_ids[0])

    @staticmethod
    def get_body_pos_w(env, asset_cfg: SceneEntityCfg | str, body_name: str) -> torch.Tensor:
        """返回指定 body 的世界坐标位置，形状为 `(num_envs, 3)`。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        body_id = AuboToolFns.get_first_body_id(asset, body_name)
        return asset.data.body_pose_w[:, body_id, :3]

    @staticmethod
    def get_body_lin_vel_w(env, asset_cfg: SceneEntityCfg | str, body_name: str) -> torch.Tensor:
        """返回指定 body 的世界坐标系线速度，形状为 `(num_envs, 3)`。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        body_id = AuboToolFns.get_first_body_id(asset, body_name)
        return asset.data.body_lin_vel_w[:, body_id, :3]

    @staticmethod
    def get_root_pos_w(env, asset_cfg: SceneEntityCfg | str) -> torch.Tensor:
        """返回资产根节点世界坐标位置，形状为 `(num_envs, 3)`。"""
        asset = AuboToolFns.get_asset(env, asset_cfg)
        return asset.data.root_pos_w[:, :3]

    @staticmethod
    def ee_to_target_vec_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_cfg: SceneEntityCfg | str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到目标的世界坐标位移向量，形状为 `(num_envs, 3)`。"""
        ee_pos_w = AuboToolFns.get_body_pos_w(env, robot_asset_cfg, ee_body_name)
        target_pos_w = AuboToolFns.get_root_pos_w(env, target_asset_cfg)
        return target_pos_w - ee_pos_w

    @staticmethod
    def ee_target_distance_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        target_asset_cfg: SceneEntityCfg | str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到目标的欧氏距离，形状为 `(num_envs,)`。"""
        return torch.norm(
            AuboToolFns.ee_to_target_vec_w(env, robot_asset_cfg, ee_body_name, target_asset_cfg),
            dim=-1,
        )

    @staticmethod
    def ee_goal_distance_w(
        env,
        robot_asset_cfg: SceneEntityCfg | str = ROBOT_ASSET_NAME,
        ee_body_name: str = EE_BODY_NAME,
        goal_pos_name: str = "goal_pos_w",
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """返回末端到“目标点”(goal) 的欧氏距离，形状为 `(num_envs,)`。"""
        ee_pos_w = AuboToolFns.get_body_pos_w(env, robot_asset_cfg, ee_body_name)
        goal_pos_w = AuboToolFns.resolve_goal_pos_w(
            env,
            goal_pos_name=goal_pos_name,
            target_asset_name=target_asset_name,
        )
        return torch.norm(ee_pos_w - goal_pos_w, dim=-1)

    @staticmethod
    def resolve_goal_pos_w(
        env,
        goal_pos_name: str = "goal_pos_w",
        target_asset_name: str = TARGET_ASSET_NAME,
    ) -> torch.Tensor:
        """按优先级解析目标点世界坐标张量。"""
        # 1) 直接从 env 属性读取
        if hasattr(env, goal_pos_name):
            goal_pos = getattr(env, goal_pos_name)
            if isinstance(goal_pos, torch.Tensor) and goal_pos.ndim == 2 and goal_pos.shape[-1] >= 3:
                return goal_pos[:, :3]

        # 2) 若名称是 scene key，则读取该资产根坐标
        try:
            asset = env.scene[goal_pos_name]
            if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
                return asset.data.root_pos_w[:, :3]
        except Exception:
            pass

        # 3) 回退到本项目常用缓存字段 goal_pos_w
        if goal_pos_name != "goal_pos_w" and hasattr(env, "goal_pos_w"):
            goal_pos = getattr(env, "goal_pos_w")
            if isinstance(goal_pos, torch.Tensor) and goal_pos.ndim == 2 and goal_pos.shape[-1] >= 3:
                return goal_pos[:, :3]

        # 4) 最后回退到当前目标资产根坐标
        return AuboToolFns.get_root_pos_w(env, target_asset_name)
