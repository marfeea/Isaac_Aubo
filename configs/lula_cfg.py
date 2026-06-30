from pathlib import Path

from isaaclab.utils import configclass

LULA_CONFIG_ROOT = Path(__file__).resolve().parent / "lula"
AUBO_LULA_ROBOT_DESCRIPTION = LULA_CONFIG_ROOT / "aubo_e5_robot_description.yaml"
AUBO_LULA_URDF = LULA_CONFIG_ROOT / "aubo_e5.urdf"


@configclass
class AuboLulaIKControllerCfg:
    """AUBO E5 的 Lula IK 求解配置。"""

    robot_description_path: str = str(AUBO_LULA_ROBOT_DESCRIPTION)
    urdf_path: str = str(AUBO_LULA_URDF)
    end_effector_frame_name: str = "AUBO_E5_Flange"

    # 顺序必须与 ActionTerm 解析出的 Isaac 关节顺序一致。
    lula_joint_names: tuple[str, ...] = (
        "Joint1",
        "Joint2",
        "Joint3",
        "Joint4",
        "Joint5",
        "Link_05_Flange",
    )

    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05

    # 每个物理步允许变化的最大关节目标，避免非线性 IK 全量解直接形成目标跳变。
    max_joint_delta: float | None = 0.025
