from pathlib import Path
from typing import NotRequired, TypedDict

# === 类型别名 ===
# 三维位置/缩放向量，按 (x, y, z) 存储。
Vec3 = tuple[float, float, float]
# 四元数，按 Isaac Lab 常用的 (w, x, y, z) 存储。
Quat = tuple[float, float, float, float]

# === 基础资产目录 ===
# 本地 Isaac/S2R 资产统一根目录。后续所有 USD 文件路径都从这里派生。
ASSET_ROOT = Path("D:/project/S2R/Asset")

# === 场景对象命名 ===
# 这些名称用于 Isaac Lab scene 注册、env.scene[...] 访问，以及 SceneEntityCfg 引用。
ROBOT_ASSET_NAME = "AUBObot"
# 第一台 AUBO 机器人在场景中的 key，当前复用默认机器人名。
ROBOT_ASSET_NAME_1 = ROBOT_ASSET_NAME
# 第二台 AUBO 机器人在场景中的 key。
ROBOT_ASSET_NAME_2 = "AUBObot_2"
# 目标物体的场景 key，保留给需要 target 的任务或调试逻辑使用。
TARGET_ASSET_NAME = "target"
# AUBO 末端/法兰 body 名称，用于解析末端执行器位姿和雅可比。
EE_BODY_NAME = "Flange"
# TEST_ASSET_NAME = ""

# === 工作台基准位姿 ===
# 工作台在每个 env 局部世界坐标系中的基准位置。
WORKSTATION_POS = (1.3, 0.0, 0.0)
# 工作台在每个 env 局部世界坐标系中的基准旋转四元数。
WORKSTATION_ROT = (0.70711, 0.0, 0.0, -0.70711)
# 拆分后的工作台零部件 USD 根目录。
WORKSTATION_PART_ROOT = ASSET_ROOT / "QKL-HX-300-II-00" / "Part"

# === CameraSensor 传感器相机初始化部分 ===
# 这里配置的是场景里的 CameraSensor 传感器相机，用于采集 rgb/depth/segmentation 等图像。
# 入口值：传感器相机相对工作台基准位置的偏移。
CAMERA_WORKSTATION_OFFSET = (-0.5, 0.0, 0.9)
# 派生值：传感器相机在每个 env 局部世界坐标系中的初始位置。
CAMERA_INITIAL_POS = (
    WORKSTATION_POS[0] + CAMERA_WORKSTATION_OFFSET[0],
    WORKSTATION_POS[1] + CAMERA_WORKSTATION_OFFSET[1],
    WORKSTATION_POS[2] + CAMERA_WORKSTATION_OFFSET[2],
)
# 入口值：传感器相机初始朝向四元数。
CAMERA_INITIAL_ROT = (0.70711, 0.0, 0.70711, 0.0)
# 入口值：CameraCfg/AuboCameraFns 使用的相机坐标约定。
CAMERA_POSE_CONVENTION = "opengl"

# === 视物相机初始化部分 ===
# 这里配置的是 Isaac Sim GUI 视物窗口相机，不是 CameraSensor 传感器相机。
# 入口值只放在下面两个常量里：相对工作台的位置偏移，以及最终朝向的世界方向。
VIEWPORT_CAMERA_WORKSTATION_OFFSET = (-1.5, 0.2, 1.0)
VIEWPORT_CAMERA_FORWARD_W = (1.0, 0.0, 0.0)
# 派生值：视物窗口相机的 eye 位置，由工作台基准位置和相对偏移计算得到。
VIEWPORT_CAMERA_EYE = (
    WORKSTATION_POS[0] + VIEWPORT_CAMERA_WORKSTATION_OFFSET[0],
    WORKSTATION_POS[1] + VIEWPORT_CAMERA_WORKSTATION_OFFSET[1],
    WORKSTATION_POS[2] + VIEWPORT_CAMERA_WORKSTATION_OFFSET[2],
)
# 派生值：视物窗口相机的 target 点，由 eye 位置沿世界朝向方向前进一步得到。
VIEWPORT_CAMERA_TARGET = (
    VIEWPORT_CAMERA_EYE[0] + VIEWPORT_CAMERA_FORWARD_W[0],
    VIEWPORT_CAMERA_EYE[1] + VIEWPORT_CAMERA_FORWARD_W[1],
    VIEWPORT_CAMERA_EYE[2] + VIEWPORT_CAMERA_FORWARD_W[2],
)

# === AUBO 机器人安装位姿 ===
# 两台 AUBO 相对工作台 prim 的局部安装位置，作为人工记录/对齐参考。
AUBO_STATION_ROBOT_POS_1 = (0.034, -0.013, 0.816)
AUBO_STATION_ROBOT_POS_2 = (-0.81, 0.21, 0.816)

# 两台 AUBO 在每个 env 局部世界坐标系中的实际初始化位置。
# 当前值已经包含工作台旋转后的结果；WORKSTATION_ROT 约等于绕 Z 轴 -90 度。
AUBO_WORLD_ROBOT_POS_1 = (1.287, -0.034, 0.816)
AUBO_WORLD_ROBOT_POS_2 = (1.51, 0.81, 0.816)
# 两台 AUBO 的世界初始旋转，保持与工作台坐标系一致。
AUBO_WORLD_ROBOT_ROT = WORKSTATION_ROT

# === 工作台交互物体初始化基准 ===
# 入口值：交互物体整体相对工作台基准位置的高度/平移偏移。
# 后续 reagent、cap、dropper、syringe 等物体的局部表格都会基于该偏移派生世界位置。
WORKSTATION_INTERACTIVE_BASE_OFFSET = (0.285, 0.414, 0.94)

# 派生值：交互物体坐标基准点在每个 env 局部世界坐标系中的位置。
WORKSTATION_INTERACTIVE_BASE_POS = (
    WORKSTATION_POS[0] + WORKSTATION_INTERACTIVE_BASE_OFFSET[0],
    WORKSTATION_POS[1] + WORKSTATION_INTERACTIVE_BASE_OFFSET[1],
    WORKSTATION_POS[2] + WORKSTATION_INTERACTIVE_BASE_OFFSET[2],
)

# WORKSTATION_INTERACTIVE_BASE_ROT = WORKSTATION_ROT

# 入口值：交互物体局部表格转到世界时使用的基准旋转。当前使用单位四元数，不继承工作台旋转。
WORKSTATION_INTERACTIVE_BASE_ROT = (0.0, 0.0, 0.0, 1.0)

# 试剂瓶/瓶盖等零件常用的局部 Z 轴旋转四元数。
REAGENT_LOCAL_ROT_Z_POS_90 = (0.70711, 0.0, 0.0, 0.70711)
REAGENT_LOCAL_ROT_Z_NEG_90 = (0.70711, 0.0, 0.0, -0.70711)
# 交互物体默认缩放。只有个别资产需要显式写入 scale 时才引用。
REAGENT_DEFAULT_SCALE = (1.0, 1.0, 1.0)


class WorkstationInteractiveAssetPlacement(TypedDict):
    """工作台交互物体放置表项。local_* 是入口值，pos/rot 是派生后的世界位姿。"""

    name: str
    source_name: str
    group_label: str
    usd_path: Path
    scene_key: str
    local_pos: Vec3
    local_rot: Quat
    pos: Vec3
    rot: Quat
    scale: NotRequired[Vec3]


def _clean_pose_value(value: float) -> float:
    cleaned = round(value, 6)
    return 0.0 if cleaned == -0.0 else cleaned


def _clean_vec3(values: Vec3) -> Vec3:
    return (
        _clean_pose_value(values[0]),
        _clean_pose_value(values[1]),
        _clean_pose_value(values[2]),
    )


def _clean_quat(values: Quat) -> Quat:
    return (
        _clean_pose_value(values[0]),
        _clean_pose_value(values[1]),
        _clean_pose_value(values[2]),
        _clean_pose_value(values[3]),
    )


def _normalize_quat(quat: Quat) -> Quat:
    length = sum(component * component for component in quat) ** 0.5
    if length == 0.0:
        raise ValueError("Quaternion length must be non-zero.")
    return (
        quat[0] / length,
        quat[1] / length,
        quat[2] / length,
        quat[3] / length,
    )


def _quat_mul(
    lhs: Quat,
    rhs: Quat,
) -> Quat:
    lhs_w, lhs_x, lhs_y, lhs_z = lhs
    rhs_w, rhs_x, rhs_y, rhs_z = rhs
    return (
        lhs_w * rhs_w - lhs_x * rhs_x - lhs_y * rhs_y - lhs_z * rhs_z,
        lhs_w * rhs_x + lhs_x * rhs_w + lhs_y * rhs_z - lhs_z * rhs_y,
        lhs_w * rhs_y - lhs_x * rhs_z + lhs_y * rhs_w + lhs_z * rhs_x,
        lhs_w * rhs_z + lhs_x * rhs_y - lhs_y * rhs_x + lhs_z * rhs_w,
    )


def _quat_conjugate(quat: Quat) -> Quat:
    quat_w, quat_x, quat_y, quat_z = quat
    return (quat_w, -quat_x, -quat_y, -quat_z)


def _rotate_pos(
    rot: Quat,
    pos: Vec3,
) -> Vec3:
    normalized_rot = _normalize_quat(rot)
    rotated = _quat_mul(_quat_mul(normalized_rot, (0.0, *pos)), _quat_conjugate(normalized_rot))
    return (rotated[1], rotated[2], rotated[3])


def _interactive_world_pos(local_pos: Vec3) -> Vec3:
    rotated_pos = _rotate_pos(WORKSTATION_INTERACTIVE_BASE_ROT, local_pos)
    return _clean_vec3(
        (
            WORKSTATION_INTERACTIVE_BASE_POS[0] + rotated_pos[0],
            WORKSTATION_INTERACTIVE_BASE_POS[1] + rotated_pos[1],
            WORKSTATION_INTERACTIVE_BASE_POS[2] + rotated_pos[2],
        )
    )


def _interactive_world_rot(local_rot: Quat) -> Quat:
    return _clean_quat(
        _quat_mul(
            _normalize_quat(WORKSTATION_INTERACTIVE_BASE_ROT),
            _normalize_quat(local_rot),
        )
    )


def _interactive_asset_placement(
    *,
    name: str,
    source_name: str,
    group_label: str,
    usd_path: Path,
    scene_key: str,
    local_pos: Vec3,
    local_rot: Quat,
    scale: Vec3 | None = None,
) -> WorkstationInteractiveAssetPlacement:
    placement: WorkstationInteractiveAssetPlacement = {
        "name": name,
        "source_name": source_name,
        "group_label": group_label,
        "usd_path": usd_path,
        "scene_key": scene_key,
        "local_pos": local_pos,
        "local_rot": local_rot,
        "pos": _interactive_world_pos(local_pos),
        "rot": _interactive_world_rot(local_rot),
    }
    if scale is not None:
        placement["scale"] = scale
    return placement


# === 工作台交互物体放置表 ===
# 入口值：每个物体的本地位置 local_pos、本地旋转 local_rot、可选缩放 scale。
# 派生值：_interactive_asset_placement 会根据交互物体基准位姿计算世界 pos/rot。
# scene_key 会作为 Isaac Lab scene 中的对象 key；usd_path 指向对应拆分 USD 文件。
WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS: tuple[WorkstationInteractiveAssetPlacement, ...] = (
    _interactive_asset_placement(
        name="Reagent_01_sample_bottle",
        source_name="Reagent_01",
        group_label="sample_bottles",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_01" / "M_Reagent_01.usd",
        scene_key="ws_interactive_reagent_01_sample_bottle",
        local_pos=(0.158, 0.211, 0.0),
        local_rot=REAGENT_LOCAL_ROT_Z_POS_90,
    ),
    _interactive_asset_placement(
        name="Reagent_02_tray_bottle",
        source_name="Reagent_02",
        group_label="tray_bottles",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_02" / "M_Reagent_02.usd",
        scene_key="ws_interactive_reagent_02_tray_bottle",
        local_pos=(0.441, -0.17, 0.003),
        local_rot=REAGENT_LOCAL_ROT_Z_NEG_90,
    ),
    _interactive_asset_placement(
        name="ReagentCap_01_tray_head",
        source_name="ReagentCap_01",
        group_label="tray_caps",
        usd_path=WORKSTATION_PART_ROOT / "ReagentCap_01" / "M_ReagentCap_01.usd",
        scene_key="ws_interactive_reagent_cap_01_tray_head",
        local_pos=(0.441, -0.167, 0.048),
        local_rot=REAGENT_LOCAL_ROT_Z_NEG_90,
    ),
    _interactive_asset_placement(
        name="Reagent_03_brown_bottle_1",
        source_name="Reagent_03",
        group_label="brown_reagent_bottles",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_03" / "M_Reagent_03.usd",
        scene_key="ws_interactive_reagent_03_brown_bottle_1",
        local_pos=(0.651, -0.325, 0.003),
        local_rot=REAGENT_LOCAL_ROT_Z_POS_90,
    ),
    _interactive_asset_placement(
        name="Reagent_03_brown_bottle_2",
        source_name="Reagent_03",
        group_label="brown_reagent_bottles",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_03" / "M_Reagent_03.usd",
        scene_key="ws_interactive_reagent_03_brown_bottle_2",
        local_pos=(0.443, -0.388, -0.02),
        local_rot=REAGENT_LOCAL_ROT_Z_POS_90,
    ),
    _interactive_asset_placement(
        name="Reagent_04_dropper",
        source_name="Reagent_04",
        group_label="droppers",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_04" / "M_Reagent_04.usd",
        scene_key="ws_interactive_reagent_04_dropper",
        local_pos=(0.629, -0.451, 0.1),
        local_rot=REAGENT_LOCAL_ROT_Z_POS_90,
        scale=REAGENT_DEFAULT_SCALE,
    ),
    _interactive_asset_placement(
        name="Reagent_05_syringe",
        source_name="Reagent_05",
        group_label="syringes",
        usd_path=WORKSTATION_PART_ROOT / "Reagent_05" / "M_Reagent_05.usd",
        scene_key="ws_interactive_reagent_05_syringe",
        local_pos=(0.465, 0.056, 0.0),
        local_rot=REAGENT_LOCAL_ROT_Z_POS_90,
    ),
)

# === 常用 USD 资产路径 ===
# AUBO 机器人模型 USD 路径。
AUBO_ROBOT_USD = ASSET_ROOT / "AUBO_E5" / "AUBO_E5.usd"
# 实验室背景环境 USD 路径。
ENV_ASSET_USD  = ASSET_ROOT / "Laboratory" / "M_Laboratory.usd"
# 未拆分的工作台整包 USD 路径，保留给需要整站加载的脚本或调试逻辑使用。
STATION_ASSET_USD  = ASSET_ROOT / "QKL-HX-300-II-00" / "WorkStation_All.usd"
