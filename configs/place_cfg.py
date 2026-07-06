"""Workstation split-USD asset placement configuration.

This module owns asset grouping, USD paths, and initial poses. It intentionally
does not add or modify physics collision schemas.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NotRequired, TypedDict

from configs.asset import ASSET_ROOT

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class WorkstationPoseCfg:
    """Base workstation pose in each environment's local world frame."""

    pos: Vec3 = (1.3, 0.0, 0.0)
    rot: Quat = (0.70711, 0.0, 0.0, -0.70711)
    part_root: Path = ASSET_ROOT / "QKL-HX-300-II-00" / "Part"


WORKSTATION_POSE_CFG = WorkstationPoseCfg()


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


def _quat_mul(lhs: Quat, rhs: Quat) -> Quat:
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


def rotate_vec3(rot: Quat, pos: Vec3) -> Vec3:
    normalized_rot = _normalize_quat(rot)
    rotated = _quat_mul(_quat_mul(normalized_rot, (0.0, *pos)), _quat_conjugate(normalized_rot))
    return (rotated[1], rotated[2], rotated[3])


def workstation_local_to_world_pos(local_pos: Vec3, pose_cfg: WorkstationPoseCfg = WORKSTATION_POSE_CFG) -> Vec3:
    rotated_pos = rotate_vec3(pose_cfg.rot, local_pos)
    return _clean_vec3(
        (
            pose_cfg.pos[0] + rotated_pos[0],
            pose_cfg.pos[1] + rotated_pos[1],
            pose_cfg.pos[2] + rotated_pos[2],
        )
    )


class WorkstationInteractiveAssetPlacement(TypedDict):
    """Placement table item. local_* values are inputs; pos/rot are derived world poses."""

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
    rigid_object: NotRequired[bool]


@dataclass(frozen=True)
class WorkstationInteractivePlacementCfg:
    """Base transform for movable tabletop assets."""

    base_offset: Vec3 = (0.285, 0.414, 0.94)
    base_rot: Quat = (0.0, 0.0, 0.0, 1.0)
    local_rot_z_pos_90: Quat = (0.70711, 0.0, 0.0, 0.70711)
    local_rot_z_neg_90: Quat = (0.70711, 0.0, 0.0, -0.70711)
    default_scale: Vec3 = (1.0, 1.0, 1.0)
    workstation_pose: WorkstationPoseCfg = WORKSTATION_POSE_CFG

    @property
    def base_pos(self) -> Vec3:
        return _clean_vec3(
            (
                self.workstation_pose.pos[0] + self.base_offset[0],
                self.workstation_pose.pos[1] + self.base_offset[1],
                self.workstation_pose.pos[2] + self.base_offset[2],
            )
        )


WORKSTATION_INTERACTIVE_PLACEMENT_CFG = WorkstationInteractivePlacementCfg()


def _interactive_world_pos(
    local_pos: Vec3,
    cfg: WorkstationInteractivePlacementCfg = WORKSTATION_INTERACTIVE_PLACEMENT_CFG,
) -> Vec3:
    rotated_pos = rotate_vec3(cfg.base_rot, local_pos)
    return _clean_vec3(
        (
            cfg.base_pos[0] + rotated_pos[0],
            cfg.base_pos[1] + rotated_pos[1],
            cfg.base_pos[2] + rotated_pos[2],
        )
    )


def _interactive_world_rot(
    local_rot: Quat,
    cfg: WorkstationInteractivePlacementCfg = WORKSTATION_INTERACTIVE_PLACEMENT_CFG,
) -> Quat:
    return _clean_quat(
        _quat_mul(
            _normalize_quat(cfg.base_rot),
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
    rigid_object: bool = False,
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
    if rigid_object:
        placement["rigid_object"] = True
    return placement


def _numbered(prefix: str, start: int, end: int) -> tuple[str, ...]:
    """Generate prim names with two-digit suffixes, for example M_Reagent_01."""
    return tuple(f"{prefix}_{index:02d}" for index in range(start, end + 1))


class CollisionBodyKind(str, Enum):
    """Semantic group for workstation tabletop assets."""

    STATIC = "static"
    INTERACTIVE = "interactive"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class CollisionBodyGroup:
    """A named group of workstation prims with the same semantic behavior."""

    label: str
    kind: CollisionBodyKind
    names: tuple[str, ...]

    @property
    def movable(self) -> bool:
        return self.kind in {CollisionBodyKind.INTERACTIVE, CollisionBodyKind.DYNAMIC}


STATIC_WORKSTATION = CollisionBodyGroup(
    label="workstation",
    kind=CollisionBodyKind.STATIC,
    names=(
        "WorkStation",
        "M_MainFrame_01",
        "M_MainFrame_02",
    ),
)

STATIC_SUPPORT_TRAYS = CollisionBodyGroup(
    label="support_trays",
    kind=CollisionBodyKind.STATIC,
    names=tuple(name for name in _numbered("M_SupportTray", 1, 14) if name != "M_SupportTray_07"),
)

SAMPLE_BOTTLES = CollisionBodyGroup(
    label="sample_bottles",
    kind=CollisionBodyKind.INTERACTIVE,
    names=("Reagent_01_sample_bottle",),
)

TRAY_BOTTLES = CollisionBodyGroup(
    label="tray_bottles",
    kind=CollisionBodyKind.INTERACTIVE,
    names=("Reagent_02_tray_bottle",),
)

TRAY_CAPS = CollisionBodyGroup(
    label="tray_caps",
    kind=CollisionBodyKind.INTERACTIVE,
    names=("ReagentCap_01_tray_head",),
)

BROWN_REAGENT_BOTTLES = CollisionBodyGroup(
    label="brown_reagent_bottles",
    kind=CollisionBodyKind.INTERACTIVE,
    names=(
        "Reagent_03_brown_bottle_1",
        "Reagent_03_brown_bottle_2",
    ),
)

DROPPERS = CollisionBodyGroup(
    label="droppers",
    kind=CollisionBodyKind.INTERACTIVE,
    names=("Reagent_04_dropper",),
)

SYRINGES = CollisionBodyGroup(
    label="syringes",
    kind=CollisionBodyKind.INTERACTIVE,
    names=("Reagent_05_syringe",),
)

CLAW_TOOLS = CollisionBodyGroup(
    label="claw_tools",
    kind=CollisionBodyKind.DYNAMIC,
    names=(
        "ClawTool_A",
        "ClawTool_B",
        "ClawTool_C",
    ),
)

WORKSTATION_COLLISION_GROUPS: tuple[CollisionBodyGroup, ...] = (
    STATIC_WORKSTATION,
    STATIC_SUPPORT_TRAYS,
    SAMPLE_BOTTLES,
    TRAY_BOTTLES,
    TRAY_CAPS,
    BROWN_REAGENT_BOTTLES,
    DROPPERS,
    SYRINGES,
    CLAW_TOOLS,
)

STATIC_COLLIDER_NAMES = frozenset(
    name for group in WORKSTATION_COLLISION_GROUPS if group.kind == CollisionBodyKind.STATIC for name in group.names
)
INTERACTIVE_COLLIDER_NAMES = frozenset(
    name
    for group in WORKSTATION_COLLISION_GROUPS
    if group.kind == CollisionBodyKind.INTERACTIVE
    for name in group.names
)
DYNAMIC_COLLIDER_NAMES = frozenset(
    name for group in WORKSTATION_COLLISION_GROUPS if group.kind == CollisionBodyKind.DYNAMIC for name in group.names
)
MOVABLE_COLLIDER_NAMES = INTERACTIVE_COLLIDER_NAMES | DYNAMIC_COLLIDER_NAMES
ALL_COLLIDER_NAMES = STATIC_COLLIDER_NAMES | MOVABLE_COLLIDER_NAMES


@dataclass(frozen=True)
class CollisionApplyReport:
    """Report which configured prim names were found in a stage."""

    matched: dict[str, tuple[str, ...]]
    missing: tuple[str, ...]

    @property
    def matched_count(self) -> int:
        return sum(len(paths) for paths in self.matched.values())

    @property
    def missing_count(self) -> int:
        return len(self.missing)


WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS: tuple[WorkstationInteractiveAssetPlacement, ...] = (
    _interactive_asset_placement(
        name="Reagent_01_sample_bottle",
        source_name="Reagent_01",
        group_label=SAMPLE_BOTTLES.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_01" / "M_Reagent_01.usd",
        scene_key="ws_interactive_reagent_01_sample_bottle",
        local_pos=(0.048, 0.211, 0.0),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_pos_90,
        rigid_object=True,
    ),
    _interactive_asset_placement(
        name="Reagent_02_tray_bottle",
        source_name="Reagent_02",
        group_label=TRAY_BOTTLES.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_02" / "M_Reagent_02.usd",
        scene_key="ws_interactive_reagent_02_tray_bottle",
        local_pos=(0.441, -0.17, 0.003),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_neg_90,
    ),
    _interactive_asset_placement(
        name="ReagentCap_01_tray_head",
        source_name="ReagentCap_01",
        group_label=TRAY_CAPS.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "ReagentCap_01" / "M_ReagentCap_01.usd",
        scene_key="ws_interactive_reagent_cap_01_tray_head",
        local_pos=(0.441, -0.167, 0.048),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_neg_90,
    ),
    _interactive_asset_placement(
        name="Reagent_03_brown_bottle_1",
        source_name="Reagent_03",
        group_label=BROWN_REAGENT_BOTTLES.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_03" / "M_Reagent_03.usd",
        scene_key="ws_interactive_reagent_03_brown_bottle_1",
        local_pos=(0.651, -0.325, 0.003),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_pos_90,
    ),
    _interactive_asset_placement(
        name="Reagent_03_brown_bottle_2",
        source_name="Reagent_03",
        group_label=BROWN_REAGENT_BOTTLES.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_03" / "M_Reagent_03.usd",
        scene_key="ws_interactive_reagent_03_brown_bottle_2",
        local_pos=(0.443, -0.388, -0.02),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_pos_90,
    ),
    _interactive_asset_placement(
        name="Reagent_04_dropper",
        source_name="Reagent_04",
        group_label=DROPPERS.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_04" / "M_Reagent_04.usd",
        scene_key="ws_interactive_reagent_04_dropper",
        local_pos=(0.629, -0.451, 0.1),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_pos_90,
        scale=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.default_scale,
    ),
    _interactive_asset_placement(
        name="Reagent_05_syringe",
        source_name="Reagent_05",
        group_label=SYRINGES.label,
        usd_path=WORKSTATION_POSE_CFG.part_root / "Reagent_05" / "M_Reagent_05.usd",
        scene_key="ws_interactive_reagent_05_syringe",
        local_pos=(0.465, 0.056, 0.0),
        local_rot=WORKSTATION_INTERACTIVE_PLACEMENT_CFG.local_rot_z_pos_90,
    ),
)


@dataclass(frozen=True)
class WorkstationAssetSpec:
    """Load description for one split workstation USD asset."""

    name: str
    kind: CollisionBodyKind
    group_label: str
    usd_path: Path
    scene_key: str
    init_pos: tuple[float, float, float] | None = None
    init_rot: tuple[float, float, float, float] | None = None
    scale: tuple[float, float, float] | None = None
    rigid_object: bool = False

    @property
    def exists(self) -> bool:
        return self.usd_path.exists()




@dataclass(frozen=True)
class WorkstationTabletopLoadCfg:
    """Build Isaac Lab scene cfg entries for split workstation tabletop USDs.

    This module only loads assets and keeps their semantic grouping/placement.
    It does not author collision or rigid-body schemas onto the spawned USD prims.
    """

    prim_root: str = "{ENV_REGEX_NS}/station"
    station_pos: tuple[float, float, float] = WORKSTATION_POSE_CFG.pos
    station_rot: tuple[float, float, float, float] = WORKSTATION_POSE_CFG.rot
    specs: tuple[WorkstationAssetSpec, ...] = ()
    include_missing_assets: bool = False
    strict_assets: bool = False
    create_parent_xforms: bool = True

    def __post_init__(self) -> None:
        if not self.specs:
            object.__setattr__(self, "specs", WORKSTATION_TABLETOP_ASSET_SPECS)

    def missing_asset_specs(self) -> tuple[WorkstationAssetSpec, ...]:
        return tuple(spec for spec in self.specs if not spec.exists)

    def to_scene_cfgs(self) -> dict[str, object]:
        missing_specs = self.missing_asset_specs()
        if self.strict_assets and missing_specs:
            missing_names = ", ".join(spec.name for spec in missing_specs)
            raise FileNotFoundError(f"Missing workstation split USD assets: {missing_names}")

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

        scene_cfgs: dict[str, object] = {}
        xform_spawn = sim_utils.SpawnerCfg(func=sim_utils.clone(_spawn_empty_xform_prim))

        if self.create_parent_xforms:
            scene_cfgs["AA_ws_station_root"] = AssetBaseCfg(
                prim_path=self.prim_root,
                spawn=xform_spawn,
            )
            for kind in CollisionBodyKind:
                scene_cfgs[f"AB_ws_{kind.value}_root"] = AssetBaseCfg(
                    prim_path=f"{self.prim_root}/{kind.value}",
                    spawn=xform_spawn,
                )

        for spec in self.specs:
            if not spec.exists and not self.include_missing_assets:
                continue

            usd_spawn_kwargs = {"usd_path": str(spec.usd_path)}
            if spec.scale is not None:
                usd_spawn_kwargs["scale"] = spec.scale

            asset_cfg_type = RigidObjectCfg if spec.rigid_object else AssetBaseCfg
            scene_cfgs[spec.scene_key] = asset_cfg_type(
                prim_path=f"{self.prim_root}/{spec.kind.value}/{spec.name}",
                spawn=sim_utils.UsdFileCfg(**usd_spawn_kwargs),
                init_state=asset_cfg_type.InitialStateCfg(
                    pos=spec.init_pos or self.station_pos,
                    rot=spec.init_rot or self.station_rot,
                ),
            )

        return scene_cfgs


def make_workstation_tabletop_scene_cfgs(
    *,
    prim_root: str = "{ENV_REGEX_NS}/station",
    station_pos: tuple[float, float, float] | None = None,
    station_rot: tuple[float, float, float, float] | None = None,
    include_missing_assets: bool = False,
    strict_assets: bool = False,
    create_parent_xforms: bool = True,
) -> dict[str, object]:
    return WorkstationTabletopLoadCfg(
        prim_root=prim_root,
        station_pos=station_pos or WORKSTATION_POSE_CFG.pos,
        station_rot=station_rot or WORKSTATION_POSE_CFG.rot,
        include_missing_assets=include_missing_assets,
        strict_assets=strict_assets,
        create_parent_xforms=create_parent_xforms,
    ).to_scene_cfgs()


def install_workstation_tabletop_scene_cfgs(
    namespace: dict[str, object],
    load_cfg: WorkstationTabletopLoadCfg | None = None,
) -> None:
    namespace.update((load_cfg or WorkstationTabletopLoadCfg()).to_scene_cfgs())


def _spawn_empty_xform_prim(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """SpawnerCfg callback that creates an empty Xform parent prim."""
    del cfg, kwargs

    import isaaclab.sim as sim_utils

    return sim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
    )


def _workstation_part_usd_path(name: str, part_root: Path = WORKSTATION_POSE_CFG.part_root) -> Path:
    if name == "WorkStation":
        return part_root / "WorkStation" / "WorkStation.usd"
    if name.startswith("M_MainFrame_"):
        suffix = name.rsplit("_", maxsplit=1)[-1]
        return part_root / f"Mainframe_{suffix}" / f"M_Mainframe_{suffix}.usd"
    if name.startswith("M_SupportTray_"):
        suffix = name.rsplit("_", maxsplit=1)[-1]
        return part_root / f"SupportTray_{suffix}" / f"M_SupportTray_{suffix}.usd"
    if name.startswith("M_ReagentCap_"):
        suffix = name.rsplit("_", maxsplit=1)[-1]
        return part_root / f"ReagentCap_{suffix}" / f"M_ReagentCap_{suffix}.usd"
    if name.startswith("M_Reagent_"):
        suffix = name.rsplit("_", maxsplit=1)[-1]
        return part_root / f"Reagent_{suffix}" / f"M_Reagent_{suffix}.usd"
    if name.startswith("ClawTool_"):
        return part_root / name / f"{name}.usd"
    return part_root / name / f"{name}.usd"


def _workstation_scene_key(kind: CollisionBodyKind, name: str) -> str:
    return f"ws_{kind.value}_{name.lower()}"


def _make_workstation_asset_specs(
    groups: Iterable[CollisionBodyGroup] = WORKSTATION_COLLISION_GROUPS,
) -> tuple[WorkstationAssetSpec, ...]:
    specs: list[WorkstationAssetSpec] = []
    for group in groups:
        if group.kind == CollisionBodyKind.INTERACTIVE:
            continue
        for name in group.names:
            specs.append(
                WorkstationAssetSpec(
                    name=name,
                    kind=group.kind,
                    group_label=group.label,
                    usd_path=_workstation_part_usd_path(name),
                    scene_key=_workstation_scene_key(group.kind, name),
                )
            )

    for placement in WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS:
        specs.append(
            WorkstationAssetSpec(
                name=placement["name"],
                kind=CollisionBodyKind.INTERACTIVE,
                group_label=placement["group_label"],
                usd_path=placement["usd_path"],
                scene_key=placement["scene_key"],
                init_pos=placement["pos"],
                init_rot=placement["rot"],
                scale=placement.get("scale"),
                rigid_object=placement.get("rigid_object", False),
            )
        )
    return tuple(specs)


WORKSTATION_TABLETOP_ASSET_SPECS = _make_workstation_asset_specs()


def get_collision_body_kind(name: str) -> CollisionBodyKind | None:
    if name in STATIC_COLLIDER_NAMES:
        return CollisionBodyKind.STATIC
    if name in INTERACTIVE_COLLIDER_NAMES:
        return CollisionBodyKind.INTERACTIVE
    if name in DYNAMIC_COLLIDER_NAMES:
        return CollisionBodyKind.DYNAMIC
    return None


def scan_workstation_placement_config(
    scene_or_stage,
    *,
    station_path_fragment: str = "/station/",
    groups: Iterable[CollisionBodyGroup] = WORKSTATION_COLLISION_GROUPS,
    verbose: bool = True,
) -> CollisionApplyReport:
    """Scan a stage and report configured workstation placement prim names.

    This is a placement diagnostic. It does not apply CollisionAPI,
    RigidBodyAPI, or any other physics schema.
    """
    stage = scene_or_stage.stage if hasattr(scene_or_stage, "stage") else scene_or_stage
    name_to_paths: dict[str, list[str]] = {name: [] for group in groups for name in group.names}

    for prim in stage.TraverseAll():
        prim_name = prim.GetName()
        if prim_name not in name_to_paths:
            continue

        prim_path = str(prim.GetPath())
        if station_path_fragment and station_path_fragment not in prim_path:
            continue
        name_to_paths[prim_name].append(prim_path)

    matched = {name: tuple(paths) for name, paths in name_to_paths.items() if paths}
    missing = tuple(sorted(name for name, paths in name_to_paths.items() if not paths))
    report = CollisionApplyReport(matched=matched, missing=missing)

    if verbose:
        print(
            "[INFO] Workstation placement config scan complete: "
            f"matched={report.matched_count}, missing={report.missing_count}"
        )
        if report.missing:
            print("[WARN] Missing workstation prim names:")
            for name in report.missing:
                print(f"  {name}")

    return report


# Backwards-compatible alias for older debug scripts. New code should import
# scan_workstation_placement_config from configs.place_cfg.
apply_workstation_collision_config = scan_workstation_placement_config
