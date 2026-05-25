from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from configs.asset import (
    ASSET_ROOT,
    WORKSTATION_INTERACTIVE_ASSET_PLACEMENTS,
    WORKSTATION_POS,
    WORKSTATION_ROT,
)


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
    names=_numbered("M_SupportTray", 1, 14),
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


WORKSTATION_PART_ROOT = ASSET_ROOT / "QKL-HX-300-II-00" / "Part"


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

    @property
    def exists(self) -> bool:
        return self.usd_path.exists()




@dataclass(frozen=True)
class WorkstationTabletopLoadCfg:
    """Build Isaac Lab scene cfg entries for split workstation tabletop USDs.

    This module only loads assets and keeps their semantic grouping. It no
    longer authors collision or rigid-body schemas onto the spawned USD prims.
    """

    prim_root: str = "{ENV_REGEX_NS}/station"
    station_pos: tuple[float, float, float] = WORKSTATION_POS
    station_rot: tuple[float, float, float, float] = WORKSTATION_ROT
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
        from isaaclab.assets import AssetBaseCfg

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

            scene_cfgs[spec.scene_key] = AssetBaseCfg(
                prim_path=f"{self.prim_root}/{spec.kind.value}/{spec.name}",
                spawn=sim_utils.UsdFileCfg(**usd_spawn_kwargs),
                init_state=AssetBaseCfg.InitialStateCfg(
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
        station_pos=station_pos or WORKSTATION_POS,
        station_rot=station_rot or WORKSTATION_ROT,
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


def _workstation_part_usd_path(name: str, part_root: Path = WORKSTATION_PART_ROOT) -> Path:
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


def apply_workstation_collision_config(
    scene_or_stage,
    *,
    station_path_fragment: str = "/station/",
    groups: Iterable[CollisionBodyGroup] = WORKSTATION_COLLISION_GROUPS,
    verbose: bool = True,
) -> CollisionApplyReport:
    """Scan a stage and report configured workstation prim names.

    Kept for compatibility with existing callers. It no longer applies
    CollisionAPI, RigidBodyAPI, or any other physics schema.
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
            "[INFO] Workstation collision config scan complete: "
            f"matched={report.matched_count}, missing={report.missing_count}"
        )
        if report.missing:
            print("[WARN] Missing workstation prim names:")
            for name in report.missing:
                print(f"  {name}")

    return report
