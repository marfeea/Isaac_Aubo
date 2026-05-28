from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from configs.asset import ROBOT_ASSET_NAME
from configs.place_cfg import (
    ALL_COLLIDER_NAMES,
    DYNAMIC_COLLIDER_NAMES,
    INTERACTIVE_COLLIDER_NAMES,
    MOVABLE_COLLIDER_NAMES,
    STATIC_COLLIDER_NAMES,
    WORKSTATION_COLLISION_GROUPS,
    CollisionBodyGroup,
    CollisionBodyKind,
    WorkstationTabletopLoadCfg,
    get_collision_body_kind,
    install_workstation_tabletop_scene_cfgs,
    make_workstation_tabletop_scene_cfgs,
    scan_workstation_placement_config,
)

from isaaclab.sensors import ContactSensorCfg


ROBOT_CONTACT_SENSOR_NAME = "robot_contact_sensor"
ROBOT_CONTACT_SENSOR_PRIM_PATH = f"{{ENV_REGEX_NS}}/{ROBOT_ASSET_NAME}/.*"
ROBOT_CONTACT_FORCE_THRESHOLD = 1.0e-6

ROBOT_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path=ROBOT_CONTACT_SENSOR_PRIM_PATH,
    update_period=0.0,
    history_length=3,
    debug_vis=False,
    track_pose=True,
    track_air_time=False,
    force_threshold=ROBOT_CONTACT_FORCE_THRESHOLD,
)


TEMPORARY_DISABLED_WORKSTATION_COLLIDER_NAMES = frozenset(
    {
        "M_SupportTray_07",
        "M_Reagent_05",
    }
)


@dataclass(frozen=True)
class CollisionApplyReport:
    """Report which configured workstation prim names were found or changed."""

    matched: dict[str, tuple[str, ...]]
    missing: tuple[str, ...]
    disabled: dict[str, tuple[str, ...]]

    @property
    def matched_count(self) -> int:
        return sum(len(paths) for paths in self.matched.values())

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def disabled_count(self) -> int:
        return sum(len(paths) for paths in self.disabled.values())


def apply_workstation_collision_config(
    scene_or_stage,
    *,
    station_path_fragment: str = "/station/",
    groups: Iterable[CollisionBodyGroup] = WORKSTATION_COLLISION_GROUPS,
    disabled_names: Iterable[str] = TEMPORARY_DISABLED_WORKSTATION_COLLIDER_NAMES,
    verbose: bool = True,
) -> CollisionApplyReport:
    """Apply runtime workstation collision overrides for the split-USD station.

    The split assets already carry their authored USD physics schemas. This
    function keeps the semantic scan from the old debug path and only disables
    explicit temporary problem colliders by setting CollisionAPI enabled=false.
    """
    placement_report = scan_workstation_placement_config(
        scene_or_stage,
        station_path_fragment=station_path_fragment,
        groups=groups,
        verbose=False,
    )
    disabled = disable_workstation_collision_prims(
        scene_or_stage,
        names=disabled_names,
        station_path_fragment=station_path_fragment,
        verbose=False,
    )
    report = CollisionApplyReport(
        matched=placement_report.matched,
        missing=placement_report.missing,
        disabled=disabled,
    )

    if verbose:
        print(
            "[INFO] Workstation collision config applied: "
            f"matched={report.matched_count}, missing={report.missing_count}, "
            f"disabled={report.disabled_count}"
        )
        if report.missing:
            print("[WARN] Missing workstation prim names:")
            for name in report.missing:
                print(f"  {name}")
        if report.disabled:
            print("[INFO] Disabled temporary workstation collider prims:")
            for prim_path in sorted(path for paths in report.disabled.values() for path in paths):
                print(f"  {prim_path}")

    return report


def disable_workstation_collision_prims(
    scene_or_stage,
    *,
    names: Iterable[str] = TEMPORARY_DISABLED_WORKSTATION_COLLIDER_NAMES,
    station_path_fragment: str = "/station/",
    verbose: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Disable CollisionAPI for named workstation prims or their descendants."""
    from pxr import UsdPhysics

    stage = scene_or_stage.stage if hasattr(scene_or_stage, "stage") else scene_or_stage
    disabled_names = frozenset(names)
    disabled: dict[str, list[str]] = {name: [] for name in disabled_names}

    for prim in stage.TraverseAll():
        if not prim.IsValid() or not prim.IsActive():
            continue

        prim_path = str(prim.GetPath())
        if station_path_fragment and station_path_fragment not in prim_path:
            continue

        matched_name = _path_matched_disabled_name(prim_path, disabled_names)
        if matched_name is None or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue

        collision_api = UsdPhysics.CollisionAPI(prim)
        collision_api.CreateCollisionEnabledAttr(False).Set(False)
        disabled[matched_name].append(prim_path)

    disabled_report = {name: tuple(paths) for name, paths in disabled.items() if paths}
    if verbose:
        if disabled_report:
            print("[INFO] Disabled temporary workstation collider prims:")
            for prim_path in sorted(path for paths in disabled_report.values() for path in paths):
                print(f"  {prim_path}")
        else:
            print("[WARN] No temporary workstation collider prims were found to disable.")

    return disabled_report


def _path_matched_disabled_name(prim_path: str, disabled_names: frozenset[str]) -> str | None:
    path_parts = prim_path.split("/")
    for name in disabled_names:
        if name in path_parts:
            return name
    return None


__all__ = [
    "ALL_COLLIDER_NAMES",
    "DYNAMIC_COLLIDER_NAMES",
    "INTERACTIVE_COLLIDER_NAMES",
    "MOVABLE_COLLIDER_NAMES",
    "ROBOT_CONTACT_FORCE_THRESHOLD",
    "ROBOT_CONTACT_SENSOR_CFG",
    "ROBOT_CONTACT_SENSOR_NAME",
    "ROBOT_CONTACT_SENSOR_PRIM_PATH",
    "STATIC_COLLIDER_NAMES",
    "TEMPORARY_DISABLED_WORKSTATION_COLLIDER_NAMES",
    "WORKSTATION_COLLISION_GROUPS",
    "CollisionApplyReport",
    "CollisionBodyGroup",
    "CollisionBodyKind",
    "WorkstationTabletopLoadCfg",
    "apply_workstation_collision_config",
    "disable_workstation_collision_prims",
    "get_collision_body_kind",
    "install_workstation_tabletop_scene_cfgs",
    "make_workstation_tabletop_scene_cfgs",
]
