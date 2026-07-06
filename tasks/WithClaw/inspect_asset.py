from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _load_usd_modules():
    """从 Isaac Sim 安装包定位 USD Python 与 DLL，不启动 SimulationApp。"""
    import isaacsim

    extension_root = Path(isaacsim.__file__).resolve().parent / "extscache"
    candidates = sorted(extension_root.glob("omni.usd.libs-*"))
    if len(candidates) != 1:
        raise RuntimeError(f"无法唯一定位 omni.usd.libs：{candidates}")
    usd_root = candidates[0]
    sys.path.insert(0, str(usd_root))
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(usd_root / "bin"))
        os.add_dll_directory(str(usd_root / "bin" / "deps"))
    from pxr import Gf, Usd, UsdGeom

    return Gf, Usd, UsdGeom


def _tuple3(vector) -> tuple[float, float, float]:
    return tuple(round(float(value), 6) for value in vector)


def main() -> None:
    parser = argparse.ArgumentParser(description="只读检查带夹爪 USD 的 Flange 与夹爪几何。")
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path("D:/project/S2R/Asset/AUBO_E5/AUBO_E5_Withclaw.usd"),
    )
    args = parser.parse_args()
    Gf, Usd, UsdGeom = _load_usd_modules()
    stage = Usd.Stage.Open(str(args.usd))
    if stage is None:
        raise RuntimeError(f"无法打开 USD：{args.usd}")
    flange = stage.GetPrimAtPath("/Root/AUBO_E5/Flange")
    if not flange.IsA(UsdGeom.Xform):
        raise RuntimeError("USD 中缺少预期的 Flange Xform body。")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    flange_world_inverse = cache.GetLocalToWorldTransform(flange).GetInverse()
    claw_root = stage.GetPrimAtPath("/Root/AUBO_E5/ClawTool_C")
    claw_in_flange = cache.GetLocalToWorldTransform(claw_root) * flange_world_inverse
    points_in_flange = []
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith(str(claw_root.GetPath())):
            continue
        mesh_count += 1
        mesh_in_flange = cache.GetLocalToWorldTransform(prim) * flange_world_inverse
        points_in_flange.extend(
            mesh_in_flange.Transform(Gf.Vec3d(point))
            for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
        )
    if not points_in_flange:
        raise RuntimeError("未能读取夹爪 mesh 顶点。")

    bounds_min = tuple(min(float(point[axis]) for point in points_in_flange) for axis in range(3))
    bounds_max = tuple(max(float(point[axis]) for point in points_in_flange) for axis in range(3))
    print(f"[ASSET] usd={args.usd}")
    print(f"[ASSET] flange_body={flange.GetPath()}")
    print(f"[ASSET] claw_root={claw_root.GetPath()}")
    print(f"[ASSET] claw_origin_in_flange={_tuple3(claw_in_flange.ExtractTranslation())}")
    print(f"[ASSET] claw_mesh_count={mesh_count}")
    print(f"[ASSET] claw_bounds_min_in_flange={_tuple3(bounds_min)}")
    print(f"[ASSET] claw_bounds_max_in_flange={_tuple3(bounds_max)}")


if __name__ == "__main__":
    main()
