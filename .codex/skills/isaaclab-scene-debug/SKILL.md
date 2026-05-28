---
name: isaaclab-scene-debug
description: Debug IsaacLab scene setup issues in this AUBO project, especially asset placement, split workstation USD loading, collision/contact behavior, camera initialization, and single-source numeric configuration cleanup.
---

# IsaacLab Scene Debug

Use this skill for scene-level IsaacLab work in this repository when symptoms involve misplaced assets, unexpected rotations, missing contact reports, blocked robot motion, camera pose not applying, or duplicated numeric tuning inputs.

## First Pass

1. Inspect the current file layout and avoid assuming old paths:
   - Runtime entries: `scripts/test.py`, `scripts/train.py`, `scripts/eval.py`
   - Scene configuration: `configs/Testcfg.py`, `configs/RLcfg.py`
   - Canonical constants: `configs/asset.py`
   - Workstation split-USD placement: `configs/place_cfg.py`
   - Robot contact/collision training config: `configs/collision_cfg.py`
   - Utilities: `tools/camera.py`, `tools/contact.py`, `tools/logic.py`, `tools/scene.py`
2. Read `git status --short` before editing. Preserve unrelated user changes.
3. Use `rg` to trace the exact value, name, or prim path from the symptom. Prefer source references over memory from prior sessions.
4. Separate three concerns before changing code:
   - Placement: which prims are loaded, where they are spawned, and which config owns the pose.
   - Physics/collision: whether USD assets already carry collider or rigid-body schemas, and whether this project should add, remove, or only observe collision behavior.
   - Runtime instrumentation: whether `test.py` has the relevant debug print, contact subscription, camera pose report, or screenshot capture.

## Numeric Input Cleanup

When the user asks to reduce duplicated numeric inputs or make a pose/rotation adjustment easier to reason about:

1. Choose one canonical input location, usually `configs/asset.py`.
2. Treat world poses and final rotations as derived values when they can be computed from canonical constants.
3. Remove or deprecate duplicate defaults in helper function signatures when they repeat business values.
4. After editing, state exactly which variables the user should tune and which old entry points are no longer authoritative.
5. Verify with `python -m py_compile` for edited Python files. Add a small import-level check only when it proves derived values without launching Isaac.

## Placement Workflow

For workstation, interactive object, or robot base placement issues:

1. Start in `configs/asset.py` for `WORKSTATION_POS`, `WORKSTATION_ROT`, camera constants, robot world poses, and interactive object local pose tables.
2. Check `configs/place_cfg.py` for split-USD asset grouping and scene key generation. This module should own loading descriptions and should not author physics collision schemas.
3. Check `configs/Testcfg.py` or `configs/RLcfg.py` to confirm the placement config is actually installed into the scene.
4. In `scripts/test.py`, use existing reports such as workstation placement scan, camera pose report, asset report, and interactive object report before adding new instrumentation.
5. If a loaded scene object is an `XformPrimView`, do not assume it has `.data`; use configured pose, stage prim inspection, or type-aware reporting.

## Collision And Contact Workflow

For stuck robot motion, missing contact prints, collision termination, or support tray interference:

1. Distinguish `configs/place_cfg.py` from `configs/collision_cfg.py`:
   - `place_cfg.py` describes workstation asset placement and semantic groups.
   - `collision_cfg.py` describes robot contact sensor and collision termination settings for training.
2. Confirm robot USD spawn enables `activate_contact_sensors=True` in the active scene config.
3. Confirm `ROBOT_CONTACT_SENSOR_CFG` is mounted in the scene under `ROBOT_CONTACT_SENSOR_NAME`.
4. In `scripts/test.py`, contact subscriptions and report enabling should happen early enough to observe initialization contacts.
5. When disabling temporary workstation collisions for testing, prefer disabling `CollisionAPI` for explicit prim names instead of deactivating whole prims.
6. If old logic searches `/station/WorkStation_All/`, update it for split-USD paths under `/station/`.

## Camera Workflow

For camera pose or image capture issues:

1. Separate the CameraSensor from the Isaac Sim viewport camera:
   - CameraSensor constants: `CAMERA_WORKSTATION_OFFSET`, `CAMERA_INITIAL_POS`, `CAMERA_INITIAL_ROT`, `CAMERA_POSE_CONVENTION`.
   - Viewport constants: `VIEWPORT_CAMERA_WORKSTATION_OFFSET`, `VIEWPORT_CAMERA_FORWARD_W`, `VIEWPORT_CAMERA_EYE`, `VIEWPORT_CAMERA_TARGET`.
2. Confirm `configs/Testcfg.py` passes the CameraSensor constants into `CameraCfg`.
3. Confirm `tools/camera.py::AuboCameraFns.set_camera_pose` is called in the runtime entry and handles env origins when `relative_to_env_origins=True`.
4. Verify image output by checking `picture/` timestamps or by using the existing capture path in `scripts/test.py`.

## Verification

Use the lightest verification that matches the change:

- Pure config or helper changes: `python -m py_compile <edited files>`.
- Derived pose changes: add or run an import-level print/assertion that checks the derived tuple values.
- Isaac runtime behavior: run the project entry only when the local Isaac environment is available. If it fails because `isaaclab` is missing from the active Python, report that limitation and do not claim simulation verification.

## Final Response

Report:

1. The root cause or confirmed uncertainty.
2. The files changed, with the specific source of truth after the change.
3. The verification command and result.
4. Any Isaac runtime behavior that still needs visual or simulation confirmation.
