# Vendored models, and where each came from

Nothing in `models/` was authored here. Both descriptions are someone else's
work, copied in and then adapted to MuJoCo. The adaptations are listed so the
provenance claim stays honest.

## UR3e — `models/ur3e/`

| | |
|---|---|
| source | `github.com/UTS-RI/UR3_RG2_rosbridge` (MIT, © 2026 Robotics Institute, University of Technology Sydney) |
| taken from | that repository's `models/ur3e_official/` |
| what | `ur3e.xml` (body tree, inertials, joint ranges) and the seven STLs in `assets/` |
| unmodified? | meshes yes — copied byte-for-byte, SHA-256 verified. The MJCF is adapted (see below) |
| licence | [`LICENSE`](ur3e/LICENSE) — the source repository's MIT text, alongside the files |

That repository carries the manufacturer's UR3e description; the link inertials
in `ur3e.xml` are the UR3e's, not estimates.

Adapted for this project:

- `wrist_3_joint` gained a range. The source omitted it, and this project drives
  that joint to its limit during a wrist rotation.
- Actuators were added: `<position>` servos with identified gains. The source is
  a plant description and has no controller; this project needs one to measure
  servo behaviour against.
- The `home` keyframe is overwritten at load from `SIM_HOME_Q` in
  `src/config.py`, so the one in the file is not authoritative.
- `models/ur3e/arm_dynamics.json` is **generated**, not vendored — see
  `tools/build_arm_dynamics.py`. It must be rebuilt whenever the payload changes.

## OnRobot RG2 — `models/onrobot_rg2/`

| | |
|---|---|
| source | `github.com/tonydle/OnRobot_ROS2_Description` (MIT, © 2025 Tony Le) |
| upstream of that | `github.com/Osaka-University-Harada-Laboratory/onrobot` (MIT) — where the RG2 meshes originate, per the source package's own README |
| what | `onrobot_rg2.xml` (four-bar linkage, dimensions) and the meshes in `meshes/{collision,visual}/` |
| unmodified? | meshes yes. The MJCF is a hand-written conversion of the package's XACRO |
| licence | [`LICENSE`](onrobot_rg2/LICENSE) — MIT, © 2025 Tony Le |

Adapted for this project:

- XACRO → MJCF: the linkage is modelled with its real ratio and limits, and the
  driven DOF is the package's `finger_width`.
- The pads are given the measured friction and a `forcerange`, so the simulated
  squeeze is a number the real RG2 can actually deliver.

## ArUco calibration marker — `models/aruco/`

| | |
|---|---|
| source | generated here, with OpenCV's `aruco` module |
| dictionary / id | `DICT_4X4_50`, id 0 |
| physical size | **100.0 mm** black square, 140 mm page (20 mm quiet zone) |
| files | `marker0_DICT_4X4_50_100mm.pdf` (print-ready) and `..._300dpi.png` |
| licence | none needed — an ArUco marker is not creative work |

Both are the same artwork. The PDF's page is 140 mm, so it must be printed at
**100%, never "fit to page"** — scaled to A4 the square comes out at 150 mm while
`--marker-size` still says 0.10. `tools/calibrate_camera.py --wrist` defaults to
100 mm to match this file.

## Why the arm is not from `mujoco_menagerie`

`mujoco_menagerie` ships no UR3e. The alternative there is the UR5e, which has
~850 mm of reach against the UR3e's ~500 mm; using it would put every tuned
workspace constant — spawn ranges, table geometry, standoff — on the wrong arm.
`model_loader` therefore resolves `'ur3e'` to this directory, and only falls back
to menagerie for a name that `models/` does not provide.
