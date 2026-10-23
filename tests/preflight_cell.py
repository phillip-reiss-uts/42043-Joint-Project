#!/usr/bin/env python3
"""
preflight_cell.py -- verify the cell stack with NO hardware attached.

Lives in tests/ because it is a test, but it is also the gate you run before
travelling to the cell. With the camera unplugged and the UR3e off the network:

    cd ~/"assignment 4"
    ~/cell/bin/python tests/preflight_cell.py

It checks everything that can be checked offline:

  1. the four hardware modules import into ONE interpreter
  2. ur_rtde exposes the methods src/robot_backend_ur.py actually calls, under
     the name that build really uses (servoJ vs servoj -- this bit twice)
  3. tests/test_ur_backend.py passes: the backend plumbing, against fakes
  4. tests/test_arm_control.py --config-only passes: src/config.py is the
     measured cell, not the simulator's defaults
  5. pyrealsense2 loads and enumerates   (0 devices is CORRECT here)
  6. opencv has aruco + solvePnP         (informational; calibration only)

What it CANNOT prove, and what you still run at the bench:
  tests/test_arm_control.py      does the UR3e accept these servoJ calls
  tests/test_camera_enum.py      does the D435i stream on this box
  tests/test_camera_ruler.py     is CAM_MOUNT biased

Exit code 0 means every offline-checkable thing is verified.
"""

import argparse
import importlib
import os
import subprocess
import sys
import traceback

RESULTS = []
NOTES = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  -- " + detail) if detail else ""))
    return bool(ok)


def note(text):
    NOTES.append(text)
    print("  [info] " + text)


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def last_line(exc):
    parts = traceback.format_exc().strip().splitlines()
    return parts[-1] if parts else repr(exc)


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


# --------------------------------------------------------------- 1. imports
def check_imports():
    section("1. hardware modules import into one interpreter")
    mods = {}
    for name in ("rtde_control", "rtde_receive", "pyrealsense2", "cv2"):
        try:
            mods[name] = importlib.import_module(name)
        except Exception:
            record("import %s" % name, False, last_line(sys.exc_info()[1]))
            continue
        record("import %s" % name, True,
               getattr(mods[name], "__file__", "builtin"))
    for name in ("run_hardware", "src.config"):
        try:
            importlib.import_module(name)
            record("import %s" % name, True)
        except Exception:
            record("import %s" % name, False, last_line(sys.exc_info()[1]))
    note("ur_rtde %s | opencv %s | pyrealsense2 %s" % (
        pkg_version("ur-rtde"),
        getattr(mods.get("cv2"), "__version__", "?"),
        pkg_version("pyrealsense2")))
    return mods


# ------------------------------------------------------------ 2. ur_rtde API
def check_ur_rtde_api(mods):
    section("2. ur_rtde exposes what src/robot_backend_ur.py calls")
    ctrl_mod, recv_mod = mods.get("rtde_control"), mods.get("rtde_receive")
    if ctrl_mod is None or recv_mod is None:
        record("ur_rtde modules present", False, "skipped")
        return

    ctrl = getattr(ctrl_mod, "RTDEControlInterface", None)
    recv = getattr(recv_mod, "RTDEReceiveInterface", None)
    record("RTDEControlInterface exists", ctrl is not None)
    record("RTDEReceiveInterface exists", recv is not None)
    if ctrl is None or recv is None:
        return

    # The single most important check. URBackend.step() resolves this name at
    # runtime; if neither spelling exists it has nothing to call.
    servo = next((n for n in ("servoJ", "servoj") if hasattr(ctrl, n)), None)
    record("a servoJ/servoj method exists", servo is not None,
           "build uses %s()" % servo if servo else
           "NEITHER name found -- step() cannot stream targets")
    if servo:
        doc = (getattr(ctrl, servo).__doc__ or "").strip()
        doc = " ".join(doc.split())
        note("%s signature: %s" % (servo, doc[:140] if doc else "(no docstring)"))

    record("RTDEReceiveInterface.getActualQ exists",
           hasattr(recv, "getActualQ"),
           "URBackend.__init__ calls it immediately")

    for optional in ("servoStop", "getActualQd", "getTimestamp",
                     "getActualTCPPose"):
        target = ctrl if optional == "servoStop" else recv
        note("%s(): %s (code falls back if absent)"
             % (optional, "present" if hasattr(target, optional) else "absent"))


# ------------------------------------------------------- 3+4. project tests
def run_test(label, script, extra_args=()):
    path = os.path.join(os.getcwd(), script)
    if not os.path.exists(path):
        return record(label, False, "missing file: %s" % script)
    proc = subprocess.run([sys.executable, script, *extra_args],
                          capture_output=True, text=True)
    ok = proc.returncode == 0
    record(label, ok, "exit %d" % proc.returncode)
    out = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
    for line in [ln for ln in out if ln.strip()][-4:]:
        print("         | " + line)
    return ok


def check_backend_plumbing():
    section("3. tests/test_ur_backend.py  (fakes: no robot, no ur_rtde)")
    run_test("hardware backend plumbing", "tests/test_ur_backend.py")


def check_config():
    section("4. tests/test_arm_control.py --config-only  (no robot contact)")
    run_test("config is the measured cell", "tests/test_arm_control.py",
             ("--config-only",))


# --------------------------------------------------------------- 5. camera
def check_realsense(mods):
    section("5. pyrealsense2 loads and enumerates")
    rs = mods.get("pyrealsense2")
    if rs is None:
        return record("pyrealsense2 importable", False)
    try:
        ctx = rs.context()
        devices = list(ctx.query_devices())
    except Exception:
        record("rs.context() enumerates", False, last_line(sys.exc_info()[1]))
        return False
    if devices:
        record("rs.context() enumerates", True,
               "%d device(s) attached" % len(devices))
    else:
        record("rs.context() enumerates", True,
               "0 devices -- correct with the camera unplugged")
    return True


# --------------------------------------------------------------- 6. opencv
def check_opencv(mods):
    section("6. opencv (calibration only -- not on the pick path)")
    cv2 = mods.get("cv2")
    if cv2 is None:
        note("cv2 not importable; src/calibration_camera.py will not run")
        return
    note("cv2 %s" % getattr(cv2, "__version__", "?"))
    note("cv2.aruco: %s" % ("present" if hasattr(cv2, "aruco") else "ABSENT"))
    note("cv2.solvePnP: %s" % ("present" if hasattr(cv2, "solvePnP") else "ABSENT"))


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Verify the cell stack offline, with no hardware.")
    ap.add_argument("--repo", default=os.getcwd(),
                    help="project root containing src/ and tests/ (default: cwd)")
    args = ap.parse_args(argv)

    repo = os.path.abspath(os.path.expanduser(args.repo))
    if not os.path.isdir(os.path.join(repo, "tests")):
        print("not a project root (no tests/ directory): %s" % repo)
        print("run from the project root:")
        print('  cd ~/"assignment 4" && ~/cell/bin/python tests/preflight_cell.py')
        return 2
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    print()
    print("PREFLIGHT -- cell stack, no hardware")
    print("  repo        : %s" % repo)
    print("  interpreter : %s" % sys.executable)
    print("  python      : %s" % sys.version.split()[0])

    mods = check_imports()
    check_ur_rtde_api(mods)
    check_backend_plumbing()
    check_config()
    check_realsense(mods)
    check_opencv(mods)

    failed = [r for r in RESULTS if not r[1]]
    print()
    print("=" * 78)
    if failed:
        print("  %d of %d checks FAILED" % (len(failed), len(RESULTS)))
        for name, _ok, detail in failed:
            print("   - %s%s" % (name, ("  (" + detail + ")") if detail else ""))
        print("=" * 78)
        return 1
    print("  all %d checks passed" % len(RESULTS))
    print("=" * 78)
    print("  Still needs hardware, in this order:")
    print("    python tests/test_arm_control.py     # UR3e accepts servoJ")
    print("    python tests/test_camera_enum.py     # D435i streams here")
    print("    python tests/test_camera_ruler.py    # CAM_MOUNT bias")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
