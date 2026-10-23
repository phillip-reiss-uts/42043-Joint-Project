# RealSense D435i first contact: is the device there, and can it stream?
#
# WHAT THIS PROVES
# ----------------
# Three separate questions, in the order they fail:
#
#   1. does librealsense's device list contain anything?   (USB / driver)
#   2. can a raw pipeline be started, with no project code? (SDK / OS)
#   3. does `RealSenseSource` -- the class the pick actually uses -- work?
#      Its docstring says it has never been executed, so this is its first run.
#
# Keeping 2 separate from 3 is the point: `RealSenseSource` is unverified code,
# so a failure there must not be confused with the camera or the SDK being
# broken. Run the steps in order and stop at the first one that fails.
#
# PLATFORM NOTES (measured 2026-10-23)
# ------------------------------------
# macOS fails at step 2 with "failed to set power state": the camera enumerates
# (system_profiler shows it at 5 Gb/s, 720 mA, and Photo Booth can open it) but
# librealsense cannot take exclusive UVC control. The same camera streams fine
# on Debian, so that is a platform limitation, not a cable or a code fault.
# On macOS the thing to try is a libusb-backed build (FORCE_RSUSB_BACKEND);
# the pip wheel never gets there.
#
# Run: python tests/test_camera_enum.py

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

W, H, FPS = 640, 480, 30


def step1_list(rs):
    print('=' * 70)
    print('1. devices librealsense can see')
    print('=' * 70)
    ctx = rs.context()
    devs = ctx.query_devices()
    print('  count: %d' % len(devs))
    if len(devs) == 0:
        print('  FAIL: no device. Check the USB 3 port and the cable.')
        return False
    # Any attribute access on a device can fail even when it is listed --
    # that is the macOS power-state failure, and it is worth reporting
    # separately from "not listed at all".
    d = devs[0]
    for key in ('name', 'serial_number', 'usb_type', 'firmware_version',
                'physical_port'):
        try:
            print('  %-18s %s' % (key, d.get_info(getattr(rs.camera_info, key))))
        except Exception as exc:
            print('  %-18s ERROR %s' % (key, exc))
            print('  FAIL: the device is listed but cannot be queried.')
            return False
    return True


def step2_pipeline(rs, w=W, h=H, fps=FPS):
    print()
    print('=' * 70)
    print('2. raw pipeline, no project code  (%dx%d @ %d)' % (w, h, fps))
    print('=' * 70)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    try:
        prof = pipe.start(cfg)
    except Exception as exc:
        print('  FAIL: %s' % exc)
        print('  If this is macOS "failed to set power state", try a lower')
        print('  resolution/fps, or a libusb-backed librealsense build.')
        return None, None
    try:
        scale = float(prof.get_device().first_depth_sensor()
                      .get_depth_scale())
        intr = (prof.get_stream(rs.stream.depth)
                .as_video_stream_profile().get_intrinsics())
        print('  started. depth_scale %.6f m/unit' % scale)
        print('  intrinsics fx %.2f fy %.2f cx %.2f cy %.2f'
              % (intr.fx, intr.fy, intr.ppx, intr.ppy))
        # A started pipeline that returns no frames is not a working camera.
        for _ in range(5):
            frames = pipe.wait_for_frames()
            depth = np.asanyarray(frames.get_depth_frame().get_data())
        nz = depth[depth > 0]
        print('  frame %s  valid %d/%d' % (depth.shape, nz.size, depth.size))
        if nz.size == 0:
            print('  WARNING: every pixel is zero -- lens covered, or too'
                  ' close/far.')
        else:
            print('  range mm: min %d  median %d  max %d'
                  % (nz.min(), int(np.median(nz)), nz.max()))
            print('  UNITS: median > 100 means the stream is already'
                  ' millimetres, which is what _to_millimetres expects.')
    finally:
        pipe.stop()
    return intr.fx, intr.fy


def step3_project_source(rs):
    print()
    print('=' * 70)
    print('3. RealSenseSource -- the class the pick uses')
    print('=' * 70)
    try:
        from src.world_real import RealSenseSource
    except Exception as exc:
        print('  FAIL to import: %s' % exc)
        return False
    src = None
    try:
        src = RealSenseSource(W, H, FPS)
        print('  intrinsics_report:', src.intrinsics_report())
        z16, K = src.capture()
        print('  depth %s %s   reported as mm: %s'
              % (z16.shape, z16.dtype, src._units_mm))
        print('  K:\n%s' % np.round(K, 2))
        nz = z16[z16 > 0]
        if nz.size:
            print('  range mm: min %d  median %d  max %d'
                  % (nz.min(), int(np.median(nz)), nz.max()))
        return True
    except Exception as exc:
        print('  FAIL: %s: %s' % (type(exc).__name__, exc))
        return False
    finally:
        if src is not None:
            src.close()


def main():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        print('pyrealsense2 is not installed: %s' % exc)
        print('  Debian:  pip3 install --break-system-packages pyrealsense2')
        print('           (plus the 99-realsense-libusb.rules udev file)')
        print('  macOS:   a FORCE_RSUSB_BACKEND build; the wheel does not work')
        return 2
    print('pyrealsense2: %s' % getattr(rs, '__file__', '<no path>'))
    print()

    if not step1_list(rs):
        return 1
    fx, fy = step2_pipeline(rs)
    if fx is None:
        return 1
    ok = step3_project_source(rs)

    print()
    print('=' * 70)
    print('  device listed, pipeline streamed, RealSenseSource %s'
          % ('OK' if ok else 'FAILED'))
    print('  Next: the ruler check -- put the cube at three measured spots and')
    print('  compare the detected centre. That is checklist Test 5.')
    print('=' * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
