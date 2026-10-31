"""Object motion for the robustness demonstration.

SIMULATION ONLY. This module drives the cube as a ghost: it writes the
object's velocity directly into MuJoCo's state every step. There is no
hardware equivalent -- you cannot write `qvel` on a real table -- so nothing
here is part of the robot. Robot-side code must never import this module, and
tests/test_layering.py enforces that.

Motion is a straight glide at one of three speed bands, in a direction that
is re-randomised on a timer and mirrored off the spawn bounds, so it stays on
the table and inside the camera footprint.

NOTE: this driver deliberately does NOT know where the robot is. An earlier
version took the gripper position and halted the cube when the arm came
close, which made the cube's motion law depend on the robot. That coupling is
gone: the robot now waits for the object to hold still rather than forcing it
to stop, so the demo needs no knowledge of the arm at all.
"""

import numpy as np

from src.config import (
    OBJ_MOVE_BANDS, OBJ_MOVE_TURN_S,
    SPAWN_X_RANGE, SPAWN_Y_RANGE,
)

# The speed band is a live demo toggle, so it lives in module state rather than
# being threaded through clear_table -> pick_one -> servo on every call.
_state = {'band': 0}


def band_names():
    return [name for name, _ in OBJ_MOVE_BANDS]


def current_band():
    return OBJ_MOVE_BANDS[_state['band']]


def current_speed():
    return float(current_band()[1])


def set_band(i):
    _state['band'] = int(i) % len(OBJ_MOVE_BANDS)
    return current_band()


def cycle_band():
    # Step to the next band, wrapping. Returns (name, speed).
    return set_band(_state['band'] + 1)


class TargetMover:
    # Glides one registry object around the table during the demonstration.

    def __init__(self, model, data, reg, index, rng=None, speed=None,
                 turn_s=None):
        self.reg = reg
        self.index = int(index)
        self.rng = np.random.default_rng() if rng is None else rng
        # speed=None means track the live band, so the '.' toggle takes effect
        # mid-pick rather than only on the next one
        self.follow_band = speed is None
        self.speed = current_speed() if speed is None else float(speed)
        self.turn_s = OBJ_MOVE_TURN_S if turn_s is None else float(turn_s)

        self.bounds = (tuple(SPAWN_X_RANGE), tuple(SPAWN_Y_RANGE))
        h = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self._dir = np.array([np.cos(h), np.sin(h)], float)
        self._last_turn = None
        self.turned = 0                 # direction changes, for reporting
        self.moved = 0.0                # path length, for reporting
        self._prev_xy = None
        self.dt = float(model.opt.timestep)

    # state
    def describe(self):
        name, sp = current_band()
        return '%s (%.0f mm/s)' % (name, sp * 1e3)

    def _zero(self, data):
        vadr = self.reg.qvel_adrs[self.index]
        data.qvel[vadr:vadr + 6] = 0.0

    # LOOP
    def step(self, data):
        # Advance the object one control step. Returns True while it is moving.
        #
        # Takes no robot state: the cube's motion is independent of the arm.
        adr = self.reg.qpos_adrs[self.index]
        vadr = self.reg.qvel_adrs[self.index]
        p = np.array(data.qpos[adr:adr + 3], float)
        self._prev_xy = p[:2].copy()

        # Safe to leave installed for a whole session: a parked object (already
        # delivered to the chute) must not be given velocity.
        if self.index not in self.reg.active():
            self._zero(data)
            return False

        # the band can change while the arm is moving
        if self.follow_band:
            self.speed = current_speed()

        if self.speed <= 0.0:
            self._zero(data)
            return False

        # re-aim on a timer, so the chase is not one long straight line
        now = float(data.time)
        if self._last_turn is None:
            self._last_turn = now
        elif now - self._last_turn >= self.turn_s:
            h = float(self.rng.uniform(0.0, 2.0 * np.pi))
            self._dir = np.array([np.cos(h), np.sin(h)], float)
            self._last_turn = now
            self.turned += 1

        # mirror off the spawn bounds rather than clamping, so it keeps moving
        nxt = p[:2] + self._dir * self.speed * self.dt
        for k, (lo, hi) in enumerate(self.bounds):
            if nxt[k] < lo or nxt[k] > hi:
                self._dir[k] = -self._dir[k]
                self.turned += 1

        v = self._dir * self.speed
        data.qvel[vadr] = v[0]
        data.qvel[vadr + 1] = v[1]
        data.qvel[vadr + 2] = 0.0
        data.qvel[vadr + 3:vadr + 6] = 0.0
        self.moved += float(np.linalg.norm(v)) * self.dt
        return True

