# Object motion for the robustness demonstration.
#
# Drives the tracked object around the table while the arm is servoing, so the
# servo has to chase a target that will not hold still. Motion is a straight
# glide at one of three speed bands, in a direction that is re-randomised on a
# timer and mirrored off the spawn bounds, so it stays on the table and inside
# the camera footprint.
#
# It stops on approach: once the gripper is close enough that the next phase
# would be the open-loop commit, the object holds still, because committing to a
# target that is still walking is not a grasp, it is a guess.

import numpy as np

from src.config import (
    OBJ_MOVE_BANDS, OBJ_MOVE_STOP_DIST, OBJ_MOVE_TURN_S,
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
    # Glides one registry object around the table while the servo runs.

    def __init__(self, model, data, reg, index, rng=None, speed=None,
                 stop_distance=None, turn_s=None):
        self.reg = reg
        self.index = int(index)
        self.rng = np.random.default_rng() if rng is None else rng
        # speed=None means track the live band, so the '.' toggle takes effect
        # mid-pick rather than only on the next one
        self.follow_band = speed is None
        self.speed = current_speed() if speed is None else float(speed)
        self.stop_distance = (OBJ_MOVE_STOP_DIST if stop_distance is None
                              else float(stop_distance))
        self.turn_s = OBJ_MOVE_TURN_S if turn_s is None else float(turn_s)

        self.bounds = (tuple(SPAWN_X_RANGE), tuple(SPAWN_Y_RANGE))
        h = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self._dir = np.array([np.cos(h), np.sin(h)], float)
        self._last_turn = None
        self.turned = 0                 # direction changes, for reporting
        self.stopped = False
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
    def step(self, data, gripper_pos):
        # Advance the object one control step. Returns True while it is moving.
        # gripper_pos is the full 3D gripper-base position: the xy separation
        # alone is misleading, because at the home pose the arm is already
        # within 0.25 m in xy while sitting half a metre above the table.
        adr = self.reg.qpos_adrs[self.index]
        vadr = self.reg.qvel_adrs[self.index]
        p = np.array(data.qpos[adr:adr + 3], float)
        self._prev_xy = p[:2].copy()

        # Safe to leave installed on the arm for a whole session: a parked
        # object (already delivered to the chute) must not be given velocity.
        if self.index not in self.reg.active():
            self._zero(data)
            return False

        # the band can change while the arm is moving
        if self.follow_band:
            self.speed = current_speed()

        if self.speed <= 0.0 or self.stopped:
            self._zero(data)
            return False

        # hold still once the gripper is close enough to be committing soon.
        # A stop_distance of 0 disables this: the robot now waits for the object
        # to stop rather than forcing it to, so the chase can run its course.
        if self.stop_distance > 0.0 and float(
                np.linalg.norm(p - np.asarray(gripper_pos, float)[:3])) \
                < self.stop_distance:
            self.stopped = True
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
