# Constant-velocity Kalman filter on the tracked object's world position.
#
# The camera gives an independent position measurement every perceive_every
# servo iterations. At 320x240 that measurement carries ~1.5 mm of error, it is
# biased toward the visible part of the object once the jaws start clipping it,
# and it disappears entirely for stretches when the object sits in the camera's
# own shadow. The filter turns that intermittent, noisy stream into a continuous
# estimate: it smooths the noise, propagates through the gaps, supplies a
# velocity the servo can lead with, and gates out gross outliers.
#
# State is [x, y, z, vx, vy, vz]; the measurement is position only.

import numpy as np


class TargetFilter:
    # Kalman filter for a target moving at roughly constant velocity.

    def __init__(self, sigma_meas=1.5e-3, sigma_acc=0.6, gate=3.0,
                 p0=1e-3, v0=1.0):
        self.sigma_meas = float(sigma_meas)
        self.sigma_acc = float(sigma_acc)
        self.gate = float(gate)         # innovation gate, in sigmas

        n = 6
        self.x = np.zeros(n)
        self.P = np.eye(n)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.R = (self.sigma_meas ** 2) * np.eye(3)

        self.p0 = float(p0)             # initial position variance
        self.v0 = float(v0)             # initial velocity variance
        self.t = None                   # timestamp of the current belief
        self.ready = False
        self.updates = 0                # accepted measurements
        self.rejects = 0                # gated-out measurements
        self.last_nis = 0.0             # normalised innovation squared
        self.last_accept = False

    # LIFECYCLE
    def reset(self, p, t=None):
        # Seed the filter at a first measurement; velocity stays unknown-wide.
        self.x = np.r_[np.asarray(p, float)[:3], np.zeros(3)]
        self.P = np.diag([self.p0] * 3 + [self.v0] * 3)
        self.t = t
        self.ready = True
        self.updates = 0
        self.rejects = 0

    def _F(self, dt):
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        return F

    def _Q(self, dt):
        # Continuous white-acceleration model, so the process noise scales
        # correctly with however long the gap between updates was.
        s = self.sigma_acc ** 2
        q = np.zeros((6, 6))
        q[:3, :3] = s * (dt ** 4) / 4.0 * np.eye(3)
        q[:3, 3:] = s * (dt ** 3) / 2.0 * np.eye(3)
        q[3:, :3] = s * (dt ** 3) / 2.0 * np.eye(3)
        q[3:, 3:] = s * (dt ** 2) * np.eye(3)
        return q

    # PREDICT / UPDATE
    def predict(self, dt):
        if not self.ready or dt <= 0.0:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        if self.t is not None:
            self.t += dt

    def advance_to(self, t):
        # Propagate to an absolute time, so callers never have to track dt.
        if self.t is None:
            self.t = t
            return
        self.predict(float(t) - float(self.t))

    def update(self, z):
        # Fold in one position measurement. Returns False when the measurement
        # is gated out as an outlier, in which case the prediction stands.
        z = np.asarray(z, float)[:3]
        if not self.ready:
            self.reset(z, self.t)
            self.last_accept = True
            return True

        y = z - self.H @ self.x                 # innovation
        S = self.H @ self.P @ self.H.T + self.R
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        self.last_nis = float(y @ Si @ y)
        if self.gate > 0.0 and self.last_nis > self.gate ** 2:
            self.rejects += 1
            self.last_accept = False
            return False

        K = self.P @ self.H.T @ Si
        self.x = self.x + K @ y
        # Joseph form keeps the covariance symmetric and positive definite
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        self.updates += 1
        self.last_accept = True
        return True

    # READOUT
    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:].copy()

    @property
    def speed(self):
        return float(np.linalg.norm(self.x[3:]))

    def lead(self, seconds):
        # Position advanced by the estimated velocity, for aiming ahead of a
        # moving target to cancel the servo's own lag.
        return self.x[:3] + self.x[3:] * float(seconds)
