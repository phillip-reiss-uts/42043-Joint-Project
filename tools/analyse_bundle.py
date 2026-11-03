#!/usr/bin/env python3
"""Analyse a capture bundle (tools/capture_bundle.py) for the state-space
observer: measurement noise, and the command->joint dynamics.

Three measurements, all from the recorded data:

  still   per-joint standard deviation of q and qd. R for the Kalman filter.
  step    first-order + dead-time fit to every step, both directions. Gives
          gain, delay and lag per joint, and shows direction asymmetry.
  chirp   the same model fitted to a frequency sweep, once with a free delay
          and once with the delay forced to zero. The second is what a fit
          without a delay term implicitly does: it trades delay for a bigger
          lag, which is why a step-only fit cannot separate the two.

    python tools/analyse_bundle.py --bundle hw_bundle_2026-11-02_1435.csv
    python tools/analyse_bundle.py --bundle <csv> --out ../notes/bundle_fit.json
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import config as cfg


def load(path):
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    n = cfg.ARM_DOF
    t = np.array([float(r['t']) for r in rows])
    ph = np.array([r['phase'] for r in rows])
    Q = np.array([[float(r['q%d' % i]) for i in range(n)] for r in rows])
    QD = np.array([[float(r['qd%d' % i]) for i in range(n)] for r in rows])
    U = np.array([[float(r['u%d' % i]) for i in range(n)] for r in rows])
    return t, ph, Q, QD, U


def first_order(t, u, q, delay_free=True):
    # q(t) = q0 + K * x(t-Td),  x' = (u - q0 - x) / tau
    # The input is a DEVIATION from the start pose: using the absolute command
    # makes x settle at q0 and the fit diverges by exactly q0.
    t = np.asarray(t, float) - t[0]
    q0 = float(q[0])
    # The reference is the POSE THE JOINT IS HOLDING at the phase start, not
    # u[0]: a step phase begins with the new command already applied, so
    # u - u[0] is identically zero and the optimiser has no input to fit.
    u = np.asarray(u, float) - q0
    q = np.asarray(q, float) - q0
    dt = float(np.mean(np.diff(t)))

    def sim(K, Td, tau):
        # BEFORE the dead time the joint is still held at its start pose, so
        # the input there is zero deviation, not the new command. Filling with
        # u[0] removes the dead time from the model entirely.
        ud = np.interp(t - Td, t, u, left=0.0, right=u[-1])
        x = np.zeros(len(t))
        a = dt / max(tau, 1e-3)
        for k in range(1, len(t)):
            x[k] = x[k - 1] + a * (ud[k - 1] - x[k - 1])
        return K * x

    if delay_free:
        sol = least_squares(lambda p: (sim(*p) - q) * 1e3,
                            [1.0, 0.3, 1.0],
                            bounds=([0.5, 0.0, 0.02], [1.5, 1.0, 3.0]))
        K, Td, tau = sol.x
        rms = float(np.sqrt(np.mean((sim(K, Td, tau) - q) ** 2)) * 1e3)
        return dict(gain=float(K), delay_ms=float(Td * 1e3),
                    tau_ms=float(tau * 1e3), rms_mrad=rms)

    sol = least_squares(lambda p: (sim(p[0], 0.0, p[1]) - q) * 1e3,
                        [1.0, 0.5], bounds=([0.5, 0.02], [1.5, 5.0]))
    K, tau = sol.x
    rms = float(np.sqrt(np.mean((sim(K, 0.0, tau) - q) ** 2)) * 1e3)
    return dict(gain=float(K), delay_ms=0.0, tau_ms=float(tau * 1e3),
                rms_mrad=rms)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bundle', required=True, help='capture_bundle CSV')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    t, ph, Q, QD, U = load(args.bundle)
    out = dict(bundle=os.path.abspath(args.bundle), phases=[])

    print('%-14s %7s %9s %8s %8s' % ('phase', 'n', 'dur s', 'dt ms', 'rate Hz'))
    for p in dict.fromkeys(ph):
        m = ph == p
        i = np.nonzero(m)[0]
        dt = float(np.mean(np.diff(t[i])))
        out['phases'].append(dict(name=p, n=int(m.sum()),
                                  duration_s=float(t[i[-1]] - t[i[0]]),
                                  dt_ms=dt * 1e3, rate_hz=1.0 / dt))
        print('%-14s %7d %9.2f %8.2f %8.1f'
              % (p, m.sum(), t[i[-1]] - t[i[0]], dt * 1e3, 1.0 / dt))

    # ---- still: measurement noise, R --------------------------------
    m = ph == 'still'
    if m.sum() > 100:
        out['still'] = dict(
            q_std=[float(v) for v in np.std(Q[m], axis=0)],
            q_ptp=[float(v) for v in np.ptp(Q[m], axis=0)],
            qd_std=[float(v) for v in np.std(QD[m], axis=0)])
        print('\nSTILL  (measurement noise; R = sigma^2)')
        print('  q  std (rad)  %s' % ' '.join('%.2e' % v
                                              for v in out['still']['q_std']))
        print('  qd std (rad/s)%s' % ' '.join('%.2e' % v
                                              for v in out['still']['qd_std']))
        print('  config ENCODER_NOISE = %.1e' % cfg.ENCODER_NOISE)

    # ---- steps: gain, delay, lag, both directions -------------------
    print('\nSTEPS  first-order + dead time')
    print('  %-12s %6s %9s %8s %8s'
          % ('phase', 'gain', 'delay ms', 'tau ms', 'rms mrad'))
    steps = {}
    for p in dict.fromkeys(ph):
        if not p.startswith('step'):
            continue
        j = int(p.split('_j')[1].split('_')[0])
        i = np.nonzero(ph == p)[0]
        r = first_order(t[i], U[i, j], Q[i, j])
        steps[p] = r
        print('  %-12s %6.3f %9.0f %8.0f %8.2f'
              % (p, r['gain'], r['delay_ms'], r['tau_ms'], r['rms_mrad']))
    if steps:
        a = np.array([[v['gain'], v['delay_ms'], v['tau_ms']]
                      for v in steps.values()])
        out['steps'] = steps
        out['steps_median'] = dict(gain=float(np.median(a[:, 0])),
                                   delay_ms=float(np.median(a[:, 1])),
                                   tau_ms=float(np.median(a[:, 2])))
        print('  median        %6.3f %9.0f %8.0f'
              % (out['steps_median']['gain'],
                 out['steps_median']['delay_ms'],
                 out['steps_median']['tau_ms']))

    # ---- chirp: the delay is only visible with frequency content ----
    chirp = {}
    for p in dict.fromkeys(ph):
        if not p.startswith('chirp'):
            continue
        j = int(p.split('_j')[1])
        i = np.nonzero(ph == p)[0]
        free = first_order(t[i], U[i, j], Q[i, j], delay_free=True)
        forced = first_order(t[i], U[i, j], Q[i, j], delay_free=False)
        chirp[p] = dict(free=free, forced_no_delay=forced)
        print('\n%s' % p)
        print('  free delay    gain %.3f  delay %4.0f ms  tau %5.0f ms  '
              'rms %5.2f mrad'
              % (free['gain'], free['delay_ms'], free['tau_ms'],
                 free['rms_mrad']))
        print('  delay = 0     gain %.3f  delay    0 ms  tau %5.0f ms  '
              'rms %5.2f mrad'
              % (forced['gain'], forced['tau_ms'], forced['rms_mrad']))
    if chirp:
        out['chirp'] = chirp

    # What the observer assumes: no dead time at all. Say so explicitly.
    if chirp:
        d = np.median([v['free']['delay_ms'] for v in chirp.values()])
        tau = np.median([v['free']['tau_ms'] for v in chirp.values()])
        print('\nPLANT (median over chirps): dead time %.0f ms, lag %.0f ms'
              % (d, tau))
        print('The observer model has NEITHER: A_cl is the undelayed PD servo.')

    path = args.out or os.path.splitext(args.bundle)[0] + '_analysis.json'
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('wrote %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
