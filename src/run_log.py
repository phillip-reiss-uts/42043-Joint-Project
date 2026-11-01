"""Tee stdout/stderr to a dated run log.

Used by the hardware front-end so every terminal line from a cell run is kept
with the exact command that produced it. The log is written under run_logs/
at the repo root; stdout/stderr still go to the terminal.
"""

import atexit
import datetime
import os
import sys


class _Tee:
    def __init__(self, terminal, log):
        self._terminal = terminal
        self._log = log

    def write(self, text):
        self._terminal.write(text)
        self._log.write(text)
        self._terminal.flush()
        self._log.flush()
        return len(text)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def isatty(self):
        return bool(getattr(self._terminal, 'isatty', lambda: False)())

    @property
    def encoding(self):
        return getattr(self._terminal, 'encoding', 'utf-8')


def start_run_log(args, root, tag='run'):
    """Start logging to run_logs/<tag>_YYYY-MM-DD_HHMMSS.txt.

    args is the argument list after the program name. The header is written
    before any run output. Returns the log path, or None if logging could not
    be opened (the run continues either way).
    """
    try:
        logdir = os.path.join(root, 'run_logs')
        os.makedirs(logdir, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        path = os.path.join(logdir, '%s_%s.txt' % (tag, stamp))
        fh = open(path, 'w', encoding='utf-8', buffering=1)
    except Exception as exc:
        print('[LOG] could not open run log: %s' % exc)
        return None

    fh.write('=' * 78 + '\n')
    fh.write('RUN LOG  %s\n' % datetime.datetime.now().isoformat(timespec='seconds'))
    fh.write('program  %s\n' % sys.argv[0])
    fh.write('cwd      %s\n' % os.getcwd())
    fh.write('python   %s\n' % sys.version.replace('\n', ' '))
    fh.write('argv     %s\n' % ' '.join(str(a) for a in args))
    fh.write('=' * 78 + '\n\n')
    fh.flush()

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, fh)
    sys.stderr = _Tee(old_err, fh)

    closed = [False]

    def close():
        if closed[0]:
            return
        closed[0] = True
        try:
            sys.stdout = old_out
            sys.stderr = old_err
        except Exception:
            pass
        try:
            fh.write('\n[LOG] closed %s\n'
                     % datetime.datetime.now().isoformat(timespec='seconds'))
            fh.flush()
            fh.close()
        except Exception:
            pass

    atexit.register(close)
    print('[LOG] writing terminal output to %s' % path)
    return path
