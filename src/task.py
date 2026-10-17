# The manipulation task: what the robot does, independent of how it is driven.
#
# HARDWARE PORTABILITY
# --------------------
# This module is part of the robot side and must run on a real UR3e. It
# therefore contains NO mujoco, NO viewer, NO scene, NO key bindings and NO
# object pool. It expresses the task purely in terms of:
#
#   * an arm-like object with move()/hold()/grasp_state()
#   * an ObjectWorld (src/world.py) it can observe
#   * callbacks for "should I stop?" and "should I pause?"
#
# Everything simulation-specific -- the passive viewer, the keyboard, the cube
# spawn, the moving-target demo -- lives in run_simulation.py, which drives the
# TaskStateMachine below and supplies those callbacks.
#
# The state machine itself is the part worth reusing on hardware: it encodes
# WHEN to search, how to handle a pause mid-motion, and what an interrupted
# search means. None of that depends on being a simulation.
#
# tests/test_layering.py enforces the no-mujoco rule for this module.


# Task modes. The robot is either waiting for an object, or clearing the table.
WAIT = 'WAIT'
SEARCH = 'SEARCH'


class Commands:
    # A small command queue, filled by whatever front-end owns the input.
    #
    # Simulation fills it from keys 6-9; a hardware front-end would fill it from
    # an operator panel or a service call. The task logic never knows which.
    #
    #   place  -- put an object on the table
    #   start  -- begin / stop a search
    #   pause  -- hold position / resume
    #   reset  -- return to a known state immediately
    def __init__(self):
        self.place = False
        self.start = False
        self.pause = False
        self.reset = False

    def take(self, name):
        # Consume a flag: True exactly once per request.
        if getattr(self, name):
            setattr(self, name, False)
            return True
        return False

    def pending(self):
        # Non-consuming peek. Safe to call every step.
        return self.place or self.start or self.reset

    def clear(self):
        self.place = self.start = self.pause = self.reset = False


class TaskStateMachine:
    # WAIT / SEARCH, plus pause and abort semantics.
    #
    # Owns the policy, not the mechanism:
    #   * pause is consumed per step, not at loop boundaries (see estop below)
    #   * RESET pre-empts everything
    #   * an interrupted search is classified in priority order
    #
    # The front-end supplies these callables:
    #   say(msg)          -- report state (print, overlay, log; anything)
    #   at_home()         -- is the arm at its home pose?
    #   go_home()         -- drive there
    #   place_object()    -- put an object on the table
    #   clear_table()     -- run the search/pick loop
    #   running()         -- is the front-end still alive?

    def __init__(self, commands, say, at_home, go_home, place_object,
                 clear_table, running, arm, world):
        self.cmd = commands
        self.arm = arm
        self.world = world
        self._say = say
        self._at_home = at_home
        self._go_home = go_home
        self._place_object = place_object
        self._clear_table = clear_table
        self._running = running

        self.mode = WAIT
        self.paused = False

    # MOTION LOOP CALLBACKS
    #
    # Motion loops (Arm.move, PBVSServo.servo) need an estop callable and a
    # stop callable. They are bound methods here rather than closures in the
    # front-end, so the front-end never has to reference the task object before
    # it exists. Grab them once, after construction:
    #     task = TaskStateMachine(...)
    #     arm.move(q, grip, estop=task.estop, stop=task.stopped)
    def stopped(self):
        # True when the front-end is shutting down. Used as `stop=` in the
        # motion loops so they can return promptly.
        return not self._running()

    # STATE REPORTING
    def say(self, msg):
        self._say(msg)
        return msg

    # PAUSE / E-STOP
    def estop(self):
        # Called from inside the motion loops. Consumes the pause request here,
        # NOT at the outer-loop boundary: that boundary is not reached while
        # clear_table is running, so a mid-search press would otherwise queue up
        # and fire late. Consuming it per step also stops two quick presses
        # collapsing into one toggle.
        if self.cmd.pause:
            self.cmd.pause = False
            self.paused = not self.paused
            self.say('PAUSED  (holding position)' if self.paused
                     else 'RESUMED')
        return self.paused

    def at_home(self):
        return self._at_home()

    # MODE TRANSITIONS
    def enter_wait(self, reason):
        self.mode = WAIT
        self.say('WAIT  (%s)' % reason)
        if not self.at_home():
            self.say('not at home -- returning home')
            self._go_home()
            self.say('WAIT  (at home)')

    def hard_reset(self):
        # Back to a known state, clearing the pause flag and the queue.
        self.paused = False
        self.mode = WAIT
        self.cmd.clear()

    # ONE STEP OF THE OUTER LOOP
    def step(self):
        # Advance one iteration. Returns False when the front-end has stopped.
        if not self._running():
            return False

        # RESET pre-empts everything and needs no motion
        if self.cmd.take('reset'):
            self._on_reset()
            return True

        # PAUSE toggled at the outer boundary too, so a press between searches
        # takes effect even when no motion loop is running.
        if self.cmd.take('pause'):
            self.paused = not self.paused
            self.say('PAUSED  (holding position)' if self.paused
                     else 'RESUMED')

        if self.mode == WAIT:
            self._step_wait()
        else:
            self._step_search()

        return self._running()

    def _step_wait(self):
        # place an object; the robot stays in WAIT
        if self.cmd.take('place'):
            self.say('SPAWN  (place)')
            self._place_object()

        # start a search
        if self.cmd.take('start'):
            if not self.world.active():
                self.say('SEARCH ignored -- nothing on the table '
                         '(place an object first)')
            else:
                self.mode = SEARCH

    def _step_search(self):
        self.say('SEARCH')
        self._clear_table()
        if not self._running():
            return

        # Decide what the interrupted search means, in priority order.
        if self.cmd.take('reset'):
            self._on_reset()
        elif self.cmd.take('place'):
            self.say('SPAWN  (place during search) -- aborting, returning home')
            self._place_object()
            self.enter_wait('after place')
        elif self.cmd.take('start'):
            self.say('STOP  -- placing the object back on the table')
            self._put_down()
            self.enter_wait('after stop')
        else:
            self.enter_wait('table clear')

    # Overridden by the front-end via attributes, so task.py needs no imports
    # from the scene layer.
    def _on_reset(self):
        hook = getattr(self, 'on_reset', None)
        if hook is not None:
            hook()
        else:
            self.hard_reset()

    def _put_down(self):
        hook = getattr(self, 'on_put_down', None)
        if hook is not None:
            hook()
