"""The manipulation task: what the robot does, independent of how it is driven.

HARDWARE PORTABILITY
--------------------
This module is part of the robot side and must run on a real UR3e. It
therefore contains NO mujoco, NO viewer, NO scene, NO key bindings and NO
object pool. It expresses the task purely in terms of:

  * an arm-like object with move()/hold()/grasp_state()
  * an ObjectWorld (src/world.py) it can observe
  * callbacks for "should I stop?" and "should I pause?"

Everything simulation-specific -- the passive viewer, the keyboard, the cube
spawn, the moving-target demo -- lives in run_simulation.py, which drives the
TaskStateMachine below and supplies those callbacks.

The state machine itself is the part worth reusing on hardware: it encodes
WHEN to search, how to handle a pause mid-motion, and what an interrupted
search means. None of that depends on being a simulation.

tests/test_layering.py enforces the no-mujoco rule for this module.
"""

# Task modes.
#
#   WAIT   -- idle at home, nothing to do
#   TRACK  -- an object is on the table; follow it at the hover height, but
#             do NOT descend. This is the resting state whenever
#             there is something to look at.
#   PICK   -- an explicit command was given; run the grasp and only the grasp.
#
# The robot no longer has a "search" mode of its own: searching is a recovery
# step INSIDE a pick (the object left the view), not a mode the operator enters.
WAIT = 'WAIT'
TRACK = 'TRACK'
PICK = 'PICK'
# HOLD: the cube is in the jaws and the arm is waiting for the next command.
# The pickup key now toggles: press once to grasp, again to put it back down,
# again to pick it up. Between those the robot waits in HOLD rather than
# tracking, because there is nothing on the table to track -- the cube is in
# the gripper.
HOLD = 'HOLD'
# PUTDOWN: setting a held cube back on the table. Entered and completed inside
# one call, like PICK, so a new command cannot interleave with it.
PUTDOWN = 'PUTDOWN'


class Commands:
    # A small command queue, filled by whatever front-end owns the input.
    #
    # Simulation fills it from keys; a hardware front-end would fill it from an
    # operator panel or a service call. The task logic never knows which.
    #
    #   place   -- put an object on the table
    #   pickup  -- grab the tracked object NOW (the only way a grasp starts)
    #   pause   -- hold position / resume
    #   reset   -- return to a known state immediately
    def __init__(self):
        self.place = False
        self.pickup = False
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
        return self.place or self.pickup or self.reset

    def clear(self):
        self.place = self.pickup = self.pause = self.reset = False


class TaskStateMachine:
    # WAIT / TRACK / PICK, plus pause and abort semantics.
    #
    # Owns the policy, not the mechanism:
    #   * pause is consumed per step, not at loop boundaries (see estop below)
    #   * RESET pre-empts everything
    #   * a pick only ever starts from an explicit command
    #
    # The front-end supplies these callables:
    #   say(msg)          -- report state (print, overlay, log; anything)
    #   at_home()         -- is the arm at its home pose?
    #   go_home()         -- drive there
    #   place_object()    -- put an object on the table
    #   track_step()      -- one step of "follow the object without descending"
    #   pick_object()     -- run one grasp attempt; returns True if it held
    #   running()         -- is the front-end still alive?

    def __init__(self, commands, say, at_home, go_home, place_object,
                 track_step, pick_object, running, arm, world):
        self.cmd = commands
        self.arm = arm
        self.world = world
        self._say = say
        self._at_home = at_home
        self._go_home = go_home
        self._place_object = place_object
        self._track_step = track_step
        self._pick_object = pick_object
        self._running = running

        self.mode = WAIT
        self.paused = False
        # True while the jaws are holding a cube. This, not `mode`, decides what
        # the pickup key means: holding -> put down, not holding -> pick up.
        self.holding = False

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

    def enter_track(self, reason):
        # Tracking holds the arm where it is (over the object); it does not
        # re-home, because the point is to stay with the target.
        self.mode = TRACK
        self.say('TRACK  (%s)' % reason)

    def hard_reset(self):
        # Back to a known state, clearing the pause flag and the queue.
        self.paused = False
        self.mode = WAIT
        self.holding = False
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
        elif self.mode == TRACK:
            self._step_track()
        elif self.mode == HOLD:
            self._step_hold()
        else:
            self._step_pick()

        return self._running()

    def _step_wait(self):
        # Place an object, then start tracking it. There is nothing to do in
        # WAIT while something is already on the table.
        if self.cmd.take('place'):
            self.say('SPAWN  (place)')
            self._place_object()
            if self.world.active():
                self.enter_track('object on the table')
            return

        # The pickup key is a toggle on whether a cube is HELD:
        #   not holding -> grasp;  holding -> put it back down.
        # A press while nothing is on the table is a no-op, not an error.
        #
        # Note this grasps directly rather than only entering TRACK: the key
        # means "grab it now", and the pick itself does the tracking and waiting
        # for stillness internally. Entering TRACK on the first press would make
        # the toggle off by one -- press one would merely start watching, press
        # two would grasp, and press three would put down, which is not what the
        # operator expects from a pick/put-down toggle.
        if self.cmd.take('pickup'):
            if self.holding:
                self._run_put_down()
                return
            if not self.world.active():
                self.say('PICKUP ignored -- nothing on the table')
            else:
                self._run_pick()
            return

        # Resume tracking if an object appeared without a `place` command
        # (e.g. after a reset that parked everything, then a spawn).
        if self.world.active():
            self.enter_track('object on the table')

    def _step_track(self):
        # Follow the object at standoff height. This never descends and never
        # grasps: the tool only moves vertically on an explicit pickup.
        if self.cmd.take('place'):
            self.say('SPAWN  (place during track) -- re-tracking')
            self._place_object()
            if not self.world.active():
                self.enter_wait('table empty')
            return

        if self.cmd.take('pickup'):
            if not self.world.active():
                self.say('PICKUP ignored -- nothing on the table')
                self.enter_wait('table empty')
            else:
                self._run_pick()
            return

        if not self.world.active():
            self.enter_wait('table empty')
            return

        # One tracking step. Returns False if the object left the view, in which
        # case we stay in TRACK and let the front-end's recovery find it again.
        self._track_step()

    def _run_pick(self):
        # One explicit grasp attempt, then back to tracking whatever remains.
        #
        # The pick itself follows the established rules: track until the target
        # has been still for the dwell time, descend, abandon the descent if the
        # target moves, and on a failed grip re-home and search.
        self.mode = PICK
        self.say('PICK  (explicit command)')
        held = self._pick_object()
        if not self._running():
            return

        if self.cmd.take('reset'):
            self._on_reset()
            return

        if held:
            self.holding = True
            self.say('PICK  (grasped -- press 7 to put it back down)')
            self.mode = HOLD
            return
        else:
            self.say('PICK failed -- re-homing and searching')
            self._go_home()
            self._recover_search()

        if self.world.active():
            self.enter_track('after pick')
        else:
            self.enter_wait('table empty after pick')

    def _run_put_down(self):
        # Second press of the pickup key: set the held cube back on the table.
        #
        # The front-end supplies put_object(), which lowers gently at the cube's
        # own xy and opens the jaws. This runs to completion before the mode
        # changes, so a third press cannot start a new pick while the cube is
        # still being placed -- the queue is only consumed here, and the loop
        # does not see the outer step until this returns.
        self.mode = PUTDOWN
        self.say('PUTDOWN  (placing the cube back on the table)')
        ok = self._put_object()
        self.holding = False
        if not self._running():
            return

        if self.cmd.take('reset'):
            self._on_reset()
            return

        if ok:
            self.say('PUTDOWN  (cube released)')
        else:
            self.say('PUTDOWN failed -- cube may still be held')

        # Retract to the home pose. The put-down leaves the tool low over the
        # table where it released, which is neither a useful standby pose nor
        # where the operator expects the arm after the cube is gone. This is the
        # same retract the failed-grip path above uses, so both exits from a
        # grasp end in the same place.
        self._go_home()
        if not self._running():
            return

        # The cube is on the table again, so go back to watching it.
        if self.world.active():
            self.enter_track('after put-down')
        else:
            self.enter_wait('after put-down')

    def _step_hold(self):
        # Holding a cube: nothing to track, nothing to do until the pickup key
        # is pressed again. If we are in HOLD but not holding, the state is
        # inconsistent -- fall back to waiting rather than guessing.
        if self.cmd.take('place'):
            self.say('SPAWN ignored -- the gripper is holding a cube')
            return
        if self.cmd.take('pickup'):
            self._run_put_down()
            return
        if not self.holding:
            self.enter_wait('idle while holding nothing')

    def _step_pick(self):
        # PICK is entered and completed within _run_pick; if we are ever seen
        # here it is because a front-end interrupted mid-pick. Fall back to a
        # safe, non-moving state.
        self.enter_track('pick interrupted')

    def _put_object(self):
        hook = getattr(self, 'on_put_object', None)
        if hook is None:
            return False
        return bool(hook())

    def _recover_search(self):
        hook = getattr(self, 'on_recover_search', None)
        if hook is not None:
            hook()

    # Overridden by the front-end via attributes, so task.py needs no imports
    # from the scene layer.
    def _on_reset(self):
        hook = getattr(self, 'on_reset', None)
        if hook is not None:
            hook()
        else:
            self.hard_reset()
