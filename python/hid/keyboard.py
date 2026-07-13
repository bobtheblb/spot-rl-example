# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Terminal raw-input keyboard controller.

Drop-in replacement for the Gamepad: it runs a background thread that maps held
keys to a [x_vel, y_vel, yaw, height, roll, pitch] command and writes it to
context.velocity_cmd, exactly like hid.gamepad.Gamepad does.

The first three axes are *velocity* commands: a terminal only delivers key-press
events (no key-release), so "holding" a key is emulated via the OS key-repeat
stream: each key sets its axis and stamps the time; an axis is zeroed once no
fresh press has arrived for `key_timeout` seconds. Tune `key_timeout` (and/or
your OS key-repeat delay) if movement stutters or coasts too long after release.

The last three axes are *pose* commands (base height, roll, pitch): unlike the
velocity axes they hold their value between presses. Each press nudges the value
by a fixed step and clamps it to its allowed range. Height ranges from 0.4 to
0.6 (default 0.5); roll and pitch range from -pi/9 to pi/9 (default 0).
"""

import math
import select
import sys
import termios
import time
import tty
from threading import Thread


# key -> (axis, signed magnitude as a fraction of that axis' speed)
KEY_BINDINGS = {
    "w": ("x", 1.0),   # forward
    "s": ("x", -1.0),  # backward
    "a": ("y", 1.0),   # strafe left
    "d": ("y", -1.0),  # strafe right
    "q": ("yaw", 1.0),  # turn left
    "e": ("yaw", -1.0),  # turn right
}

# pose command limits (see module docstring)
HEIGHT_MIN, HEIGHT_MAX, HEIGHT_DEFAULT = 0.4, 0.6, 0.5
ANGLE_LIMIT, ANGLE_DEFAULT = math.pi / 9, 0.0

# per-press step applied to a held pose axis
HEIGHT_STEP = 0.01
ANGLE_STEP = math.pi / 90  # ~2 degrees

# key -> (pose axis, signed step) for the held pose commands
POSE_BINDINGS = {
    "r": ("height", HEIGHT_STEP),   # raise body
    "f": ("height", -HEIGHT_STEP),  # lower body
    "z": ("roll", ANGLE_STEP),      # roll +
    "x": ("roll", -ANGLE_STEP),     # roll -
    "c": ("pitch", ANGLE_STEP),     # pitch +
    "v": ("pitch", -ANGLE_STEP),    # pitch -
}

# min/max clamp per pose axis
POSE_LIMITS = {
    "height": (HEIGHT_MIN, HEIGHT_MAX),
    "roll": (-ANGLE_LIMIT, ANGLE_LIMIT),
    "pitch": (-ANGLE_LIMIT, ANGLE_LIMIT),
}

# pose axis defaults, applied on start-up and reset
POSE_DEFAULTS = {
    "height": HEIGHT_DEFAULT,
    "roll": ANGLE_DEFAULT,
    "pitch": ANGLE_DEFAULT,
}

QUIT_KEYS = ("\x1b", "\x03")  # ESC, Ctrl-C
STOP_KEY = " "  # spacebar: zero velocities and reset pose to defaults

HELP_TEXT = """\
[INFO] Keyboard control:
    w / s : forward / backward
    a / d : strafe left / right
    q / e : turn left / right
    r / f : body height up / down
    z / x : roll + / -
    c / v : pitch + / -
    space : stop (zero velocity, reset pose)
    esc   : quit driving
"""


class Keyboard:
    def __init__(
        self,
        context,
        forward_speed: float = 1.0,
        lateral_speed: float = 1.0,
        yaw_speed: float = 1.0,
        key_timeout: float = 0.4,
    ):
        self._context = context
        self._speed = {"x": forward_speed, "y": lateral_speed, "yaw": yaw_speed}
        self._timeout = key_timeout

        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw = 0.0

        # pose commands hold their value between presses
        self.pose = dict(POSE_DEFAULTS)

        # seed the command with the 6-element default so the first observations
        # (collected before this thread starts writing) have the right shape
        self._context.velocity_cmd = self._command()

        self._stopping = False
        self._listening_thread = None
        self._fd = sys.stdin.fileno()
        self._old_term = None

    def _command(self):
        """assemble the 6-element command from the current velocity and pose state"""
        return [
            self.x_vel,
            self.y_vel,
            self.yaw,
            self.pose["height"],
            self.pose["roll"],
            self.pose["pitch"],
        ]

    def start_listening(self):
        print(HELP_TEXT, end="")
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # disables line buffering/echo, keeps Ctrl-C as SIGINT
        self._stopping = False
        self._listening_thread = Thread(target=self.listen)
        self._listening_thread.start()

    def listen(self):
        # current commanded magnitude per axis and the time it was last refreshed
        target = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        last_press = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        while not self._stopping:
            now = time.monotonic()

            # poll stdin at ~100Hz so we always have fresh data for the command loop
            readable, _, _ = select.select([sys.stdin], [], [], 0.01)
            if readable:
                ch = sys.stdin.read(1)
                key = ch.lower()
                if ch in QUIT_KEYS:
                    self._stopping = True
                    break
                elif ch == STOP_KEY:
                    target = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                    self.pose = dict(POSE_DEFAULTS)
                elif key in KEY_BINDINGS:
                    axis, sign = KEY_BINDINGS[key]
                    target[axis] = sign * self._speed[axis]
                    last_press[axis] = now
                elif key in POSE_BINDINGS:
                    axis, step = POSE_BINDINGS[key]
                    lo, hi = POSE_LIMITS[axis]
                    self.pose[axis] = min(max(self.pose[axis] + step, lo), hi)

            # no key-release events from a terminal: zero an axis once its
            # key-repeat stream goes quiet
            for axis in target:
                if target[axis] != 0.0 and (now - last_press[axis]) > self._timeout:
                    target[axis] = 0.0

            self.x_vel = target["x"]
            self.y_vel = target["y"]
            self.yaw = target["yaw"]
            self._context.velocity_cmd = self._command()

        # leave the robot stationary at the default pose on the way out
        self.x_vel = self.y_vel = self.yaw = 0.0
        self.pose = dict(POSE_DEFAULTS)
        self._context.velocity_cmd = self._command()

    def wait_until_stopped(self):
        """Block until the user presses a quit key (or the thread is stopped)."""
        if self._listening_thread is not None:
            self._listening_thread.join()

    def stop_listening(self):
        self._stopping = True
        if self._listening_thread is not None:
            self._listening_thread.join()
            self._listening_thread = None
        if self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            self._old_term = None
