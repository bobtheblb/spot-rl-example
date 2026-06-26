# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Terminal raw-input keyboard controller.

Drop-in replacement for the Gamepad: it runs a background thread that maps held
keys to a [x_vel, y_vel, yaw] command and writes it to context.velocity_cmd,
exactly like hid.gamepad.Gamepad does.

Unlike a joystick, a terminal only delivers key-press events (no key-release),
so "holding" a key is emulated via the OS key-repeat stream: each key sets its
axis and stamps the time; an axis is zeroed once no fresh press has arrived for
`key_timeout` seconds. Tune `key_timeout` (and/or your OS key-repeat delay) if
movement stutters or coasts too long after release.
"""

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

QUIT_KEYS = ("\x1b", "\x03")  # ESC, Ctrl-C
STOP_KEY = " "  # spacebar: zero all axes immediately

HELP_TEXT = """\
[INFO] Keyboard control:
    w / s : forward / backward
    a / d : strafe left / right
    q / e : turn left / right
    space : stop (zero all)
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

        self._stopping = False
        self._listening_thread = None
        self._fd = sys.stdin.fileno()
        self._old_term = None

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
                if ch in QUIT_KEYS:
                    self._stopping = True
                    break
                elif ch == STOP_KEY:
                    target = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                elif ch.lower() in KEY_BINDINGS:
                    axis, sign = KEY_BINDINGS[ch.lower()]
                    target[axis] = sign * self._speed[axis]
                    last_press[axis] = now

            # no key-release events from a terminal: zero an axis once its
            # key-repeat stream goes quiet
            for axis in target:
                if target[axis] != 0.0 and (now - last_press[axis]) > self._timeout:
                    target[axis] = 0.0

            self.x_vel = target["x"]
            self.y_vel = target["y"]
            self.yaw = target["yaw"]
            self._context.velocity_cmd = [self.x_vel, self.y_vel, self.yaw]

        # leave the robot stationary on the way out
        self._context.velocity_cmd = [0, 0, 0]

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
