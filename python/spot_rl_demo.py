# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

import bosdyn.client.util
import orbit.orbit_configuration
from hid.gamepad import (
    Gamepad,
    GamepadConfig,
    joystick_connected,
    load_gamepad_configuration,
)
from hid.keyboard import Keyboard
from orbit.onnx_command_generator import (
    OnnxCommandGenerator,
    OnnxControllerContext,
    StateHandler,
)
from spot.mock_spot import MockSpot
from spot.spot import Spot
from utils.event_divider import EventDivider


def main():
    """Command line interface. change that is ok"""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("-m", "--mock", action="store_true")
    parser.add_argument("--gamepad-config", type=Path)
    parser.add_argument(
        "-k",
        "--keyboard",
        action="store_true",
        help="drive with the keyboard (terminal raw input) instead of a gamepad",
    )
    parser.add_argument(
        "--stand_bit",
        action="store_true",
        help="include the standstill bit in the model observations",
    )
    options = parser.parse_args()

    conf_file = orbit.orbit_configuration.detect_config_file(options.policy_file_path)
    policy_file = orbit.orbit_configuration.detect_policy_file(options.policy_file_path)

    context = OnnxControllerContext()
    config = orbit.orbit_configuration.load_configuration(conf_file)
    print(config)

    state_handler = StateHandler(context)
    print(options.verbose)
    command_generator = OnnxCommandGenerator(context, config, policy_file, options.verbose, options.stand_bit)

    # 333 Hz state update / 6 => ~56 Hz control updates
    timeing_policy = EventDivider(context.event, 6)

    controller = None
    if options.keyboard:
        print("[INFO] using keyboard control")
        controller = Keyboard(context)
        # the keyboard owns the terminal, so it is started after the input()
        # prompt below rather than here
    elif joystick_connected():
        if options.gamepad_config is not None:
            print("[INFO] loading gamepad config from file")
            gamepad_config = load_gamepad_configuration(options.gamepad_config)
        else:
            print("[INFO] using default gamepad configuration")
            gamepad_config = GamepadConfig()

        controller = Gamepad(context, gamepad_config)
        controller.start_listening()

    if options.mock:
        spot = MockSpot()
    else:
        spot = Spot(options)

    with spot.lease_keep_alive():
        try:
            spot.power_on()
            spot.stand(0.0)
            spot.start_state_stream(state_handler)

            input()
            spot.start_command_stream(command_generator, timeing_policy)

            if isinstance(controller, Keyboard):
                # grab the terminal and drive until the user presses a quit key
                controller.start_listening()
                controller.wait_until_stopped()
            else:
                input()

        except KeyboardInterrupt:
            print("killed with ctrl-c")

        finally:
            print("stop command stream")
            spot.stop_command_stream()
            print("stop state stream")
            spot.stop_state_stream()
            print("stop controller")
            if controller is not None:
                controller.stop_listening()
            print("all stopped")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
