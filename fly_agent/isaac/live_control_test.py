import time

import pydirectinput
import win32gui

from brain.fly import FlyAgentBrain
from learning.plasticity import RewardPlasticity

from isaac.environment import IsaacEnvironment
from isaac.perception import IsaacPerception
from isaac.lc10_vision import IsaacLC10Vision
from isaac.motor_map import IsaacMotorMap
from isaac.neural_explorer import NeuralExplorer


# =========================================
# SETTINGS
# =========================================

RUN_SECONDS = 45

ENABLE_TAPS = False

ENABLE_EXPLORATION = True


def focus_isaac(
    env,
):

    window = env.controls.window

    print()
    print(
        "Trying to focus Isaac..."
    )

    window.focus()

    time.sleep(
        0.5
    )

    foreground = (
        win32gui.GetForegroundWindow()
    )

    if foreground == window.hwnd:

        print(
            "Isaac successfully focused."
        )

        return True

    print(
        "Normal focus failed."
    )

    print(
        "Using click fallback..."
    )

    rect = window.client_rect()

    center_x = (
        rect["left"]
        + rect["width"] // 2
    )

    center_y = (
        rect["top"]
        + rect["height"] // 2
    )

    pydirectinput.click(
        center_x,
        center_y,
    )

    time.sleep(
        0.7
    )

    foreground = (
        win32gui.GetForegroundWindow()
    )

    if foreground == window.hwnd:

        print(
            "Isaac successfully focused "
            "using click fallback."
        )

        return True

    foreground_title = (
        win32gui.GetWindowText(
            foreground
        )
    )

    print()
    print(
        "ERROR: Isaac is NOT focused."
    )

    print(
        "Current foreground window:"
    )

    print(
        foreground_title
    )

    return False


def press_escape():

    print(
        "Sending ESC..."
    )

    pydirectinput.keyDown(
        "esc"
    )

    time.sleep(
        0.15
    )

    pydirectinput.keyUp(
        "esc"
    )


def main():

    print("==============================")
    print(" MALECNS ISAAC EXPLORATION")
    print("==============================")

    # ==================================
    # ISAAC
    # ==================================

    env = IsaacEnvironment()

    perception = IsaacPerception(
        motion_threshold=12
    )

    # ==================================
    # MALECNS
    # ==================================

    brain = FlyAgentBrain()

    plasticity = RewardPlasticity(
        brain,
        learning_rate=0.0,
        homeostasis=0.0,
    )

    plasticity.load_checkpoint(
        "learned_brain.npz"
    )

    # ==================================
    # VISUAL INPUT
    # ==================================

    vision = IsaacLC10Vision(
        brain,
        gain=4.0,
        max_drive=0.8,
        minimum_activity=0.01,
    )

    # ==================================
    # MOTOR DECODER
    # ==================================

    motor = IsaacMotorMap(
        brain,
        window=10,
        enable_taps=ENABLE_TAPS,
    )

    # ==================================
    # NEURAL EXPLORATION
    # ==================================

    explorer = NeuralExplorer(
        brain,
        drive=0.8,

        min_wait_steps=15,
        max_wait_steps=30,

        min_burst_steps=5,
        max_burst_steps=8,

        seed=2026,
    )

    # ==================================
    # RESET
    # ==================================

    brain.brain.reset(
        seed=12345
    )

    perception.reset()
    motor.reset()
    explorer.reset()

    env.reset_controls()

    print()
    print(
        f"Control time: "
        f"{RUN_SECONDS} seconds"
    )

    print()
    print(
        "REAL Isaac vision: ON"
    )

    print(
        "Neural exploration:",
        ENABLE_EXPLORATION
    )

    print()
    print(
        "Allowed:"
    )

    print(
        "W A S D"
    )

    print(
        "Arrow keys"
    )

    print()
    print(
        "Disabled:"
    )

    print(
        "E Q SPACE"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Learning is OFF."
    )

    print(
        "This is only an exploration test."
    )

    print()
    print(
        "Isaac will be focused "
        "in 3 seconds."
    )

    time.sleep(
        3
    )

    # ==================================
    # FOCUS ISAAC
    # ==================================

    if not focus_isaac(
        env
    ):

        env.stop()

        return

    time.sleep(
        1
    )

    # ==================================
    # UNPAUSE
    # ==================================

    press_escape()

    time.sleep(
        1
    )

    foreground = (
        win32gui.GetForegroundWindow()
    )

    if (
        foreground
        != env.controls.window.hwnd
    ):

        print(
            "Isaac lost focus after ESC."
        )

        env.stop()

        return

    print()
    print(
        "MaleCNS control started."
    )

    print()

    # ==================================
    # STATS
    # ==================================

    start_time = (
        time.perf_counter()
    )

    step = 0

    motor_events = 0
    exploration_bursts = 0

    try:

        while True:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            if (
                elapsed
                >= RUN_SECONDS
            ):
                break

            # --------------------------
            # CHECK FOCUS
            # --------------------------

            foreground = (
                win32gui.GetForegroundWindow()
            )

            if (
                foreground
                != env.controls.window.hwnd
            ):

                print()
                print(
                    "Isaac lost focus!"
                )

                print(
                    "Stopping."
                )

                break

            # --------------------------
            # 1. SEE ISAAC
            # --------------------------

            frame = env.observe()

            features = (
                perception.process(
                    frame
                )
            )

            # --------------------------
            # 2. EXPLORATION
            # --------------------------

            extra_inject = []

            explore_action = None

            started_burst = False

            if ENABLE_EXPLORATION:

                (
                    extra_inject,
                    explore_action,
                    started_burst,
                ) = explorer.step()

            if started_burst:

                exploration_bursts += 1

                print(
                    f"EXPLORE step={step:05d} | "
                    f"neural={explore_action}"
                )

            # --------------------------
            # 3. ISAAC VISION
            #    +
            #    EXPLORATION
            #    -> MaleCNS
            # --------------------------

            fired = vision.step(
                features,
                extra_inject=extra_inject,
            )

            # --------------------------
            # 4. MALECNS -> ACTIONS
            # --------------------------

            held, taps, totals = (
                motor.decode(
                    fired
                )
            )

            # --------------------------
            # 5. REAL KEYBOARD
            # --------------------------

            env.act(
                held_actions=held,

                # Still NEVER send
                # E/Q/Space.
                tap_actions=[],
            )

            step += 1

            if held:

                motor_events += 1

                print(
                    f"ACTION  step={step:05d} | "
                    f"{sorted(held)}"
                )

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopped manually."
        )

    finally:

        env.stop()

    # ==================================
    # SUMMARY
    # ==================================

    print()
    print("==============================")
    print("       TEST FINISHED")
    print("==============================")

    print(
        "Brain steps:",
        step
    )

    print(
        "Exploration bursts:",
        exploration_bursts
    )

    print(
        "Motor-active steps:",
        motor_events
    )

    print()
    print(
        "Learning was OFF."
    )

    print(
        "No checkpoint was changed."
    )

    print(
        "All keys released."
    )


if __name__ == "__main__":
    main()