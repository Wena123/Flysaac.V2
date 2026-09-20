import time

import pydirectinput
import win32gui


def _focus_isaac(
    env,
):
    window = env.controls.window

    foreground = (
        win32gui.GetForegroundWindow()
    )

    if foreground == window.hwnd:
        return True

    print(
        "Refocusing Isaac..."
    )

    window.focus()

    time.sleep(
        0.5
    )

    foreground = (
        win32gui.GetForegroundWindow()
    )

    if foreground == window.hwnd:
        return True

    # -----------------------------------------
    # Click fallback
    # -----------------------------------------

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

    return (
        win32gui.GetForegroundWindow()
        == window.hwnd
    )


def _press_space():

    pydirectinput.keyDown(
        "space"
    )

    time.sleep(
        0.20
    )

    pydirectinput.keyUp(
        "space"
    )


def restart_after_death(
    env,
    rewards,
    initial_wait=2.5,
    retry_wait=3.0,
):
    """
    Restart Isaac after death.

    Instead of assuming SPACE worked, this waits
    until the reward bridge reports RUN_START.

    It retries indefinitely until:
        - Isaac successfully restarts
        - or the user presses CTRL+C
    """

    print()
    print(
        "================================"
    )

    print(
        "        ISAAC DIED"
    )

    print(
        "================================"
    )

    # ========================================================
    # RELEASE ALL CURRENT INPUTS
    # ========================================================

    env.reset_controls()

    print(
        "All held keys released."
    )

    # The PLAYER_DEATH event usually arrives before the
    # death screen is fully ready.

    print(
        "Waiting for death screen..."
    )

    time.sleep(
        initial_wait
    )

    attempt = 0

    collected_events = []

    # ========================================================
    # KEEP TRYING UNTIL LUA REPORTS RUN_START
    # ========================================================

    while True:

        attempt += 1

        print()
        print(
            f"Restart attempt #{attempt}"
        )

        # -----------------------------------------
        # Ensure Isaac receives the SPACE press.
        # -----------------------------------------

        if not _focus_isaac(
            env
        ):

            print(
                "Could not focus Isaac."
            )

            print(
                "Trying again..."
            )

            time.sleep(
                1.0
            )

            continue

        # -----------------------------------------
        # SPACE
        # -----------------------------------------

        print(
            "Pressing SPACE..."
        )

        _press_space()

        # -----------------------------------------
        # Wait for RUN_START from the Lua bridge.
        # -----------------------------------------

        deadline = (
            time.perf_counter()
            + retry_wait
        )

        while (
            time.perf_counter()
            < deadline
        ):

            reward, events = (
                rewards.poll()
            )

            if events:

                for event in events:

                    collected_events.append(
                        event.copy()
                    )

                    print(
                        "RESTART EVENT | "
                        f"{event['name']} | "
                        f"reward="
                        f"{event['reward']:+.2f}"
                    )

                    if (
                        event["name"]
                        == "RUN_START"
                    ):

                        print()
                        print(
                            "NEW ISAAC RUN CONFIRMED!"
                        )

                        print(
                            "================================"
                        )

                        print()

                        return (
                            True,
                            collected_events,
                        )

            time.sleep(
                0.10
            )

        print(
            "No RUN_START detected."
        )

        print(
            "SPACE will be tried again."
        )