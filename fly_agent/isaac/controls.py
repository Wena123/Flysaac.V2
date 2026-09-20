import time

import pydirectinput

from isaac.window import IsaacWindow


class IsaacControls:

    ACTIONS = {

        # Movement
        "move_up": "w",
        "move_down": "s",
        "move_left": "a",
        "move_right": "d",

        # Shooting
        "shoot_up": "up",
        "shoot_down": "down",
        "shoot_left": "left",
        "shoot_right": "right",

        # Other Isaac actions
        "bomb": "e",
        "consumable": "q",
        "active_item": "space",
    }

    # These actions can remain held.
    HOLD_ACTIONS = {
        "move_up",
        "move_down",
        "move_left",
        "move_right",

        "shoot_up",
        "shoot_down",
        "shoot_left",
        "shoot_right",
    }

    def __init__(self):

        self.window = IsaacWindow()

        self.held = set()

        # Remove PyAutoGUI-style delay
        # between commands.
        pydirectinput.PAUSE = 0.0

    def focus(self):

        self.window.focus()

    def hold(self, action):

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unknown action: {action}"
            )

        if action in self.held:
            return

        key = self.ACTIONS[action]

        pydirectinput.keyDown(
            key
        )

        self.held.add(action)

    def release(self, action):

        if action not in self.held:
            return

        key = self.ACTIONS[action]

        pydirectinput.keyUp(
            key
        )

        self.held.remove(action)

    def set_held(self, actions):
        """
        Make exactly these continuous
        movement/shooting actions active.
        """

        actions = set(actions)

        invalid = (
            actions
            - self.HOLD_ACTIONS
        )

        if invalid:
            raise ValueError(
                f"Cannot hold: {invalid}"
            )

        # Release actions no longer wanted.
        for action in list(
            self.held
        ):

            if action not in actions:
                self.release(action)

        # Hold new actions.
        for action in actions:

            if action not in self.held:
                self.hold(action)

    def tap(
        self,
        action,
        duration=0.08
    ):

        if action not in self.ACTIONS:
            raise ValueError(
                f"Unknown action: {action}"
            )

        key = self.ACTIONS[action]

        pydirectinput.keyDown(
            key
        )

        time.sleep(
            duration
        )

        pydirectinput.keyUp(
            key
        )

    def release_all(self):

        for action in list(
            self.held
        ):
            self.release(action)

        # Safety release every known key too.
        for key in set(
            self.ACTIONS.values()
        ):

            try:
                pydirectinput.keyUp(
                    key
                )
            except Exception:
                pass