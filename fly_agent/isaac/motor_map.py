from collections import deque

import numpy as np


class IsaacMotorMap:
    """
    Converts MaleCNS motor-population firing into Isaac actions.

    HOLD actions:
        W A S D
        arrow keys

    TAP actions:
        E      = bomb
        Q      = consumable
        SPACE  = active item
    """

    ACTION_GROUPS = {
        "move_left": "steer_L",
        "move_right": "steer_R",

        "move_up": "forward_L",
        "move_down": "forward_R",

        "shoot_left": "punch_L",
        "shoot_right": "punch_R",

        "shoot_up": "kick_L",
        "shoot_down": "kick_R",

        "bomb": "escape_L",
        "active_item": "escape_R",
        "consumable": "backward_L",
    }

    HOLD_ACTIONS = (
        "move_left",
        "move_right",
        "move_up",
        "move_down",

        "shoot_left",
        "shoot_right",
        "shoot_up",
        "shoot_down",
    )

    TAP_ACTIONS = (
        "bomb",
        "consumable",
        "active_item",
    )

    def __init__(
        self,
        brain,
        window=10,
        enable_taps=True,
        tap_cooldown_steps=20,
    ):

        self.brain = brain

        self.window = int(
            window
        )

        self.enable_taps = bool(
            enable_taps
        )

        self.tap_cooldown_steps = int(
            tap_cooldown_steps
        )

        groups = brain.groups

        self.cells = {}

        for action, group_name in (
            self.ACTION_GROUPS.items()
        ):

            self.cells[action] = np.asarray(
                groups[group_name],
                dtype=np.int64,
            )

        # Based on our previous no-input calibration.
        self.thresholds = {
            "move_left": 3,
            "move_right": 2,

            "move_up": 2,
            "move_down": 2,

            "shoot_left": 5,
            "shoot_right": 5,

            "shoot_up": 2,
            "shoot_down": 2,

            "bomb": 2,
            "consumable": 2,
            "active_item": 2,
        }

        self.history = deque(
            maxlen=self.window
        )

        self.tap_latched = {
            action: False
            for action in self.TAP_ACTIONS
        }

        self.tap_cooldown = {
            action: 0
            for action in self.TAP_ACTIONS
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):

        self.history.clear()

        for action in self.TAP_ACTIONS:

            self.tap_latched[
                action
            ] = False

            self.tap_cooldown[
                action
            ] = 0

    # =========================================================
    # DECODE
    # =========================================================

    def decode(
        self,
        fired,
    ):

        fired = np.asarray(
            fired,
            dtype=np.int64,
        )

        fired_set = set(
            fired.tolist()
        )

        # -----------------------------------------------------
        # Count motor-population spikes THIS step.
        # -----------------------------------------------------

        current = {}

        for action, cells in (
            self.cells.items()
        ):

            count = 0

            for neuron_id in cells:

                if int(
                    neuron_id
                ) in fired_set:

                    count += 1

            current[action] = count

        self.history.append(
            current
        )

        # -----------------------------------------------------
        # Sum over rolling temporal window.
        # -----------------------------------------------------

        totals = {
            action: 0
            for action in self.cells
        }

        for sample in self.history:

            for action, count in (
                sample.items()
            ):

                totals[action] += count

        # =====================================================
        # HELD ACTIONS
        # =====================================================

        held = set()

        for action in self.HOLD_ACTIONS:

            if (
                totals[action]
                >= self.thresholds[action]
            ):

                held.add(
                    action
                )

        # =====================================================
        # TAP COOLDOWNS
        # =====================================================

        for action in self.TAP_ACTIONS:

            if (
                self.tap_cooldown[action]
                > 0
            ):

                self.tap_cooldown[
                    action
                ] -= 1

        # =====================================================
        # TAP ACTIONS
        # =====================================================

        taps = []

        if self.enable_taps:

            for action in self.TAP_ACTIONS:

                active = (
                    totals[action]
                    >= self.thresholds[action]
                )

                # Rising edge only.
                if (
                    active
                    and not self.tap_latched[action]
                ):

                    if (
                        self.tap_cooldown[action]
                        <= 0
                    ):

                        taps.append(
                            action
                        )

                        self.tap_cooldown[
                            action
                        ] = (
                            self.tap_cooldown_steps
                        )

                    # Latch even if cooldown prevented
                    # a tap. This avoids a delayed tap
                    # while the same neural burst remains
                    # above threshold.
                    self.tap_latched[
                        action
                    ] = True

                elif not active:

                    self.tap_latched[
                        action
                    ] = False

        return (
            held,
            taps,
            totals,
        )