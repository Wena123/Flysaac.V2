import numpy as np


class NeuralExplorer:
    """
    Exploration performed INSIDE MaleCNS.

    Continuous actions receive longer neural bursts.

    Bomb / consumable / active-item actions receive
    short neural pulses so they behave like taps.
    """

    def __init__(
        self,
        brain,
        drive=0.8,
        min_wait_steps=8,
        max_wait_steps=18,
        min_burst_steps=5,
        max_burst_steps=10,
        tap_burst_steps=4,
        tap_probability=0.20,
        seed=2026,
    ):

        self.brain = brain

        self.drive = float(
            drive
        )

        self.min_wait_steps = int(
            min_wait_steps
        )

        self.max_wait_steps = int(
            max_wait_steps
        )

        self.min_burst_steps = int(
            min_burst_steps
        )

        self.max_burst_steps = int(
            max_burst_steps
        )

        self.tap_burst_steps = int(
            tap_burst_steps
        )

        self.tap_probability = float(
            tap_probability
        )

        self.rng = np.random.default_rng(
            seed
        )

        groups = brain.groups

        # =====================================================
        # CONTINUOUS ACTIONS
        # =====================================================

        self.channels = {
            "move_left":
                np.asarray(
                    groups["steer_L"],
                    dtype=np.int64,
                ),

            "move_right":
                np.asarray(
                    groups["steer_R"],
                    dtype=np.int64,
                ),

            "move_up":
                np.asarray(
                    groups["forward_L"],
                    dtype=np.int64,
                ),

            "move_down":
                np.asarray(
                    groups["forward_R"],
                    dtype=np.int64,
                ),

            "shoot_left":
                np.asarray(
                    groups["punch_L"],
                    dtype=np.int64,
                ),

            "shoot_right":
                np.asarray(
                    groups["punch_R"],
                    dtype=np.int64,
                ),

            "shoot_up":
                np.asarray(
                    groups["kick_L"],
                    dtype=np.int64,
                ),

            "shoot_down":
                np.asarray(
                    groups["kick_R"],
                    dtype=np.int64,
                ),

            # =================================================
            # TAP ACTIONS
            # =================================================

            "bomb":
                np.asarray(
                    groups["escape_L"],
                    dtype=np.int64,
                ),

            "active_item":
                np.asarray(
                    groups["escape_R"],
                    dtype=np.int64,
                ),

            "consumable":
                np.asarray(
                    groups["backward_L"],
                    dtype=np.int64,
                ),
        }

        self.continuous_actions = [
            "move_left",
            "move_right",
            "move_up",
            "move_down",

            "shoot_left",
            "shoot_right",
            "shoot_up",
            "shoot_down",
        ]

        self.tap_actions = [
            "bomb",
            "consumable",
            "active_item",
        ]

        self.current_action = None

        self.remaining_burst = 0

        self.wait_remaining = (
            self._random_wait()
        )

    # =========================================================
    # RANDOM DELAYS
    # =========================================================

    def _random_wait(
        self,
    ):

        return int(
            self.rng.integers(
                self.min_wait_steps,
                self.max_wait_steps + 1,
            )
        )

    def _random_continuous_burst(
        self,
    ):

        return int(
            self.rng.integers(
                self.min_burst_steps,
                self.max_burst_steps + 1,
            )
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):

        self.current_action = None

        self.remaining_burst = 0

        self.wait_remaining = (
            self._random_wait()
        )

    # =========================================================
    # CHOOSE ACTION
    # =========================================================

    def _choose_action(
        self,
    ):

        if (
            self.rng.random()
            < self.tap_probability
        ):

            action = str(
                self.rng.choice(
                    self.tap_actions
                )
            )

            burst = (
                self.tap_burst_steps
            )

        else:

            action = str(
                self.rng.choice(
                    self.continuous_actions
                )
            )

            burst = (
                self._random_continuous_burst()
            )

        return (
            action,
            burst,
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(
        self,
    ):

        # -----------------------------------------------------
        # Existing burst.
        # -----------------------------------------------------

        if (
            self.remaining_burst
            > 0
        ):

            action = (
                self.current_action
            )

            cells = (
                self.channels[action]
            )

            self.remaining_burst -= 1

            if (
                self.remaining_burst
                == 0
            ):

                self.current_action = None

                self.wait_remaining = (
                    self._random_wait()
                )

            return (
                [
                    (
                        cells,
                        self.drive,
                    )
                ],
                action,
                False,
            )

        # -----------------------------------------------------
        # Waiting between exploration events.
        # -----------------------------------------------------

        if (
            self.wait_remaining
            > 0
        ):

            self.wait_remaining -= 1

            return (
                [],
                None,
                False,
            )

        # -----------------------------------------------------
        # Start new exploration event.
        # -----------------------------------------------------

        (
            action,
            burst,
        ) = self._choose_action()

        self.current_action = action

        # Current call counts as first burst step.
        self.remaining_burst = max(
            0,
            burst - 1,
        )

        cells = (
            self.channels[action]
        )

        if (
            self.remaining_burst
            == 0
        ):

            self.current_action = None

            self.wait_remaining = (
                self._random_wait()
            )

        return (
            [
                (
                    cells,
                    self.drive,
                )
            ],
            action,
            True,
        )