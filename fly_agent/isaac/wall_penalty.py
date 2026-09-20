import time


class WallCollisionPenalty:
    """
    Detect persistent attempts to move through collision.

    Important:

    This module ONLY identifies the behavioural consequence.

    The trainer decides how that reward modifies synapses.

    WALL_STUCK events contain the specific action responsible,
    allowing action-specific plasticity.

    A tiny WALL_ESCAPE reward is emitted if the agent changes
    behaviour and successfully starts moving after being stuck.
    """

    ACTION_SENSOR = {
        "move_up":
            "collision_up",

        "move_down":
            "collision_down",

        "move_left":
            "collision_left",

        "move_right":
            "collision_right",
    }

    OPPOSITE = {
        "move_up":
            "move_down",

        "move_down":
            "move_up",

        "move_left":
            "move_right",

        "move_right":
            "move_left",
    }

    # Time threshold -> one-time penalty.
    #
    # Total:
    # -0.25 -0.25 -0.50 -0.50 -0.50 = -2.00

    PENALTY_STAGES = [
        (0.50, -0.25),
        (1.00, -0.25),
        (1.50, -0.50),
        (2.00, -0.50),
        (2.50, -0.50),
    ]

    def __init__(
        self,
        collision_threshold=0.55,
        escape_reward=0.10,
        escape_window=2.0,
    ):

        self.collision_threshold = float(
            collision_threshold
        )

        self.escape_reward = float(
            escape_reward
        )

        self.escape_window = float(
            escape_window
        )

        # Per movement direction.

        self.states = {}

        for action in self.ACTION_SENSOR:

            self.states[
                action
            ] = self._fresh_state()

        self.last_room = None
        self.last_player_grid = None

        # Once a real penalty has happened, moving away shortly
        # afterwards can receive one tiny escape reward.

        self.escape_pending_until = None

        self.last_blocked_action = None

    # =========================================================
    # STATE
    # =========================================================

    @staticmethod
    def _fresh_state():

        return {
            "started": None,
            "next_stage": 0,
            "total_penalty": 0.0,
        }

    def _reset_action(
        self,
        action,
    ):

        self.states[
            action
        ] = self._fresh_state()

    def _reset_actions(
        self,
    ):

        for action in self.states:

            self._reset_action(
                action
            )

    def reset(
        self,
    ):

        self._reset_actions()

        self.last_room = None
        self.last_player_grid = None

        self.escape_pending_until = None
        self.last_blocked_action = None

    # =========================================================
    # FIND SUCCESSFUL ESCAPE ACTION
    # =========================================================

    def _escape_action(
        self,
        held_actions,
        features,
    ):

        held = set(
            held_actions
        )

        # Prefer a movement action that is currently not blocked.

        for (
            action,
            sensor,
        ) in self.ACTION_SENSOR.items():

            if action not in held:
                continue

            opposite = (
                self.OPPOSITE[
                    action
                ]
            )

            if opposite in held:
                continue

            collision = float(
                features.get(
                    sensor,
                    0.0,
                )
            )

            if (
                collision
                < self.collision_threshold
            ):

                return action

        return None

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        held_actions,
        features,
        now=None,
    ):

        if now is None:

            now = (
                time.perf_counter()
            )

        now = float(
            now
        )

        events = []

        held = set(
            held_actions
        )

        room = int(
            features.get(
                "room_index",
                -1,
            )
        )

        player_grid = int(
            features.get(
                "player_grid",
                -1,
            )
        )

        # =====================================================
        # ROOM CHANGE
        # =====================================================

        if (
            self.last_room is not None
            and room != self.last_room
        ):

            self._reset_actions()

            self.escape_pending_until = None
            self.last_blocked_action = None

        # =====================================================
        # PLAYER ACTUALLY MOVED
        # =====================================================

        moved = (
            self.last_player_grid is not None
            and player_grid >= 0
            and player_grid
            != self.last_player_grid
            and room == self.last_room
        )

        if moved:

            # If we were recently punished for being stuck and
            # have now genuinely changed grid position, reinforce
            # the movement that successfully escaped.

            if (
                self.escape_pending_until
                is not None
                and now
                <= self.escape_pending_until
            ):

                escape_action = (
                    self._escape_action(
                        held,
                        features,
                    )
                )

                if (
                    escape_action
                    is not None
                ):

                    events.append(
                        {
                            "name":
                                "WALL_ESCAPE",

                            "reward":
                                self.escape_reward,

                            "action":
                                escape_action,

                            "extra":
                                (
                                    f"{escape_action} "
                                    "movement resumed"
                                ),
                        }
                    )

            # Any actual movement breaks the stuck episode.

            self._reset_actions()

            self.escape_pending_until = None
            self.last_blocked_action = None

        # =====================================================
        # REMEMBER CURRENT POSITION
        # =====================================================

        self.last_room = (
            room
        )

        if player_grid >= 0:

            self.last_player_grid = (
                player_grid
            )

        # =====================================================
        # EXPIRE ESCAPE OPPORTUNITY
        # =====================================================

        if (
            self.escape_pending_until
            is not None
            and now
            > self.escape_pending_until
        ):

            self.escape_pending_until = None
            self.last_blocked_action = None

        # =====================================================
        # CHECK EACH MOVEMENT ACTION
        # =====================================================

        for (
            action,
            sensor_name,
        ) in self.ACTION_SENSOR.items():

            opposite = (
                self.OPPOSITE[
                    action
                ]
            )

            collision = float(
                features.get(
                    sensor_name,
                    0.0,
                )
            )

            action_held = (
                action in held
            )

            opposite_held = (
                opposite in held
            )

            blocked = (
                collision
                >= self.collision_threshold
            )

            # Don't infer causality when opposite keys are
            # simultaneously being requested.

            valid_attempt = (
                action_held
                and not opposite_held
                and blocked
            )

            if not valid_attempt:

                self._reset_action(
                    action
                )

                continue

            state = (
                self.states[
                    action
                ]
            )

            # -----------------------------------------------
            # BEGIN STUCK TIMER
            # -----------------------------------------------

            if (
                state[
                    "started"
                ]
                is None
            ):

                state[
                    "started"
                ] = now

                continue

            duration = (
                now
                - state[
                    "started"
                ]
            )

            stage_index = int(
                state[
                    "next_stage"
                ]
            )

            if (
                stage_index
                >= len(
                    self.PENALTY_STAGES
                )
            ):

                # Episode has reached the -2 cap.

                continue

            (
                threshold,
                penalty,
            ) = (
                self.PENALTY_STAGES[
                    stage_index
                ]
            )

            if (
                duration
                < threshold
            ):

                continue

            # -----------------------------------------------
            # ONE STAGE PENALTY
            # -----------------------------------------------

            state[
                "next_stage"
            ] += 1

            state[
                "total_penalty"
            ] += float(
                penalty
            )

            self.escape_pending_until = (
                now
                + self.escape_window
            )

            self.last_blocked_action = (
                action
            )

            events.append(
                {
                    "name":
                        "WALL_STUCK",

                    "reward":
                        float(
                            penalty
                        ),

                    "action":
                        action,

                    "extra":
                        (
                            f"{action} "
                            f"{duration:.2f}s "
                            f"collision="
                            f"{collision:.2f} "
                            f"episode="
                            f"{state['total_penalty']:+.2f}"
                        ),
                }
            )

        return events