import time


class RoomStallPenalty:
    """
    Penalizes the agent for staying too long in the same room
    without meaningful progress.

    Idea:
        - timer starts when entering a room
        - timer resets on "progress" events
        - after grace_seconds, a penalty is applied every tick_seconds
        - the penalty ramps up over time
        - penalty resets when entering a new room
    """

    def __init__(
        self,
        grace_seconds=120.0,
        tick_seconds=5.0,
        base_penalty=0.05,
        ramp_per_tick=0.03,
        max_penalty_per_tick=0.50,
        max_room_penalty=8.0,
    ):
        self.grace_seconds = float(
            grace_seconds
        )

        self.tick_seconds = float(
            tick_seconds
        )

        self.base_penalty = float(
            base_penalty
        )

        self.ramp_per_tick = float(
            ramp_per_tick
        )

        self.max_penalty_per_tick = float(
            max_penalty_per_tick
        )

        self.max_room_penalty = float(
            max_room_penalty
        )

        self.reset()

    def reset(self):
        self.current_room_key = None
        self.room_enter_time = None
        self.last_progress_time = None
        self.last_penalty_time = None
        self.room_penalty_total = 0.0

    def _room_key(
        self,
        room_state,
    ):
        """
        Build a room identity from whatever fields are available.
        Adjust this if your room_state uses different names.
        """

        floor = getattr(
            room_state,
            "floor_stage",
            None,
        )

        if floor is None:
            floor = getattr(
                room_state,
                "floor_id",
                None,
            )

        if floor is None:
            floor = getattr(
                room_state,
                "floor",
                None,
            )

        room = getattr(
            room_state,
            "room_id",
            None,
        )

        if room is None:
            room = getattr(
                room_state,
                "current_room",
                None,
            )

        if room is None:
            room = getattr(
                room_state,
                "room_index",
                None,
            )

        shape = getattr(
            room_state,
            "room_shape",
            None,
        )

        return (
            str(floor),
            str(room),
            str(shape),
        )

    def _ensure_room(
        self,
        room_state,
        now,
    ):
        key = self._room_key(
            room_state
        )

        if key != self.current_room_key:
            self.current_room_key = key
            self.room_enter_time = now
            self.last_progress_time = now
            self.last_penalty_time = None
            self.room_penalty_total = 0.0
            return True

        return False

    def note_progress(
        self,
        room_state,
        reason="progress",
        now=None,
    ):
        if now is None:
            now = time.perf_counter()

        now = float(now)

        self._ensure_room(
            room_state,
            now,
        )

        self.last_progress_time = now

    def update(
        self,
        room_state,
        now=None,
    ):
        """
        Returns a list of reward events.
        Usually [] or one ROOM_STALL event.
        """

        if now is None:
            now = time.perf_counter()

        now = float(now)

        self._ensure_room(
            room_state,
            now,
        )

        if self.last_progress_time is None:
            self.last_progress_time = now
            return []

        stagnant_for = (
            now - self.last_progress_time
        )

        if stagnant_for < self.grace_seconds:
            return []

        if (
            self.last_penalty_time
            is not None
            and (
                now - self.last_penalty_time
            ) < self.tick_seconds
        ):
            return []

        steps_over = int(
            (
                stagnant_for
                - self.grace_seconds
            )
            // self.tick_seconds
        )

        penalty_amount = (
            self.base_penalty
            + steps_over
            * self.ramp_per_tick
        )

        penalty_amount = min(
            penalty_amount,
            self.max_penalty_per_tick,
        )

        remaining_budget = (
            self.max_room_penalty
            - self.room_penalty_total
        )

        if remaining_budget <= 0.0:
            return []

        penalty_amount = min(
            penalty_amount,
            remaining_budget,
        )

        if penalty_amount <= 0.0:
            return []

        self.room_penalty_total += (
            penalty_amount
        )

        self.last_penalty_time = now

        return [
            {
                "name": "ROOM_STALL",
                "reward": -float(
                    penalty_amount
                ),
                "extra": (
                    f"stagnant={stagnant_for:.1f}s "
                    f"room_total=-{self.room_penalty_total:.2f}"
                ),
            }
        ]