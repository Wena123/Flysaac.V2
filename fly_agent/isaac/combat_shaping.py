import math
import time


SHOOT_ACTIONS = {
    "shoot_left",
    "shoot_right",
    "shoot_up",
    "shoot_down",
}


MOVE_VECTOR = {
    "move_left":
        (-1.0, 0.0),

    "move_right":
        (1.0, 0.0),

    "move_up":
        (0.0, -1.0),

    "move_down":
        (0.0, 1.0),
}


class CombatShaping:

    def __init__(
        self,

        no_enemy_penalty=-0.05,
        no_enemy_cooldown=0.50,

        projectile_penalty=-0.08,
        projectile_radius=120.0,
        projectile_dot_threshold=0.25,
        projectile_cooldown=0.40,
        projectile_episode_cap=-0.32,
    ):

        self.no_enemy_penalty = float(
            no_enemy_penalty
        )

        self.no_enemy_cooldown = float(
            no_enemy_cooldown
        )

        self.projectile_penalty = float(
            projectile_penalty
        )

        self.projectile_radius = float(
            projectile_radius
        )

        self.projectile_dot_threshold = float(
            projectile_dot_threshold
        )

        self.projectile_cooldown = float(
            projectile_cooldown
        )

        self.projectile_episode_cap = float(
            projectile_episode_cap
        )

        self.last_no_enemy = {
            action: None
            for action in SHOOT_ACTIONS
        }

        self.last_projectile_penalty = {
            action: None
            for action in MOVE_VECTOR
        }

        self.projectile_episode_total = {
            action: 0.0
            for action in MOVE_VECTOR
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        for action in self.last_no_enemy:

            self.last_no_enemy[
                action
            ] = None

        for action in self.last_projectile_penalty:

            self.last_projectile_penalty[
                action
            ] = None

            self.projectile_episode_total[
                action
            ] = 0.0

    # =========================================================
    # SHOOTING IN EMPTY ROOM
    # =========================================================

    def filter_shooting(
        self,
        requested_held,
        enemies_alive,
        now=None,
    ):
        """
        Shooting is NEVER blocked.

        If MaleCNS shoots while there are no enemies,
        the action still reaches Isaac normally.

        We only create a small negative reward event so
        the fly can learn that pointless shooting is bad.
        """

        if now is None:

            now = (
                time.perf_counter()
            )

        now = float(
            now
        )

        requested = set(
            requested_held
        )

        events = []

        # -----------------------------------------------------
        # ENEMIES EXIST
        # -----------------------------------------------------

        if (
            enemies_alive
            > 0
        ):

            return (
                requested,
                events,
            )

        # -----------------------------------------------------
        # NO ENEMIES
        # -----------------------------------------------------

        attempted = (
            requested
            & SHOOT_ACTIONS
        )

        for action in attempted:

            previous = (
                self.last_no_enemy[
                    action
                ]
            )

            # Avoid punishing every single brain step while
            # the same shoot key remains held.

            if (
                previous is not None
                and (
                    now - previous
                )
                < self.no_enemy_cooldown
            ):

                continue

            self.last_no_enemy[
                action
            ] = now

            events.append(
                {
                    "name":
                        "NO_ENEMY_SHOT",

                    "reward":
                        self.no_enemy_penalty,

                    "action":
                        action,

                    "extra":
                        "shot fired with no enemies",
                }
            )

        # -----------------------------------------------------
        # IMPORTANT
        # -----------------------------------------------------
        #
        # Return requested actions UNCHANGED.
        #
        # Shooting is NOT removed.
        # Python is no longer deciding whether shooting is
        # allowed.
        #
        # MaleCNS shoots normally and learns from reward.
        # -----------------------------------------------------

        return (
            requested,
            events,
        )

    # =========================================================
    # PROJECTILE DANGER
    # =========================================================

    def projectile_events(
        self,
        held_actions,
        combat_state,
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

        distance = float(
            combat_state
            .projectile_distance
        )

        count = int(
            combat_state
            .projectile_count
        )

        # -----------------------------------------------------
        # NO CURRENT PROJECTILE DANGER
        # -----------------------------------------------------

        if (
            count <= 0
            or distance < 0
            or distance
            > self.projectile_radius
        ):

            # New danger episode can begin next time a projectile
            # enters the radius.

            for action in MOVE_VECTOR:

                self.projectile_episode_total[
                    action
                ] = 0.0

            return events

        # -----------------------------------------------------
        # PROJECTILE POSITION RELATIVE TO PLAYER
        # -----------------------------------------------------

        dx = float(
            combat_state
            .projectile_dx
        )

        dy = float(
            combat_state
            .projectile_dy
        )

        length = math.hypot(
            dx,
            dy,
        )

        if (
            length
            <= 0.001
        ):

            return events

        # Unit vector:
        #
        # player -> projectile

        projectile_x = (
            dx
            / length
        )

        projectile_y = (
            dy
            / length
        )

        held = set(
            held_actions
        )

        # -----------------------------------------------------
        # CHECK MOVEMENT DIRECTIONS
        # -----------------------------------------------------

        for (
            action,
            (
                move_x,
                move_y,
            ),
        ) in MOVE_VECTOR.items():

            if (
                action
                not in held
            ):

                continue

            # Positive dot product means the movement direction
            # points toward the nearby projectile.

            toward = (
                move_x
                * projectile_x
                +
                move_y
                * projectile_y
            )

            if (
                toward
                <= self.projectile_dot_threshold
            ):

                continue

            total = (
                self.projectile_episode_total[
                    action
                ]
            )

            # Already reached the maximum punishment for this
            # danger episode.

            if (
                total
                <= self.projectile_episode_cap
            ):

                continue

            previous = (
                self.last_projectile_penalty[
                    action
                ]
            )

            if (
                previous is not None
                and (
                    now - previous
                )
                < self.projectile_cooldown
            ):

                continue

            reward = float(
                self.projectile_penalty
            )

            remaining = (
                self.projectile_episode_cap
                - total
            )

            # -------------------------------------------------
            # DO NOT PASS EPISODE CAP
            # -------------------------------------------------

            if (
                reward
                < remaining
            ):

                reward = (
                    remaining
                )

            if (
                reward
                >= 0
            ):

                continue

            self.last_projectile_penalty[
                action
            ] = now

            self.projectile_episode_total[
                action
            ] += reward

            events.append(
                {
                    "name":
                        "PROJECTILE_TOWARD",

                    "reward":
                        reward,

                    "action":
                        action,

                    "extra":
                        (
                            f"dist={distance:.1f} "
                            f"toward={toward:.2f}"
                        ),
                }
            )

        return events