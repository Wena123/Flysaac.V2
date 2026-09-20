import numpy as np


class SensoryGridDebugger:
    """
    Full sensory audit for the Isaac -> LC10 -> MaleCNS pipeline.

    This debugger answers, as directly as possible:

        Is the object present in Python sensory data?
        Does it exist on the LC10 grid?
        Were the matching LC10 neurons ACTUALLY injected this step?
        Did any matching LC10 neurons actually FIRE in MaleCNS?
        If the object has a semantic WHAT-tag, was that tag injected/fired?

    It does not control the agent and does not change rewards or weights.
    """

    CATEGORY_SPECS = (
        # label, LC10 attribute, semantic tag (or None)
        ("MAP",              "last_map_grid",                    None),
        ("SELF",             "last_self_grid",                   None),
        ("ENEMY",            "last_enemy_grid",                  None),

        ("ENEMY_TEAR",       "last_projectile_grid",             "projectile"),
        ("TEAR_FUTURE",      "last_projectile_future_grid",      "projectile"),

        ("DOOR",             "last_door_grid",                   "door"),
        ("LOCKED_DOOR",      "last_locked_door_grid",            "locked_door"),
        ("UNEXPLORED_DOOR",  "last_unexplored_door_grid",        "unexplored_door"),
        ("TREASURE_DOOR",    "last_treasure_door_grid",          "treasure_door"),
        ("SHOP_DOOR",        "last_shop_door_grid",              "shop_door"),
        ("BOSS_DOOR",        "last_boss_door_grid",              "boss_door"),
        ("DEVIL_DOOR",       "last_devil_door_grid",             "devil_door"),
        ("ANGEL_DOOR",       "last_angel_door_grid",             "angel_door"),

        ("ITEM",             "last_item_grid",                   "item"),
        ("COIN",             "last_coin_grid",                   "coin"),
        ("KEY",              "last_key_grid",                    "key"),
        ("BOMB_PICKUP",      "last_bomb_pickup_grid",            "bomb_pickup"),
        ("HEART",            "last_heart_grid",                  "heart"),

        ("HAZARD",           "last_hazard_grid",                 "hazard"),
        ("ROCK",             "last_rock_grid",                   "rock"),
    )

    def __init__(
        self,
        print_every_steps=50,
    ):
        self.print_every_steps = max(
            1,
            int(
                print_every_steps
            ),
        )

        self.last_report = {}

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _set_from_injections(
        injections,
    ):
        result = set()

        for neurons, _drive in (
            injections or []
        ):
            for neuron in np.asarray(
                neurons
            ).ravel():
                result.add(
                    int(
                        neuron
                    )
                )

        return result

    @staticmethod
    def _set_from_fired(
        fired,
    ):
        if fired is None:
            return set()

        return {
            int(
                neuron
            )
            for neuron
            in np.asarray(
                fired
            ).ravel()
        }

    @staticmethod
    def _active_cells(
        grid,
        threshold=0.05,
    ):
        if not isinstance(
            grid,
            np.ndarray,
        ):
            return []

        coords = np.argwhere(
            np.asarray(
                grid,
                dtype=np.float32,
            )
            > float(
                threshold
            )
        )

        return [
            (
                int(row),
                int(col),
            )
            for row, col
            in coords
        ]

    @staticmethod
    def _grid_peak(
        grid,
    ):
        if not isinstance(
            grid,
            np.ndarray,
        ):
            return 0.0

        if grid.size == 0:
            return 0.0

        return float(
            np.max(
                grid
            )
        )

    @staticmethod
    def _format_bool(
        value,
    ):
        return (
            "YES"
            if value
            else "no"
        )

    @staticmethod
    def _tag_cells(
        lc10,
        tag_name,
    ):
        if tag_name is None:
            return set()

        cells = set()

        for mapping_name in (
            "left_tags",
            "right_tags",
        ):
            mapping = getattr(
                lc10,
                mapping_name,
                {},
            )

            values = mapping.get(
                tag_name,
                [],
            )

            for neuron in np.asarray(
                values
            ).ravel():
                cells.add(
                    int(
                        neuron
                    )
                )

        return cells

    @staticmethod
    def _semantic_activity(
        lc10,
        tag_name,
    ):
        if tag_name is None:
            return 0.0

        left = float(
            getattr(
                lc10,
                "last_left_semantics",
                {},
            ).get(
                tag_name,
                0.0,
            )
        )

        right = float(
            getattr(
                lc10,
                "last_right_semantics",
                {},
            ).get(
                tag_name,
                0.0,
            )
        )

        return max(
            left,
            right,
        )

    @staticmethod
    def _matching_spatial_neurons(
        lc10,
        active_cells,
    ):
        ids = set()

        for row, col in active_cells:
            try:
                expected = (
                    lc10.expected_spatial_neurons(
                        row,
                        col,
                    )
                )
            except Exception:
                continue

            if not isinstance(
                expected,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            ):
                expected = [
                    expected
                ]

            for neuron in np.asarray(
                expected
            ).ravel():
                ids.add(
                    int(
                        neuron
                    )
                )

        return ids

    # ========================================================
    # ADAPTER SELF-TEST
    # ========================================================

    def run_adapter_self_test(
        self,
        lc10,
    ):
        rows = int(
            getattr(
                lc10,
                "GRID_ROWS",
                -1,
            )
        )

        cols = int(
            getattr(
                lc10,
                "GRID_COLS",
                -1,
            )
        )

        print()
        print(
            "=============================================="
        )
        print(
            " LC10 SENSORY AUDIT SELF-TEST"
        )
        print(
            "=============================================="
        )
        print(
            f"LC10 retina: {rows}x{cols}"
        )
        print(
            "Expected current build: 12x20"
        )

        ok = (
            rows == 12
            and cols == 20
        )

        print(
            "RESULT:",
            (
                "PASS"
                if ok
                else "FAIL"
            ),
        )
        print(
            "=============================================="
        )
        print()

        return ok

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ):
        self.last_report = {}

        print(
            "Sensory debugger reset for new Isaac run."
        )

    # ========================================================
    # RUNTIME AUDIT
    # ========================================================

    def update(
        self,
        features,
        lc10,
        step,
        fired=None,
    ):
        injected_ids = (
            self._set_from_injections(
                getattr(
                    lc10,
                    "last_injections",
                    [],
                )
            )
        )

        fired_ids = (
            self._set_from_fired(
                fired
            )
        )

        report = {}

        for (
            label,
            grid_attr,
            tag_name,
        ) in self.CATEGORY_SPECS:

            grid = getattr(
                lc10,
                grid_attr,
                None,
            )

            active_cells = (
                self._active_cells(
                    grid
                )
            )

            spatial_ids = (
                self._matching_spatial_neurons(
                    lc10,
                    active_cells,
                )
            )

            spatial_injected = (
                spatial_ids
                & injected_ids
            )

            spatial_fired = (
                spatial_ids
                & fired_ids
            )

            tag_cells = (
                self._tag_cells(
                    lc10,
                    tag_name,
                )
            )

            tag_activity = (
                self._semantic_activity(
                    lc10,
                    tag_name,
                )
            )

            tag_injected = (
                tag_cells
                & injected_ids
            )

            tag_fired = (
                tag_cells
                & fired_ids
            )

            present = (
                len(
                    active_cells
                )
                > 0
            )

            if not present:
                status = (
                    "ABSENT"
                )

            elif len(
                spatial_injected
            ) == 0:
                status = (
                    "NOT_INJECTED"
                )

            elif (
                tag_name is not None
                and len(
                    tag_injected
                ) == 0
            ):
                status = (
                    "SPATIAL_ONLY"
                )

            elif (
                len(
                    spatial_fired
                ) > 0
                or len(
                    tag_fired
                ) > 0
            ):
                status = (
                    "SEEN+FIRED"
                )

            else:
                status = (
                    "INJECTED"
                )

            report[
                label
            ] = {
                "present":
                    present,

                "cells":
                    len(
                        active_cells
                    ),

                "peak":
                    self._grid_peak(
                        grid
                    ),

                "spatial_neurons":
                    len(
                        spatial_ids
                    ),

                "spatial_injected":
                    len(
                        spatial_injected
                    ),

                "spatial_fired":
                    len(
                        spatial_fired
                    ),

                "tag":
                    tag_name,

                "tag_activity":
                    tag_activity,

                "tag_neurons":
                    len(
                        tag_cells
                    ),

                "tag_injected":
                    len(
                        tag_injected
                    ),

                "tag_fired":
                    len(
                        tag_fired
                    ),

                "status":
                    status,
            }

        self.last_report = report

        # Make it available to the dashboard or any future UI.
        features[
            "full_sensory_audit"
        ] = {
            key:
                dict(
                    value
                )
            for key, value
            in report.items()
        }

        # Also expose all LC10-facing grids for dashboard inspection.
        for (
            feature_name,
            attr_name,
        ) in (
            ("lc10_combined_grid", "last_combined_spatial"),
            ("lc10_camera_grid", "last_camera_grid"),
            ("lc10_map_grid", "last_map_grid"),
            ("lc10_enemy_grid", "last_enemy_grid"),
            ("lc10_self_grid", "last_self_grid"),
            ("lc10_door_grid", "last_door_grid"),
            ("lc10_locked_door_grid", "last_locked_door_grid"),
            ("lc10_unexplored_door_grid", "last_unexplored_door_grid"),
            ("lc10_treasure_door_grid", "last_treasure_door_grid"),
            ("lc10_shop_door_grid", "last_shop_door_grid"),
            ("lc10_boss_door_grid", "last_boss_door_grid"),
            ("lc10_devil_door_grid", "last_devil_door_grid"),
            ("lc10_angel_door_grid", "last_angel_door_grid"),
            ("lc10_item_grid", "last_item_grid"),
            ("lc10_coin_grid", "last_coin_grid"),
            ("lc10_key_grid", "last_key_grid"),
            ("lc10_bomb_pickup_grid", "last_bomb_pickup_grid"),
            ("lc10_heart_grid", "last_heart_grid"),
            ("lc10_projectile_grid", "last_projectile_grid"),
            ("lc10_projectile_future_grid", "last_projectile_future_grid"),
            ("lc10_hazard_grid", "last_hazard_grid"),
            ("lc10_rock_grid", "last_rock_grid"),
            ("lc10_injected_grid", "last_injected_drive_grid"),
        ):
            value = getattr(
                lc10,
                attr_name,
                None,
            )

            if isinstance(
                value,
                np.ndarray,
            ):
                features[
                    feature_name
                ] = value.copy()

        if (
            int(
                step
            )
            % self.print_every_steps
            != 0
        ):
            return

        print()
        print(
            "================ FULL SENSORY AUDIT ================"
        )
        print(
            "STATUS: ABSENT / NOT_INJECTED / SPATIAL_ONLY / "
            "INJECTED / SEEN+FIRED"
        )
        print(
            "-----------------------------------------------------"
        )
        print(
            "THING             CELLS  SP.INJ  SP.FIRE  TAG.INJ "
            "TAG.FIRE  STATUS"
        )
        print(
            "-----------------------------------------------------"
        )

        for (
            label,
            _grid_attr,
            _tag_name,
        ) in self.CATEGORY_SPECS:

            info = report[
                label
            ]

            print(
                f"{label:17s} "
                f"{info['cells']:5d} "
                f"{info['spatial_injected']:7d} "
                f"{info['spatial_fired']:8d} "
                f"{info['tag_injected']:7d} "
                f"{info['tag_fired']:8d}  "
                f"{info['status']}"
            )

        print(
            "-----------------------------------------------------"
        )

        warnings = [
            label
            for label, info
            in report.items()
            if (
                info["present"]
                and info["status"]
                in {
                    "NOT_INJECTED",
                    "SPATIAL_ONLY",
                }
            )
        ]

        if warnings:
            print(
                "WARNING:",
                ", ".join(
                    warnings
                ),
                "is present but is not fully reaching MaleCNS.",
            )
        else:
            print(
                "No present sensory category is currently blocked."
            )

        print(
            "====================================================="
        )
        print()
