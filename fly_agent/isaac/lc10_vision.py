import cv2
import numpy as np
from itertools import combinations


class IsaacLC10Vision:
    """
    Experimental 12x20 retinotopic adapter into LC10a.

    IMPORTANT ARCHITECTURE:
        external/source room grids keep their native current-room shape
        LC10 is FIXED at 12x20 in this experimental build
        the complete current room is compressed into 12x20
        MaleCNS receives the fixed 12x20 LC10 representation

    Examples:
        7x13  -> 12x20
        14x13 -> 12x20
        7x26  -> 12x20
        14x26 -> 12x20

    20 columns split exactly:

        LEFT  -> columns 0..9
        RIGHT -> columns 10..19

    Each hemisphere receives 12 x 10 = 120 spatial LC10 positions.

    LC10a has 135 LEFT and 140 RIGHT cells, so 15/20 cells remain
    for semantic identity. Because there are more semantic categories
    than spare cells, semantic tags use unique overlapping combinations
    of spare neurons instead of one exclusive pool per category.

    Python does not choose an action here. It only supplies sensory input.
    """

    GRID_ROWS = 12
    GRID_COLS = 20

    HALF_COLS = GRID_COLS // 2
    SIDE_COLS = HALF_COLS

    POSITIONS_PER_SIDE = (
        GRID_ROWS
        * SIDE_COLS
    )

    def __init__(
        self,
        brain,
        gain=1.5,
        max_drive=0.40,
        minimum_activity=0.07,
        top_k_spatial=6,
        color_gain=1.0,
        color_max_drive=0.20,
        color_minimum=0.08,
        color_top_k=3,
        enemy_visual_strength=0.85,
        map_visual_strength=0.22,
        self_visual_strength=0.55,
    ):

        self.brain = brain

        self.gain = float(
            gain
        )

        self.max_drive = float(
            max_drive
        )

        self.minimum_activity = float(
            minimum_activity
        )

        self.top_k_spatial = int(
            top_k_spatial
        )

        self.color_gain = float(
            color_gain
        )

        self.color_max_drive = float(
            color_max_drive
        )

        self.color_minimum = float(
            color_minimum
        )

        self.color_top_k = int(
            color_top_k
        )

        self.enemy_visual_strength = float(
            enemy_visual_strength
        )

        self.map_visual_strength = float(
            map_visual_strength
        )

        self.self_visual_strength = float(
            self_visual_strength
        )

        # =====================================================
        # LC10a CELLS
        # =====================================================

        self.left_cells = np.asarray(
            self.brain.brain.cells(
                ["LC10a"],
                side="L",
            ),
            dtype=np.int64,
        )

        self.right_cells = np.asarray(
            self.brain.brain.cells(
                ["LC10a"],
                side="R",
            ),
            dtype=np.int64,
        )

        print(
            "LC10a LEFT:",
            len(
                self.left_cells
            ),
        )

        print(
            "LC10a RIGHT:",
            len(
                self.right_cells
            ),
        )

        required = int(
            self.POSITIONS_PER_SIDE
        )

        if (
            len(
                self.left_cells
            )
            < required
        ):

            raise RuntimeError(
                "Not enough LEFT LC10a neurons "
                "for 12x20 spatial vision."
            )

        if (
            len(
                self.right_cells
            )
            < required
        ):

            raise RuntimeError(
                "Not enough RIGHT LC10a neurons "
                "for 12x20 spatial vision."
            )

        self.left_spatial = (
            self.left_cells[
                :required
            ]
        )

        self.right_spatial = (
            self.right_cells[
                :required
            ]
        )

        left_extra = (
            self.left_cells[
                required:
            ]
        )

        right_extra = (
            self.right_cells[
                required:
            ]
        )

        # =====================================================
        # CATEGORY TAG NEURONS
        # =====================================================
        #
        # Spatial LC10 cells say WHERE something is.
        # These extra LC10a groups say WHAT KIND of thing is present
        # on that hemisphere.
        #
        # This is sensory encoding, not an external policy.
        # MaleCNS still decides what to do with the signal.
        # =====================================================

        tag_names = (
            "red",
            "green",
            "blue",
            "door",
            "locked_door",
            "unexplored_door",
            "treasure_door",
            "shop_door",
            "boss_door",
            "devil_door",
            "angel_door",
            "projectile",
            "hazard",
            "rock",
            "coin",
            "bomb_pickup",
            "key",
            "heart",
            "item",
        )

        def make_tag_codes(
            extra_cells,
            names,
        ):
            """
            Give every semantic category a unique sparse combination of
            leftover LC10a neurons.

            12x20 uses 120 spatial neurons per hemisphere, leaving:
                LEFT  15
                RIGHT 20

            Exclusive pools would leave some categories with zero neurons.
            Overlapping 3-neuron codes preserve every category while keeping
            the 12x20 spatial map.
            """

            extra_cells = np.asarray(
                extra_cells,
                dtype=np.int64,
            )

            if len(extra_cells) == 0:
                return {
                    name: np.asarray(
                        [],
                        dtype=np.int64,
                    )
                    for name in names
                }

            code_size = min(
                3,
                len(extra_cells),
            )

            codes = list(
                combinations(
                    extra_cells.tolist(),
                    code_size,
                )
            )

            if len(codes) < len(names):
                # Only possible with an unexpectedly tiny LC10 population.
                codes = [
                    tuple(
                        extra_cells[
                            [
                                i
                                % len(extra_cells)
                            ]
                        ].tolist()
                    )
                    for i in range(
                        len(names)
                    )
                ]

            return {
                name: np.asarray(
                    codes[index],
                    dtype=np.int64,
                )
                for index, name
                in enumerate(names)
            }

        self.left_tags = make_tag_codes(
            left_extra,
            tag_names,
        )

        self.right_tags = make_tag_codes(
            right_extra,
            tag_names,
        )

        self.left_red = self.left_tags[
            "red"
        ]
        self.left_green = self.left_tags[
            "green"
        ]
        self.left_blue = self.left_tags[
            "blue"
        ]

        self.right_red = self.right_tags[
            "red"
        ]
        self.right_green = self.right_tags[
            "green"
        ]
        self.right_blue = self.right_tags[
            "blue"
        ]

        print(
            "LC10 semantic tag groups:"
        )

        for name in tag_names[3:]:
            print(
                f"  {name:16s} "
                f"L={len(self.left_tags[name])} "
                f"R={len(self.right_tags[name])}"
            )

        print()
        print(
            "Spatial LC10a mapping: 12 x 20"
        )

        print(
            " LEFT:",
            len(
                self.left_spatial
            ),
            "positions",
        )

        print(
            " RIGHT:",
            len(
                self.right_spatial
            ),
            "positions",
        )

        print(
            " Split: cols 0-9 | 10-19"
        )

        print(
            " External/source room grids: DYNAMIC"
        )

        print(
            " LC10 fixed retina:",
            f"{self.GRID_ROWS} x {self.GRID_COLS}",
        )

        print()
        print(
            "Sensory strengths:"
        )

        print(
            " enemy:",
            self.enemy_visual_strength,
        )

        print(
            " map:",
            self.map_visual_strength,
        )

        print(
            " self:",
            self.self_visual_strength,
        )

        # =====================================================
        # DASHBOARD / DEBUG STATE
        # =====================================================

        shape = (
            self.GRID_ROWS,
            self.GRID_COLS,
        )

        self.last_left_activity = 0.0
        self.last_right_activity = 0.0

        self.last_left_drive = 0.0
        self.last_right_drive = 0.0

        self.last_spatial_left = 0
        self.last_spatial_right = 0

        self.last_left_colors = {
            "red": 0.0,
            "green": 0.0,
            "blue": 0.0,
        }

        self.last_right_colors = {
            "red": 0.0,
            "green": 0.0,
            "blue": 0.0,
        }

        self.last_left_semantics = {}
        self.last_right_semantics = {}

        self.last_camera_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.last_map_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.last_enemy_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.last_self_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        self.last_combined_spatial = np.zeros(
            shape,
            dtype=np.float32,
        )

        # Exact 12x20 cells selected and injected on the latest step.
        self.last_injected_drive_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        # Exact neuron IDs/drives handed to MaleCNS on the latest step.
        # Used only for auditing/proof; it does not affect decisions.
        self.last_injections = []

    def reset_dynamic_state(self):
        """
        Immediately blank all cached visual/semantic LC10 state.

        This prevents the dashboard/debugger from showing the last room's
        doors during the short gap before fresh bridge data arrives.
        """

        shape = (
            self.GRID_ROWS,
            self.GRID_COLS,
        )

        for name, value in list(
            self.__dict__.items()
        ):
            if (
                isinstance(
                    value,
                    np.ndarray,
                )
                and value.shape == shape
            ):
                value.fill(
                    0.0
                )

        self.last_left_semantics = {}
        self.last_right_semantics = {}
        self.last_injections = []

        for name in (
            "last_left_activity",
            "last_right_activity",
            "last_left_drive",
            "last_right_drive",
        ):
            if hasattr(
                self,
                name,
            ):
                setattr(
                    self,
                    name,
                    0.0,
                )

        print(
            "LC10 cached sensory state HARD RESET."
        )

    # =========================================================
    # DEBUG MAPPING HELPER
    # =========================================================

    def expected_spatial_neurons(
        self,
        row,
        col,
    ):
        """
        Return the LC10 neuron(s) representing one 12x20 grid cell.
        With the even 20-column split, every cell belongs to exactly
        one hemisphere.
        """

        row = int(
            row
        )

        col = int(
            col
        )

        if not (
            0
            <= row
            < self.GRID_ROWS
            and 0
            <= col
            < self.GRID_COLS
        ):

            raise IndexError(
                (
                    row,
                    col,
                )
            )

        if (
            col
            < self.HALF_COLS
        ):

            index = (
                row
                * self.SIDE_COLS
                + col
            )

            return [
                int(
                    self.left_spatial[
                        index
                    ]
                )
            ]

        local_col = (
            col
            - self.HALF_COLS
        )

        index = (
            row
            * self.SIDE_COLS
            + local_col
        )

        return [
            int(
                self.right_spatial[
                    index
                ]
            )
        ]

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _top_mean(
        values,
        k,
    ):

        values = np.asarray(
            values,
            dtype=np.float32,
        ).ravel()

        if values.size == 0:
            return 0.0

        k = max(
            1,
            min(
                int(
                    k
                ),
                values.size,
            ),
        )

        strongest = np.partition(
            values,
            -k,
        )[
            -k:
        ]

        return float(
            np.mean(
                strongest
            )
        )

    def _to_brain_grid(
        self,
        values,
        name,
        allow_none=False,
        interpolation=cv2.INTER_AREA,
    ):
        """
        Convert any 2D external spatial source to the fixed 12x20
        MaleCNS retina.

        Source grids are allowed to be native room shapes such as
        7x13, 14x13, 7x26, 14x26, and narrow I-room variants.
        """

        if (
            values
            is None
        ):

            if allow_none:

                return np.zeros(
                    (
                        self.GRID_ROWS,
                        self.GRID_COLS,
                    ),
                    dtype=np.float32,
                )

            raise RuntimeError(
                (
                    "Missing required Isaac sensory grid: "
                    f"{name}"
                )
            )

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if (
            array.ndim
            != 2
        ):

            raise RuntimeError(
                (
                    f"Expected 2D {name} grid, "
                    f"got shape {array.shape}"
                )
            )

        expected = (
            self.GRID_ROWS,
            self.GRID_COLS,
        )

        if (
            array.shape
            == expected
        ):

            return array

        resized = cv2.resize(
            array,
            (
                self.GRID_COLS,
                self.GRID_ROWS,
            ),
            interpolation=interpolation,
        )

        return np.asarray(
            resized,
            dtype=np.float32,
        )

    def _to_brain_grid_categorical(
        self,
        values,
        name,
        allow_none=False,
    ):
        """
        Resize discrete spatial sources (map/enemy/self) to 12x20
        WITHOUT allowing small occupied cells to disappear.

        cv2.INTER_NEAREST alone can drop a single source cell when a
        wide 26-column room is compressed to 20 columns. We therefore:

        1. create a nearest-neighbor baseline, then
        2. forward-map every source cell into the corresponding LC10 bin
           using MAX aggregation.

        This keeps LC10 fixed at 12x20 while preserving walls, enemies,
        and self markers from large rooms.
        """

        if values is None:

            if allow_none:

                return np.zeros(
                    (
                        self.GRID_ROWS,
                        self.GRID_COLS,
                    ),
                    dtype=np.float32,
                )

            raise RuntimeError(
                (
                    "Missing required Isaac sensory grid: "
                    f"{name}"
                )
            )

        source = np.asarray(
            values,
            dtype=np.float32,
        )

        if source.ndim != 2:

            raise RuntimeError(
                (
                    f"Expected 2D {name} grid, "
                    f"got shape {source.shape}"
                )
            )

        if source.shape == (
            self.GRID_ROWS,
            self.GRID_COLS,
        ):

            return source

        result = cv2.resize(
            source,
            (
                self.GRID_COLS,
                self.GRID_ROWS,
            ),
            interpolation=cv2.INTER_NEAREST,
        ).astype(
            np.float32,
            copy=False,
        )

        src_rows, src_cols = (
            source.shape
        )

        # Peak-preserving forward aggregation.
        for src_row in range(
            src_rows
        ):

            dst_row = min(
                self.GRID_ROWS - 1,
                max(
                    0,
                    int(
                        (
                            src_row + 0.5
                        )
                        * self.GRID_ROWS
                        / src_rows
                    ),
                ),
            )

            for src_col in range(
                src_cols
            ):

                value = float(
                    source[
                        src_row,
                        src_col
                    ]
                )

                if value <= 0.0:
                    continue

                dst_col = min(
                    self.GRID_COLS - 1,
                    max(
                        0,
                        int(
                            (
                                src_col + 0.5
                            )
                            * self.GRID_COLS
                            / src_cols
                        ),
                    ),
                )

                result[
                    dst_row,
                    dst_col
                ] = max(
                    result[
                        dst_row,
                        dst_col
                    ],
                    value,
                )

        return result

    def _map_source(
        self,
        features,
    ):
        """
        Prefer the sparse near-collision map when it is available.
        Otherwise use the full room collision grid.
        """

        navigation = features.get(
            "navigation_grid"
        )

        if isinstance(
            navigation,
            np.ndarray,
        ):
            if navigation.ndim == 2:
                return navigation

        near = features.get(
            "near_collision_grid"
        )

        if isinstance(
            near,
            np.ndarray,
        ):

            if (
                near.ndim == 2
                and np.any(
                    near > 0.0
                )
            ):

                return near

        return features.get(
            "collision_grid"
        )

    # =========================================================
    # SPATIAL INJECTION
    # =========================================================

    def _spatial_injections(
        self,
        region,
        cells,
    ):

        flat = np.asarray(
            region,
            dtype=np.float32,
        ).ravel()

        candidates = np.flatnonzero(
            flat
            >= self.minimum_activity
        )

        if (
            len(
                candidates
            )
            == 0
        ):

            return (
                [],
                0,
                0.0,
                np.empty(
                    0,
                    dtype=np.int64,
                ),
                np.empty(
                    0,
                    dtype=np.float32,
                ),
            )

        if (
            len(
                candidates
            )
            > self.top_k_spatial
        ):

            candidate_values = (
                flat[
                    candidates
                ]
            )

            order = np.argsort(
                candidate_values
            )[
                -self.top_k_spatial:
            ]

            candidates = (
                candidates[
                    order
                ]
            )

        inject = []
        actual_indices = []
        actual_drives = []

        max_drive = 0.0

        for index in candidates:

            drive = float(
                np.clip(
                    flat[
                        index
                    ]
                    * self.gain,
                    0.0,
                    self.max_drive,
                )
            )

            if (
                drive
                <= 0.0
            ):

                continue

            neuron = np.asarray(
                [
                    cells[
                        int(
                            index
                        )
                    ]
                ],
                dtype=np.int64,
            )

            inject.append(
                (
                    neuron,
                    drive,
                )
            )

            actual_indices.append(
                int(
                    index
                )
            )

            actual_drives.append(
                drive
            )

            max_drive = max(
                max_drive,
                drive,
            )

        return (
            inject,
            len(
                inject
            ),
            max_drive,
            np.asarray(
                actual_indices,
                dtype=np.int64,
            ),
            np.asarray(
                actual_drives,
                dtype=np.float32,
            ),
        )

    # =========================================================
    # COLOUR
    # =========================================================

    def _add_color(
        self,
        inject,
        values,
        cells,
    ):

        if (
            len(
                cells
            )
            == 0
        ):

            return 0.0

        activity = self._top_mean(
            values,
            self.color_top_k,
        )

        if (
            activity
            < self.color_minimum
        ):

            return activity

        drive = float(
            np.clip(
                activity
                * self.color_gain,
                0.0,
                self.color_max_drive,
            )
        )

        inject.append(
            (
                cells,
                drive,
            )
        )

        return activity

    def _add_semantic_tag(
        self,
        inject,
        values,
        cells,
        minimum=0.05,
        gain=0.55,
        max_drive=0.28,
    ):
        """
        Inject a category identity tag for one hemisphere.

        Spatial LC10 tells MaleCNS WHERE.
        Tag neurons tell MaleCNS WHAT CATEGORY is present.
        """

        if len(cells) == 0:
            return 0.0

        activity = self._top_mean(
            values,
            3,
        )

        if activity < float(minimum):
            return activity

        drive = float(
            np.clip(
                activity
                * float(gain),
                0.0,
                float(max_drive),
            )
        )

        inject.append(
            (
                cells,
                drive,
            )
        )

        return activity

    # =========================================================
    # BUILD 12 x 20 SPATIAL FIELD
    # =========================================================

    def _make_spatial_field(
        self,
        features,
    ):

        base_source = features.get(
            "salience_grid"
        )

        if (
            base_source
            is None
        ):

            base_source = features.get(
                "grid"
            )

        camera = (
            self._to_brain_grid(
                base_source,
                "salience",
                allow_none=False,
                interpolation=(
                    cv2.INTER_AREA
                ),
            )
            .copy()
        )

        exact_enemy = features.get(
            "enemy_lc10_hitbox_grid"
        )

        if (
            isinstance(
                exact_enemy,
                np.ndarray,
            )
            and exact_enemy.shape
            == (
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        ):
            enemy = exact_enemy.astype(
                np.float32,
                copy=True,
            )

        else:
            enemy = (
                self._to_brain_grid_categorical(
                    features.get(
                        "enemy_grid"
                    ),
                    "enemy",
                    allow_none=True,
                )
                .copy()
            )

        exact_self = features.get(
            "self_lc10_hitbox_grid"
        )

        if (
            isinstance(
                exact_self,
                np.ndarray,
            )
            and exact_self.shape
            == (
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        ):
            self_grid = exact_self.astype(
                np.float32,
                copy=True,
            )

        else:
            self_grid = (
                self._to_brain_grid_categorical(
                    features.get(
                        "self_grid"
                    ),
                    "self",
                    allow_none=True,
                )
                .copy()
            )

        map_grid = (
            self._to_brain_grid_categorical(
                self._map_source(
                    features
                ),
                "map",
                allow_none=True,
            )
            .copy()
        )

        door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "door_grid_exact"
                ),
                "door",
                allow_none=True,
            )
            .copy()
        )

        locked_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "locked_door_grid_exact"
                ),
                "locked_door",
                allow_none=True,
            )
            .copy()
        )

        unexplored_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "unexplored_door_grid_exact"
                ),
                "unexplored_door",
                allow_none=True,
            )
            .copy()
        )

        treasure_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "treasure_door_grid"
                ),
                "treasure_door",
                allow_none=True,
            )
            .copy()
        )

        shop_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "shop_door_grid"
                ),
                "shop_door",
                allow_none=True,
            )
            .copy()
        )

        boss_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "boss_door_grid"
                ),
                "boss_door",
                allow_none=True,
            )
            .copy()
        )

        devil_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "devil_door_grid"
                ),
                "devil_door",
                allow_none=True,
            )
            .copy()
        )

        angel_door_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "angel_door_grid"
                ),
                "angel_door",
                allow_none=True,
            )
            .copy()
        )

        hazard_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "hazard_grid"
                ),
                "hazard",
                allow_none=True,
            )
            .copy()
        )

        rock_grid = (
            self._to_brain_grid_categorical(
                features.get(
                    "rock_grid"
                ),
                "rock",
                allow_none=True,
            )
            .copy()
        )

        def fixed_grid(
            key,
        ):
            value = features.get(
                key
            )

            if (
                isinstance(
                    value,
                    np.ndarray,
                )
                and value.shape
                == (
                    self.GRID_ROWS,
                    self.GRID_COLS,
                )
            ):
                return value.astype(
                    np.float32,
                    copy=True,
                )

            return np.zeros(
                (
                    self.GRID_ROWS,
                    self.GRID_COLS,
                ),
                dtype=np.float32,
            )

        coin_grid = fixed_grid(
            "coin_lc10_grid"
        )

        bomb_pickup_grid = fixed_grid(
            "bomb_pickup_lc10_grid"
        )

        key_grid = fixed_grid(
            "key_lc10_grid"
        )

        heart_grid = fixed_grid(
            "heart_lc10_grid"
        )

        item_grid = fixed_grid(
            "item_lc10_grid"
        )

        projectile_grid = fixed_grid(
            "projectile_lc10_grid"
        )

        projectile_future_grid = fixed_grid(
            "projectile_future_lc10_grid"
        )

        camera = np.clip(
            camera,
            0.0,
            1.0,
        )

        enemy = np.clip(
            enemy,
            0.0,
            1.0,
        )

        self_grid = np.clip(
            self_grid,
            0.0,
            1.0,
        )

        map_grid = np.clip(
            map_grid,
            0.0,
            1.0,
        )

        door_grid = np.clip(
            door_grid,
            0.0,
            1.0,
        )

        locked_door_grid = np.clip(
            locked_door_grid,
            0.0,
            1.0,
        )

        unexplored_door_grid = np.clip(
            unexplored_door_grid,
            0.0,
            1.0,
        )

        treasure_door_grid = np.clip(
            treasure_door_grid,
            0.0,
            1.0,
        )

        shop_door_grid = np.clip(
            shop_door_grid,
            0.0,
            1.0,
        )

        boss_door_grid = np.clip(
            boss_door_grid,
            0.0,
            1.0,
        )

        devil_door_grid = np.clip(
            devil_door_grid,
            0.0,
            1.0,
        )

        angel_door_grid = np.clip(
            angel_door_grid,
            0.0,
            1.0,
        )

        hazard_grid = np.clip(
            hazard_grid,
            0.0,
            1.0,
        )

        rock_grid = np.clip(
            rock_grid,
            0.0,
            1.0,
        )

        coin_grid = np.clip(
            coin_grid,
            0.0,
            1.0,
        )

        bomb_pickup_grid = np.clip(
            bomb_pickup_grid,
            0.0,
            1.0,
        )

        key_grid = np.clip(
            key_grid,
            0.0,
            1.0,
        )

        heart_grid = np.clip(
            heart_grid,
            0.0,
            1.0,
        )

        item_grid = np.clip(
            item_grid,
            0.0,
            1.0,
        )

        projectile_grid = np.clip(
            projectile_grid,
            0.0,
            1.0,
        )

        projectile_future_grid = np.clip(
            projectile_future_grid,
            0.0,
            1.0,
        )

        enemy_activity = np.clip(
            enemy
            * self.enemy_visual_strength,
            0.0,
            1.0,
        )

        map_activity = np.clip(
            map_grid
            * self.map_visual_strength,
            0.0,
            1.0,
        )

        self_activity = np.clip(
            self_grid
            * self.self_visual_strength,
            0.0,
            1.0,
        )

        # Semantic sensory strengths.
        # This does not choose actions; it only makes different world
        # features physically distinguishable to MaleCNS.
        rock_activity = np.clip(
            rock_grid * 0.32,
            0.0,
            1.0,
        )

        hazard_activity = np.clip(
            hazard_grid * 0.78,
            0.0,
            1.0,
        )

        door_activity = np.clip(
            door_grid * 0.72,
            0.0,
            1.0,
        )

        coin_activity = np.clip(
            coin_grid * 0.58,
            0.0,
            1.0,
        )

        bomb_pickup_activity = np.clip(
            bomb_pickup_grid * 0.62,
            0.0,
            1.0,
        )

        key_activity = np.clip(
            key_grid * 0.66,
            0.0,
            1.0,
        )

        heart_activity = np.clip(
            heart_grid * 0.60,
            0.0,
            1.0,
        )

        item_activity = np.clip(
            item_grid * 0.92,
            0.0,
            1.0,
        )

        # Hostile projectile now gets exact position plus a faint future
        # position. This gives MaleCNS motion information without Python
        # choosing a dodge action.
        projectile_activity = np.clip(
            projectile_grid * 0.95,
            0.0,
            1.0,
        )

        projectile_future_activity = np.clip(
            projectile_future_grid * 0.50,
            0.0,
            1.0,
        )

        spatial = np.maximum.reduce(
            [
                camera,
                enemy_activity,
                map_activity,
                self_activity,
                rock_activity,
                hazard_activity,
                door_activity,
                coin_activity,
                bomb_pickup_activity,
                key_activity,
                heart_activity,
                item_activity,
                projectile_activity,
                projectile_future_activity,
            ]
        ).astype(
            np.float32,
            copy=False,
        )

        self.last_camera_grid = (
            camera.copy()
        )

        self.last_map_grid = (
            map_grid.copy()
        )

        self.last_enemy_grid = (
            enemy.copy()
        )

        self.last_self_grid = (
            self_grid.copy()
        )

        self.last_door_grid = (
            door_grid.copy()
        )

        self.last_locked_door_grid = (
            locked_door_grid.copy()
        )

        self.last_unexplored_door_grid = (
            unexplored_door_grid.copy()
        )

        self.last_treasure_door_grid = (
            treasure_door_grid.copy()
        )

        self.last_shop_door_grid = (
            shop_door_grid.copy()
        )

        self.last_boss_door_grid = (
            boss_door_grid.copy()
        )

        self.last_devil_door_grid = (
            devil_door_grid.copy()
        )

        self.last_angel_door_grid = (
            angel_door_grid.copy()
        )

        self.last_hazard_grid = (
            hazard_grid.copy()
        )

        self.last_rock_grid = (
            rock_grid.copy()
        )

        self.last_coin_grid = (
            coin_grid.copy()
        )

        self.last_bomb_pickup_grid = (
            bomb_pickup_grid.copy()
        )

        self.last_key_grid = (
            key_grid.copy()
        )

        self.last_heart_grid = (
            heart_grid.copy()
        )

        self.last_item_grid = (
            item_grid.copy()
        )

        self.last_projectile_grid = (
            projectile_grid.copy()
        )

        self.last_projectile_future_grid = (
            projectile_future_grid.copy()
        )

        self.last_combined_spatial = (
            spatial.copy()
        )

        return spatial

    # =========================================================
    # MAKE INJECTIONS
    # =========================================================

    def make_injections(
        self,
        features,
    ):

        spatial = (
            self._make_spatial_field(
                features
            )
        )

        # Exact even split:
        #
        # LEFT  -> cols 0..6
        # RIGHT -> cols 7..13

        left_field = spatial[
            :,
            :self.HALF_COLS,
        ]

        right_field = spatial[
            :,
            self.HALF_COLS:,
        ]

        inject = []

        (
            left_inject,
            left_count,
            left_max_drive,
            left_indices,
            left_drives,
        ) = self._spatial_injections(
            left_field,
            self.left_spatial,
        )

        (
            right_inject,
            right_count,
            right_max_drive,
            right_indices,
            right_drives,
        ) = self._spatial_injections(
            right_field,
            self.right_spatial,
        )

        inject.extend(
            left_inject
        )

        inject.extend(
            right_inject
        )

        # =====================================================
        # EXACT 12x20 POSITIONS ACTUALLY INJECTED
        # =====================================================

        drive_grid = np.zeros(
            (
                self.GRID_ROWS,
                self.GRID_COLS,
            ),
            dtype=np.float32,
        )

        for (
            index,
            drive,
        ) in zip(
            left_indices,
            left_drives,
        ):

            row = (
                int(
                    index
                )
                // self.SIDE_COLS
            )

            col = (
                int(
                    index
                )
                % self.SIDE_COLS
            )

            drive_grid[
                row,
                col
            ] = max(
                drive_grid[
                    row,
                    col
                ],
                float(
                    drive
                ),
            )

        for (
            index,
            drive,
        ) in zip(
            right_indices,
            right_drives,
        ):

            row = (
                int(
                    index
                )
                // self.SIDE_COLS
            )

            local_col = (
                int(
                    index
                )
                % self.SIDE_COLS
            )

            col = (
                local_col
                + self.HALF_COLS
            )

            drive_grid[
                row,
                col
            ] = max(
                drive_grid[
                    row,
                    col
                ],
                float(
                    drive
                ),
            )

        self.last_injected_drive_grid = (
            drive_grid
        )

        self.last_spatial_left = (
            left_count
        )

        self.last_spatial_right = (
            right_count
        )

        self.last_left_activity = (
            self._top_mean(
                left_field,
                self.top_k_spatial,
            )
        )

        self.last_right_activity = (
            self._top_mean(
                right_field,
                self.top_k_spatial,
            )
        )

        self.last_left_drive = (
            left_max_drive
        )

        self.last_right_drive = (
            right_max_drive
        )

        # =====================================================
        # COLOUR SIGNALS
        # =====================================================

        red = self._to_brain_grid(
            features.get(
                "red_grid"
            ),
            "red",
            allow_none=True,
        )

        green = self._to_brain_grid(
            features.get(
                "green_grid"
            ),
            "green",
            allow_none=True,
        )

        blue = self._to_brain_grid(
            features.get(
                "blue_grid"
            ),
            "blue",
            allow_none=True,
        )

        self.last_left_colors[
            "red"
        ] = self._add_color(
            inject,
            red[
                :,
                :self.HALF_COLS
            ],
            self.left_red,
        )

        self.last_left_colors[
            "green"
        ] = self._add_color(
            inject,
            green[
                :,
                :self.HALF_COLS
            ],
            self.left_green,
        )

        self.last_left_colors[
            "blue"
        ] = self._add_color(
            inject,
            blue[
                :,
                :self.HALF_COLS
            ],
            self.left_blue,
        )

        self.last_right_colors[
            "red"
        ] = self._add_color(
            inject,
            red[
                :,
                self.HALF_COLS:
            ],
            self.right_red,
        )

        self.last_right_colors[
            "green"
        ] = self._add_color(
            inject,
            green[
                :,
                self.HALF_COLS:
            ],
            self.right_green,
        )

        self.last_right_colors[
            "blue"
        ] = self._add_color(
            inject,
            blue[
                :,
                self.HALF_COLS:
            ],
            self.right_blue,
        )

        # =====================================================
        # CATEGORY IDENTITY TAGS
        # =====================================================
        #
        # The position still comes from the 12x20 spatial cells.
        # These tag groups stop "door", "coin", "rock", etc. from
        # being just different strengths of the same signal.
        # =====================================================

        semantic_layers = {
            "door":
                getattr(
                    self,
                    "last_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "locked_door":
                getattr(
                    self,
                    "last_locked_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "unexplored_door":
                getattr(
                    self,
                    "last_unexplored_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "treasure_door":
                getattr(
                    self,
                    "last_treasure_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "shop_door":
                getattr(
                    self,
                    "last_shop_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "boss_door":
                getattr(
                    self,
                    "last_boss_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "devil_door":
                getattr(
                    self,
                    "last_devil_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "angel_door":
                getattr(
                    self,
                    "last_angel_door_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "projectile":
                np.maximum(
                    getattr(
                        self,
                        "last_projectile_grid",
                        np.zeros(
                            (
                                self.GRID_ROWS,
                                self.GRID_COLS,
                            ),
                            dtype=np.float32,
                        ),
                    ),
                    getattr(
                        self,
                        "last_projectile_future_grid",
                        np.zeros(
                            (
                                self.GRID_ROWS,
                                self.GRID_COLS,
                            ),
                            dtype=np.float32,
                        ),
                    ),
                ),
            "hazard":
                getattr(
                    self,
                    "last_hazard_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "rock":
                getattr(
                    self,
                    "last_rock_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "coin":
                getattr(
                    self,
                    "last_coin_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "bomb_pickup":
                getattr(
                    self,
                    "last_bomb_pickup_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "key":
                getattr(
                    self,
                    "last_key_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "heart":
                getattr(
                    self,
                    "last_heart_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
            "item":
                getattr(
                    self,
                    "last_item_grid",
                    np.zeros(
                        (
                            self.GRID_ROWS,
                            self.GRID_COLS,
                        ),
                        dtype=np.float32,
                    ),
                ),
        }

        self.last_left_semantics = {}
        self.last_right_semantics = {}

        for (
            name,
            layer,
        ) in semantic_layers.items():

            self.last_left_semantics[
                name
            ] = self._add_semantic_tag(
                inject,
                layer[
                    :,
                    :self.HALF_COLS
                ],
                self.left_tags[
                    name
                ],
            )

            self.last_right_semantics[
                name
            ] = self._add_semantic_tag(
                inject,
                layer[
                    :,
                    self.HALF_COLS:
                ],
                self.right_tags[
                    name
                ],
            )

        self.last_injections = [
            (
                np.asarray(
                    neurons,
                    dtype=np.int64,
                ).copy(),
                float(
                    drive
                ),
            )
            for neurons, drive
            in inject
        ]

        return inject

    # =========================================================
    # STEP MALECNS
    # =========================================================

    def step(
        self,
        features,
        extra_inject=None,
    ):

        inject = self.make_injections(
            features
        )

        if extra_inject:

            inject.extend(
                extra_inject
            )

        if inject:

            return self.brain.brain.step(
                inject=inject
            )

        return self.brain.brain.step()
