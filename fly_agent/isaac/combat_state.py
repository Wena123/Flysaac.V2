from pathlib import Path

import numpy as np


class IsaacCombatState:
    """
    Combat state with two coordinate paths:

    1. OLD/FALLBACK:
       room-normalized x/y/rx/ry

    2. ACCURATE V4:
       gx/gy/grx/gry in Isaac RAW GRID coordinates

    The V4 grid coordinates are aligned to the same Isaac grid used by
    room_state.py, so enemy/self centers no longer drift relative to walls.

    Source grids remain dynamic:
        1x1      ->  7x13
        1x2      -> 14x13
        2x1      ->  7x26
        2x2 / L  -> 14x26

    LC10 remains separately fixed at 12x20.
    """

    SUPERSAMPLE = 16

    FALLBACK_ENEMY_RX = 0.018
    FALLBACK_ENEMY_RY = 0.030

    FALLBACK_PLAYER_RX = 0.016
    FALLBACK_PLAYER_RY = 0.027

    SHAPE_DIMS = {
        1: (7, 13),
        2: (3, 13),
        3: (7, 5),
        4: (14, 13),
        5: (14, 5),
        6: (7, 26),
        7: (3, 26),
        8: (14, 26),
        9: (14, 26),
        10: (14, 26),
        11: (14, 26),
        12: (14, 26),
    }

    def __init__(
        self,
        log_path=None,
    ):
        if log_path is None:
            log_path = (
                Path.home()
                / "Documents"
                / "My Games"
                / "Binding of Isaac Repentance+"
                / "log.txt"
            )

        self.log_path = Path(
            log_path
        )

        self.file = None
        self.position = 0

        self.enemies_alive = 0
        self.visible_enemies = 0

        # Backwards-compatible normalized player hitbox.
        self.player_x = 0.5
        self.player_y = 0.5
        self.player_rx = (
            self.FALLBACK_PLAYER_RX
        )
        self.player_ry = (
            self.FALLBACK_PLAYER_RY
        )

        # Accurate raw-grid coordinates from combat bridge V4.
        self.player_gx = None
        self.player_gy = None
        self.player_grx = None
        self.player_gry = None

        # Exact screen-space coordinates, normalized 0..1.
        # Dashboard can draw these directly over the captured frame.
        self.player_sx = None
        self.player_sy = None
        self.player_srx = None
        self.player_sry = None

        self.enemies = []

        self.projectile_count = 0
        self.projectile_distance = -1.0
        self.projectile_dx = 0.0
        self.projectile_dy = 0.0

        # V5 exact hostile projectile vision.
        self.projectiles = []

        self._open()

    # ========================================================
    # LOG
    # ========================================================

    def _open(self):
        if not self.log_path.exists():
            return

        self.file = open(
            self.log_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        )

        self.file.seek(
            0,
            2,
        )

        self.position = (
            self.file.tell()
        )

        print(
            "Isaac combat bridge:"
        )

        print(
            self.log_path
        )

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

        self.file = None

    def reset(self):
        """
        Clear all old-run combat sensory state and skip unread combat lines.
        """

        self.enemies_alive = 0
        self.visible_enemies = 0

        self.player_x = 0.5
        self.player_y = 0.5
        self.player_rx = 0.0
        self.player_ry = 0.0

        self.player_gx = None
        self.player_gy = None
        self.player_grx = None
        self.player_gry = None

        self.player_sx = None
        self.player_sy = None
        self.player_srx = None
        self.player_sry = None

        self.enemies = []

        self.projectile_count = 0
        self.projectile_distance = -1.0
        self.projectile_dx = 0.0
        self.projectile_dy = 0.0
        self.projectiles = []

        if self.file is not None:
            try:
                self.file.seek(0, 2)
                self.position = self.file.tell()
            except Exception:
                pass

        print(
            "Combat state HARD RESET for new Isaac run."
        )

    # ========================================================
    # PARSING
    # ========================================================

    @staticmethod
    def _clip01(value):
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )

    @staticmethod
    def _optional_float(
        pieces,
        index,
    ):
        if index >= len(
            pieces
        ):
            return None

        try:
            return float(
                pieces[
                    index
                ]
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _parse_enemies(
        cls,
        text,
    ):
        """
        V4 entry:
            x,y,rx,ry,vulnerable,boss,
            gx,gy,grx,gry,
            sx,sy,srx,sry

        V3 entry:
            x,y,rx,ry,vulnerable,boss

        Older entry:
            x,y,vulnerable,boss
        """

        enemies = []

        if (
            not text
            or text == "-"
        ):
            return enemies

        for enemy_text in str(
            text
        ).split(";"):

            pieces = (
                enemy_text.split(
                    ","
                )
            )

            try:
                if len(
                    pieces
                ) >= 6:

                    x = cls._clip01(
                        pieces[0]
                    )

                    y = cls._clip01(
                        pieces[1]
                    )

                    rx = max(
                        0.0,
                        float(
                            pieces[2]
                        ),
                    )

                    ry = max(
                        0.0,
                        float(
                            pieces[3]
                        ),
                    )

                    vulnerable = bool(
                        int(
                            pieces[4]
                        )
                    )

                    boss = bool(
                        int(
                            pieces[5]
                        )
                    )

                elif len(
                    pieces
                ) == 4:

                    x = cls._clip01(
                        pieces[0]
                    )

                    y = cls._clip01(
                        pieces[1]
                    )

                    rx = (
                        cls.FALLBACK_ENEMY_RX
                    )

                    ry = (
                        cls.FALLBACK_ENEMY_RY
                    )

                    vulnerable = bool(
                        int(
                            pieces[2]
                        )
                    )

                    boss = bool(
                        int(
                            pieces[3]
                        )
                    )

                else:
                    continue

            except (
                TypeError,
                ValueError,
            ):
                continue

            enemy = {
                "x": x,
                "y": y,
                "rx": rx,
                "ry": ry,
                "vulnerable":
                    vulnerable,
                "boss":
                    boss,

                # V4 exact grid coordinates.
                "gx":
                    cls._optional_float(
                        pieces,
                        6,
                    ),
                "gy":
                    cls._optional_float(
                        pieces,
                        7,
                    ),
                "grx":
                    cls._optional_float(
                        pieces,
                        8,
                    ),
                "gry":
                    cls._optional_float(
                        pieces,
                        9,
                    ),

                # V4 exact screen coordinates.
                "sx":
                    cls._optional_float(
                        pieces,
                        10,
                    ),
                "sy":
                    cls._optional_float(
                        pieces,
                        11,
                    ),
                "srx":
                    cls._optional_float(
                        pieces,
                        12,
                    ),
                "sry":
                    cls._optional_float(
                        pieces,
                        13,
                    ),
            }

            enemies.append(
                enemy
            )

        return enemies

    @classmethod
    def _parse_projectiles(
        cls,
        text,
    ):
        """
        V5 hostile projectile entry:
            sx,sy,srx,sry,svx,svy,seed

        sx/sy/srx/sry:
            normalized exact screen-space hitbox

        svx/svy:
            normalized screen-space velocity per game update
        """

        projectiles = []

        if (
            not text
            or text == "-"
        ):
            return projectiles

        for projectile_text in str(
            text
        ).split(";"):

            pieces = projectile_text.split(",")

            if len(pieces) < 7:
                continue

            try:
                projectiles.append(
                    {
                        "sx": cls._clip01(
                            pieces[0]
                        ),
                        "sy": cls._clip01(
                            pieces[1]
                        ),
                        "srx": max(
                            0.0,
                            float(
                                pieces[2]
                            ),
                        ),
                        "sry": max(
                            0.0,
                            float(
                                pieces[3]
                            ),
                        ),
                        "svx": float(
                            pieces[4]
                        ),
                        "svy": float(
                            pieces[5]
                        ),
                        "seed": int(
                            float(
                                pieces[6]
                            )
                        ),
                    }
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return projectiles

    # ========================================================
    # POLL
    # ========================================================

    def poll(self):
        events = []

        if self.file is None:
            self._open()

            if self.file is None:
                return events

        try:
            size = (
                self.log_path
                .stat()
                .st_size
            )

            if size < self.position:
                self.file.close()
                self.file = None
                self._open()
                return events

        except OSError:
            return events

        self.file.seek(
            self.position
        )

        while True:
            line = (
                self.file.readline()
            )

            if not line:
                break

            marker = (
                line.find(
                    "FLYCOMBAT|"
                )
            )

            if marker < 0:
                continue

            payload = (
                line[
                    marker:
                ]
                .strip()
            )

            parts = (
                payload.split(
                    "|"
                )
            )

            if len(
                parts
            ) < 2:
                continue

            kind = (
                parts[
                    1
                ]
            )

            if kind == "STATE":

                if len(
                    parts
                ) >= 11:

                    try:
                        self.enemies_alive = int(
                            parts[2]
                        )

                        self.projectile_count = int(
                            parts[3]
                        )

                        self.projectile_distance = float(
                            parts[4]
                        )

                        self.projectile_dx = float(
                            parts[5]
                        )

                        self.projectile_dy = float(
                            parts[6]
                        )

                        self.player_x = self._clip01(
                            parts[7]
                        )

                        self.player_y = self._clip01(
                            parts[8]
                        )

                        self.visible_enemies = int(
                            parts[9]
                        )

                        self.enemies = (
                            self._parse_enemies(
                                parts[10]
                            )
                        )

                        if len(
                            parts
                        ) >= 13:

                            self.player_rx = max(
                                0.0,
                                float(
                                    parts[11]
                                ),
                            )

                            self.player_ry = max(
                                0.0,
                                float(
                                    parts[12]
                                ),
                            )

                        # V4 accurate grid-space player hitbox.
                        if len(
                            parts
                        ) >= 17:

                            self.player_gx = float(
                                parts[13]
                            )

                            self.player_gy = float(
                                parts[14]
                            )

                            self.player_grx = max(
                                0.0,
                                float(
                                    parts[15]
                                ),
                            )

                            self.player_gry = max(
                                0.0,
                                float(
                                    parts[16]
                                ),
                            )

                        # V4 exact screen-space player hitbox.
                        if len(
                            parts
                        ) >= 21:

                            self.player_sx = float(
                                parts[17]
                            )

                            self.player_sy = float(
                                parts[18]
                            )

                            self.player_srx = max(
                                0.0,
                                float(
                                    parts[19]
                                ),
                            )

                            self.player_sry = max(
                                0.0,
                                float(
                                    parts[20]
                                ),
                            )

                        # V5 exact hostile projectile list.
                        if len(
                            parts
                        ) >= 22:

                            self.projectiles = (
                                self._parse_projectiles(
                                    parts[21]
                                )
                            )

                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

                elif len(
                    parts
                ) >= 7:

                    # Very old bridge.
                    try:
                        self.enemies_alive = int(
                            parts[2]
                        )

                        self.projectile_count = int(
                            parts[3]
                        )

                        self.projectile_distance = float(
                            parts[4]
                        )

                        self.projectile_dx = float(
                            parts[5]
                        )

                        self.projectile_dy = float(
                            parts[6]
                        )

                    except ValueError:
                        pass

                continue

            if (
                kind
                in {
                    "SHOT_HIT",
                    "SHOT_MISS",
                }
                and len(
                    parts
                ) >= 4
            ):
                try:
                    reward = float(
                        parts[2]
                    )

                except ValueError:
                    reward = 0.0

                events.append(
                    {
                        "name":
                            kind,
                        "reward":
                            reward,
                        "action":
                            parts[3],
                        "extra":
                            (
                                parts[4]
                                if len(
                                    parts
                                ) >= 5
                                else ""
                            ),
                    }
                )

        self.position = (
            self.file.tell()
        )

        return events

    # ========================================================
    # SOURCE SHAPE
    # ========================================================

    @classmethod
    def _target_shape(
        cls,
        features,
    ):
        collision = (
            features.get(
                "collision_grid"
            )
        )

        if (
            isinstance(
                collision,
                np.ndarray,
            )
            and collision.ndim == 2
            and collision.shape[0] > 0
            and collision.shape[1] > 0
        ):
            return (
                int(
                    collision.shape[
                        0
                    ]
                ),
                int(
                    collision.shape[
                        1
                    ]
                ),
            )

        room_shape = int(
            features.get(
                "room_shape",
                1,
            )
        )

        return cls.SHAPE_DIMS.get(
            room_shape,
            (
                7,
                13,
            ),
        )

    # ========================================================
    # NORMALIZED FALLBACK RASTER
    # ========================================================

    @classmethod
    def _normalized_ellipses_to_grid(
        cls,
        ellipses,
        rows,
        cols,
        valid_mask=None,
    ):
        rows = max(
            1,
            int(
                rows
            ),
        )

        cols = max(
            1,
            int(
                cols
            ),
        )

        hi_rows = (
            rows
            * cls.SUPERSAMPLE
        )

        hi_cols = (
            cols
            * cls.SUPERSAMPLE
        )

        high = np.zeros(
            (
                hi_rows,
                hi_cols,
            ),
            dtype=np.float32,
        )

        xs = (
            np.arange(
                hi_cols,
                dtype=np.float32,
            )
            + 0.5
        ) / float(
            hi_cols
        )

        ys = (
            np.arange(
                hi_rows,
                dtype=np.float32,
            )
            + 0.5
        ) / float(
            hi_rows
        )

        xx = xs[
            None,
            :
        ]

        yy = ys[
            :,
            None
        ]

        centers = []

        for (
            cx,
            cy,
            rx,
            ry,
            strength,
        ) in ellipses:

            cx = cls._clip01(
                cx
            )

            cy = cls._clip01(
                cy
            )

            rx = max(
                float(
                    rx
                ),
                1e-5,
            )

            ry = max(
                float(
                    ry
                ),
                1e-5,
            )

            strength = float(
                np.clip(
                    strength,
                    0.0,
                    1.0,
                )
            )

            mask = (
                (
                    (
                        xx - cx
                    )
                    / rx
                ) ** 2
                +
                (
                    (
                        yy - cy
                    )
                    / ry
                ) ** 2
                <= 1.0
            )

            high[
                mask
            ] = np.maximum(
                high[
                    mask
                ],
                strength,
            )

            centers.append(
                (
                    cx,
                    cy,
                    strength,
                )
            )

        grid = (
            high
            .reshape(
                rows,
                cls.SUPERSAMPLE,
                cols,
                cls.SUPERSAMPLE,
            )
            .mean(
                axis=(
                    1,
                    3,
                )
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        for (
            cx,
            cy,
            strength,
        ) in centers:

            col = min(
                cols - 1,
                max(
                    0,
                    int(
                        cx
                        * cols
                    ),
                ),
            )

            row = min(
                rows - 1,
                max(
                    0,
                    int(
                        cy
                        * rows
                    ),
                ),
            )

            grid[
                row,
                col
            ] = max(
                grid[
                    row,
                    col
                ],
                strength,
            )

        if (
            isinstance(
                valid_mask,
                np.ndarray,
            )
            and valid_mask.shape
            == grid.shape
        ):
            grid *= (
                valid_mask
                > 0.5
            ).astype(
                np.float32
            )

        return grid

    # ========================================================
    # EXACT ISAAC GRID-SPACE RASTER
    # ========================================================

    @classmethod
    def _grid_ellipses_to_grid(
        cls,
        ellipses,
        rows,
        cols,
        valid_mask=None,
    ):
        """
        ellipses are already in SOURCE CELL UNITS:

            cx = 0.5 means center of source cell 0
            cx = 1.5 means center of source cell 1
            rx = 0.5 means half a grid cell radius

        This keeps enemy/self geometry in exactly the same coordinate system
        as the collision map.
        """

        rows = max(
            1,
            int(
                rows
            ),
        )

        cols = max(
            1,
            int(
                cols
            ),
        )

        hi_rows = (
            rows
            * cls.SUPERSAMPLE
        )

        hi_cols = (
            cols
            * cls.SUPERSAMPLE
        )

        high = np.zeros(
            (
                hi_rows,
                hi_cols,
            ),
            dtype=np.float32,
        )

        # High-resolution sample coordinates measured in SOURCE CELLS.
        xs = (
            np.arange(
                hi_cols,
                dtype=np.float32,
            )
            + 0.5
        ) / float(
            cls.SUPERSAMPLE
        )

        ys = (
            np.arange(
                hi_rows,
                dtype=np.float32,
            )
            + 0.5
        ) / float(
            cls.SUPERSAMPLE
        )

        xx = xs[
            None,
            :
        ]

        yy = ys[
            :,
            None
        ]

        centers = []

        for (
            cx,
            cy,
            rx,
            ry,
            strength,
        ) in ellipses:

            cx = float(
                cx
            )

            cy = float(
                cy
            )

            rx = max(
                float(
                    rx
                ),
                1e-4,
            )

            ry = max(
                float(
                    ry
                ),
                1e-4,
            )

            strength = float(
                np.clip(
                    strength,
                    0.0,
                    1.0,
                )
            )

            mask = (
                (
                    (
                        xx - cx
                    )
                    / rx
                ) ** 2
                +
                (
                    (
                        yy - cy
                    )
                    / ry
                ) ** 2
                <= 1.0
            )

            high[
                mask
            ] = np.maximum(
                high[
                    mask
                ],
                strength,
            )

            centers.append(
                (
                    cx,
                    cy,
                    strength,
                )
            )

        grid = (
            high
            .reshape(
                rows,
                cls.SUPERSAMPLE,
                cols,
                cls.SUPERSAMPLE,
            )
            .mean(
                axis=(
                    1,
                    3,
                )
            )
            .astype(
                np.float32,
                copy=False,
            )
        )

        # Preserve the exact center cell so a small entity cannot vanish.
        for (
            cx,
            cy,
            strength,
        ) in centers:

            col = int(
                np.floor(
                    cx
                )
            )

            row = int(
                np.floor(
                    cy
                )
            )

            if (
                0 <= row < rows
                and 0 <= col < cols
            ):
                grid[
                    row,
                    col
                ] = max(
                    grid[
                        row,
                        col
                    ],
                    strength,
                )

        if (
            isinstance(
                valid_mask,
                np.ndarray,
            )
            and valid_mask.shape
            == grid.shape
        ):
            grid *= (
                valid_mask
                > 0.5
            ).astype(
                np.float32
            )

        return grid

    # ========================================================
    # ENEMY / SELF GRIDS
    # ========================================================

    def enemy_grid(
        self,
        rows,
        cols,
        origin_row,
        origin_col,
        valid_mask=None,
    ):
        exact = []
        fallback = []

        for enemy in self.enemies:

            strength = (
                1.0
                if enemy[
                    "vulnerable"
                ]
                else 0.40
            )

            if enemy[
                "boss"
            ]:
                strength = max(
                    strength,
                    1.0,
                )

            if all(
                enemy.get(
                    key
                )
                is not None
                for key in (
                    "gx",
                    "gy",
                    "grx",
                    "gry",
                )
            ):
                # Raw grid -> source grid.
                exact.append(
                    (
                        float(
                            enemy[
                                "gx"
                            ]
                        )
                        - float(
                            origin_col
                        ),

                        float(
                            enemy[
                                "gy"
                            ]
                        )
                        - float(
                            origin_row
                        ),

                        float(
                            enemy[
                                "grx"
                            ]
                        ),

                        float(
                            enemy[
                                "gry"
                            ]
                        ),

                        strength,
                    )
                )

            else:
                fallback.append(
                    (
                        enemy[
                            "x"
                        ],
                        enemy[
                            "y"
                        ],
                        enemy[
                            "rx"
                        ],
                        enemy[
                            "ry"
                        ],
                        strength,
                    )
                )

        result = np.zeros(
            (
                rows,
                cols,
            ),
            dtype=np.float32,
        )

        if exact:
            result = np.maximum(
                result,
                self._grid_ellipses_to_grid(
                    exact,
                    rows,
                    cols,
                    valid_mask=(
                        valid_mask
                    ),
                ),
            )

        if fallback:
            result = np.maximum(
                result,
                self._normalized_ellipses_to_grid(
                    fallback,
                    rows,
                    cols,
                    valid_mask=(
                        valid_mask
                    ),
                ),
            )

        return result

    def self_grid(
        self,
        rows,
        cols,
        origin_row,
        origin_col,
        valid_mask=None,
    ):
        if all(
            value is not None
            for value in (
                self.player_gx,
                self.player_gy,
                self.player_grx,
                self.player_gry,
            )
        ):
            return self._grid_ellipses_to_grid(
                [
                    (
                        float(
                            self.player_gx
                        )
                        - float(
                            origin_col
                        ),

                        float(
                            self.player_gy
                        )
                        - float(
                            origin_row
                        ),

                        float(
                            self.player_grx
                        ),

                        float(
                            self.player_gry
                        ),

                        1.0,
                    )
                ],
                rows,
                cols,
                valid_mask=(
                    valid_mask
                ),
            )

        return self._normalized_ellipses_to_grid(
            [
                (
                    self.player_x,
                    self.player_y,
                    self.player_rx,
                    self.player_ry,
                    1.0,
                )
            ],
            rows,
            cols,
            valid_mask=(
                valid_mask
            ),
        )

    # ========================================================
    # EXACT SCREEN HITBOX -> FIXED LC10 12x20
    # ========================================================

    @staticmethod
    def _ellipse_to_lc10_grid(
        ellipses,
        rows=12,
        cols=20,
        samples=16,
    ):
        """
        Convert the REAL screen-space collision ellipse directly into
        the LC10 cell(s) it physically overlaps.

        This intentionally bypasses the old:
            hitbox -> room source grid -> resize -> LC10
        path for enemy/self location, because repeated resampling can
        shift small entities by one or two cells.

        Input coordinates are normalized screen coordinates:
            sx, sy   = ellipse center
            srx,sry  = ellipse radii
        """

        rows = int(rows)
        cols = int(cols)
        samples = max(
            4,
            int(samples),
        )

        grid = np.zeros(
            (
                rows,
                cols,
            ),
            dtype=np.float32,
        )

        offsets = (
            np.arange(
                samples,
                dtype=np.float32,
            )
            + 0.5
        ) / float(
            samples
        )

        for ellipse in ellipses:
            (
                sx,
                sy,
                srx,
                sry,
                strength,
            ) = ellipse

            if any(
                value is None
                for value in (
                    sx,
                    sy,
                    srx,
                    sry,
                )
            ):
                continue

            try:
                sx = float(sx)
                sy = float(sy)
                srx = abs(
                    float(srx)
                )
                sry = abs(
                    float(sry)
                )
                strength = float(
                    strength
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                srx <= 0.0
                or sry <= 0.0
            ):
                continue

            # Bounding cells potentially touched by the ellipse.
            c0 = max(
                0,
                int(
                    np.floor(
                        (
                            sx - srx
                        )
                        * cols
                    )
                ),
            )

            c1 = min(
                cols - 1,
                int(
                    np.floor(
                        (
                            sx + srx
                        )
                        * cols
                    )
                ),
            )

            r0 = max(
                0,
                int(
                    np.floor(
                        (
                            sy - sry
                        )
                        * rows
                    )
                ),
            )

            r1 = min(
                rows - 1,
                int(
                    np.floor(
                        (
                            sy + sry
                        )
                        * rows
                    )
                ),
            )

            for row in range(
                r0,
                r1 + 1,
            ):
                ys = (
                    row
                    + offsets
                ) / float(
                    rows
                )

                for col in range(
                    c0,
                    c1 + 1,
                ):
                    xs = (
                        col
                        + offsets
                    ) / float(
                        cols
                    )

                    xx, yy = (
                        np.meshgrid(
                            xs,
                            ys,
                        )
                    )

                    inside = (
                        (
                            (
                                xx - sx
                            )
                            / srx
                        ) ** 2
                        +
                        (
                            (
                                yy - sy
                            )
                            / sry
                        ) ** 2
                        <= 1.0
                    )

                    coverage = float(
                        np.mean(
                            inside
                        )
                    )

                    if coverage <= 0.0:
                        continue

                    # Any genuine overlap should survive the LC10 threshold.
                    # Larger overlap gives a stronger sensory value.
                    value = max(
                        0.15,
                        min(
                            1.0,
                            coverage
                            * 4.0,
                        ),
                    )

                    grid[
                        row,
                        col
                    ] = max(
                        grid[
                            row,
                            col
                        ],
                        value
                        * strength,
                    )

            # Always preserve the exact center cell.
            center_col = min(
                cols - 1,
                max(
                    0,
                    int(
                        sx
                        * cols
                    ),
                ),
            )

            center_row = min(
                rows - 1,
                max(
                    0,
                    int(
                        sy
                        * rows
                    ),
                ),
            )

            grid[
                center_row,
                center_col
            ] = max(
                grid[
                    center_row,
                    center_col
                ],
                strength,
            )

        return grid

    def exact_enemy_lc10_grid(
        self,
    ):
        ellipses = []

        for enemy in self.enemies:
            if any(
                enemy.get(
                    key
                )
                is None
                for key in (
                    "sx",
                    "sy",
                    "srx",
                    "sry",
                )
            ):
                continue

            strength = (
                1.0
                if enemy.get(
                    "vulnerable",
                    True,
                )
                else 0.40
            )

            ellipses.append(
                (
                    enemy.get(
                        "sx"
                    ),
                    enemy.get(
                        "sy"
                    ),
                    enemy.get(
                        "srx"
                    ),
                    enemy.get(
                        "sry"
                    ),
                    strength,
                )
            )

        return self._ellipse_to_lc10_grid(
            ellipses,
            rows=12,
            cols=20,
        )

    def exact_self_lc10_grid(
        self,
    ):
        if any(
            value is None
            for value in (
                self.player_sx,
                self.player_sy,
                self.player_srx,
                self.player_sry,
            )
        ):
            return np.zeros(
                (
                    12,
                    20,
                ),
                dtype=np.float32,
            )

        return self._ellipse_to_lc10_grid(
            [
                (
                    self.player_sx,
                    self.player_sy,
                    self.player_srx,
                    self.player_sry,
                    1.0,
                )
            ],
            rows=12,
            cols=20,
        )

    def exact_projectile_lc10_grid(
        self,
    ):
        ellipses = []

        for projectile in self.projectiles:
            ellipses.append(
                (
                    projectile.get(
                        "sx"
                    ),
                    projectile.get(
                        "sy"
                    ),
                    max(
                        projectile.get(
                            "srx",
                            0.0,
                        ),
                        0.006,
                    ),
                    max(
                        projectile.get(
                            "sry",
                            0.0,
                        ),
                        0.008,
                    ),
                    1.0,
                )
            )

        return self._ellipse_to_lc10_grid(
            ellipses,
            rows=12,
            cols=20,
        )

    def predicted_projectile_lc10_grid(
        self,
        frames_ahead=6.0,
    ):
        """
        Faint future-position trail so MaleCNS can sense projectile motion.
        No dodge direction is selected here.
        """

        ellipses = []

        for projectile in self.projectiles:
            sx = float(
                projectile.get(
                    "sx",
                    0.0,
                )
            )

            sy = float(
                projectile.get(
                    "sy",
                    0.0,
                )
            )

            svx = float(
                projectile.get(
                    "svx",
                    0.0,
                )
            )

            svy = float(
                projectile.get(
                    "svy",
                    0.0,
                )
            )

            predicted_x = max(
                0.0,
                min(
                    1.0,
                    sx
                    + svx
                    * float(
                        frames_ahead
                    ),
                ),
            )

            predicted_y = max(
                0.0,
                min(
                    1.0,
                    sy
                    + svy
                    * float(
                        frames_ahead
                    ),
                ),
            )

            ellipses.append(
                (
                    predicted_x,
                    predicted_y,
                    max(
                        projectile.get(
                            "srx",
                            0.0,
                        ),
                        0.006,
                    ),
                    max(
                        projectile.get(
                            "sry",
                            0.0,
                        ),
                        0.008,
                    ),
                    0.45,
                )
            )

        return self._ellipse_to_lc10_grid(
            ellipses,
            rows=12,
            cols=20,
        )

    # ========================================================
    # FEATURES
    # ========================================================

    def enrich_features(
        self,
        features,
    ):
        rows, cols = (
            self._target_shape(
                features
            )
        )

        valid_mask = (
            features.get(
                "room_valid_mask"
            )
        )

        origin_row = int(
            features.get(
                "room_source_origin_row",
                1,
            )
        )

        origin_col = int(
            features.get(
                "room_source_origin_col",
                1,
            )
        )

        enemy_grid = (
            self.enemy_grid(
                rows,
                cols,
                origin_row,
                origin_col,
                valid_mask=(
                    valid_mask
                ),
            )
        )

        self_grid = (
            self.self_grid(
                rows,
                cols,
                origin_row,
                origin_col,
                valid_mask=(
                    valid_mask
                ),
            )
        )

        features[
            "enemy_grid"
        ] = enemy_grid

        features[
            "self_grid"
        ] = self_grid

        # Direct 12x20 cell occupancy made from the exact red/cyan
        # collision ellipses shown on the dashboard.
        features[
            "enemy_lc10_hitbox_grid"
        ] = self.exact_enemy_lc10_grid()

        features[
            "self_lc10_hitbox_grid"
        ] = self.exact_self_lc10_grid()

        features[
            "enemy_count"
        ] = int(
            self.visible_enemies
        )

        features[
            "enemies_alive"
        ] = int(
            self.enemies_alive
        )

        features[
            "enemy_player_x"
        ] = float(
            self.player_x
        )

        features[
            "enemy_player_y"
        ] = float(
            self.player_y
        )

        features[
            "enemy_player_rx"
        ] = float(
            self.player_rx
        )

        features[
            "enemy_player_ry"
        ] = float(
            self.player_ry
        )

        features[
            "enemy_player_gx"
        ] = (
            self.player_gx
        )

        features[
            "enemy_player_gy"
        ] = (
            self.player_gy
        )

        features[
            "enemy_player_grx"
        ] = (
            self.player_grx
        )

        features[
            "enemy_player_gry"
        ] = (
            self.player_gry
        )

        features[
            "enemy_player_screen_x"
        ] = (
            self.player_sx
        )

        features[
            "enemy_player_screen_y"
        ] = (
            self.player_sy
        )

        features[
            "enemy_player_screen_rx"
        ] = (
            self.player_srx
        )

        features[
            "enemy_player_screen_ry"
        ] = (
            self.player_sry
        )

        features[
            "enemy_hitboxes"
        ] = [
            dict(
                enemy
            )
            for enemy
            in self.enemies
        ]

        features[
            "projectile_count"
        ] = int(
            self.projectile_count
        )

        features[
            "projectile_distance"
        ] = float(
            self.projectile_distance
        )

        features[
            "projectile_dx"
        ] = float(
            self.projectile_dx
        )

        features[
            "projectile_dy"
        ] = float(
            self.projectile_dy
        )

        features[
            "projectile_lc10_grid"
        ] = (
            self.exact_projectile_lc10_grid()
        )

        features[
            "projectile_future_lc10_grid"
        ] = (
            self.predicted_projectile_lc10_grid()
        )

        features[
            "projectile_hitboxes"
        ] = [
            dict(
                projectile
            )
            for projectile
            in self.projectiles
        ]

        features[
            "combat_grid_rows"
        ] = int(
            rows
        )

        features[
            "combat_grid_cols"
        ] = int(
            cols
        )

        features[
            "combat_hitbox_mode"
        ] = (
            "grid_exact"
            if (
                self.player_gx
                is not None
            )
            else "normalized_fallback"
        )

        return features
