from pathlib import Path
import math
import re

import numpy as np


class IsaacRoomState:
    """
    Reads FLYROOM state emitted by the Isaac room bridge.

    SOURCE GRID POLICY
    ------------------
    Room/map geometry stays in Isaac-native interior tile dimensions:

        1x1       ->  7 x 13
        IH        ->  3 x 13
        IV        ->  7 x 5
        1x2       -> 14 x 13
        IIV       -> 14 x 5
        2x1       ->  7 x 26
        IIH       ->  3 x 26
        2x2       -> 14 x 26
        L-shapes  -> 14 x 26 bounding grid with the missing quadrant invalid

    Array shapes are always (rows, columns).

    LC10 is NOT handled here. LC10 may independently resize the complete
    current-room source grid to its fixed 8x14 retina.
    """

    PREFIX = "FLYROOM|"

    # Repentance RoomShape integer -> (source rows, source cols)
    SHAPE_DIMS = {
        1: (7, 13),    # 1x1
        2: (3, 13),    # IH
        3: (7, 5),     # IV
        4: (14, 13),   # 1x2
        5: (14, 5),    # IIV
        6: (7, 26),    # 2x1
        7: (3, 26),    # IIH
        8: (14, 26),   # 2x2
        9: (14, 26),   # LTL
        10: (14, 26),  # LTR
        11: (14, 26),  # LBL
        12: (14, 26),  # LBR
    }

    SHAPE_NAMES = {
        1: "1x1",
        2: "IH",
        3: "IV",
        4: "1x2",
        5: "IIV",
        6: "2x1",
        7: "IIH",
        8: "2x2",
        9: "LTL",
        10: "LTR",
        11: "LBL",
        12: "LBR",
    }

    SLOT_DIRECTION = {
        0: "left",
        1: "up",
        2: "right",
        3: "down",
        4: "left",
        5: "up",
        6: "right",
        7: "down",
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

        self.log_path = Path(log_path)

        self.file = None
        self.position = 0

        self.has_state = False

        self.stage = 0
        self.stage_type = 0
        self.room_index = -1
        self.previous_room = -1
        self.starting_room = -1
        self.room_type = 0
        self.room_shape = 1

        self.grid_width = 15
        self.grid_size = 135
        self.raw_grid_height = 9

        self.player_grid = -1

        self.doors = []
        self.raw_collision = []

        self._source_collision = np.zeros(
            self.SHAPE_DIMS[1],
            dtype=np.float32,
        )

        self._source_collision_classes = np.zeros(
            self.SHAPE_DIMS[1],
            dtype=np.int16,
        )

        self._valid_mask = np.ones(
            self.SHAPE_DIMS[1],
            dtype=np.float32,
        )

        self._source_origin = (1, 1)

        self.visited_by_floor = {}
        self.known_by_floor = {}

        self._open()

    # ========================================================
    # LOG TAIL
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

        # Tail only new state, consistent with combat/reward readers.
        self.file.seek(0, 2)
        self.position = self.file.tell()

        print("Isaac room bridge:")
        print(self.log_path)

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

        self.file = None

    def reset_run_history(self):
        """
        Hard reset of every per-run room/map cache.

        Also skips unread log lines from the previous run. Without this,
        stale FLYROOM lines already sitting in log.txt can immediately
        repopulate old doors after we clear the dictionaries.
        """

        self.visited_by_floor.clear()
        self.known_by_floor.clear()

        self.has_state = False

        self.stage = 0
        self.stage_type = 0
        self.room_index = -1
        self.previous_room = -1
        self.starting_room = -1
        self.room_type = 0
        self.room_shape = 1

        self.grid_width = 15
        self.grid_size = 135
        self.raw_grid_height = 9

        self.player_grid = -1
        self.doors = []
        self.raw_collision = []

        self._source_collision = np.zeros(
            self.SHAPE_DIMS[1],
            dtype=np.float32,
        )

        self._source_collision_classes = np.zeros(
            self.SHAPE_DIMS[1],
            dtype=np.int16,
        )

        self._valid_mask = np.ones(
            self.SHAPE_DIMS[1],
            dtype=np.float32,
        )

        self._source_origin = (
            1,
            1,
        )

        # CRITICAL: drop unread bridge output from the dead run.
        if self.file is not None:
            try:
                self.file.seek(0, 2)
                self.position = self.file.tell()
            except Exception:
                pass

        print(
            "Room history HARD RESET for new Isaac run."
        )

    # ========================================================
    # PARSING HELPERS
    # ========================================================

    @staticmethod
    def _to_int(value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _parse_ints(text):
        """
        Parse Isaac room collision data safely.

        The room bridge may emit collision classes in either form:

            comma/space separated:
                4,4,4,0,0,5,...

            compact single-digit stream:
                444005000...

        GridCollisionClass values are single digits (0..5). The old parser
        treated a compact stream as ONE enormous integer, which caused:

            OverflowError: Python int too large to convert to C long

        when that value was assigned into the NumPy collision array.
        """

        if text is None:
            return []

        raw = str(
            text
        ).strip()

        if (
            not raw
            or raw == "-"
        ):
            return []

        # ----------------------------------------------------
        # Compact room-bridge format.
        # Every character is one GridCollisionClass.
        # ----------------------------------------------------

        if all(
            ch in "012345"
            for ch in raw
        ):
            return [
                int(
                    ch
                )
                for ch in raw
            ]

        # ----------------------------------------------------
        # Delimited / verbose format.
        # ----------------------------------------------------

        values = []

        for token in re.findall(
            r"-?\d+",
            raw,
        ):
            try:
                value = int(
                    token
                )
            except ValueError:
                continue

            # Collision classes outside 0..5 are malformed data.
            # Ignore them instead of ever allowing a huge integer into
            # the NumPy map.
            if 0 <= value <= 5:
                values.append(
                    value
                )

        return values

    @classmethod
    def _parse_doors(cls, text):
        """
        Bridge format:
            slot:TargetRoomIndex:open:locked
        with multiple entries normally separated by ';'.

        The parser is intentionally tolerant of commas/whitespace between
        records, because Lua debug output formatting can vary.
        """
        if not text or text == "-":
            return []

        matches = re.findall(
            r"(-?\d+):(-?\d+):(-?\d+):(-?\d+)",
            str(text),
        )

        doors = []

        for slot, target, opened, locked in matches:
            slot = int(slot)

            doors.append(
                {
                    "slot": slot,
                    "target": int(target),
                    "open": bool(int(opened)),
                    "locked": bool(int(locked)),
                    "direction": cls.SLOT_DIRECTION.get(
                        slot,
                        "?",
                    ),
                }
            )

        return doors

    # ========================================================
    # ROOM SHAPE
    # ========================================================

    @classmethod
    def source_dims_for_shape(
        cls,
        room_shape,
    ):
        return cls.SHAPE_DIMS.get(
            int(room_shape),
            cls.SHAPE_DIMS[1],
        )

    @classmethod
    def shape_name(
        cls,
        room_shape,
    ):
        return cls.SHAPE_NAMES.get(
            int(room_shape),
            f"shape-{room_shape}",
        )

    @staticmethod
    def _collision_strength(value):
        """
        Isaac GridCollisionClass values:

            0 = COLLISION_NONE
            1 = COLLISION_PIT
            2 = COLLISION_OBJECT
            3 = COLLISION_SOLID
            4 = COLLISION_WALL
            5 = COLLISION_WALL_EXCEPT_PLAYER

        For PLAYER movement sensing, class 5 must be CLEAR.
        The old code treated every non-zero value as a wall, which made
        walkable wall-edge tiles appear blocked and caused false WALL_STUCK
        penalties.

        Pits stay blocked here for a normal grounded player. We preserve
        the original class grid separately so flight can be handled later.
        """
        value = int(value)

        if value in (
            0,  # none
            5,  # wall except player -> player can pass
        ):
            return 0.0

        return 1.0

    @classmethod
    def _l_valid_mask(
        cls,
        room_shape,
        rows,
        cols,
    ):
        mask = np.ones(
            (
                rows,
                cols,
            ),
            dtype=np.float32,
        )

        if rows != 14 or cols != 26:
            return mask

        # Missing 13x7 quadrant in each L-shaped room.
        if room_shape == 9:       # LTL: missing top-left
            mask[0:7, 0:13] = 0.0

        elif room_shape == 10:    # LTR: missing top-right
            mask[0:7, 13:26] = 0.0

        elif room_shape == 11:    # LBL: missing bottom-left
            mask[7:14, 0:13] = 0.0

        elif room_shape == 12:    # LBR: missing bottom-right
            mask[7:14, 13:26] = 0.0

        return mask

    @classmethod
    def raw_collision_to_source(
        cls,
        room_shape,
        grid_width,
        grid_size,
        collision_values,
    ):
        """
        Convert Isaac's raw grid into the native interior source map.

        IMPORTANT:
        The raw Isaac grid includes the outside border. The source room map
        removes that border but does NOT delete the outermost PLAYABLE row
        or column.

        Returns:
            collision_grid          float32 player-blocked map
            collision_class_grid    exact GridCollisionClass values
            valid_mask              L-room geometry mask
            origin_row_in_raw
            origin_col_in_raw
        """
        room_shape = int(room_shape)
        grid_width = max(
            1,
            int(grid_width),
        )
        grid_size = max(
            0,
            int(grid_size),
        )

        raw_rows = max(
            1,
            int(
                math.ceil(
                    grid_size
                    / float(grid_width)
                )
            ),
        )

        # Unknown / missing raw entries default to a real wall class,
        # not PIT. This makes malformed bridge data obvious and safe.
        raw_classes = np.full(
            (
                raw_rows,
                grid_width,
            ),
            4,
            dtype=np.int16,
        )

        values = list(
            collision_values
            or []
        )

        count = min(
            len(values),
            raw_rows * grid_width,
        )

        if count:
            flat = raw_classes.ravel()

            for i in range(count):
                try:
                    value = int(
                        values[i]
                    )
                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ):
                    value = 4

                if not (
                    0
                    <= value
                    <= 5
                ):
                    value = 4

                flat[i] = value

        # Remove only Isaac's raw one-tile OUTSIDE border.
        # The resulting first/last rows are still legitimate playable
        # room tiles and must not be deleted.
        if (
            raw_rows >= 3
            and grid_width >= 3
        ):
            interior_classes = raw_classes[
                1:-1,
                1:-1,
            ]

            base_row = 1
            base_col = 1

        else:
            interior_classes = (
                raw_classes
            )

            base_row = 0
            base_col = 0

        target_rows, target_cols = (
            cls.source_dims_for_shape(
                room_shape
            )
        )

        # Narrow I-shapes occupy a centered subsection of the raw box.
        crop_row = max(
            0,
            (
                interior_classes.shape[0]
                - target_rows
            )
            // 2,
        )

        crop_col = max(
            0,
            (
                interior_classes.shape[1]
                - target_cols
            )
            // 2,
        )

        source_classes = interior_classes[
            crop_row:
            crop_row + target_rows,
            crop_col:
            crop_col + target_cols,
        ]

        if source_classes.shape != (
            target_rows,
            target_cols,
        ):
            padded_classes = np.full(
                (
                    target_rows,
                    target_cols,
                ),
                4,
                dtype=np.int16,
            )

            rr = min(
                target_rows,
                source_classes.shape[0],
            )

            cc = min(
                target_cols,
                source_classes.shape[1],
            )

            padded_classes[
                :rr,
                :cc,
            ] = source_classes[
                :rr,
                :cc,
            ]

            source_classes = (
                padded_classes
            )

        valid = cls._l_valid_mask(
            room_shape,
            target_rows,
            target_cols,
        )

        # Player-passability map from the exact collision classes.
        source = np.zeros(
            (
                target_rows,
                target_cols,
            ),
            dtype=np.float32,
        )

        for row in range(
            target_rows
        ):
            for col in range(
                target_cols
            ):
                source[
                    row,
                    col
                ] = cls._collision_strength(
                    source_classes[
                        row,
                        col
                    ]
                )

        # Missing quadrant of an L-room is outside the room.
        source = np.where(
            valid > 0.5,
            source,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        source_classes = np.where(
            valid > 0.5,
            source_classes,
            4,
        ).astype(
            np.int16,
            copy=False,
        )

        return (
            source,
            source_classes,
            valid,
            base_row + crop_row,
            base_col + crop_col,
        )

    # ========================================================
    # PLAYER
    # ========================================================

    def player_source_position(
        self,
    ):
        if (
            self.player_grid < 0
            or self.grid_width <= 0
        ):
            return (
                -1,
                -1,
            )

        raw_row = (
            int(self.player_grid)
            // int(self.grid_width)
        )

        raw_col = (
            int(self.player_grid)
            % int(self.grid_width)
        )

        origin_row, origin_col = (
            self._source_origin
        )

        row = (
            raw_row
            - origin_row
        )

        col = (
            raw_col
            - origin_col
        )

        rows, cols = (
            self._source_collision.shape
        )

        if not (
            0 <= row < rows
            and 0 <= col < cols
        ):
            return (
                -1,
                -1,
            )

        if (
            self._valid_mask[
                row,
                col
            ]
            <= 0.5
        ):
            return (
                -1,
                -1,
            )

        return (
            int(row),
            int(col),
        )

    def directional_collision(
        self,
    ):
        row, col = (
            self.player_source_position()
        )

        result = {
            "up": 0.0,
            "down": 0.0,
            "left": 0.0,
            "right": 0.0,
        }

        if row < 0 or col < 0:
            return result

        grid = (
            self._source_collision
        )

        rows, cols = (
            grid.shape
        )

        for (
            name,
            rr,
            cc,
        ) in (
            (
                "up",
                row - 1,
                col,
            ),
            (
                "down",
                row + 1,
                col,
            ),
            (
                "left",
                row,
                col - 1,
            ),
            (
                "right",
                row,
                col + 1,
            ),
        ):
            if (
                rr < 0
                or rr >= rows
                or cc < 0
                or cc >= cols
            ):
                result[
                    name
                ] = 1.0
            else:
                result[
                    name
                ] = float(
                    grid[
                        rr,
                        cc
                    ]
                )

        return result

    def near_collision_grid(
        self,
        radius=1,
    ):
        grid = np.zeros_like(
            self._source_collision
        )

        row, col = (
            self.player_source_position()
        )

        if row < 0 or col < 0:
            return grid

        r0 = max(
            0,
            row - int(radius),
        )

        r1 = min(
            grid.shape[0],
            row + int(radius) + 1,
        )

        c0 = max(
            0,
            col - int(radius),
        )

        c1 = min(
            grid.shape[1],
            col + int(radius) + 1,
        )

        grid[
            r0:r1,
            c0:c1,
        ] = self._source_collision[
            r0:r1,
            c0:c1,
        ]

        return grid

    # ========================================================
    # DOOR GRIDS
    # ========================================================

    @staticmethod
    def _segment_centers(
        length,
        doubled_threshold,
    ):
        if length >= doubled_threshold:
            return (
                max(
                    0,
                    min(
                        length - 1,
                        length // 4,
                    ),
                ),
                max(
                    0,
                    min(
                        length - 1,
                        (3 * length) // 4,
                    ),
                ),
            )

        center = (
            length // 2
        )

        return (
            center,
            center,
        )

    @classmethod
    def _door_cell(
        cls,
        slot,
        rows,
        cols,
    ):
        row0, row1 = (
            cls._segment_centers(
                rows,
                14,
            )
        )

        col0, col1 = (
            cls._segment_centers(
                cols,
                26,
            )
        )

        second = (
            int(slot)
            >= 4
        )

        slot_base = (
            int(slot)
            % 4
        )

        if slot_base == 0:  # left
            return (
                row1
                if second
                else row0,
                0,
            )

        if slot_base == 1:  # up
            return (
                0,
                col1
                if second
                else col0,
            )

        if slot_base == 2:  # right
            return (
                row1
                if second
                else row0,
                cols - 1,
            )

        # down
        return (
            rows - 1,
            col1
            if second
            else col0,
        )

    def door_grids(
        self,
    ):
        shape = (
            self._source_collision.shape
        )

        door_grid = np.zeros(
            shape,
            dtype=np.float32,
        )

        unexplored = np.zeros(
            shape,
            dtype=np.float32,
        )

        locked = np.zeros(
            shape,
            dtype=np.float32,
        )

        floor_key = (
            self.stage,
            self.stage_type,
        )

        visited = (
            self.visited_by_floor.get(
                floor_key,
                set(),
            )
        )

        rows, cols = (
            shape
        )

        door_info = []

        for door in self.doors:
            row, col = (
                self._door_cell(
                    door["slot"],
                    rows,
                    cols,
                )
            )

            # If an L-shaped missing quadrant catches the nominal point,
            # search inward on that edge for the nearest valid cell.
            if (
                self._valid_mask[
                    row,
                    col
                ]
                <= 0.5
            ):
                candidates = []

                if col == 0:
                    candidates = [
                        (
                            rr,
                            col,
                        )
                        for rr in range(
                            rows
                        )
                    ]

                elif col == cols - 1:
                    candidates = [
                        (
                            rr,
                            col,
                        )
                        for rr in range(
                            rows
                        )
                    ]

                elif row == 0:
                    candidates = [
                        (
                            row,
                            cc,
                        )
                        for cc in range(
                            cols
                        )
                    ]

                else:
                    candidates = [
                        (
                            row,
                            cc,
                        )
                        for cc in range(
                            cols
                        )
                    ]

                valid_candidates = [
                    cell
                    for cell in candidates
                    if self._valid_mask[
                        cell[0],
                        cell[1]
                    ] > 0.5
                ]

                if valid_candidates:
                    row, col = min(
                        valid_candidates,
                        key=lambda cell: (
                            abs(
                                cell[0] - row
                            )
                            + abs(
                                cell[1] - col
                            )
                        ),
                    )

            door_grid[
                row,
                col
            ] = 1.0

            target = int(
                door["target"]
            )

            was_visited = (
                target
                in visited
            )

            if not was_visited:
                unexplored[
                    row,
                    col
                ] = 1.0

            if door["locked"]:
                locked[
                    row,
                    col
                ] = 1.0

            info = dict(
                door
            )

            info[
                "visited"
            ] = was_visited

            info[
                "row"
            ] = int(
                row
            )

            info[
                "col"
            ] = int(
                col
            )

            door_info.append(
                info
            )

        return (
            door_grid,
            unexplored,
            locked,
            door_info,
        )

    # ========================================================
    # STATE UPDATE
    # ========================================================

    def _apply_state(
        self,
        parts,
    ):
        """
        Expected bridge payload:

        FLYROOM|STATE|
        stage|stageType|roomIndex|previousRoom|startingRoom|
        roomType|roomShape|gridWidth|gridSize|playerGrid|
        doors|collision
        """
        if len(parts) < 14:
            return False

        try:
            stage = self._to_int(
                parts[2]
            )

            stage_type = self._to_int(
                parts[3]
            )

            room_index = self._to_int(
                parts[4],
                -1,
            )

            previous_room = self._to_int(
                parts[5],
                -1,
            )

            starting_room = self._to_int(
                parts[6],
                -1,
            )

            room_type = self._to_int(
                parts[7]
            )

            room_shape = self._to_int(
                parts[8],
                1,
            )

            grid_width = self._to_int(
                parts[9],
                15,
            )

            grid_size = self._to_int(
                parts[10],
                135,
            )

            player_grid = self._to_int(
                parts[11],
                -1,
            )

            doors = self._parse_doors(
                parts[12]
            )

            collision_text = (
                parts[13]
            )

            collision_values = (
                self._parse_ints(
                    collision_text
                )
            )

            # If the bridge sent a compact collision stream, its length
            # should normally match grid_size. Do not crash on a malformed
            # line; raw_collision_to_source() safely pads missing cells.

        except Exception:
            return False

        (
            source_collision,
            source_collision_classes,
            valid_mask,
            origin_row,
            origin_col,
        ) = self.raw_collision_to_source(
            room_shape,
            grid_width,
            grid_size,
            collision_values,
        )

        self.stage = stage
        self.stage_type = stage_type
        self.room_index = room_index
        self.previous_room = previous_room
        self.starting_room = starting_room
        self.room_type = room_type
        self.room_shape = room_shape

        self.grid_width = grid_width
        self.grid_size = grid_size
        self.raw_grid_height = max(
            1,
            int(
                math.ceil(
                    grid_size
                    / float(
                        max(
                            1,
                            grid_width,
                        )
                    )
                )
            ),
        )

        self.player_grid = (
            player_grid
        )

        self.doors = doors
        self.raw_collision = (
            collision_values
        )

        self._source_collision = (
            source_collision
        )

        self._source_collision_classes = (
            source_collision_classes
        )

        self._valid_mask = (
            valid_mask
        )

        self._source_origin = (
            origin_row,
            origin_col,
        )

        floor_key = (
            stage,
            stage_type,
        )

        visited = (
            self.visited_by_floor
            .setdefault(
                floor_key,
                set(),
            )
        )

        known = (
            self.known_by_floor
            .setdefault(
                floor_key,
                set(),
            )
        )

        if room_index >= 0:
            visited.add(
                room_index
            )
            known.add(
                room_index
            )

        if previous_room >= 0:
            known.add(
                previous_room
            )

        if starting_room >= 0:
            known.add(
                starting_room
            )

        for door in doors:
            if door[
                "target"
            ] >= 0:
                known.add(
                    door[
                        "target"
                    ]
                )

        self.has_state = True

        return True

    def poll(self):
        updates = 0

        if self.file is None:
            self._open()

            if self.file is None:
                return updates

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
                return updates

        except OSError:
            return updates

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
                    self.PREFIX
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

            if (
                len(parts) >= 2
                and parts[1]
                == "STATE"
            ):
                if self._apply_state(
                    parts
                ):
                    updates += 1

        self.position = (
            self.file.tell()
        )

        return updates

    # ========================================================
    # FEATURES
    # ========================================================

    def enrich_features(
        self,
        features,
    ):
        if not self.has_state:
            # Never allow a reused feature dictionary to retain room/map
            # information from the previous run.
            for key in (
            'collision_class_grid',
            'collision_down',
            'collision_grid',
            'collision_left',
            'collision_right',
            'collision_up',
            'door_grid',
            'door_info',
            'known_room_indices',
            'locked_door_grid',
            'near_collision_grid',
            'player_retina_col',
            'player_retina_row',
            'previous_room',
            'room_grid_cols',
            'room_grid_rows',
            'room_grid_size',
            'room_index',
            'room_raw_grid_height',
            'room_raw_grid_width',
            'room_shape',
            'room_shape_name',
            'room_source_origin_col',
            'room_source_origin_row',
            'room_stage',
            'room_type',
            'room_valid_mask',
            'rooms_known',
            'rooms_visited',
            'stage',
            'stage_type',
            'starting_room',
            'unexplored_door_grid',
            'visited_room_indices'
            ):
                features.pop(
                    key,
                    None,
                )

            return features

        collision_grid = (
            self._source_collision
            .copy()
        )

        valid_mask = (
            self._valid_mask
            .copy()
        )

        near = (
            self.near_collision_grid(
                radius=1
            )
        )

        directions = (
            self.directional_collision()
        )

        (
            door_grid,
            unexplored,
            locked,
            door_info,
        ) = self.door_grids()

        player_row, player_col = (
            self.player_source_position()
        )

        floor_key = (
            self.stage,
            self.stage_type,
        )

        visited = (
            self.visited_by_floor
            .get(
                floor_key,
                set(),
            )
        )

        known = (
            self.known_by_floor
            .get(
                floor_key,
                set(),
            )
        )

        features[
            "collision_grid"
        ] = collision_grid

        features[
            "collision_class_grid"
        ] = (
            self._source_collision_classes
            .copy()
        )

        features[
            "room_source_origin_row"
        ] = int(
            self._source_origin[0]
        )

        features[
            "room_source_origin_col"
        ] = int(
            self._source_origin[1]
        )

        features[
            "near_collision_grid"
        ] = near

        features[
            "room_valid_mask"
        ] = valid_mask

        features[
            "door_grid"
        ] = door_grid

        features[
            "unexplored_door_grid"
        ] = unexplored

        features[
            "locked_door_grid"
        ] = locked

        features[
            "door_info"
        ] = door_info

        features[
            "collision_up"
        ] = directions[
            "up"
        ]

        features[
            "collision_down"
        ] = directions[
            "down"
        ]

        features[
            "collision_left"
        ] = directions[
            "left"
        ]

        features[
            "collision_right"
        ] = directions[
            "right"
        ]

        features[
            "player_retina_row"
        ] = player_row

        features[
            "player_retina_col"
        ] = player_col

        features[
            "room_stage"
        ] = (
            f"{self.stage}:"
            f"{self.stage_type}"
        )

        features[
            "stage"
        ] = self.stage

        features[
            "stage_type"
        ] = self.stage_type

        features[
            "room_index"
        ] = self.room_index

        features[
            "previous_room"
        ] = self.previous_room

        features[
            "starting_room"
        ] = self.starting_room

        features[
            "room_type"
        ] = self.room_type

        features[
            "room_shape"
        ] = self.room_shape

        features[
            "room_shape_name"
        ] = self.shape_name(
            self.room_shape
        )

        features[
            "room_grid_rows"
        ] = int(
            collision_grid.shape[0]
        )

        features[
            "room_grid_cols"
        ] = int(
            collision_grid.shape[1]
        )

        features[
            "room_raw_grid_width"
        ] = self.grid_width

        features[
            "room_raw_grid_height"
        ] = self.raw_grid_height

        features[
            "room_grid_size"
        ] = self.grid_size

        features[
            "rooms_visited"
        ] = len(
            visited
        )

        features[
            "rooms_known"
        ] = len(
            known
        )

        features[
            "visited_room_indices"
        ] = sorted(
            visited
        )

        features[
            "known_room_indices"
        ] = sorted(
            known
        )

        return features

    # Old code used both names during development.
    enrich = enrich_features


# Compatibility aliases.
RoomStateReader = IsaacRoomState
IsaacRoomStateReader = IsaacRoomState
RoomState = IsaacRoomState
