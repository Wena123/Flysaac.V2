from pathlib import Path
import numpy as np


class IsaacWorldState:
    PREFIX = "FLYWORLD|"

    GRID_ROCK_TYPES = {2, 3, 4, 5, 6, 22, 25, 26, 27}
    GRID_PIT = 7
    GRID_SPIKE_TYPES = {8, 9, 25}
    GRID_TNT = 12
    GRID_POOP = 14

    PICKUP_HEART = 10
    PICKUP_COIN = 20
    PICKUP_KEY = 30
    PICKUP_BOMB = 40
    PICKUP_COLLECTIBLE = 100
    PICKUP_SHOPITEM = 150

    def __init__(self, log_path=None):
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

        self.room_index = -1
        self.grid_width = 0
        self.grid_size = 0

        self.grid_entities = []
        self.pickups = []
        self.doors = []

        self.has_state = False
        self._open()

    def _open(self):
        if not self.log_path.exists():
            return

        self.file = open(
            self.log_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        )
        self.file.seek(0, 2)
        self.position = self.file.tell()

        print("Isaac world bridge:")
        print(self.log_path)

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
        self.file = None

    def reset(self):
        """
        Hard reset live world objects and discard unread old-run log state.
        """

        self.room_index = -1
        self.grid_width = 0
        self.grid_size = 0

        self.grid_entities = []
        self.pickups = []
        self.doors = []

        self.has_state = False

        # CRITICAL: skip any FLYWORLD lines written by the dead run that
        # were not consumed before the restart completed.
        if self.file is not None:
            try:
                self.file.seek(0, 2)
                self.position = self.file.tell()
            except Exception:
                pass

        print(
            "World state HARD RESET for new Isaac run."
        )

    @staticmethod
    def _int(value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _parse_grid_entities(cls, text):
        if not text or text == "-":
            return []

        result = []

        for entry in str(text).split(";"):
            p = entry.split(",")

            if len(p) < 4:
                continue

            result.append(
                {
                    "index": cls._int(p[0], -1),
                    "type": cls._int(p[1]),
                    "variant": cls._int(p[2]),
                    "state": cls._int(p[3]),
                }
            )

        return result

    @classmethod
    def _parse_pickups(cls, text):
        if not text or text == "-":
            return []

        result = []

        for entry in str(text).split(";"):
            p = entry.split(",")

            if len(p) < 11:
                continue

            result.append(
                {
                    "variant": cls._int(p[0]),
                    "subtype": cls._int(p[1]),
                    "gx": cls._float(p[2]),
                    "gy": cls._float(p[3]),
                    "sx": cls._float(p[4]),
                    "sy": cls._float(p[5]),
                    "srx": abs(cls._float(p[6])),
                    "sry": abs(cls._float(p[7])),
                    "price": cls._int(p[8]),
                    "shop": bool(cls._int(p[9])),
                    "seed": cls._int(p[10]),
                }
            )

        return result

    @classmethod
    def _parse_doors(cls, text):
        if not text or text == "-":
            return []

        result = []

        for entry in str(text).split(";"):
            p = entry.split(",")

            if len(p) < 6:
                continue

            result.append(
                {
                    "slot": cls._int(p[0], -1),
                    "gx": cls._float(p[1]),
                    "gy": cls._float(p[2]),
                    "open": bool(cls._int(p[3])),
                    "locked": bool(cls._int(p[4])),
                    "target": cls._int(p[5], -1),
                    "target_type": (
                        cls._int(p[6], 0)
                        if len(p) >= 7
                        else 0
                    ),
                }
            )

        return result

    def poll(self):
        updates = 0

        if self.file is None:
            self._open()

            if self.file is None:
                return updates

        try:
            size = self.log_path.stat().st_size
            if size < self.position:
                self.file.close()
                self.file = None
                self._open()
                return updates
        except OSError:
            return updates

        self.file.seek(self.position)

        while True:
            line = self.file.readline()

            if not line:
                break

            marker = line.find(self.PREFIX)

            if marker < 0:
                continue

            payload = line[marker:].strip()
            parts = payload.split("|")

            if (
                len(parts) >= 8
                and parts[1] == "STATE"
            ):
                self.room_index = self._int(parts[2], -1)
                self.grid_width = self._int(parts[3])
                self.grid_size = self._int(parts[4])
                self.grid_entities = self._parse_grid_entities(parts[5])
                self.pickups = self._parse_pickups(parts[6])
                self.doors = self._parse_doors(parts[7])
                self.has_state = True
                updates += 1

        self.position = self.file.tell()
        return updates

    @staticmethod
    def _shape(features):
        collision = features.get("collision_grid")

        if (
            isinstance(collision, np.ndarray)
            and collision.ndim == 2
        ):
            return collision.shape

        return (7, 13)

    @staticmethod
    def _origin(features):
        return (
            int(features.get("room_source_origin_row", 1)),
            int(features.get("room_source_origin_col", 1)),
        )

    @staticmethod
    def _blank(rows, cols):
        return np.zeros((rows, cols), dtype=np.float32)

    @staticmethod
    def _index_to_source(
        index,
        raw_width,
        origin_row,
        origin_col,
        rows,
        cols,
    ):
        if index < 0 or raw_width <= 0:
            return None

        raw_row = int(index) // int(raw_width)
        raw_col = int(index) % int(raw_width)

        row = raw_row - int(origin_row)
        col = raw_col - int(origin_col)

        if not (
            0 <= row < rows
            and 0 <= col < cols
        ):
            return None

        return (row, col)

    @staticmethod
    def _float_to_source(
        gx,
        gy,
        origin_row,
        origin_col,
        rows,
        cols,
        clamp=False,
    ):
        col = int(np.floor(float(gx) - float(origin_col)))
        row = int(np.floor(float(gy) - float(origin_row)))

        if clamp:
            return (
                min(rows - 1, max(0, row)),
                min(cols - 1, max(0, col)),
            )

        if not (
            0 <= row < rows
            and 0 <= col < cols
        ):
            return None

        return (row, col)

    @staticmethod
    def _ellipse_to_lc10(entries, rows=12, cols=20, samples=12):
        grid = np.zeros(
            (rows, cols),
            dtype=np.float32,
        )

        offsets = (
            np.arange(
                samples,
                dtype=np.float32,
            )
            + 0.5
        ) / float(samples)

        for entry in entries:
            try:
                sx = float(entry["sx"])
                sy = float(entry["sy"])

                srx = max(
                    abs(float(entry["srx"])),
                    0.006,
                )

                sry = max(
                    abs(float(entry["sry"])),
                    0.008,
                )

            except (KeyError, TypeError, ValueError):
                continue

            c0 = max(
                0,
                int(
                    np.floor(
                        (sx - srx)
                        * cols
                    )
                ),
            )

            c1 = min(
                cols - 1,
                int(
                    np.floor(
                        (sx + srx)
                        * cols
                    )
                ),
            )

            r0 = max(
                0,
                int(
                    np.floor(
                        (sy - sry)
                        * rows
                    )
                ),
            )

            r1 = min(
                rows - 1,
                int(
                    np.floor(
                        (sy + sry)
                        * rows
                    )
                ),
            )

            for row in range(r0, r1 + 1):
                ys = (row + offsets) / float(rows)

                for col in range(c0, c1 + 1):
                    xs = (col + offsets) / float(cols)
                    xx, yy = np.meshgrid(xs, ys)

                    inside = (
                        ((xx - sx) / srx) ** 2
                        + ((yy - sy) / sry) ** 2
                        <= 1.0
                    )

                    coverage = float(
                        np.mean(inside)
                    )

                    if coverage > 0.0:
                        grid[row, col] = max(
                            grid[row, col],
                            max(
                                0.18,
                                min(
                                    1.0,
                                    coverage * 4.0,
                                ),
                            ),
                        )

            center_col = min(
                cols - 1,
                max(
                    0,
                    int(sx * cols),
                ),
            )

            center_row = min(
                rows - 1,
                max(
                    0,
                    int(sy * rows),
                ),
            )

            grid[center_row, center_col] = 1.0

        return grid

    def enrich_features(self, features):
        if not self.has_state:
            # Explicitly remove stale semantic overlays from a reused
            # vision feature dictionary.
            for key in (
            'angel_door_grid',
            'bomb_pickup_count',
            'bomb_pickup_lc10_grid',
            'boss_door_grid',
            'closed_door_grid',
            'coin_count',
            'coin_lc10_grid',
            'destructible_grid',
            'devil_door_grid',
            'door_grid_exact',
            'hazard_grid',
            'heart_lc10_grid',
            'heart_pickup_count',
            'item_lc10_grid',
            'item_pedestal_count',
            'key_count',
            'key_lc10_grid',
            'locked_door_grid_exact',
            'navigation_grid',
            'open_door_grid',
            'other_pickup_lc10_grid',
            'pit_grid',
            'rock_grid',
            'shop_door_grid',
            'spike_grid',
            'treasure_door_grid',
            'unexplored_door_grid_exact',
            'world_doors',
            'world_pickups'
            ):
                features.pop(
                    key,
                    None,
                )

            return features

        rows, cols = self._shape(features)
        origin_row, origin_col = self._origin(features)

        rocks = self._blank(rows, cols)
        pits = self._blank(rows, cols)
        spikes = self._blank(rows, cols)
        destructible = self._blank(rows, cols)

        for obj in self.grid_entities:
            cell = self._index_to_source(
                obj["index"],
                self.grid_width,
                origin_row,
                origin_col,
                rows,
                cols,
            )

            if cell is None:
                continue

            row, col = cell
            grid_type = int(obj["type"])

            if grid_type in self.GRID_ROCK_TYPES:
                rocks[row, col] = 1.0

            if grid_type == self.GRID_PIT:
                pits[row, col] = 1.0

            if grid_type in self.GRID_SPIKE_TYPES:
                spikes[row, col] = 1.0

            if grid_type in {
                self.GRID_TNT,
                self.GRID_POOP,
            }:
                destructible[row, col] = 1.0

        door_grid = self._blank(rows, cols)
        open_doors = self._blank(rows, cols)
        closed_doors = self._blank(rows, cols)
        locked_doors = self._blank(rows, cols)
        unexplored_doors = self._blank(rows, cols)

        treasure_doors = self._blank(rows, cols)
        shop_doors = self._blank(rows, cols)
        boss_doors = self._blank(rows, cols)
        devil_doors = self._blank(rows, cols)
        angel_doors = self._blank(rows, cols)

        visited_rooms = set(
            int(x)
            for x in features.get(
                "visited_room_indices",
                []
            )
        )

        for door in self.doors:
            cell = self._float_to_source(
                door["gx"],
                door["gy"],
                origin_row,
                origin_col,
                rows,
                cols,
                clamp=True,
            )

            row, col = cell
            door_grid[row, col] = 1.0

            if door["target"] >= 0 and door["target"] not in visited_rooms:
                unexplored_doors[row, col] = 1.0

            target_type = int(
                door.get(
                    "target_type",
                    0,
                )
            )

            # RoomType enum:
            # 2 shop, 4 treasure, 5 boss, 14 devil, 15 angel.
            if target_type == 2:
                shop_doors[row, col] = 1.0

            elif target_type == 4:
                treasure_doors[row, col] = 1.0

            elif target_type == 5:
                boss_doors[row, col] = 1.0

            elif target_type == 14:
                devil_doors[row, col] = 1.0

            elif target_type == 15:
                angel_doors[row, col] = 1.0

            if door["locked"]:
                locked_doors[row, col] = 1.0

            elif door["open"]:
                open_doors[row, col] = 1.0

            else:
                closed_doors[row, col] = 1.0

        collision = features.get("collision_grid")

        if (
            isinstance(collision, np.ndarray)
            and collision.shape == (rows, cols)
        ):
            navigation = collision.astype(
                np.float32,
                copy=True,
            )
        else:
            navigation = self._blank(rows, cols)

        # Crucial door distinction:
        # an open door is a real hole through the wall.
        navigation[open_doors > 0.5] = 0.0

        navigation[
            (locked_doors + closed_doors)
            > 0.5
        ] = 1.0

        coins = []
        bombs = []
        keys = []
        hearts = []
        items = []
        other = []

        for pickup in self.pickups:
            variant = int(pickup["variant"])

            if variant == self.PICKUP_COIN:
                coins.append(pickup)

            elif variant == self.PICKUP_BOMB:
                bombs.append(pickup)

            elif variant == self.PICKUP_KEY:
                keys.append(pickup)

            elif variant == self.PICKUP_HEART:
                hearts.append(pickup)

            elif variant in {
                self.PICKUP_COLLECTIBLE,
                self.PICKUP_SHOPITEM,
            }:
                items.append(pickup)

            else:
                other.append(pickup)

        features["rock_grid"] = rocks
        features["pit_grid"] = pits
        features["spike_grid"] = spikes
        features["destructible_grid"] = destructible

        features["hazard_grid"] = np.maximum(
            pits,
            spikes,
        ).astype(
            np.float32,
            copy=False,
        )

        features["navigation_grid"] = navigation

        features["door_grid_exact"] = door_grid
        features["open_door_grid"] = open_doors
        features["closed_door_grid"] = closed_doors
        features["locked_door_grid_exact"] = locked_doors
        features["unexplored_door_grid_exact"] = unexplored_doors

        features["treasure_door_grid"] = treasure_doors
        features["shop_door_grid"] = shop_doors
        features["boss_door_grid"] = boss_doors
        features["devil_door_grid"] = devil_doors
        features["angel_door_grid"] = angel_doors

        features["coin_lc10_grid"] = self._ellipse_to_lc10(coins)
        features["bomb_pickup_lc10_grid"] = self._ellipse_to_lc10(bombs)
        features["key_lc10_grid"] = self._ellipse_to_lc10(keys)
        features["heart_lc10_grid"] = self._ellipse_to_lc10(hearts)
        features["item_lc10_grid"] = self._ellipse_to_lc10(items)
        features["other_pickup_lc10_grid"] = self._ellipse_to_lc10(other)

        features["world_doors"] = [
            dict(x)
            for x in self.doors
        ]

        features["world_pickups"] = [
            dict(x)
            for x in self.pickups
        ]

        features["coin_count"] = len(coins)
        features["bomb_pickup_count"] = len(bombs)
        features["key_count"] = len(keys)
        features["heart_pickup_count"] = len(hearts)
        features["item_pedestal_count"] = len(items)

        return features


WorldState = IsaacWorldState
