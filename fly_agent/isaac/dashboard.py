"""FlyBrain Isaac dashboard — room/combat source grid 7x13."""

__all__ = ["IsaacDashboard"]
DASHBOARD_GRID_MODE = "ROOM_7x13__LC10_12x20"

import pygame
import numpy as np


class IsaacDashboard:

    WIDTH = 1400
    HEIGHT = 1000

    ACTIONS = [
        ("W", "move_up"),
        ("A", "move_left"),
        ("S", "move_down"),
        ("D", "move_right"),

        ("UP", "shoot_up"),
        ("LEFT", "shoot_left"),
        ("DOWN", "shoot_down"),
        ("RIGHT", "shoot_right"),

        ("E", "bomb"),
        ("Q", "consumable"),
        ("SPACE", "active_item"),
    ]

    def __init__(self):

        pygame.init()

        self.screen = (
            pygame.display.set_mode(
                (
                    self.WIDTH,
                    self.HEIGHT,
                )
            )
        )

        pygame.display.set_caption(
            "FlyBrain - MaleCNS Isaac"
        )

        self.font_small = (
            pygame.font.SysFont(
                "consolas",
                16,
            )
        )

        self.font = (
            pygame.font.SysFont(
                "consolas",
                19,
            )
        )

        self.font_medium = (
            pygame.font.SysFont(
                "consolas",
                23,
                bold=True,
            )
        )

        self.font_big = (
            pygame.font.SysFont(
                "consolas",
                30,
                bold=True,
            )
        )

        self.running = True

    # =========================================================
    # HELPERS
    # =========================================================

    def _text(
        self,
        text,
        x,
        y,
        color=(225, 225, 225),
        font=None,
    ):

        if font is None:
            font = self.font

        surface = font.render(
            str(text),
            True,
            color,
        )

        self.screen.blit(
            surface,
            (
                int(x),
                int(y),
            ),
        )

    def _panel(
        self,
        rect,
        title,
    ):

        pygame.draw.rect(
            self.screen,
            (24, 26, 31),
            rect,
            border_radius=8,
        )

        pygame.draw.rect(
            self.screen,
            (65, 68, 78),
            rect,
            width=1,
            border_radius=8,
        )

        self._text(
            title,
            rect.x + 14,
            rect.y + 10,
            (245, 245, 245),
            self.font_medium,
        )

    @staticmethod
    def _resize_grid_nearest(
        values,
        rows,
        cols,
    ):

        if not isinstance(
            values,
            np.ndarray,
        ):

            return values

        array = np.asarray(
            values,
            dtype=np.float32,
        )

        if (
            array.ndim != 2
        ):

            return values

        if array.shape == (
            rows,
            cols,
        ):

            return array

        src_rows, src_cols = (
            array.shape
        )

        row_index = np.clip(
            np.round(
                (
                    np.arange(
                        rows,
                        dtype=np.float32,
                    )
                    + 0.5
                )
                * src_rows
                / rows
                - 0.5
            ).astype(
                np.int64
            ),
            0,
            src_rows - 1,
        )

        col_index = np.clip(
            np.round(
                (
                    np.arange(
                        cols,
                        dtype=np.float32,
                    )
                    + 0.5
                )
                * src_cols
                / cols
                - 0.5
            ).astype(
                np.int64
            ),
            0,
            src_cols - 1,
        )

        return array[
            np.ix_(
                row_index,
                col_index,
            )
        ]

    # =========================================================
    # FRAME
    # =========================================================

    def _draw_vision(
        self,
        rect,
        frame,
        features,
    ):

        self._panel(
            rect,
            "FLY VISION / SENSORY RETINA",
        )

        if frame is None:
            return

        rgb = (
            frame[
                :,
                :,
                ::-1
            ]
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        surface = (
            pygame.surfarray
            .make_surface(
                np.transpose(
                    rgb,
                    (
                        1,
                        0,
                        2,
                    ),
                )
            )
        )

        target = pygame.Rect(
            rect.x + 15,
            rect.y + 48,
            rect.width - 30,
            500,
        )

        source_w = (
            surface.get_width()
        )

        source_h = (
            surface.get_height()
        )

        scale = min(
            target.width / source_w,
            target.height / source_h,
        )

        width = int(
            source_w * scale
        )

        height = int(
            source_h * scale
        )

        image = (
            pygame.transform
            .smoothscale(
                surface,
                (
                    width,
                    height,
                ),
            )
        )

        image_rect = pygame.Rect(
            target.x
            + (
                target.width
                - width
            ) // 2,

            target.y
            + (
                target.height
                - height
            ) // 2,

            width,
            height,
        )

        self.screen.blit(
            image,
            image_rect,
        )

        # =====================================================
        # NORMAL / COMBINED SENSORY SALIENCE
        # =====================================================

        grid = features.get(
            "lc10_camera_grid"
        )

        if not isinstance(
            grid,
            np.ndarray,
        ):

            grid = features.get(
                "grid"
            )

        if isinstance(
            grid,
            np.ndarray,
        ):

            rows, cols = (
                grid.shape
            )

            overlay = pygame.Surface(
                (
                    image_rect.width,
                    image_rect.height,
                ),
                pygame.SRCALPHA,
            )

            cell_w = (
                image_rect.width
                / cols
            )

            cell_h = (
                image_rect.height
                / rows
            )

            for row in range(rows):

                for col in range(cols):

                    value = float(
                        grid[
                            row,
                            col
                        ]
                    )

                    if value < 0.06:
                        continue

                    alpha = int(
                        min(
                            120,
                            20
                            + value
                            * 95,
                        )
                    )

                    cell = pygame.Rect(
                        int(
                            col * cell_w
                        ),
                        int(
                            row * cell_h
                        ),
                        max(
                            1,
                            int(
                                cell_w + 1
                            ),
                        ),
                        max(
                            1,
                            int(
                                cell_h + 1
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        overlay,
                        (
                            255,
                            145,
                            40,
                            alpha,
                        ),
                        cell,
                    )

            self.screen.blit(
                overlay,
                image_rect,
            )

        # =====================================================
        # EXPLICIT ENEMY RETINA
        # =====================================================

        enemy_grid = features.get(
            "lc10_enemy_grid"
        )

        if not isinstance(
            enemy_grid,
            np.ndarray,
        ):

            enemy_grid = features.get(
                "enemy_grid"
            )

        if isinstance(
            enemy_grid,
            np.ndarray,
        ):

            enemy_rows, enemy_cols = (
                enemy_grid.shape
            )

            enemy_overlay = pygame.Surface(
                (
                    image_rect.width,
                    image_rect.height,
                ),
                pygame.SRCALPHA,
            )

            enemy_cell_w = (
                image_rect.width
                / enemy_cols
            )

            enemy_cell_h = (
                image_rect.height
                / enemy_rows
            )

            for row in range(
                enemy_rows
            ):

                for col in range(
                    enemy_cols
                ):

                    value = float(
                        enemy_grid[
                            row,
                            col
                        ]
                    )

                    if value <= 0.05:
                        continue

                    alpha = int(
                        min(
                            220,
                            55
                            + value
                            * 165,
                        )
                    )

                    cell = pygame.Rect(
                        int(
                            col
                            * enemy_cell_w
                        ),
                        int(
                            row
                            * enemy_cell_h
                        ),
                        max(
                            1,
                            int(
                                enemy_cell_w + 1
                            ),
                        ),
                        max(
                            1,
                            int(
                                enemy_cell_h + 1
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        enemy_overlay,
                        (
                            235,
                            55,
                            70,
                            alpha,
                        ),
                        cell,
                    )

                    # Strong center mark for actual enemy cells.
                    if value >= 0.75:

                        center = (
                            int(
                                (
                                    col + 0.5
                                )
                                * enemy_cell_w
                            ),
                            int(
                                (
                                    row + 0.5
                                )
                                * enemy_cell_h
                            ),
                        )

                        pygame.draw.circle(
                            enemy_overlay,
                            (
                                255,
                                220,
                                220,
                                235,
                            ),
                            center,
                            max(
                                3,
                                int(
                                    min(
                                        enemy_cell_w,
                                        enemy_cell_h,
                                    )
                                    * 0.16
                                ),
                            ),
                        )

            self.screen.blit(
                enemy_overlay,
                image_rect,
            )

        # =====================================================
        # SELF HITBOX RETINA
        # =====================================================

        self_grid = features.get(
            "lc10_self_grid"
        )

        if not isinstance(
            self_grid,
            np.ndarray,
        ):

            self_grid = features.get(
                "self_grid"
            )

        if isinstance(
            self_grid,
            np.ndarray,
        ):

            self_rows, self_cols = (
                self_grid.shape
            )

            self_overlay = pygame.Surface(
                (
                    image_rect.width,
                    image_rect.height,
                ),
                pygame.SRCALPHA,
            )

            self_cell_w = (
                image_rect.width
                / self_cols
            )

            self_cell_h = (
                image_rect.height
                / self_rows
            )

            for row in range(
                self_rows
            ):

                for col in range(
                    self_cols
                ):

                    value = float(
                        self_grid[
                            row,
                            col
                        ]
                    )

                    if value <= 0.05:
                        continue

                    alpha = int(
                        min(
                            170,
                            35
                            + value
                            * 120,
                        )
                    )

                    cell = pygame.Rect(
                        int(
                            col
                            * self_cell_w
                        ),
                        int(
                            row
                            * self_cell_h
                        ),
                        max(
                            1,
                            int(
                                self_cell_w + 1
                            ),
                        ),
                        max(
                            1,
                            int(
                                self_cell_h + 1
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        self_overlay,
                        (
                            70,
                            220,
                            245,
                            alpha,
                        ),
                        cell,
                    )

            self.screen.blit(
                self_overlay,
                image_rect,
            )

        # =====================================================
        # EXACT SPATIAL CELLS ACTUALLY INJECTED INTO LC10
        # =====================================================

        injected_grid = features.get(
            "lc10_injected_grid"
        )

        if isinstance(
            injected_grid,
            np.ndarray,
        ):

            inj_rows, inj_cols = (
                injected_grid.shape
            )

            inj_cell_w = (
                image_rect.width
                / inj_cols
            )

            inj_cell_h = (
                image_rect.height
                / inj_rows
            )

            for row in range(
                inj_rows
            ):

                for col in range(
                    inj_cols
                ):

                    if (
                        float(
                            injected_grid[
                                row,
                                col
                            ]
                        )
                        <= 0.0
                    ):
                        continue

                    cell = pygame.Rect(
                        image_rect.x
                        + int(
                            col
                            * inj_cell_w
                        ),
                        image_rect.y
                        + int(
                            row
                            * inj_cell_h
                        ),
                        max(
                            1,
                            int(
                                inj_cell_w
                            ),
                        ),
                        max(
                            1,
                            int(
                                inj_cell_h
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        self.screen,
                        (
                            250,
                            250,
                            250,
                        ),
                        cell,
                        width=2,
                    )

        # =====================================================
        # LUA PLAYER POSITION
        # =====================================================

        player_x = features.get(
            "enemy_player_x"
        )

        player_y = features.get(
            "enemy_player_y"
        )

        if (
            player_x is not None
            and player_y is not None
        ):

            try:

                player_x = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            player_x
                        ),
                    ),
                )

                player_y = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            player_y
                        ),
                    ),
                )

                px = int(
                    image_rect.x
                    + player_x
                    * image_rect.width
                )

                py = int(
                    image_rect.y
                    + player_y
                    * image_rect.height
                )

                pygame.draw.circle(
                    self.screen,
                    (
                        80,
                        225,
                        245,
                    ),
                    (
                        px,
                        py,
                    ),
                    8,
                    width=2,
                )

                pygame.draw.line(
                    self.screen,
                    (
                        80,
                        225,
                        245,
                    ),
                    (
                        px - 11,
                        py,
                    ),
                    (
                        px + 11,
                        py,
                    ),
                    width=1,
                )

                pygame.draw.line(
                    self.screen,
                    (
                        80,
                        225,
                        245,
                    ),
                    (
                        px,
                        py - 11,
                    ),
                    (
                        px,
                        py + 11,
                    ),
                    width=1,
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        # =====================================================
        # EXACT GAME COLLISION HITBOXES (COMBAT BRIDGE V4)
        # =====================================================
        #
        # These thin ellipses are NOT the coarse LC10 cells.
        # They are Isaac's physical Entity.Size * SizeMulti hitboxes,
        # converted by Lua with WorldToScreen().
        #
        # Cyan outline = exact player hitbox
        # Red outline  = exact enemy hitbox
        # =====================================================

        def draw_exact_hitbox(
            sx,
            sy,
            srx,
            sry,
            color,
        ):
            try:
                sx = float(
                    sx
                )

                sy = float(
                    sy
                )

                srx = abs(
                    float(
                        srx
                    )
                )

                sry = abs(
                    float(
                        sry
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                return

            cx = (
                image_rect.x
                + sx
                * image_rect.width
            )

            cy = (
                image_rect.y
                + sy
                * image_rect.height
            )

            rx = max(
                1.0,
                srx
                * image_rect.width,
            )

            ry = max(
                1.0,
                sry
                * image_rect.height,
            )

            hitbox_rect = pygame.Rect(
                int(
                    round(
                        cx - rx
                    )
                ),
                int(
                    round(
                        cy - ry
                    )
                ),
                max(
                    2,
                    int(
                        round(
                            rx * 2.0
                        )
                    ),
                ),
                max(
                    2,
                    int(
                        round(
                            ry * 2.0
                        )
                    ),
                ),
            )

            pygame.draw.ellipse(
                self.screen,
                color,
                hitbox_rect,
                width=2,
            )

            pygame.draw.circle(
                self.screen,
                color,
                (
                    int(
                        round(
                            cx
                        )
                    ),
                    int(
                        round(
                            cy
                        )
                    ),
                ),
                2,
            )

        # Player exact hitbox.
        draw_exact_hitbox(
            features.get(
                "enemy_player_screen_x"
            ),
            features.get(
                "enemy_player_screen_y"
            ),
            features.get(
                "enemy_player_screen_rx"
            ),
            features.get(
                "enemy_player_screen_ry"
            ),
            (
                70,
                235,
                255,
            ),
        )

        # Enemy exact hitboxes.
        for enemy in features.get(
            "enemy_hitboxes",
            [],
        ):
            if not isinstance(
                enemy,
                dict,
            ):
                continue

            draw_exact_hitbox(
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
                (
                    255,
                    65,
                    75,
                ),
            )

        # =====================================================
        # EXACT CIRCLE -> LC10 CELL HIGHLIGHT
        # =====================================================
        #
        # Red cells  = LC10 cells physically touched by enemy ellipse
        # Cyan cells = LC10 cells physically touched by Isaac ellipse
        #
        # These are the SAME 12x20 grids LC10 now uses for enemy/self.
        # =====================================================

        exact_enemy_cells = features.get(
            "enemy_lc10_hitbox_grid"
        )

        exact_self_cells = features.get(
            "self_lc10_hitbox_grid"
        )

        def draw_hitbox_cells(
            cell_grid,
            color,
        ):
            if (
                not isinstance(
                    cell_grid,
                    np.ndarray,
                )
                or cell_grid.shape
                != (
                    8,
                    14,
                )
            ):
                return

            overlay = pygame.Surface(
                (
                    image_rect.width,
                    image_rect.height,
                ),
                pygame.SRCALPHA,
            )

            cell_w = (
                image_rect.width
                / 20.0
            )

            cell_h = (
                image_rect.height
                / 12.0
            )

            for row in range(
                8
            ):
                for col in range(
                    14
                ):
                    value = float(
                        cell_grid[
                            row,
                            col
                        ]
                    )

                    if value <= 0.0:
                        continue

                    alpha = int(
                        35
                        + 85
                        * min(
                            1.0,
                            value,
                        )
                    )

                    cell = pygame.Rect(
                        int(
                            col
                            * cell_w
                        ),
                        int(
                            row
                            * cell_h
                        ),
                        max(
                            1,
                            int(
                                cell_w + 1
                            ),
                        ),
                        max(
                            1,
                            int(
                                cell_h + 1
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        overlay,
                        (
                            color[0],
                            color[1],
                            color[2],
                            alpha,
                        ),
                        cell,
                    )

                    pygame.draw.rect(
                        overlay,
                        (
                            color[0],
                            color[1],
                            color[2],
                            220,
                        ),
                        cell,
                        width=2,
                    )

            self.screen.blit(
                overlay,
                image_rect,
            )

        draw_hitbox_cells(
            exact_enemy_cells,
            (
                255,
                70,
                80,
            ),
        )

        draw_hitbox_cells(
            exact_self_cells,
            (
                70,
                235,
                255,
            ),
        )

        # =====================================================
        # SEMANTIC WORLD CELLS
        # =====================================================
        #
        # These overlays are only for us to verify the sensory encoding.
        # MaleCNS receives the same underlying grids through LC10.
        #
        # door   = blue
        # hazard = magenta
        # coin   = yellow
        # bomb   = orange
        # key    = light blue
        # heart  = pink
        # item   = green
        # =====================================================

        semantic_specs = (
            (
                "lc10_projectile_grid",
                "projectile_lc10_grid",
                (
                    255,
                    95,
                    235,
                ),
            ),
            (
                "lc10_projectile_future_grid",
                "projectile_future_lc10_grid",
                (
                    185,
                    110,
                    255,
                ),
            ),
            (
                "lc10_treasure_door_grid",
                "treasure_door_grid",
                (
                    255,
                    215,
                    55,
                ),
            ),
            (
                "lc10_shop_door_grid",
                "shop_door_grid",
                (
                    80,
                    220,
                    150,
                ),
            ),
            (
                "lc10_boss_door_grid",
                "boss_door_grid",
                (
                    215,
                    70,
                    70,
                ),
            ),
            (
                "lc10_devil_door_grid",
                "devil_door_grid",
                (
                    155,
                    70,
                    190,
                ),
            ),
            (
                "lc10_angel_door_grid",
                "angel_door_grid",
                (
                    245,
                    245,
                    245,
                ),
            ),
            (
                "lc10_locked_door_grid",
                "locked_door_grid_exact",
                (
                    255,
                    75,
                    75,
                ),
            ),
            (
                "lc10_unexplored_door_grid",
                "unexplored_door_grid_exact",
                (
                    255,
                    205,
                    70,
                ),
            ),
            (
                "lc10_door_grid",
                "door_grid_exact",
                (
                    75,
                    155,
                    255,
                ),
            ),
            (
                "lc10_hazard_grid",
                "hazard_grid",
                (
                    220,
                    80,
                    220,
                ),
            ),
            (
                "lc10_coin_grid",
                "coin_lc10_grid",
                (
                    245,
                    215,
                    55,
                ),
            ),
            (
                "lc10_bomb_pickup_grid",
                "bomb_pickup_lc10_grid",
                (
                    245,
                    135,
                    55,
                ),
            ),
            (
                "lc10_key_grid",
                "key_lc10_grid",
                (
                    110,
                    205,
                    255,
                ),
            ),
            (
                "lc10_heart_grid",
                "heart_lc10_grid",
                (
                    255,
                    120,
                    170,
                ),
            ),
            (
                "lc10_item_grid",
                "item_lc10_grid",
                (
                    110,
                    235,
                    120,
                ),
            ),
        )

        for (
            lc10_name,
            feature_name,
            semantic_color,
        ) in semantic_specs:

            semantic_grid = features.get(
                lc10_name
            )

            if not isinstance(
                semantic_grid,
                np.ndarray,
            ):
                semantic_grid = features.get(
                    feature_name
                )

            if (
                not isinstance(
                    semantic_grid,
                    np.ndarray,
                )
                or semantic_grid.ndim != 2
            ):
                continue

            # Source-sized room grids are only used here if they already
            # happen to be 12x20; otherwise LC10/debug features should supply
            # the converted 12x20 version.
            if semantic_grid.shape != (
                12,
                20,
            ):
                continue

            semantic_overlay = pygame.Surface(
                (
                    image_rect.width,
                    image_rect.height,
                ),
                pygame.SRCALPHA,
            )

            semantic_cell_w = (
                image_rect.width
                / 20.0
            )

            semantic_cell_h = (
                image_rect.height
                / 12.0
            )

            for row in range(12):
                for col in range(20):
                    value = float(
                        semantic_grid[
                            row,
                            col
                        ]
                    )

                    if value <= 0.05:
                        continue

                    cell = pygame.Rect(
                        int(
                            col
                            * semantic_cell_w
                        ),
                        int(
                            row
                            * semantic_cell_h
                        ),
                        max(
                            1,
                            int(
                                semantic_cell_w
                                + 1
                            ),
                        ),
                        max(
                            1,
                            int(
                                semantic_cell_h
                                + 1
                            ),
                        ),
                    )

                    pygame.draw.rect(
                        semantic_overlay,
                        (
                            semantic_color[0],
                            semantic_color[1],
                            semantic_color[2],
                            55,
                        ),
                        cell,
                    )

                    pygame.draw.rect(
                        semantic_overlay,
                        (
                            semantic_color[0],
                            semantic_color[1],
                            semantic_color[2],
                            220,
                        ),
                        cell,
                        width=2,
                    )

            self.screen.blit(
                semantic_overlay,
                image_rect,
            )

        # =====================================================
        # LC10 GRID LINES
        # =====================================================

        brain_grid = features.get(
            "lc10_combined_grid"
        )

        if isinstance(
            brain_grid,
            np.ndarray,
        ):

            brain_rows, brain_cols = (
                brain_grid.shape
            )

            for col in range(
                1,
                brain_cols,
            ):

                x = int(
                    image_rect.x
                    + col
                    * image_rect.width
                    / brain_cols
                )

                pygame.draw.line(
                    self.screen,
                    (
                        90,
                        95,
                        105,
                    ),
                    (
                        x,
                        image_rect.y,
                    ),
                    (
                        x,
                        image_rect.bottom,
                    ),
                    width=1,
                )

            for row in range(
                1,
                brain_rows,
            ):

                y_line = int(
                    image_rect.y
                    + row
                    * image_rect.height
                    / brain_rows
                )

                pygame.draw.line(
                    self.screen,
                    (
                        90,
                        95,
                        105,
                    ),
                    (
                        image_rect.x,
                        y_line,
                    ),
                    (
                        image_rect.right,
                        y_line,
                    ),
                    width=1,
                )

        # =====================================================
        # METRICS
        # =====================================================

        global_motion = float(
            features.get(
                "global_motion",
                0.0,
            )
        )

        mean_salience = 0.0
        peak_salience = 0.0

        if isinstance(
            grid,
            np.ndarray,
        ):

            mean_salience = float(
                np.mean(
                    grid
                )
            )

            peak_salience = float(
                np.max(
                    grid
                )
            )

        enemies = int(
            features.get(
                "enemy_count",
                0,
            )
        )

        alive = int(
            features.get(
                "enemies_alive",
                enemies,
            )
        )

        y = (
            rect.bottom - 91
        )

        self._text(
            (
                f"Motion {global_motion:.3f}   "
                f"Mean {mean_salience:.3f}   "
                f"Peak {peak_salience:.3f}   "
                f"Enemies {enemies}/{alive}"
            ),
            rect.x + 18,
            y,
            (
                190,
                195,
                205,
            ),
        )

        self._text(
            (
                "LC10 12x20   blocks=sensory cells   "
                "thin ellipse=REAL hitbox   filled red/cyan cells=LC10 cells touched by it   "
                "white=actual LC10 injection"
            ),
            rect.x + 18,
            y + 25,
            (
                145,
                150,
                160,
            ),
            self.font_small,
        )

        map_status = str(
            features.get(
                "debug_map_status",
                "WAIT",
            )
        )

        enemy_status = str(
            features.get(
                "debug_enemy_status",
                "WAIT",
            )
        )

        self_status = str(
            features.get(
                "debug_self_status",
                "WAIT",
            )
        )

        all_proven = (
            map_status == "PROVEN"
            and enemy_status == "PROVEN"
            and self_status == "PROVEN"
        )

        audit_color = (
            (
                100,
                225,
                125,
            )
            if all_proven
            else (
                235,
                185,
                80,
            )
        )

        self._text(
            (
                f"LC10 AUDIT  "
                f"MAP {map_status}   "
                f"ENEMY {enemy_status}   "
                f"SELF {self_status}"
            ),
            rect.x + 18,
            y + 49,
            audit_color,
            self.font_small,
        )

    # =========================================================
    # ROOM COLLISION MAP
    # =========================================================

    def _draw_collision_map(
        self,
        rect,
        features,
    ):
        """
        Draw the ORIGINAL dynamic source-room grid.

        Normal room      -> 7x13
        Tall room        -> 14x13
        Wide room        -> 7x26
        2x2 / L room     -> 14x26
        Narrow I-shapes  -> their native smaller source grid

        This method deliberately does NOT use lc10_map_grid, because LC10
        is a separate fixed 12x20 representation.
        """

        collision = features.get(
            "collision_grid"
        )

        if not isinstance(
            collision,
            np.ndarray,
        ) or collision.ndim != 2:

            self._text(
                "Waiting for room-state bridge...",
                rect.x + 10,
                rect.y + 10,
                (
                    200,
                    160,
                    120,
                ),
            )

            return

        rows, cols = (
            collision.shape
        )

        if rows <= 0 or cols <= 0:
            return

        # Preserve square tile proportions and center the complete room.
        cell_size = min(
            rect.width
            / float(
                cols
            ),
            rect.height
            / float(
                rows
            ),
        )

        map_w = (
            cell_size
            * cols
        )

        map_h = (
            cell_size
            * rows
        )

        offset_x = (
            rect.x
            + (
                rect.width
                - map_w
            )
            / 2.0
        )

        offset_y = (
            rect.y
            + (
                rect.height
                - map_h
            )
            / 2.0
        )

        def source_grid(
            name,
        ):
            value = features.get(
                name
            )

            if not isinstance(
                value,
                np.ndarray,
            ):
                return None

            if value.shape == collision.shape:
                return value

            return self._resize_grid_nearest(
                value,
                rows,
                cols,
            )

        doors = source_grid(
            "door_grid"
        )

        unexplored = source_grid(
            "unexplored_door_grid"
        )

        locked = source_grid(
            "locked_door_grid"
        )

        enemy_grid = source_grid(
            "enemy_grid"
        )

        self_grid = source_grid(
            "self_grid"
        )

        valid_mask = source_grid(
            "room_valid_mask"
        )

        for row in range(
            rows
        ):
            for col in range(
                cols
            ):
                valid = True

                if isinstance(
                    valid_mask,
                    np.ndarray,
                ):
                    valid = (
                        float(
                            valid_mask[
                                row,
                                col
                            ]
                        )
                        > 0.5
                    )

                value = float(
                    collision[
                        row,
                        col
                    ]
                )

                if not valid:
                    # Missing quadrant of an L-shaped room.
                    color = (
                        12,
                        13,
                        17,
                    )

                elif value >= 0.90:
                    color = (
                        95,
                        98,
                        108,
                    )

                elif value >= 0.60:
                    color = (
                        83,
                        66,
                        94,
                    )

                elif value > 0.05:
                    color = (
                        70,
                        63,
                        72,
                    )

                else:
                    color = (
                        30,
                        33,
                        39,
                    )

                x0 = int(
                    offset_x
                    + col
                    * cell_size
                )

                y0 = int(
                    offset_y
                    + row
                    * cell_size
                )

                x1 = int(
                    offset_x
                    + (
                        col + 1
                    )
                    * cell_size
                )

                y1 = int(
                    offset_y
                    + (
                        row + 1
                    )
                    * cell_size
                )

                cell = pygame.Rect(
                    x0,
                    y0,
                    max(
                        1,
                        x1 - x0,
                    ),
                    max(
                        1,
                        y1 - y0,
                    ),
                )

                pygame.draw.rect(
                    self.screen,
                    color,
                    cell,
                )

                pygame.draw.rect(
                    self.screen,
                    (
                        47,
                        49,
                        57,
                    ),
                    cell,
                    width=1,
                )

                if not valid:
                    # Cross hatch the nonexistent L-room quadrant.
                    pygame.draw.line(
                        self.screen,
                        (
                            35,
                            37,
                            44,
                        ),
                        cell.topleft,
                        cell.bottomright,
                        width=1,
                    )
                    continue

                # ---------------------------------------------
                # DOORS
                # ---------------------------------------------

                if (
                    isinstance(
                        doors,
                        np.ndarray,
                    )
                    and doors[
                        row,
                        col
                    ] > 0.5
                ):
                    door_color = (
                        85,
                        190,
                        220,
                    )

                    if (
                        isinstance(
                            unexplored,
                            np.ndarray,
                        )
                        and unexplored[
                            row,
                            col
                        ] > 0.5
                    ):
                        door_color = (
                            255,
                            190,
                            55,
                        )

                    if (
                        isinstance(
                            locked,
                            np.ndarray,
                        )
                        and locked[
                            row,
                            col
                        ] > 0.5
                    ):
                        door_color = (
                            220,
                            75,
                            75,
                        )

                    pygame.draw.rect(
                        self.screen,
                        door_color,
                        cell,
                        width=3,
                    )

                # ---------------------------------------------
                # ENEMIES
                # ---------------------------------------------

                if isinstance(
                    enemy_grid,
                    np.ndarray,
                ):
                    enemy_value = float(
                        enemy_grid[
                            row,
                            col
                        ]
                    )

                    if enemy_value > 0.05:
                        center_x = int(
                            offset_x
                            + (
                                col + 0.5
                            )
                            * cell_size
                        )

                        center_y = int(
                            offset_y
                            + (
                                row + 0.5
                            )
                            * cell_size
                        )

                        radius = max(
                            2,
                            int(
                                cell_size
                                * (
                                    0.16
                                    + 0.14
                                    * min(
                                        1.0,
                                        enemy_value,
                                    )
                                )
                            ),
                        )

                        pygame.draw.circle(
                            self.screen,
                            (
                                245,
                                70,
                                80,
                            ),
                            (
                                center_x,
                                center_y,
                            ),
                            radius,
                        )

        # =====================================================
        # PLAYER / SELF ON NATIVE SOURCE MAP
        # =====================================================

        player_row = -1
        player_col = -1

        if (
            isinstance(
                self_grid,
                np.ndarray,
            )
            and self_grid.shape
            == collision.shape
            and np.any(
                self_grid > 0.0
            )
        ):
            flat_index = int(
                np.argmax(
                    self_grid
                )
            )

            (
                player_row,
                player_col,
            ) = np.unravel_index(
                flat_index,
                self_grid.shape,
            )

        else:
            try:
                player_row = int(
                    features.get(
                        "player_retina_row",
                        -1,
                    )
                )

                player_col = int(
                    features.get(
                        "player_retina_col",
                        -1,
                    )
                )

            except (
                TypeError,
                ValueError,
            ):
                player_row = -1
                player_col = -1

        if (
            0
            <= player_row
            < rows
            and 0
            <= player_col
            < cols
        ):
            px = int(
                offset_x
                + (
                    player_col + 0.5
                )
                * cell_size
            )

            py = int(
                offset_y
                + (
                    player_row + 0.5
                )
                * cell_size
            )

            pygame.draw.circle(
                self.screen,
                (
                    100,
                    245,
                    130,
                ),
                (
                    px,
                    py,
                ),
                max(
                    3,
                    int(
                        cell_size
                        * 0.26
                    ),
                ),
            )

        # Small shape label inside the map panel.
        shape_name = str(
            features.get(
                "room_shape_name",
                "?",
            )
        )

        self._text(
            (
                f"{shape_name}  "
                f"{rows}x{cols}"
            ),
            rect.x + 6,
            rect.y + 5,
            (
                175,
                180,
                190,
            ),
            self.font_small,
        )

    # =========================================================
    # FLOOR LAYOUT
    # =========================================================

    def _draw_floor_layout(
        self,
        rect,
        features,
    ):

        known = set(
            features.get(
                "known_room_indices",
                [],
            )
        )

        visited = set(
            features.get(
                "visited_room_indices",
                [],
            )
        )

        current = int(
            features.get(
                "room_index",
                -1,
            )
        )

        # Isaac's normal level room-index grid is 13 x 13.
        cols = 13
        rows = 13

        cell = min(
            rect.width / cols,
            rect.height / rows,
        )

        offset_x = (
            rect.x
            + (
                rect.width
                - cell * cols
            ) / 2
        )

        offset_y = (
            rect.y
            + (
                rect.height
                - cell * rows
            ) / 2
        )

        for room in known:

            if (
                room < 0
                or room >= 169
            ):
                continue

            row = (
                room // 13
            )

            col = (
                room % 13
            )

            box = pygame.Rect(
                int(
                    offset_x
                    + col * cell
                ),
                int(
                    offset_y
                    + row * cell
                ),
                max(
                    3,
                    int(cell - 1),
                ),
                max(
                    3,
                    int(cell - 1),
                ),
            )

            if room == current:

                color = (
                    100,
                    245,
                    130,
                )

            elif room in visited:

                color = (
                    70,
                    150,
                    190,
                )

            else:

                color = (
                    220,
                    175,
                    55,
                )

            pygame.draw.rect(
                self.screen,
                color,
                box,
                border_radius=2,
            )

    # =========================================================
    # ROOM PANEL
    # =========================================================

    def _draw_room(
        self,
        rect,
        features,
    ):

        self._panel(
            rect,
            "ROOM / COMBAT STATE — DYNAMIC",
        )

        stage = features.get(
            "room_stage",
            "-",
        )

        room = features.get(
            "room_index",
            -1,
        )

        previous = features.get(
            "previous_room",
            -1,
        )

        visited = features.get(
            "rooms_visited",
            0,
        )

        known = features.get(
            "rooms_known",
            0,
        )

        enemy_count = int(
            features.get(
                "enemy_count",
                0,
            )
        )

        enemies_alive = int(
            features.get(
                "enemies_alive",
                enemy_count,
            )
        )

        self._text(
            (
                f"Floor {stage}    "
                f"Room {room}    "
                f"Previous {previous}"
            ),
            rect.x + 15,
            rect.y + 48,
        )

        self._text(
            (
                f"Visited {visited}    "
                f"Known {known}"
            ),
            rect.x + 15,
            rect.y + 76,
            (
                175,
                185,
                200,
            ),
        )

        # =====================================================
        # COMBAT SENSOR SUMMARY
        # =====================================================

        enemy_color = (
            (
                245,
                90,
                90,
            )
            if enemies_alive > 0
            else (
                120,
                185,
                135,
            )
        )

        self._text(
            (
                f"Enemies visible {enemy_count}   "
                f"alive {enemies_alive}"
            ),
            rect.x + 15,
            rect.y + 101,
            enemy_color,
            self.font_small,
        )

        projectile_count = features.get(
            "projectile_count"
        )

        projectile_distance = features.get(
            "projectile_distance"
        )

        if (
            projectile_count is not None
            or projectile_distance is not None
        ):

            try:
                p_count = int(
                    projectile_count
                    if projectile_count is not None
                    else 0
                )
            except (
                TypeError,
                ValueError,
            ):
                p_count = 0

            try:
                p_distance = float(
                    projectile_distance
                    if projectile_distance is not None
                    else -1.0
                )
            except (
                TypeError,
                ValueError,
            ):
                p_distance = -1.0

            projectile_text = (
                f"Projectiles {p_count}   "
                f"nearest {p_distance:.1f}"
            )

        else:

            projectile_text = (
                "Projectiles: combat state available when "
                "added to features"
            )

        self._text(
            projectile_text,
            rect.x + 15,
            rect.y + 123,
            (
                165,
                170,
                185,
            ),
            self.font_small,
        )

        # =====================================================
        # ROOM GEOMETRY
        # =====================================================

        collision_rect = pygame.Rect(
            rect.x + 15,
            rect.y + 150,
            rect.width - 30,
            240,
        )

        self._draw_collision_map(
            collision_rect,
            features,
        )

        source_collision = features.get(
            "collision_grid"
        )

        if isinstance(
            source_collision,
            np.ndarray,
        ):
            source_shape_text = (
                f"{source_collision.shape[0]}x"
                f"{source_collision.shape[1]}"
            )
        else:
            source_shape_text = "?"

        self._text(
            (
                f"ROOM MAP {source_shape_text}   "
                f"coins={int(features.get('coin_count', 0))}  "
                f"bombs={int(features.get('bomb_pickup_count', 0))}  "
                f"keys={int(features.get('key_count', 0))}  "
                f"items={int(features.get('item_pedestal_count', 0))}"
            ),
            rect.x + 15,
            rect.y + 396,
            (
                150,
                155,
                170,
            ),
            self.font_small,
        )

        sensors = (
            f"Touch  "
            f"U {float(features.get('collision_up', 0)):.2f}   "
            f"D {float(features.get('collision_down', 0)):.2f}   "
            f"L {float(features.get('collision_left', 0)):.2f}   "
            f"R {float(features.get('collision_right', 0)):.2f}"
        )

        self._text(
            sensors,
            rect.x + 15,
            rect.y + 420,
            (
                210,
                210,
                220,
            ),
            self.font_small,
        )

        floor_rect = pygame.Rect(
            rect.x + 15,
            rect.y + 468,
            225,
            145,
        )

        self._draw_floor_layout(
            floor_rect,
            features,
        )

        self._text(
            "KNOWN FLOOR",
            floor_rect.x,
            floor_rect.y - 22,
            (
                165,
                170,
                180,
            ),
            self.font_small,
        )

        door_x = (
            rect.x + 255
        )

        door_y = (
            rect.y + 449
        )

        self._text(
            "DOORS",
            door_x,
            door_y,
            (
                165,
                170,
                180,
            ),
            self.font_small,
        )

        door_y += 24

        door_info = features.get(
            "door_info",
            [],
        )

        if not door_info:

            self._text(
                "none",
                door_x,
                door_y,
                (
                    130,
                    135,
                    145,
                ),
                self.font_small,
            )

        for door in door_info[
            :7
        ]:

            status = (
                "VISITED"
                if door.get(
                    "visited"
                )
                else "NEW"
            )

            if door.get(
                "locked"
            ):

                status = (
                    "LOCKED"
                )

            elif door.get(
                "open"
            ):

                status += (
                    "/OPEN"
                )

            text = (
                f"{door.get('direction', '?').upper():5s}"
                f" -> "
                f"{door.get('target', -1):3d} "
                f"{status}"
            )

            color = (
                (
                    100,
                    220,
                    130,
                )
                if door.get(
                    "visited"
                )
                else (
                    245,
                    190,
                    60,
                )
            )

            if door.get(
                "locked"
            ):

                color = (
                    230,
                    90,
                    90,
                )

            self._text(
                text,
                door_x,
                door_y,
                color,
                self.font_small,
            )

            door_y += 22

    # =========================================================
    # ACTION PANEL
    # =========================================================

    def _draw_actions(
        self,
        rect,
        held,
        explore_action,
        fired_count,
        step,
        elapsed,
    ):

        self._panel(
            rect,
            "MALECNS / CONTROLS",
        )

        hz = (
            step / elapsed
            if elapsed > 0
            else 0.0
        )

        self._text(
            (
                f"Step {step:,}   "
                f"Fired {fired_count:,}   "
                f"~{hz:.1f} Hz"
            ),
            rect.x + 14,
            rect.y + 46,
            (185, 195, 210),
            self.font_small,
        )

        held = set(
            held
        )

        start_y = (
            rect.y + 80
        )

        for index, (
            key,
            action,
        ) in enumerate(
            self.ACTIONS
        ):

            col = (
                index % 2
            )

            row = (
                index // 2
            )

            x = (
                rect.x
                + 14
                + col * 185
            )

            y = (
                start_y
                + row * 34
            )

            active = (
                action in held
            )

            exploring = (
                action
                == explore_action
            )

            if active:

                color = (
                    80,
                    225,
                    115,
                )

            elif exploring:

                color = (
                    245,
                    165,
                    55,
                )

            else:

                color = (
                    120,
                    125,
                    135,
                )

            self._text(
                f"{key:5s} {action}",
                x,
                y,
                color,
                self.font_small,
            )

    # =========================================================
    # REWARDS
    # =========================================================

    def _draw_rewards(
        self,
        rect,
        recent_events,
        total_reward,
    ):

        self._panel(
            rect,
            "REWARD / LEARNING EVENTS",
        )

        reward_color = (
            (
                100,
                225,
                120,
            )
            if total_reward >= 0
            else (
                230,
                95,
                95,
            )
        )

        self._text(
            f"Total reward: {total_reward:+.2f}",
            rect.x + 14,
            rect.y + 45,
            reward_color,
        )

        # Small combat-learning summary from the visible history.

        visible = list(
            recent_events[
                -12:
            ]
        )

        hits = sum(
            1
            for event in visible
            if event.get(
                "name"
            )
            == "SHOT_HIT"
        )

        misses = sum(
            1
            for event in visible
            if event.get(
                "name"
            )
            == "SHOT_MISS"
        )

        empty_shots = sum(
            1
            for event in visible
            if event.get(
                "name"
            )
            == "NO_ENEMY_SHOT"
        )

        projectile_events = sum(
            1
            for event in visible
            if event.get(
                "name"
            )
            == "PROJECTILE_TOWARD"
        )

        self._text(
            (
                f"Recent: hit {hits}  miss {misses}  "
                f"empty {empty_shots}  danger {projectile_events}"
            ),
            rect.x + 14,
            rect.y + 70,
            (
                150,
                160,
                178,
            ),
            self.font_small,
        )

        y = (
            rect.y + 100
        )

        for event in (
            recent_events[
                -7:
            ]
        ):

            name = str(
                event.get(
                    "name",
                    "?",
                )
            )

            reward = float(
                event.get(
                    "reward",
                    0.0,
                )
            )

            action = str(
                event.get(
                    "action",
                    "",
                )
            )

            extra = str(
                event.get(
                    "extra",
                    "",
                )
            )

            if reward > 0:

                color = (
                    100,
                    220,
                    120,
                )

            elif reward < 0:

                color = (
                    235,
                    95,
                    95,
                )

            else:

                color = (
                    150,
                    155,
                    165,
                )

            # Combat events get an additional bright name marker.
            if name in {
                "SHOT_HIT",
                "SHOT_MISS",
                "NO_ENEMY_SHOT",
                "PROJECTILE_TOWARD",
            }:

                name_color = (
                    235,
                    170,
                    95,
                )

            elif name in {
                "WALL_STUCK",
                "WALL_ESCAPE",
            }:

                name_color = (
                    110,
                    190,
                    235,
                )

            else:

                name_color = color

            self._text(
                f"{name[:16]:16s}",
                rect.x + 14,
                y,
                name_color,
                self.font_small,
            )

            self._text(
                f"{reward:+5.2f}",
                rect.x + 158,
                y,
                color,
                self.font_small,
            )

            if action:

                self._text(
                    action[:13],
                    rect.x + 220,
                    y,
                    (
                        205,
                        205,
                        215,
                    ),
                    self.font_small,
                )

            elif extra:

                self._text(
                    extra[:16],
                    rect.x + 220,
                    y,
                    (
                        135,
                        140,
                        150,
                    ),
                    self.font_small,
                )

            y += 26

    # =========================================================
    # WEIGHTS
    # =========================================================

    def _draw_weights(
        self,
        rect,
        multipliers,
    ):

        self._panel(
            rect,
            "REAL SYNAPTIC MULTIPLIERS",
        )

        items = list(
            multipliers.items()
        )

        start_y = (
            rect.y + 50
        )

        for index, (
            action,
            value,
        ) in enumerate(items):

            col = (
                index % 2
            )

            row = (
                index // 2
            )

            x = (
                rect.x
                + 14
                + col * 235
            )

            y = (
                start_y
                + row * 34
            )

            delta = (
                value - 1.0
            )

            if delta > 0.001:

                color = (
                    95,
                    220,
                    120,
                )

            elif delta < -0.001:

                color = (
                    230,
                    100,
                    100,
                )

            else:

                color = (
                    180,
                    185,
                    195,
                )

            self._text(
                (
                    f"{action:12s} "
                    f"{value:.4f}"
                ),
                x,
                y,
                color,
                self.font_small,
            )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        frame,
        features,
        left_motion,
        right_motion,
        left_drive,
        right_drive,
        held,
        explore_action,
        recent_events,
        total_reward,
        multipliers,
        fired_count,
        step,
        elapsed,
    ):

        if not self.running:
            return False

        for event in pygame.event.get():

            if (
                event.type
                == pygame.QUIT
            ):

                self.running = False

                return False

        self.screen.fill(
            (14, 16, 20)
        )

        vision_rect = pygame.Rect(
            20,
            20,
            820,
            640,
        )

        room_rect = pygame.Rect(
            860,
            20,
            520,
            640,
        )

        action_rect = pygame.Rect(
            20,
            680,
            400,
            300,
        )

        reward_rect = pygame.Rect(
            440,
            680,
            420,
            300,
        )

        weight_rect = pygame.Rect(
            880,
            680,
            500,
            300,
        )

        self._draw_vision(
            vision_rect,
            frame,
            features,
        )

        self._draw_room(
            room_rect,
            features,
        )

        self._draw_actions(
            action_rect,
            held,
            explore_action,
            fired_count,
            step,
            elapsed,
        )

        self._draw_rewards(
            reward_rect,
            recent_events,
            total_reward,
        )

        self._draw_weights(
            weight_rect,
            multipliers,
        )

        pygame.display.flip()

        return True

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self.running = False

        try:
            pygame.quit()

        except Exception:
            pass