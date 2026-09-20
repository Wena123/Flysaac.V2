import cv2
import numpy as np


class IsaacPerception:
    """
    Fast retina-like visual front-end.

    Isaac is captured at its normal resolution for the dashboard,
    but visual processing is performed on a smaller image.

    Output:
        12 x 20 spatial retina

    Channels:
        - motion
        - stationary edges
        - brightness contrast
        - colour contrast
        - red / green / blue dominance

    No object recognition.
    No enemy detection.
    No decision making.
    """

    GRID_ROWS = 12
    GRID_COLS = 20

    LEGACY_ROWS = 3
    LEGACY_COLS = 3

    LEGACY_NAMES = [
        "top_left",
        "top",
        "top_right",

        "left",
        "center",
        "right",

        "bottom_left",
        "bottom",
        "bottom_right",
    ]

    def __init__(
        self,
        motion_threshold=18,
        process_scale=0.5,
    ):

        self.motion_threshold = int(
            motion_threshold
        )

        # -----------------------------------------------------
        # PERFORMANCE
        #
        # Isaac capture may be 960x540, but processing at
        # 480x270 is plenty when final retina is only 20x12.
        # -----------------------------------------------------

        self.process_scale = float(
            process_scale
        )

        self.previous_gray = None

        self.last_motion_mask = None
        self.last_edge_mask = None

        self.last_grid = np.zeros(
            (
                self.GRID_ROWS,
                self.GRID_COLS,
            ),
            dtype=np.float32,
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):

        self.previous_gray = None

        self.last_motion_mask = None
        self.last_edge_mask = None

        self.last_grid.fill(
            0.0
        )

    # =========================================================
    # PREPARE SMALL FRAME
    # =========================================================

    def _downscale(
        self,
        frame,
    ):

        if (
            self.process_scale
            >= 0.999
        ):

            return frame

        return cv2.resize(
            frame,
            None,
            fx=self.process_scale,
            fy=self.process_scale,
            interpolation=cv2.INTER_AREA,
        )

    # =========================================================
    # GRAYSCALE
    # =========================================================

    @staticmethod
    def _prepare_gray(
        frame,
    ):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        return gray

    # =========================================================
    # FAST BINARY GRID
    # =========================================================

    @staticmethod
    def _fraction_grid(
        image,
        rows,
        cols,
    ):
        """
        For binary 0/255 images, INTER_AREA computes approximately
        the fraction of active pixels inside each output cell.

        This replaces hundreds of Python array slices.
        """

        normalized = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        grid = cv2.resize(
            normalized,
            (
                cols,
                rows,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return grid.astype(
            np.float32
        )

    # =========================================================
    # FAST MEAN GRID
    # =========================================================

    @staticmethod
    def _mean_grid(
        image,
        rows,
        cols,
    ):
        """
        Downsample an image directly into cell averages using
        optimized OpenCV code.
        """

        result = cv2.resize(
            image.astype(
                np.float32
            ),
            (
                cols,
                rows,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return result.astype(
            np.float32
        )

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def process(
        self,
        frame,
    ):

        # =====================================================
        # 1. PROCESS SMALL IMAGE
        # =====================================================

        small_frame = self._downscale(
            frame
        )

        gray = self._prepare_gray(
            small_frame
        )

        # =====================================================
        # 2. MOTION
        # =====================================================

        motion = np.zeros_like(
            gray,
            dtype=np.uint8,
        )

        if self.previous_gray is not None:

            if (
                gray.shape
                == self.previous_gray.shape
            ):

                difference = cv2.absdiff(
                    gray,
                    self.previous_gray,
                )

                _, motion = cv2.threshold(
                    difference,
                    self.motion_threshold,
                    255,
                    cv2.THRESH_BINARY,
                )

                # Remove isolated capture noise.
                motion = cv2.morphologyEx(
                    motion,
                    cv2.MORPH_OPEN,
                    np.ones(
                        (3, 3),
                        dtype=np.uint8,
                    ),
                )

            else:

                print(
                    "Frame size changed:",
                    self.previous_gray.shape,
                    "->",
                    gray.shape,
                    "- resetting motion reference."
                )

        self.previous_gray = (
            gray.copy()
        )

        self.last_motion_mask = (
            motion
        )

        # =====================================================
        # 3. STATIC EDGES
        # =====================================================

        edges = cv2.Canny(
            gray,
            50,
            120,
        )

        self.last_edge_mask = (
            edges
        )

        # =====================================================
        # 4. COLOUR
        # =====================================================

        rgb = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB,
        ).astype(
            np.float32
        )

        rgb /= 255.0

        # =====================================================
        # 5. CREATE 20 x 12 RETINA
        # =====================================================

        motion_grid = (
            self._fraction_grid(
                motion,
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        )

        edge_grid = (
            self._fraction_grid(
                edges,
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        )

        brightness = (
            gray.astype(
                np.float32
            )
            / 255.0
        )

        brightness_grid = (
            self._mean_grid(
                brightness,
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        )

        rgb_grid = (
            self._mean_grid(
                rgb,
                self.GRID_ROWS,
                self.GRID_COLS,
            )
        )

        # =====================================================
        # 6. GLOBAL VISUAL REFERENCES
        # =====================================================

        global_brightness = float(
            np.mean(
                brightness_grid
            )
        )

        global_rgb = np.mean(
            rgb_grid.reshape(
                -1,
                3,
            ),
            axis=0,
        )

        # =====================================================
        # 7. BRIGHTNESS CONTRAST
        # =====================================================

        brightness_contrast = np.abs(
            brightness_grid
            - global_brightness
        ).astype(
            np.float32
        )

        # =====================================================
        # 8. COLOUR CONTRAST
        # =====================================================

        rgb_difference = (
            rgb_grid
            - global_rgb[
                None,
                None,
                :
            ]
        )

        color_contrast = (
            np.linalg.norm(
                rgb_difference,
                axis=2,
            )
            / np.sqrt(3.0)
        )

        color_contrast = np.clip(
            color_contrast,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        # =====================================================
        # 9. RGB DOMINANCE
        # =====================================================

        red = rgb_grid[
            :,
            :,
            0
        ]

        green = rgb_grid[
            :,
            :,
            1
        ]

        blue = rgb_grid[
            :,
            :,
            2
        ]

        red_excess = np.clip(
            red
            - (
                green + blue
            ) / 2.0,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        green_excess = np.clip(
            green
            - (
                red + blue
            ) / 2.0,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        blue_excess = np.clip(
            blue
            - (
                red + green
            ) / 2.0,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        # =====================================================
        # 10. FINAL SPATIAL SALIENCE
        # =====================================================
        #
        # Motion dominates.
        # Static information remains available but quieter.
        # =====================================================

        salience_grid = (
            3.0 * motion_grid
            + 0.25 * edge_grid
            + 0.20 * color_contrast
            + 0.10 * brightness_contrast
        )

        salience_grid = np.clip(
            salience_grid,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

        self.last_grid = (
            salience_grid.copy()
        )

        # =====================================================
        # 11. LEGACY 3 x 3 MOTION SIGNALS
        # =====================================================

        legacy = (
            self._fraction_grid(
                motion,
                self.LEGACY_ROWS,
                self.LEGACY_COLS,
            )
        )

        features = {}

        index = 0

        for row in range(
            self.LEGACY_ROWS
        ):

            for col in range(
                self.LEGACY_COLS
            ):

                features[
                    self.LEGACY_NAMES[
                        index
                    ]
                ] = float(
                    legacy[
                        row,
                        col
                    ]
                )

                index += 1

        # =====================================================
        # 12. OUTPUT
        # =====================================================

        features[
            "grid"
        ] = salience_grid

        features[
            "salience_grid"
        ] = salience_grid

        features[
            "motion_grid"
        ] = motion_grid

        features[
            "edge_grid"
        ] = edge_grid

        features[
            "brightness_grid"
        ] = brightness_grid

        features[
            "brightness_contrast_grid"
        ] = brightness_contrast

        features[
            "rgb_grid"
        ] = rgb_grid

        features[
            "color_contrast_grid"
        ] = color_contrast

        features[
            "red_grid"
        ] = red_excess

        features[
            "green_grid"
        ] = green_excess

        features[
            "blue_grid"
        ] = blue_excess

        features[
            "grid_rows"
        ] = self.GRID_ROWS

        features[
            "grid_cols"
        ] = self.GRID_COLS

        features[
            "global_motion"
        ] = float(
            np.count_nonzero(
                motion
            )
            / motion.size
        )

        return features