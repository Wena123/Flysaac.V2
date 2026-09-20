from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class RetinaOutput:
    luminance: np.ndarray
    adapted: np.ndarray
    contrast: np.ndarray
    edge: np.ndarray
    motion: np.ndarray
    motion_left: np.ndarray
    motion_right: np.ndarray
    motion_up: np.ndarray
    motion_down: np.ndarray
    target: np.ndarray
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    looming_left: float
    looming_right: float
    looming_global: float


class FlyRetina:
    """
    Compact fly-inspired visual front end.

    It intentionally does NOT identify enemies, doors or projectiles.
    The only input is the BGR game frame. The output is luminance,
    local contrast, edges, directional motion and an outward-motion
    (looming) signal that can be injected into real visual projection
    neurons in MaleCNS.
    """

    def __init__(
        self,
        rows=config.RETINA_ROWS,
        cols=config.RETINA_COLS,
        naka_n=config.RETINA_NAKA_N,
        naka_sigma=config.RETINA_NAKA_SIGMA,
        adapt_alpha=config.RETINA_ADAPT_ALPHA,
        contrast_gain=config.RETINA_CONTRAST_GAIN,
        edge_gain=config.RETINA_EDGE_GAIN,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.naka_n = float(naka_n)
        self.naka_sigma = float(naka_sigma)
        self.adapt_alpha = float(adapt_alpha)
        self.contrast_gain = float(contrast_gain)
        self.edge_gain = float(edge_gain)

        self._previous = None
        self._adapt_level = None

    def reset(self):
        self._previous = None
        self._adapt_level = None

    @staticmethod
    def _normalize01(x):
        x = np.asarray(x, dtype=np.float32)
        lo = float(np.percentile(x, 2.0))
        hi = float(np.percentile(x, 98.0))
        if hi <= lo + 1e-6:
            return np.clip(x, 0.0, 1.0)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    @staticmethod
    def _directional_correlator(current, previous):
        right = np.zeros_like(current, dtype=np.float32)
        left = np.zeros_like(current, dtype=np.float32)
        down = np.zeros_like(current, dtype=np.float32)
        up = np.zeros_like(current, dtype=np.float32)

        # Elementary Reichardt-like opponent correlations.
        horizontal = (
            current[:, 1:] * previous[:, :-1]
            - current[:, :-1] * previous[:, 1:]
        )
        right[:, 1:] = np.maximum(horizontal, 0.0)
        left[:, :-1] = np.maximum(-horizontal, 0.0)

        vertical = (
            current[1:, :] * previous[:-1, :]
            - current[:-1, :] * previous[1:, :]
        )
        down[1:, :] = np.maximum(vertical, 0.0)
        up[:-1, :] = np.maximum(-vertical, 0.0)

        return left, right, up, down

    @staticmethod
    def _top_mean(values, fraction=0.12):
        flat = np.asarray(values, dtype=np.float32).ravel()
        if flat.size == 0:
            return 0.0
        k = max(1, int(round(flat.size * float(fraction))))
        if k >= flat.size:
            return float(flat.mean())
        idx = np.argpartition(flat, flat.size - k)[-k:]
        return float(flat[idx].mean())

    def process(self, frame_bgr):
        if frame_bgr is None:
            raise ValueError("FlyRetina.process() received frame=None")

        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected BGR image HxWx3, got {frame.shape!r}")

        small = cv2.resize(
            frame[:, :, :3],
            (self.cols, self.rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0

        blue = small[:, :, 0]
        green = small[:, :, 1]
        red = small[:, :, 2]

        # Rec.709 luminance is enough here; biology is represented mainly by
        # the downstream real connectome rather than by pretending RGB is a fly eye.
        lum = 0.0722 * blue + 0.7152 * green + 0.2126 * red
        lum = np.clip(lum, 0.0, 1.0).astype(np.float32)

        if self._adapt_level is None:
            self._adapt_level = float(max(0.05, lum.mean()))
        else:
            self._adapt_level = (
                self.adapt_alpha * self._adapt_level
                + (1.0 - self.adapt_alpha) * float(max(0.02, lum.mean()))
            )

        # Naka-Rushton response with a slowly adapting denominator.
        scaled = np.clip(lum / max(self._adapt_level, 0.05), 0.0, 4.0) / 4.0
        numerator = np.power(scaled, self.naka_n)
        denominator = numerator + self.naka_sigma ** self.naka_n + 1e-6
        adapted = (numerator / denominator).astype(np.float32)

        blur = cv2.GaussianBlur(adapted, (0, 0), 1.0)
        contrast = np.clip(
            np.abs(adapted - blur) * self.contrast_gain,
            0.0,
            1.0,
        ).astype(np.float32)

        gx = cv2.Sobel(adapted, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(adapted, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = np.clip(edge * self.edge_gain, 0.0, 1.0).astype(np.float32)

        if self._previous is None:
            previous = adapted.copy()
        else:
            previous = self._previous

        motion_left, motion_right, motion_up, motion_down = (
            self._directional_correlator(adapted, previous)
        )
        temporal = np.abs(adapted - previous)

        direction_sum = np.maximum.reduce(
            [motion_left, motion_right, motion_up, motion_down]
        )
        motion = self._normalize01(np.maximum(temporal, direction_sum * 2.0))

        # A generic small-moving-target salience map. No object detector.
        target = np.clip(
            motion * (0.30 + 0.45 * edge + 0.25 * contrast),
            0.0,
            1.0,
        ).astype(np.float32)

        mid_x = self.cols // 2
        mid_y = self.rows // 2

        outward_left = np.maximum(
            motion_left[:, :mid_x],
            np.pad(
                motion_up[:mid_y, :mid_x],
                ((0, self.rows - mid_y), (0, 0)),
            ),
        )
        outward_right = np.maximum(
            motion_right[:, mid_x:],
            np.pad(
                motion_down[mid_y:, mid_x:],
                ((mid_y, 0), (0, 0)),
            ),
        )

        # More complete outward-motion estimate around the visual centre.
        left_half = np.maximum(
            motion_left[:, :mid_x],
            np.concatenate(
                [motion_up[:mid_y, :mid_x], motion_down[mid_y:, :mid_x]],
                axis=0,
            ),
        )
        right_half = np.maximum(
            motion_right[:, mid_x:],
            np.concatenate(
                [motion_up[:mid_y, mid_x:], motion_down[mid_y:, mid_x:]],
                axis=0,
            ),
        )

        looming_left = self._top_mean(left_half)
        looming_right = self._top_mean(right_half)
        looming_global = max(looming_left, looming_right)

        self._previous = adapted.copy()

        return RetinaOutput(
            luminance=lum,
            adapted=adapted,
            contrast=contrast,
            edge=edge,
            motion=motion,
            motion_left=motion_left,
            motion_right=motion_right,
            motion_up=motion_up,
            motion_down=motion_down,
            target=target,
            red=red.astype(np.float32),
            green=green.astype(np.float32),
            blue=blue.astype(np.float32),
            looming_left=float(np.clip(looming_left, 0.0, 1.0)),
            looming_right=float(np.clip(looming_right, 0.0, 1.0)),
            looming_global=float(np.clip(looming_global, 0.0, 1.0)),
        )
