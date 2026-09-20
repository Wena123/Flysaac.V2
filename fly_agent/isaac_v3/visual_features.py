import numpy as np

from . import config


class RetinaFeatureExtractor:
    """FlyIsaac V3.4 multiscale retina feature extractor.

    This still uses only the fly-inspired retina output: no enemy detector,
    no room labels, and no game-state coordinates. Compared with V3.3 it
    keeps substantially more spatial detail and preserves small salient
    objects by blending tile mean + tile max rather than mean alone.
    """

    FINE_MAP_NAMES = (
        "target",
        "motion",
        "edge",
        "contrast",
    )

    COARSE_MAP_NAMES = (
        "motion_left",
        "motion_right",
        "motion_up",
        "motion_down",
        "red",
        "green",
        "blue",
        "luminance",
        "adapted",
    )

    def __init__(
        self,
        fine_tiles_y=getattr(config, "V34_VISUAL_FINE_TILES_Y", 10),
        fine_tiles_x=getattr(config, "V34_VISUAL_FINE_TILES_X", 16),
        coarse_tiles_y=getattr(config, "V34_VISUAL_COARSE_TILES_Y", 6),
        coarse_tiles_x=getattr(config, "V34_VISUAL_COARSE_TILES_X", 10),
    ):
        self.fine_tiles_y = int(fine_tiles_y)
        self.fine_tiles_x = int(fine_tiles_x)
        self.coarse_tiles_y = int(coarse_tiles_y)
        self.coarse_tiles_x = int(coarse_tiles_x)
        for value in (
            self.fine_tiles_y, self.fine_tiles_x,
            self.coarse_tiles_y, self.coarse_tiles_x,
        ):
            if value <= 0:
                raise ValueError("visual tile counts must be positive")

        self.fine_dim = (
            len(self.FINE_MAP_NAMES)
            * self.fine_tiles_y
            * self.fine_tiles_x
        )
        self.coarse_dim = (
            len(self.COARSE_MAP_NAMES)
            * self.coarse_tiles_y
            * self.coarse_tiles_x
        )
        self.global_dim = 33
        self.dim = self.fine_dim + self.coarse_dim + self.global_dim

    @staticmethod
    def _pool(field, tiles_y, tiles_x, max_weight=0.65):
        field = np.asarray(field, dtype=np.float32)
        ys = np.linspace(0, field.shape[0], int(tiles_y) + 1, dtype=int)
        xs = np.linspace(0, field.shape[1], int(tiles_x) + 1, dtype=int)
        out = np.empty(int(tiles_y) * int(tiles_x), dtype=np.float32)
        k = 0
        mw = float(max_weight)
        aw = 1.0 - mw
        for iy in range(int(tiles_y)):
            for ix in range(int(tiles_x)):
                tile = field[ys[iy]:ys[iy + 1], xs[ix]:xs[ix + 1]]
                if tile.size:
                    out[k] = aw * float(tile.mean()) + mw * float(tile.max())
                else:
                    out[k] = 0.0
                k += 1
        return out

    @staticmethod
    def _moments(field):
        field = np.maximum(np.asarray(field, dtype=np.float32), 0.0)
        h, w = field.shape
        mass = float(field.sum())
        if mass <= 1e-8:
            return (0.5, 0.5, 0.0, 0.0, 0.0)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn = xx / max(1.0, float(w - 1))
        yn = yy / max(1.0, float(h - 1))
        cx = float((field * xn).sum() / mass)
        cy = float((field * yn).sum() / mass)
        sx = float(np.sqrt(max(0.0, (field * (xn - cx) ** 2).sum() / mass)))
        sy = float(np.sqrt(max(0.0, (field * (yn - cy) ** 2).sum() / mass)))
        density = float(np.clip(mass / max(1.0, float(h * w)), 0.0, 1.0))
        return (cx, cy, sx, sy, density)

    def extract(self, retina):
        if retina is None:
            raise ValueError("retina output is None")

        fine = [
            self._pool(
                getattr(retina, name),
                self.fine_tiles_y,
                self.fine_tiles_x,
                max_weight=0.70,
            )
            for name in self.FINE_MAP_NAMES
        ]
        coarse = [
            self._pool(
                getattr(retina, name),
                self.coarse_tiles_y,
                self.coarse_tiles_x,
                max_weight=0.55,
            )
            for name in self.COARSE_MAP_NAMES
        ]

        target = np.asarray(retina.target, dtype=np.float32)
        motion = np.asarray(retina.motion, dtype=np.float32)
        edge = np.asarray(retina.edge, dtype=np.float32)
        contrast = np.asarray(retina.contrast, dtype=np.float32)
        luminance = np.asarray(retina.luminance, dtype=np.float32)
        red = np.asarray(retina.red, dtype=np.float32)
        green = np.asarray(retina.green, dtype=np.float32)
        blue = np.asarray(retina.blue, dtype=np.float32)

        t_m = self._moments(target)
        m_m = self._moments(motion)
        e_m = self._moments(edge)

        chroma_rg = np.abs(red - green)
        chroma_by = np.abs(blue - 0.5 * (red + green))

        globals_ = np.asarray(
            [
                float(retina.looming_left),
                float(retina.looming_right),
                float(retina.looming_global),
                float(target.mean()),
                float(target.max()),
                float(motion.mean()),
                float(motion.max()),
                float(edge.mean()),
                float(edge.max()),
                float(contrast.mean()),
                float(contrast.max()),
                float(luminance.mean()),
                float(red.mean()),
                float(green.mean()),
                float(blue.mean()),
                float(chroma_rg.mean()),
                float(chroma_by.mean()),
                *t_m,
                *m_m,
                float(e_m[0]),
                float(e_m[1]),
                float(np.asarray(retina.motion_left).mean()),
                float(np.asarray(retina.motion_right).mean()),
                float(np.asarray(retina.motion_up).mean()),
                float(np.asarray(retina.motion_down).mean()),
            ],
            dtype=np.float32,
        )

        if globals_.size != self.global_dim:
            raise RuntimeError(
                f"V3.4 global visual size {globals_.size}, expected {self.global_dim}"
            )

        out = np.concatenate(fine + coarse + [globals_]).astype(
            np.float32, copy=False
        )
        if out.size != self.dim:
            raise RuntimeError(
                f"V3.4 visual feature size {out.size}, expected {self.dim}"
            )
        return out
