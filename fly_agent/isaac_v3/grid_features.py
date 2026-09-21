import cv2
import numpy as np

from . import config

try:
    from isaac.room_state import IsaacRoomState
except Exception:
    IsaacRoomState = None

try:
    from isaac.world_state import IsaacWorldState
except Exception:
    IsaacWorldState = None

try:
    from isaac.combat_state import IsaacCombatState
except Exception:
    IsaacCombatState = None


class GridFeatureExtractor:
    """Reuses the old Isaac room/world/combat bridges as an assisted sensor.

    The grid is deliberately kept separate from pixel vision. This lets V3.5
    compare PURE VISION against GRID ASSIST without pretending game-state data
    came from the biological retina.
    """

    def __init__(self, rows=config.V35_GRID_ROWS, cols=config.V35_GRID_COLS):
        self.rows = int(rows)
        self.cols = int(cols)
        self.channels = tuple(config.V35_GRID_CHANNELS)
        self.scalar_names = tuple(config.V35_GRID_SCALARS)
        self.dim = self.rows * self.cols * len(self.channels) + len(self.scalar_names)
        self.last_features = {}
        self.last_vector = np.zeros(self.dim, dtype=np.float32)

    def _resize(self, value, nearest=True):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
            return np.zeros((self.rows, self.cols), dtype=np.float32)
        a = np.asarray(value, dtype=np.float32)
        if a.shape != (self.rows, self.cols):
            interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
            a = cv2.resize(a, (self.cols, self.rows), interpolation=interp)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(a, 0.0, 1.0).astype(np.float32, copy=False)

    def extract(self, features):
        features = dict(features or {})
        pieces = []
        for name in self.channels:
            # topology/door/entity grids must remain crisp when native room shape varies.
            pieces.append(self._resize(features.get(name), nearest=True).reshape(-1))

        scalars = []
        for name in self.scalar_names:
            try:
                value = float(features.get(name, 0.0))
            except (TypeError, ValueError):
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            scalars.append(value)

        out = np.concatenate(
            pieces + [np.asarray(scalars, dtype=np.float32)], axis=0
        ).astype(np.float32, copy=False)
        if out.size != self.dim:
            raise RuntimeError(f"Grid feature width changed: {out.size} != {self.dim}")
        self.last_features = features
        self.last_vector = out
        return out


class IsaacGridBridge:
    """Poll old room/world/combat readers and expose one stable V3.5 vector."""

    def __init__(self, room=None, world=None, combat=None):
        self.room = room if room is not None else (IsaacRoomState() if IsaacRoomState else None)
        self.world = world if world is not None else (IsaacWorldState() if IsaacWorldState else None)
        self.combat = combat if combat is not None else (IsaacCombatState() if IsaacCombatState else None)
        self.extractor = GridFeatureExtractor()
        self.last_features = {}
        self.last_errors = set()

        # Combat readers can emit reward events (SHOT_HIT / SHOT_MISS)
        # while the same poll also updates enemy/grid state. Older V3.7 code
        # discarded poll()'s return value here, so those events never reached
        # DopamineSystem. Keep a small one-shot queue and let the runtime drain
        # it immediately after grid_bridge.poll().
        self.pending_events = []

    @property
    def dim(self):
        return self.extractor.dim

    def _poll_reader(self, reader, features, label):
        if reader is None:
            return features
        try:
            poll = getattr(reader, "poll", None)
            if callable(poll):
                poll_result = poll()

                # IsaacCombatState.poll() returns a list of reward events.
                # Room/world readers usually return counts/None, so only
                # dictionary event records are accepted here.
                if isinstance(poll_result, (list, tuple)):
                    for event in poll_result:
                        if isinstance(event, dict) and event.get("name"):
                            self.pending_events.append(dict(event))

            enrich = getattr(reader, "enrich_features", None)
            if not callable(enrich):
                enrich = getattr(reader, "enrich", None)
            if callable(enrich):
                out = enrich(features)
                if out is not None:
                    features = out
        except Exception as exc:
            key = (label, type(exc).__name__, str(exc))
            if key not in self.last_errors:
                print(f"Grid bridge {label} warning: {exc}")
                self.last_errors.add(key)
        return features

    def poll_features(self):
        f = {}
        # Room first establishes native room geometry; combat readers in the old
        # project can then align enemy/self maps to it.
        f = self._poll_reader(self.room, f, "room")
        f = self._poll_reader(self.world, f, "world")
        f = self._poll_reader(self.combat, f, "combat")
        self.last_features = f
        return f

    def poll_vector(self):
        return self.extractor.extract(self.poll_features())

    def poll(self):
        """Poll historical readers once and return (raw_features, fixed_vector)."""
        features = self.poll_features()
        vector = self.extractor.extract(features)
        return features, vector

    def drain_events(self):
        """Return combat reward events collected during the last bridge poll.

        Events are one-shot: draining clears the queue so SHOT_HIT / SHOT_MISS
        cannot be delivered twice to dopamine.
        """
        events = self.pending_events
        self.pending_events = []
        return events

    def reset_run(self):
        self.pending_events = []

        # Different historical readers used different reset method names.
        for reader in (self.room, self.world, self.combat):
            if reader is None:
                continue
            for name in ("reset_run_history", "reset_run", "reset"):
                fn = getattr(reader, name, None)
                if callable(fn):
                    try:
                        fn()
                    except TypeError:
                        pass
                    except Exception:
                        pass
                    break
        self.last_features = {}
        self.extractor.last_vector = np.zeros(self.dim, dtype=np.float32)

    def close(self):
        for reader in (self.room, self.world, self.combat):
            fn = getattr(reader, "close", None) if reader is not None else None
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
