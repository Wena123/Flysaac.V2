from pathlib import Path

import numpy as np

from . import config


class DemoBuffer:
    """Recorder buffer. Saved labels stay in the original 8-bit V3 format."""

    def __init__(self):
        self.x = []
        self.y = []

    def __len__(self):
        return len(self.x)

    def append(self, features, labels):
        self.x.append(np.asarray(features, dtype=np.float32).copy())
        self.y.append(np.asarray(labels, dtype=np.float32).copy())

    def arrays(self):
        if not self.x:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.float32),
            )
        return np.stack(self.x), np.stack(self.y)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        x, y = self.arrays()
        metadata = "" if metadata is None else str(metadata)
        np.savez_compressed(path, x=x, y=y, metadata=np.asarray([metadata]))
        print(f"Dataset saved: {path} ({len(x)} samples)")
        return path


def load_dataset_file(path):
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    x = np.asarray(data["x"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Bad dataset {path}: X/Y must be 2D")
    if len(x) != len(y):
        raise ValueError(f"Bad dataset {path}: X/Y size mismatch")
    if y.shape[1] != len(config.ACTIONS):
        raise ValueError(
            f"Bad dataset {path}: expected {len(config.ACTIONS)} legacy labels, got {y.shape[1]}"
        )
    return x, y


def load_datasets(paths):
    """Legacy helper retained for compatibility with older utilities."""
    xs, ys = [], []
    for path in paths:
        x, y = load_dataset_file(path)
        if len(x):
            xs.append(x)
            ys.append(y)
            print("Loaded", path, x.shape, y.shape)
    if not xs:
        raise RuntimeError("No samples found in the selected dataset files.")
    width = xs[0].shape[1]
    for x in xs:
        if x.shape[1] != width:
            raise RuntimeError("Datasets use different descending-neuron dimensions.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _movement_class(up, down, left, right):
    # Opposite keys on the same axis are treated as a transition overlap,
    # therefore neutral on that axis instead of teaching an impossible state.
    vertical = 0
    if up and not down:
        vertical = -1
    elif down and not up:
        vertical = 1

    horizontal = 0
    if left and not right:
        horizontal = -1
    elif right and not left:
        horizontal = 1

    mapping = {
        (0, 0): 0,
        (-1, 0): 1,
        (1, 0): 2,
        (0, -1): 3,
        (0, 1): 4,
        (-1, -1): 5,
        (-1, 1): 6,
        (1, -1): 7,
        (1, 1): 8,
    }
    return mapping[(vertical, horizontal)]


def _clean_shoot_sequence(raw_shoot, radius=2):
    """
    Convert four binary shoot keys into one cardinal class.

    If two arrow keys overlap during a key transition, prefer the previously
    selected direction when it is still held. Otherwise select the direction
    that persists most strongly in a short local temporal window.
    """
    raw = np.asarray(raw_shoot, dtype=np.float32) >= 0.5
    out = np.zeros(len(raw), dtype=np.int64)
    ambiguous = 0
    previous = 0  # 0=none, 1..4 correspond to up/down/left/right

    for i, row in enumerate(raw):
        active = np.flatnonzero(row)
        if len(active) == 0:
            choice = 0
        elif len(active) == 1:
            choice = int(active[0]) + 1
        else:
            ambiguous += 1
            previous_raw = previous - 1
            if previous_raw >= 0 and previous_raw in active:
                choice = previous
            else:
                lo = max(0, i - int(radius))
                hi = min(len(raw), i + int(radius) + 1)
                persistence = raw[lo:hi, :].sum(axis=0)
                active_scores = persistence[active]
                best = active[np.argmax(active_scores)]
                choice = int(best) + 1
        out[i] = choice
        previous = choice

    return out, ambiguous


def convert_legacy_labels(y):
    """Convert V3 eight-bit labels into V3.1 movement + shoot class ids."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != len(config.ACTIONS):
        raise ValueError(f"Expected (*, {len(config.ACTIONS)}) legacy labels, got {y.shape}")

    b = y >= 0.5
    movement = np.empty(len(y), dtype=np.int64)
    movement_conflicts = 0

    for i, row in enumerate(b):
        up, down, left, right = map(bool, row[:4])
        if (up and down) or (left and right):
            movement_conflicts += 1
        movement[i] = _movement_class(up, down, left, right)

    shoot, shoot_ambiguous = _clean_shoot_sequence(b[:, 4:8])

    stats = {
        "movement_conflicts": int(movement_conflicts),
        "shoot_ambiguous": int(shoot_ambiguous),
    }
    return movement, shoot, stats


def blocked_split(n, val_fraction=config.VALIDATION_FRACTION, guard=config.VALIDATION_GUARD_SAMPLES):
    """Time-block split with a small discarded guard between train and validation."""
    n = int(n)
    if n < 4:
        idx = np.arange(n, dtype=np.int64)
        return idx, idx

    n_val = max(1, int(round(n * float(val_fraction))))
    split = max(1, n - n_val)
    guard = min(int(guard), max(0, split - 1))
    train_end = max(1, split - guard)
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(split, n, dtype=np.int64)
    if len(val_idx) == 0:
        val_idx = np.arange(max(0, n - 1), n, dtype=np.int64)
    return train_idx, val_idx
