from collections import deque
import time

import numpy as np

from . import config


def build_temporal_examples(
    x,
    move_targets,
    shoot_targets,
    context_frames=config.TEMPORAL_CONTEXT_FRAMES,
    lag_samples=0,
):
    """
    Build [t-K+1 .. t] DN context -> action[t + lag] examples.

    Positive lag means the decoder predicts an action that occurred AFTER the
    current DN state. Negative lag is diagnostic and is not selected by default
    because it is non-causal for autonomous play.
    """
    x = np.asarray(x, dtype=np.float32)
    move_targets = np.asarray(move_targets, dtype=np.int64)
    shoot_targets = np.asarray(shoot_targets, dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got {x.shape}")
    if len(x) != len(move_targets) or len(x) != len(shoot_targets):
        raise ValueError("X / movement / shooting sample counts differ")

    k = max(1, int(context_frames))
    lag = int(lag_samples)
    anchors = []
    target_idx = []

    for t in range(k - 1, len(x)):
        target = t + lag
        if 0 <= target < len(x):
            anchors.append(t)
            target_idx.append(target)

    if not anchors:
        return (
            np.empty((0, x.shape[1] * k), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    contexts = np.empty(
        (len(anchors), x.shape[1] * k),
        dtype=np.float32,
    )

    for i, t in enumerate(anchors):
        contexts[i] = x[t - k + 1:t + 1].reshape(-1)

    target_idx = np.asarray(target_idx, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)

    return (
        contexts,
        move_targets[target_idx],
        shoot_targets[target_idx],
        anchors,
        target_idx,
    )


class TemporalDNContext:
    """
    Runtime context sampled at the same cadence as recorded demonstrations.

    MaleCNS can run at ~30 Hz, while the temporal decoder was trained on
    ~10 Hz DN snapshots. We therefore append to the context at 10 Hz and reuse
    the newest context between sample ticks.
    """

    def __init__(
        self,
        dn_dim,
        frames=config.TEMPORAL_CONTEXT_FRAMES,
        sample_hz=config.TEMPORAL_SAMPLE_HZ,
    ):
        self.dn_dim = int(dn_dim)
        self.frames = max(1, int(frames))
        self.sample_hz = float(sample_hz)
        self.interval = 1.0 / max(1e-6, self.sample_hz)
        self.buffer = deque(maxlen=self.frames)
        self.last_sample_time = None
        self.last_vector = None

    @property
    def full_dim(self):
        return self.dn_dim * self.frames

    @property
    def ready(self):
        return len(self.buffer) >= self.frames

    def reset(self):
        self.buffer.clear()
        self.last_sample_time = None
        self.last_vector = None

    def update(self, dn, now=None, force=False):
        dn = np.asarray(dn, dtype=np.float32).reshape(-1)
        if dn.size != self.dn_dim:
            raise ValueError(
                f"Temporal context expects {self.dn_dim} DN values, got {dn.size}"
            )

        if now is None:
            now = time.perf_counter()
        now = float(now)

        due = (
            force
            or self.last_sample_time is None
            or (now - self.last_sample_time) >= (self.interval - 1e-9)
        )

        if due:
            self.buffer.append(dn.copy())
            self.last_sample_time = now
            if self.ready:
                self.last_vector = np.concatenate(
                    list(self.buffer),
                    axis=0,
                ).astype(np.float32, copy=False)

        return self.last_vector

    def warmup_remaining(self):
        return max(0, self.frames - len(self.buffer))
