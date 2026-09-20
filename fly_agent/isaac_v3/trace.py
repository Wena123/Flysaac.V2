import numpy as np

from . import config


def _to_numpy_fired(fired):
    if fired is None:
        return np.empty(0, dtype=np.int64)
    if hasattr(fired, "get"):
        fired = fired.get()
    arr = np.asarray(fired)
    if arr.dtype == np.bool_:
        arr = np.flatnonzero(arr)
    return arr.astype(np.int64, copy=False).ravel()


class DescendingTrace:
    """Exponential activity trace over the real descending neurons."""

    def __init__(
        self,
        fly_agent_brain,
        decay=config.DN_TRACE_DECAY,
        clip=config.DN_TRACE_CLIP,
    ):
        self.brain = fly_agent_brain.brain
        self.decay = float(decay)
        self.clip = float(clip)

        self.cells = np.asarray(
            self.brain.cells(["descending_neuron"]),
            dtype=np.int64,
        )
        if self.cells.size == 0:
            raise RuntimeError("No descending_neuron cells found in MaleCNS.")

        lookup_size = max(200_000, int(self.cells.max()) + 1)
        self.lookup = np.full(lookup_size, -1, dtype=np.int32)
        self.lookup[self.cells] = np.arange(self.cells.size, dtype=np.int32)
        self.trace = np.zeros(self.cells.size, dtype=np.float32)

        print("FlyIsaac V3 descending neurons:", self.cells.size)

    @property
    def size(self):
        return int(self.cells.size)

    def reset(self):
        self.trace.fill(0.0)

    def update(self, fired):
        self.trace *= self.decay

        fired = _to_numpy_fired(fired)
        if fired.size:
            valid = fired[(fired >= 0) & (fired < self.lookup.size)]
            if valid.size:
                local = self.lookup[valid]
                local = local[local >= 0]
                if local.size:
                    np.add.at(self.trace, local, 1.0)

        np.minimum(self.trace, self.clip, out=self.trace)
        # Smooth bounded feature. Keeps scale stable for the small decoder.
        return np.tanh(self.trace / 2.0).astype(np.float32, copy=True)
