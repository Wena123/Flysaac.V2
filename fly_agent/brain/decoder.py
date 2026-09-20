from collections import deque
import numpy as np


class ActionDecoder:

    def __init__(self, brain, window=15):
        self.brain = brain
        self.window = window

        self.left_neurons = np.asarray(
            brain.groups["steer_L"]
        )

        self.right_neurons = np.asarray(
            brain.groups["steer_R"]
        )

        self.left_history = deque(maxlen=window)
        self.right_history = deque(maxlen=window)

    def reset(self):
        self.left_history.clear()
        self.right_history.clear()

    def decode(self, fired):
        fired = np.asarray(fired)

        left_spikes = np.intersect1d(
            fired,
            self.left_neurons
        ).size

        right_spikes = np.intersect1d(
            fired,
            self.right_neurons
        ).size

        self.left_history.append(left_spikes)
        self.right_history.append(right_spikes)

        left_score = sum(self.left_history)
        right_score = sum(self.right_history)

        if left_score == 0 and right_score == 0:
            return "none"

        if left_score > right_score:
            return "left"

        if right_score > left_score:
            return "right"

        return "none"