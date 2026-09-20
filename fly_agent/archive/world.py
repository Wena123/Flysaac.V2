import numpy as np

class SimpleWorld:

    def __init__(self, width=100):
        self.width = width
        self.reset()

    def reset(self):
        self.player_x = self.width // 2

        # Put the target somewhere other than the starting position.
        self.target_x = np.random.randint(
            10,
            self.width - 10
        )

        self.steps = 0

        return self.state()

    def state(self):
        return {
            "player_x": self.player_x,
            "target_x": self.target_x,
            "width": self.width,
        }

    def step(self, action):

        old_distance = abs(
            self.target_x - self.player_x
        )

        if action == "left":
            self.player_x -= 1

        elif action == "right":
            self.player_x += 1

        self.player_x = np.clip(
            self.player_x,
            0,
            self.width - 1
        )

        self.steps += 1

        new_distance = abs(
            self.target_x - self.player_x
        )

        reward = old_distance - new_distance

        done = (
            new_distance == 0
            or self.steps >= 200
        )

        if new_distance == 0:
            reward += 10

        return (
            self.state(),
            float(reward),
            done,
        )

