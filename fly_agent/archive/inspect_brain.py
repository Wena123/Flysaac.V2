import numpy as np

from brain.fly import FlyAgentBrain
from vision.encoder import VisionEncoder
from environment.world import SimpleWorld


def describe(name, x):
    arr = np.asarray(x)

    print(f"\n{name}")
    print("-" * 50)
    print("type:       ", type(x))
    print("shape:      ", arr.shape)
    print("dtype:      ", arr.dtype)
    print("size:       ", arr.size)

    if arr.size > 0:
        print("min:        ", arr.min())
        print("max:        ", arr.max())
        print("nonzero:    ", np.count_nonzero(arr))
        print("first 20:   ", arr.flat[:20])

        unique = np.unique(arr)

        if len(unique) <= 20:
            print("unique:     ", unique)
        else:
            print("unique count:", len(unique))


def main():

    print("==============================")
    print("      FlyBrain inspection")
    print("==============================")

    brain = FlyAgentBrain()
    vision = VisionEncoder(brain)
    world = SimpleWorld()

    # Fixed left-side stimulus
    world.player_x = 50
    world.target_x = 25

    state = world.state()

    eye_drive = vision.encode(state)

    brain.reset()

    fired = brain.step(eye_drive)

    describe("brain.step() OUTPUT", fired)

    describe(
        'groups["steer_L"]',
        brain.groups["steer_L"]
    )

    describe(
        'groups["steer_R"]',
        brain.groups["steer_R"]
    )

    describe(
        "visual neurons",
        brain.brain.visual
    )

    describe(
        "azimuth",
        brain.brain.azimuth
    )

    describe(
        "eye_drive",
        eye_drive
    )


if __name__ == "__main__":
    main()