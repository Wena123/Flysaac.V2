import numpy as np

from brain.fly import FlyAgentBrain
from environment.world import SimpleWorld
from vision.encoder import VisionEncoder


STEPS = 300
TRIALS = 5


def run_trial(brain, vision, world, target_x=None, blank=False):

    brain.reset()

    world.player_x = 50

    if target_x is not None:
        world.target_x = target_x

    left_id = int(brain.groups["steer_L"][0])
    right_id = int(brain.groups["steer_R"][0])

    left_spikes = 0
    right_spikes = 0

    total_network_spikes = 0

    first_nonempty = None

    for step in range(STEPS):

        state = world.state()

        if blank:
            eye_drive = np.zeros(
                brain.visual_count,
                dtype=np.float32
            )
        else:
            eye_drive = vision.encode(state)

        fired = brain.step(eye_drive)

        fired = np.asarray(fired)

        total_network_spikes += fired.size

        if fired.size > 0 and first_nonempty is None:
            first_nonempty = fired.copy()

        if np.any(fired == left_id):
            left_spikes += 1

        if np.any(fired == right_id):
            right_spikes += 1

    return {
        "left": left_spikes,
        "right": right_spikes,
        "network": total_network_spikes,
        "first": first_nonempty,
    }


def test_condition(
    name,
    brain,
    vision,
    world,
    target_x=None,
    blank=False
):

    print()
    print("=" * 50)
    print(name)
    print("=" * 50)

    all_left = 0
    all_right = 0
    all_network = 0

    example_fired = None

    for trial in range(TRIALS):

        result = run_trial(
            brain,
            vision,
            world,
            target_x=target_x,
            blank=blank
        )

        all_left += result["left"]
        all_right += result["right"]
        all_network += result["network"]

        if (
            example_fired is None
            and result["first"] is not None
        ):
            example_fired = result["first"]

        print(
            f"trial {trial + 1:02d}: "
            f"steer_L spikes={result['left']:3d} | "
            f"steer_R spikes={result['right']:3d} | "
            f"network spikes={result['network']:7d}"
        )

    print()
    print("TOTAL")
    print(f"steer_L: {all_left}")
    print(f"steer_R: {all_right}")
    print(f"network: {all_network}")

    if example_fired is not None:
        print()
        print("Example non-empty brain.step() output:")
        print("shape:", example_fired.shape)
        print("first IDs:", example_fired[:20])


def main():

    print("================================")
    print("     RAW neural direction test")
    print("================================")

    brain = FlyAgentBrain()
    vision = VisionEncoder(brain)
    world = SimpleWorld()

    print()
    print(
        "steer_L neuron:",
        brain.groups["steer_L"]
    )

    print(
        "steer_R neuron:",
        brain.groups["steer_R"]
    )

    test_condition(
        "BLANK / NO VISUAL INPUT",
        brain,
        vision,
        world,
        blank=True
    )

    test_condition(
        "TARGET LEFT",
        brain,
        vision,
        world,
        target_x=25
    )

    test_condition(
        "TARGET CENTER",
        brain,
        vision,
        world,
        target_x=50
    )

    test_condition(
        "TARGET RIGHT",
        brain,
        vision,
        world,
        target_x=75
    )


if __name__ == "__main__":
    main()