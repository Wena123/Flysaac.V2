import numpy as np

from brain.fly import FlyAgentBrain
from brain.decoder import ActionDecoder
from environment.world import SimpleWorld


EPISODES = 20
MAX_STEPS = 300

LC10_DRIVE = 0.8

# Shorter than our previous 15-step memory.
DECODER_WINDOW = 5


def main():

    print("================================")
    print("       LC10a chase test")
    print("================================")

    brain = FlyAgentBrain()

    world = SimpleWorld()

    decoder = ActionDecoder(
        brain,
        window=DECODER_WINDOW
    )

    # Biological target-detection populations.
    lc10_left = np.asarray(
        brain.brain.cells(
            ["LC10a"],
            side="L"
        )
    )

    lc10_right = np.asarray(
        brain.brain.cells(
            ["LC10a"],
            side="R"
        )
    )

    print()
    print("LC10a left:", len(lc10_left))
    print("LC10a right:", len(lc10_right))
    print()

    successes = 0

    left_successes = 0
    right_successes = 0

    left_episodes = 0
    right_episodes = 0

    for episode in range(EPISODES):

        state = world.reset()

        # Avoid starting directly on the target.
        while state["target_x"] == state["player_x"]:
            state = world.reset()

        brain.reset()
        decoder.reset()

        start_x = state["player_x"]
        target_x = state["target_x"]

        if target_x < start_x:
            target_side = "left"
            left_episodes += 1
        else:
            target_side = "right"
            right_episodes += 1

        left_actions = 0
        right_actions = 0
        none_actions = 0

        for step in range(MAX_STEPS):

            state = world.state()

            player_x = state["player_x"]
            target_x = state["target_x"]

            delta = target_x - player_x

            # -----------------------------
            # ENVIRONMENT -> LC10a
            # -----------------------------

            if delta < 0:

                fired = brain.brain.step(
                    inject=[
                        (
                            lc10_left,
                            LC10_DRIVE
                        )
                    ]
                )

            elif delta > 0:

                fired = brain.brain.step(
                    inject=[
                        (
                            lc10_right,
                            LC10_DRIVE
                        )
                    ]
                )

            else:

                fired = brain.brain.step()

            fired = np.asarray(
                fired,
                dtype=np.int64
            )

            # -----------------------------
            # MaleCNS -> ACTION
            # -----------------------------

            action = decoder.decode(
                fired
            )

            if action == "left":
                left_actions += 1

            elif action == "right":
                right_actions += 1

            else:
                none_actions += 1

            # -----------------------------
            # ACTION -> ENVIRONMENT
            # -----------------------------

            state, reward, done = world.step(
                action
            )

            if done:
                break

        success = (
            state["player_x"]
            == state["target_x"]
        )

        if success:

            successes += 1

            if target_side == "left":
                left_successes += 1

            else:
                right_successes += 1

        print(
            f"Episode {episode + 1:02d} | "
            f"target={target_x:03d} "
            f"{target_side:5s} | "
            f"final={state['player_x']:03d} | "
            f"steps={step + 1:03d} | "
            f"L={left_actions:03d} "
            f"R={right_actions:03d} "
            f"none={none_actions:03d} | "
            f"{'SUCCESS' if success else 'FAIL'}"
        )

    print()
    print("================================")
    print("             RESULTS")
    print("================================")

    print(
        f"Successes: "
        f"{successes}/{EPISODES}"
    )

    print(
        f"Left: "
        f"{left_successes}/{left_episodes}"
    )

    print(
        f"Right: "
        f"{right_successes}/{right_episodes}"
    )

    print()

    accuracy = (
        successes / EPISODES
    ) * 100

    print(
        f"Success rate: "
        f"{accuracy:.1f}%"
    )


if __name__ == "__main__":
    main()