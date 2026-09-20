from brain.fly import FlyAgentBrain
from brain.decoder import ActionDecoder
from environment.world import SimpleWorld
from vision.encoder import VisionEncoder


STEPS = 250
TRIALS_PER_SIDE = 10


def run_trial(brain, vision, world, decoder, target_x):

    world.reset()
    world.player_x = 50
    world.target_x = target_x

    brain.reset()
    decoder.reset()

    left_actions = 0
    right_actions = 0
    none_actions = 0

    for _ in range(STEPS):

        state = world.state()

        eye_drive = vision.encode(state)

        fired = brain.step(
            eye_drive
        )

        action = decoder.decode(
            fired
        )

        if action == "left":
            left_actions += 1

        elif action == "right":
            right_actions += 1

        else:
            none_actions += 1

        # IMPORTANT:
        # do NOT move the player in this experiment.
        #
        # We want the visual stimulus to stay
        # in exactly the same position.

    return left_actions, right_actions, none_actions


def main():

    print("==============================")
    print("   FlyAgent direction test")
    print("==============================")

    brain = FlyAgentBrain()
    vision = VisionEncoder(brain)
    decoder = ActionDecoder(brain)
    world = SimpleWorld()

    results = {
        "LEFT": [],
        "RIGHT": []
    }

    # Target clearly to the left.
    for trial in range(TRIALS_PER_SIDE):

        result = run_trial(
            brain,
            vision,
            world,
            decoder,
            target_x=25
        )

        results["LEFT"].append(result)

        print(
            f"LEFT  trial {trial + 1:02d}: "
            f"L={result[0]:03d} "
            f"R={result[1]:03d} "
            f"none={result[2]:03d}"
        )

    print()

    # Target clearly to the right.
    for trial in range(TRIALS_PER_SIDE):

        result = run_trial(
            brain,
            vision,
            world,
            decoder,
            target_x=75
        )

        results["RIGHT"].append(result)

        print(
            f"RIGHT trial {trial + 1:02d}: "
            f"L={result[0]:03d} "
            f"R={result[1]:03d} "
            f"none={result[2]:03d}"
        )

    print("\n==============================")
    print("          SUMMARY")
    print("==============================")

    for side in ["LEFT", "RIGHT"]:

        total_left = sum(x[0] for x in results[side])
        total_right = sum(x[1] for x in results[side])
        total_none = sum(x[2] for x in results[side])

        print()
        print(side)
        print(f"left actions:  {total_left}")
        print(f"right actions: {total_right}")
        print(f"none actions:  {total_none}")


if __name__ == "__main__":
    main()