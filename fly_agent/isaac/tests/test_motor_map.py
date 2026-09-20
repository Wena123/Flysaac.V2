from brain.fly import FlyAgentBrain
from learning.plasticity import RewardPlasticity
from isaac.motor_map import IsaacMotorMap


def main():

    print("==============================")
    print("     MALECNS MOTOR TEST")
    print("==============================")

    brain = FlyAgentBrain()

    # Load our known-good learned synapses.
    plasticity = RewardPlasticity(
        brain,
        learning_rate=0.0,
        homeostasis=0.0,
    )

    plasticity.load_checkpoint(
        "learned_brain.npz"
    )

    motor = IsaacMotorMap(
        brain,
        window=3
    )

    brain.brain.reset(
        seed=12345
    )

    motor.reset()

    print()
    print(
        "Watching MaleCNS for 100 steps."
    )
    print(
        "NO Isaac keys will be pressed."
    )
    print()

    for step in range(1, 101):

        # No sensory injection yet.
        # Just observe spontaneous CNS activity.
        fired = brain.brain.step()

        held, taps, totals = (
            motor.decode(fired)
        )

        if held or taps:

            print(
                f"step {step:03d} | "
                f"held={sorted(held)} | "
                f"taps={taps}"
            )

    print()
    print("Test finished.")


if __name__ == "__main__":
    main()