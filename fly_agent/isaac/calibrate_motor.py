import numpy as np

from brain.fly import FlyAgentBrain
from learning.plasticity import RewardPlasticity
from isaac.motor_map import IsaacMotorMap


STEPS = 1000


def rolling_max(values, window):

    values = np.asarray(
        values,
        dtype=np.int32
    )

    if len(values) < window:
        return int(values.sum())

    kernel = np.ones(
        window,
        dtype=np.int32
    )

    sums = np.convolve(
        values,
        kernel,
        mode="valid"
    )

    return int(sums.max())


def main():

    print("==============================")
    print("   MALECNS MOTOR CALIBRATION")
    print("==============================")

    brain = FlyAgentBrain()

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

    recordings = {
        action: []
        for action in motor.channels
    }

    print()
    print(
        f"Recording {STEPS} steps "
        "with NO sensory input..."
    )
    print()

    for step in range(STEPS):

        fired = brain.brain.step()

        for action, neurons in (
            motor.channels.items()
        ):

            count = np.intersect1d(
                fired,
                neurons
            ).size

            recordings[action].append(
                count
            )

        if (
            (step + 1) % 100 == 0
        ):
            print(
                f"{step + 1}/{STEPS}"
            )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    for action, values in recordings.items():

        values = np.asarray(
            values
        )

        total = int(
            values.sum()
        )

        active_steps = int(
            np.count_nonzero(values)
        )

        max3 = rolling_max(
            values,
            3
        )

        max5 = rolling_max(
            values,
            5
        )

        max10 = rolling_max(
            values,
            10
        )

        print()
        print(action)

        print(
            f"  total spikes:       {total}"
        )

        print(
            f"  active steps:       "
            f"{active_steps}/{STEPS}"
        )

        print(
            f"  max spikes / 3:     {max3}"
        )

        print(
            f"  max spikes / 5:     {max5}"
        )

        print(
            f"  max spikes / 10:    {max10}"
        )


if __name__ == "__main__":
    main()