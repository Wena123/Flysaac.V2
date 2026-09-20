import time
from collections import defaultdict

import numpy as np

from brain.fly import FlyAgentBrain
from learning.plasticity import RewardPlasticity

from isaac.capture import IsaacCapture
from isaac.perception import IsaacPerception
from isaac.lc10_vision import IsaacLC10Vision
from isaac.motor_map import IsaacMotorMap


def main():

    print("==============================")
    print(" ISAAC -> MALECNS DEBUG v2")
    print("==============================")

    capture = IsaacCapture()

    perception = IsaacPerception(
        motion_threshold=12
    )

    brain = FlyAgentBrain()

    plasticity = RewardPlasticity(
        brain,
        learning_rate=0.0,
        homeostasis=0.0,
    )

    plasticity.load_checkpoint(
        "learned_brain.npz"
    )

    vision = IsaacLC10Vision(
        brain,
        gain=4.0,
        max_drive=0.8,
        minimum_activity=0.01,
    )

    motor = IsaacMotorMap(
        brain,
        window=10,
        enable_taps=False,
    )

    steer_left = np.asarray(
        brain.groups["steer_L"],
        dtype=np.int64,
    )

    steer_right = np.asarray(
        brain.groups["steer_R"],
        dtype=np.int64,
    )

    brain.brain.reset(
        seed=12345
    )

    perception.reset()
    motor.reset()

    # -----------------------------
    # STATISTICS
    # -----------------------------

    max_totals = {
        action: 0
        for action in motor.channels
    }

    action_events = defaultdict(int)

    dna_left_spikes = 0
    dna_right_spikes = 0

    step = 0

    print()
    print("Isaac will be focused in 3 seconds.")
    print()
    print("NO KEYS WILL BE PRESSED.")
    print()
    print(
        "This test records every motor event,"
    )
    print(
        "not only every 10th step."
    )
    print()

    time.sleep(3)

    capture.window.focus()

    time.sleep(1)

    try:

        while True:

            # -------------------------
            # CAPTURE ISAAC
            # -------------------------

            frame = capture.frame()

            features = perception.process(
                frame
            )

            # -------------------------
            # ISAAC -> LC10a -> CNS
            # -------------------------

            fired = vision.step(
                features
            )

            # -------------------------
            # CNS -> MOTOR DECODER
            # -------------------------

            held, taps, totals = (
                motor.decode(
                    fired
                )
            )

            # -------------------------
            # RAW DNa02
            # -------------------------

            raw_left = np.intersect1d(
                fired,
                steer_left,
            ).size

            raw_right = np.intersect1d(
                fired,
                steer_right,
            ).size

            dna_left_spikes += raw_left
            dna_right_spikes += raw_right

            step += 1

            # -------------------------
            # RECORD MAX MOTOR ACTIVITY
            # -------------------------

            for action, value in totals.items():

                if value > max_totals[action]:

                    max_totals[action] = int(
                        value
                    )

            # -------------------------
            # PRINT REAL ACTION EVENTS
            # -------------------------

            if held or taps:

                for action in held:
                    action_events[action] += 1

                for action in taps:
                    action_events[action] += 1

                print(
                    f"ACTION step={step:05d} | "
                    f"held={sorted(held)} | "
                    f"taps={taps}"
                )

            # -------------------------
            # PERIODIC DEBUG
            # -------------------------

            if step % 25 == 0:

                print(
                    f"DEBUG step={step:05d} | "
                    f"motion "
                    f"L={vision.last_left_activity:.3f} "
                    f"R={vision.last_right_activity:.3f} | "
                    f"drive "
                    f"L={vision.last_left_drive:.3f} "
                    f"R={vision.last_right_drive:.3f} | "
                    f"DNa02 "
                    f"L={raw_left} "
                    f"R={raw_right}"
                )

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print("==============================")
        print("          SUMMARY")
        print("==============================")
        print()

        print(
            "Steps:",
            step
        )

        print(
            "Total DNa02 LEFT spikes:",
            dna_left_spikes
        )

        print(
            "Total DNa02 RIGHT spikes:",
            dna_right_spikes
        )

        print()
        print(
            "Maximum motor activity:"
        )

        for action, maximum in (
            max_totals.items()
        ):

            print(
                f"  {action:14s}: "
                f"{maximum}"
            )

        print()
        print(
            "Decoded action events:"
        )

        if not action_events:

            print(
                "  NONE"
            )

        else:

            for action, count in (
                sorted(
                    action_events.items()
                )
            ):

                print(
                    f"  {action:14s}: "
                    f"{count}"
                )

        print()
        print(
            "NO Isaac keys were pressed."
        )


if __name__ == "__main__":
    main()