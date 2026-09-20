import time

import numpy as np

from brain.fly import FlyAgentBrain

from isaac.environment import (
    IsaacEnvironment
)

from isaac.perception import (
    IsaacPerception
)

from isaac.lc10_vision import (
    IsaacLC10Vision
)


TEST_SECONDS = 60


def mean_or_zero(
    values,
):

    if not values:
        return 0.0

    return float(
        np.mean(
            values
        )
    )


def max_or_zero(
    values,
):

    if not values:
        return 0.0

    return float(
        np.max(
            values
        )
    )


def main():

    print(
        "================================"
    )

    print(
        "   MALECNS VISION CALIBRATION"
    )

    print(
        "================================"
    )

    print()
    print(
        "Learning: OFF"
    )

    print(
        "Keyboard control: OFF"
    )

    print(
        "Exploration: OFF"
    )

    print()
    print(
        "Just play/move Isaac manually."
    )

    # ========================================================
    # ISAAC
    # ========================================================

    env = IsaacEnvironment()

    perception = IsaacPerception(
        motion_threshold=18
    )

    # ========================================================
    # MALECNS
    # ========================================================

    brain = FlyAgentBrain()

    vision = IsaacLC10Vision(
        brain,

        gain=1.5,
        max_drive=0.40,

        minimum_activity=0.07,

        top_k_spatial=6,

        color_gain=1.0,
        color_max_drive=0.20,

        color_minimum=0.08,

        color_top_k=3,
    )

    brain.brain.reset(
        seed=12345
    )

    perception.reset()

    # ========================================================
    # STATISTICS
    # ========================================================

    fired_counts = []

    mean_salience_values = []

    max_salience_values = []

    left_active_values = []

    right_active_values = []

    left_drive_values = []

    right_drive_values = []

    global_motion_values = []

    step = 0

    start = time.perf_counter()

    try:

        while True:

            elapsed = (
                time.perf_counter()
                - start
            )

            if (
                elapsed
                >= TEST_SECONDS
            ):

                break

            # =================================================
            # CAPTURE
            # =================================================

            frame = env.observe()

            # =================================================
            # RETINA
            # =================================================

            features = (
                perception.process(
                    frame
                )
            )

            grid = np.asarray(
                features[
                    "salience_grid"
                ],
                dtype=np.float32,
            )

            # =================================================
            # MALECNS
            # =================================================

            fired = vision.step(
                features
            )

            # =================================================
            # RECORD
            # =================================================

            fired_counts.append(
                len(fired)
            )

            mean_salience_values.append(
                float(
                    np.mean(
                        grid
                    )
                )
            )

            max_salience_values.append(
                float(
                    np.max(
                        grid
                    )
                )
            )

            global_motion_values.append(
                float(
                    features.get(
                        "global_motion",
                        0.0,
                    )
                )
            )

            left_active_values.append(
                vision.last_spatial_left
            )

            right_active_values.append(
                vision.last_spatial_right
            )

            left_drive_values.append(
                vision.last_left_drive
            )

            right_drive_values.append(
                vision.last_right_drive
            )

            # =================================================
            # LIVE PRINT
            # =================================================

            if (
                step % 10
                == 0
            ):

                print(
                    f"step={step:05d} | "
                    f"motion="
                    f"{features['global_motion']:.4f} | "
                    f"salience mean="
                    f"{np.mean(grid):.4f} | "
                    f"max="
                    f"{np.max(grid):.4f} | "
                    f"LC10 spatial "
                    f"L={vision.last_spatial_left} "
                    f"R={vision.last_spatial_right} | "
                    f"drive "
                    f"L={vision.last_left_drive:.3f} "
                    f"R={vision.last_right_drive:.3f} | "
                    f"fired={len(fired)}"
                )

            step += 1

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print(
            "Calibration stopped manually."
        )

    finally:

        env.stop()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        "================================"
    )

    print(
        "        VISION SUMMARY"
    )

    print(
        "================================"
    )

    print(
        "Steps:",
        step
    )

    print()

    print(
        "Global motion"
    )

    print(
        " mean:",
        f"{mean_or_zero(global_motion_values):.4f}"
    )

    print(
        " max :",
        f"{max_or_zero(global_motion_values):.4f}"
    )

    print()

    print(
        "Visual salience"
    )

    print(
        " mean cell:",
        f"{mean_or_zero(mean_salience_values):.4f}"
    )

    print(
        " mean peak:",
        f"{mean_or_zero(max_salience_values):.4f}"
    )

    print(
        " max peak :",
        f"{max_or_zero(max_salience_values):.4f}"
    )

    print()

    print(
        "Spatial LC10 active"
    )

    print(
        " LEFT mean:",
        f"{mean_or_zero(left_active_values):.2f}",
        "/ 120"
    )

    print(
        " RIGHT mean:",
        f"{mean_or_zero(right_active_values):.2f}",
        "/ 120"
    )

    print()

    print(
        "LC10 peak drive"
    )

    print(
        " LEFT mean:",
        f"{mean_or_zero(left_drive_values):.3f}"
    )

    print(
        " RIGHT mean:",
        f"{mean_or_zero(right_drive_values):.3f}"
    )

    print(
        " LEFT max:",
        f"{max_or_zero(left_drive_values):.3f}"
    )

    print(
        " RIGHT max:",
        f"{max_or_zero(right_drive_values):.3f}"
    )

    print()

    print(
        "MaleCNS firing"
    )

    print(
        " mean:",
        f"{mean_or_zero(fired_counts):.1f}",
        "neurons / step"
    )

    print(
        " max :",
        int(
            max_or_zero(
                fired_counts
            )
        ),
        "neurons"
    )


if __name__ == "__main__":
    main()