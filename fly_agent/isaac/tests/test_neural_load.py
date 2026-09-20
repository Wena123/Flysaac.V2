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


STEPS = 250
BURN_IN = 50
SEED = 12345


ACTION_GROUPS = {
    "move_left": "steer_L",
    "move_right": "steer_R",

    "move_up": "forward_L",
    "move_down": "forward_R",

    "shoot_left": "punch_L",
    "shoot_right": "punch_R",

    "shoot_up": "kick_L",
    "shoot_down": "kick_R",

    "bomb": "escape_L",
    "active_item": "escape_R",
    "consumable": "backward_L",
}


def make_motor_sets(
    brain,
):

    result = {}

    for action, group_name in (
        ACTION_GROUPS.items()
    ):

        result[action] = set(
            int(x)
            for x in brain.groups[
                group_name
            ]
        )

    return result


def remove_direct_color(
    features,
):

    result = dict(
        features
    )

    for key in (
        "red_grid",
        "green_grid",
        "blue_grid",
    ):

        if key in result:

            result[key] = np.zeros_like(
                result[key]
            )

    return result


def run_phase(
    label,
    brain,
    perception,
    vision,
    frame,
    motor_sets,
    mode,
):

    print()
    print(
        "================================"
    )

    print(
        label
    )

    print(
        "================================"
    )

    # Same initial brain state for every phase.
    brain.brain.reset(
        seed=SEED
    )

    perception.reset()

    fired_counts = []

    motor_spikes = {
        action: 0
        for action in motor_sets
    }

    spatial_left = []
    spatial_right = []

    drive_left = []
    drive_right = []

    color_left = {
        "red": [],
        "green": [],
        "blue": [],
    }

    color_right = {
        "red": [],
        "green": [],
        "blue": [],
    }

    for step in range(
        STEPS
    ):

        # =========================================
        # A: NO INPUT
        # =========================================

        if mode == "none":

            fired = (
                brain.brain.step()
            )

        # =========================================
        # B/C: STATIC ISAAC FRAME
        # =========================================

        else:

            features = (
                perception.process(
                    frame
                )
            )

            if mode == "spatial_only":

                features = (
                    remove_direct_color(
                        features
                    )
                )

            fired = vision.step(
                features
            )

        # Ignore initial transient.
        if step < BURN_IN:
            continue

        fired = np.asarray(
            fired,
            dtype=np.int64,
        )

        fired_counts.append(
            len(fired)
        )

        fired_set = set(
            fired.tolist()
        )

        for action, cells in (
            motor_sets.items()
        ):

            motor_spikes[action] += (
                len(
                    fired_set
                    & cells
                )
            )

        if mode != "none":

            spatial_left.append(
                vision.last_spatial_left
            )

            spatial_right.append(
                vision.last_spatial_right
            )

            drive_left.append(
                vision.last_left_drive
            )

            drive_right.append(
                vision.last_right_drive
            )

            for color in (
                "red",
                "green",
                "blue",
            ):

                color_left[
                    color
                ].append(
                    vision
                    .last_left_colors[
                        color
                    ]
                )

                color_right[
                    color
                ].append(
                    vision
                    .last_right_colors[
                        color
                    ]
                )

        if (
            step % 50
            == 0
        ):

            print(
                f"step={step:03d} | "
                f"fired={len(fired):5d}"
            )

    counts = np.asarray(
        fired_counts,
        dtype=np.float32,
    )

    result = {
        "label": label,

        "mean_fired":
            float(
                counts.mean()
            ),

        "min_fired":
            int(
                counts.min()
            ),

        "max_fired":
            int(
                counts.max()
            ),

        "motor_spikes":
            motor_spikes,
    }

    if mode != "none":

        result[
            "spatial_left"
        ] = float(
            np.mean(
                spatial_left
            )
        )

        result[
            "spatial_right"
        ] = float(
            np.mean(
                spatial_right
            )
        )

        result[
            "drive_left"
        ] = float(
            np.mean(
                drive_left
            )
        )

        result[
            "drive_right"
        ] = float(
            np.mean(
                drive_right
            )
        )

        result[
            "color_left"
        ] = {
            color:
                float(
                    np.mean(
                        values
                    )
                )

            for color, values
            in color_left.items()
        }

        result[
            "color_right"
        ] = {
            color:
                float(
                    np.mean(
                        values
                    )
                )

            for color, values
            in color_right.items()
        }

    return result


def print_result(
    result,
):

    print()
    print(
        result["label"]
    )

    print(
        "  fired mean:",
        f"{result['mean_fired']:.1f}"
    )

    print(
        "  fired min :",
        result["min_fired"]
    )

    print(
        "  fired max :",
        result["max_fired"]
    )

    print()
    print(
        "  Motor population spikes:"
    )

    for action, count in (
        result[
            "motor_spikes"
        ].items()
    ):

        print(
            f"    {action:12s}: "
            f"{count}"
        )

    if (
        "spatial_left"
        in result
    ):

        print()
        print(
            "  Spatial LC10 mean:"
        )

        print(
            "    LEFT :",
            f"{result['spatial_left']:.2f}"
        )

        print(
            "    RIGHT:",
            f"{result['spatial_right']:.2f}"
        )

        print()
        print(
            "  Peak-drive mean:"
        )

        print(
            "    LEFT :",
            f"{result['drive_left']:.3f}"
        )

        print(
            "    RIGHT:",
            f"{result['drive_right']:.3f}"
        )

        print()
        print(
            "  Colour activity LEFT:",
            result[
                "color_left"
            ]
        )

        print(
            "  Colour activity RIGHT:",
            result[
                "color_right"
            ]
        )


def main():

    print(
        "================================"
    )

    print(
        "   MALECNS NEURAL LOAD TEST"
    )

    print(
        "================================"
    )

    print()
    print(
        "NO learning."
    )

    print(
        "NO keyboard control."
    )

    print(
        "NO exploration."
    )

    print()
    print(
        "Keep Isaac open on a normal "
        "stationary room."
    )

    # ========================================================
    # CAPTURE ONE EXACT ISAAC FRAME
    # ========================================================

    env = IsaacEnvironment()

    frame = env.observe()

    print()
    print(
        "Captured frame:",
        frame.shape
    )

    # We no longer need Isaac to move.
    # Every visual phase sees this exact same image.

    # ========================================================
    # BRAIN
    # ========================================================

    brain = FlyAgentBrain()

    perception = IsaacPerception(
        motion_threshold=18
    )

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

    motor_sets = (
        make_motor_sets(
            brain
        )
    )

    try:

        # ====================================================
        # A — PURE MALECNS
        # ====================================================

        baseline = run_phase(
            label=(
                "A - NO SENSORY INPUT"
            ),

            brain=brain,

            perception=perception,

            vision=vision,

            frame=frame,

            motor_sets=motor_sets,

            mode="none",
        )

        # ====================================================
        # B — SPATIAL / STATIC VISUAL SIGNAL
        #     RGB POOLS DISABLED
        # ====================================================

        spatial = run_phase(
            label=(
                "B - STATIC SPATIAL VISION "
                "(DIRECT RGB OFF)"
            ),

            brain=brain,

            perception=perception,

            vision=vision,

            frame=frame,

            motor_sets=motor_sets,

            mode="spatial_only",
        )

        # ====================================================
        # C — FULL STATIC VISION
        # ====================================================

        full = run_phase(
            label=(
                "C - FULL STATIC VISION "
                "(RGB ON)"
            ),

            brain=brain,

            perception=perception,

            vision=vision,

            frame=frame,

            motor_sets=motor_sets,

            mode="full",
        )

    finally:

        env.stop()

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print()
    print(
        "################################"
    )

    print(
        "          FINAL RESULTS"
    )

    print(
        "################################"
    )

    print_result(
        baseline
    )

    print_result(
        spatial
    )

    print_result(
        full
    )

    # ========================================================
    # RELATIVE LOAD
    # ========================================================

    baseline_mean = (
        baseline[
            "mean_fired"
        ]
    )

    print()
    print(
        "================================"
    )

    print(
        "RELATIVE TO NO-INPUT BASELINE"
    )

    print(
        "================================"
    )

    if baseline_mean > 0:

        print(
            "Spatial-only:",
            f"{spatial['mean_fired'] / baseline_mean:.3f}x"
        )

        print(
            "Full vision :",
            f"{full['mean_fired'] / baseline_mean:.3f}x"
        )

    else:

        print(
            "Baseline firing was zero."
        )


if __name__ == "__main__":
    main()