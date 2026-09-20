import numpy as np
import cupy as cp

from brain.fly import FlyAgentBrain


STEPS = 200
TRIALS = 3
DRIVE = 0.8

FACTORS = [
    1.00,
    0.75,
    0.60,
    0.50,
    0.40,
    0.30,
    0.20,
    0.10,
]


def measure(
    brain,
    stim_cells,
    target_neuron
):
    results = []

    for trial in range(TRIALS):

        brain.brain.reset(
            seed=1000 + trial
        )

        spikes = 0

        for _ in range(STEPS):

            fired = brain.brain.step(
                inject=[
                    (
                        stim_cells,
                        DRIVE
                    )
                ]
            )

            fired = np.asarray(
                fired,
                dtype=np.int64
            )

            if np.any(
                fired == target_neuron
            ):
                spikes += 1

        results.append(spikes)

    return np.mean(results)


def get_row(brain, neuron):

    W = brain.brain._W

    start = int(
        W.indptr[neuron].item()
    )

    end = int(
        W.indptr[neuron + 1].item()
    )

    original = (
        W.data[start:end]
        .copy()
    )

    return start, end, original


def apply_factor(
    W,
    start,
    end,
    original,
    factor
):

    weights = original.copy()

    # Only weaken positive inputs,
    # same as our learning experiment.
    mask = weights > 0

    weights[mask] *= factor

    W.data[start:end] = weights


def main():

    print("================================")
    print("   IMPAIRMENT THRESHOLD TEST")
    print("================================")

    brain = FlyAgentBrain()

    core = brain.brain
    W = core._W

    lc10_left = np.asarray(
        core.cells(
            ["LC10a"],
            side="L"
        )
    )

    lc10_right = np.asarray(
        core.cells(
            ["LC10a"],
            side="R"
        )
    )

    dna02_left = int(
        brain.groups["steer_L"][0]
    )

    dna02_right = int(
        brain.groups["steer_R"][0]
    )

    (
        l_start,
        l_end,
        l_original
    ) = get_row(
        brain,
        dna02_left
    )

    (
        r_start,
        r_end,
        r_original
    ) = get_row(
        brain,
        dna02_right
    )

    print()
    print(
        f"{'FACTOR':>8} | "
        f"{'LEFT':>8} | "
        f"{'RIGHT':>8}"
    )

    print("-" * 32)

    for factor in FACTORS:

        apply_factor(
            W,
            l_start,
            l_end,
            l_original,
            factor
        )

        apply_factor(
            W,
            r_start,
            r_end,
            r_original,
            factor
        )

        left = measure(
            brain,
            lc10_left,
            dna02_left
        )

        right = measure(
            brain,
            lc10_right,
            dna02_right
        )

        print(
            f"{factor:8.2f} | "
            f"{left:8.2f} | "
            f"{right:8.2f}"
        )

    # Restore original brain.
    W.data[
        l_start:l_end
    ] = l_original

    W.data[
        r_start:r_end
    ] = r_original


if __name__ == "__main__":
    main()