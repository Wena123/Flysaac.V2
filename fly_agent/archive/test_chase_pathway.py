import numpy as np

from brain.fly import FlyAgentBrain


STEPS = 200
TRIALS = 5
DRIVE = 0.8


def run_trial(brain, stim_cells, dnl, dnr):

    brain.reset()

    left_spikes = 0
    right_spikes = 0
    total_spikes = 0

    for _ in range(STEPS):

        if stim_cells is None:
            fired = brain.brain.step()
        else:
            fired = brain.brain.step(
                inject=[
                    (stim_cells, DRIVE)
                ]
            )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        total_spikes += fired.size

        if fired.size:

            left_spikes += np.intersect1d(
                fired,
                dnl
            ).size

            right_spikes += np.intersect1d(
                fired,
                dnr
            ).size

    return (
        left_spikes,
        right_spikes,
        total_spikes
    )


def run_condition(
    name,
    brain,
    stim_cells,
    dnl,
    dnr
):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    total_l = 0
    total_r = 0
    total_network = 0

    for trial in range(TRIALS):

        l, r, network = run_trial(
            brain,
            stim_cells,
            dnl,
            dnr
        )

        total_l += l
        total_r += r
        total_network += network

        print(
            f"trial {trial + 1:02d}: "
            f"DNa02_L={l:3d} | "
            f"DNa02_R={r:3d} | "
            f"network={network:8d}"
        )

    print()
    print(
        f"TOTAL: "
        f"DNa02_L={total_l} | "
        f"DNa02_R={total_r} | "
        f"network={total_network}"
    )


def main():

    print("================================")
    print("       LC10a -> DNa02 test")
    print("================================")

    brain = FlyAgentBrain()

    # Visual pursuit neurons
    lc10_l = np.asarray(
        brain.brain.cells(
            ["LC10a"],
            side="L"
        )
    )

    lc10_r = np.asarray(
        brain.brain.cells(
            ["LC10a"],
            side="R"
        )
    )

    # Identified steering descending neurons
    dna02_l = np.asarray(
        brain.brain.cells(
            ["DNa02"],
            side="L"
        )
    )

    dna02_r = np.asarray(
        brain.brain.cells(
            ["DNa02"],
            side="R"
        )
    )

    # All descending neurons
    descending = np.asarray(
        brain.brain.cells(
            ["descending_neuron"]
        )
    )

    print()
    print("LC10a left:")
    print(" count:", len(lc10_l))
    print(" IDs:", lc10_l)

    print()
    print("LC10a right:")
    print(" count:", len(lc10_r))
    print(" IDs:", lc10_r)

    print()
    print("DNa02 left:")
    print(" count:", len(dna02_l))
    print(" IDs:", dna02_l)

    print()
    print("DNa02 right:")
    print(" count:", len(dna02_r))
    print(" IDs:", dna02_r)

    print()
    print(
        "All descending neurons:",
        len(descending)
    )

    print()
    print("Existing FlyBrain groups:")
    print(
        "steer_L:",
        brain.groups["steer_L"]
    )
    print(
        "steer_R:",
        brain.groups["steer_R"]
    )

    print()
    print(
        "steer_L == DNa02_L:",
        set(brain.groups["steer_L"])
        == set(dna02_l)
    )

    print(
        "steer_R == DNa02_R:",
        set(brain.groups["steer_R"])
        == set(dna02_r)
    )

    run_condition(
        "NO STIMULUS / BASELINE",
        brain,
        None,
        dna02_l,
        dna02_r
    )

    run_condition(
        "LEFT LC10a STIMULATION",
        brain,
        lc10_l,
        dna02_l,
        dna02_r
    )

    run_condition(
        "RIGHT LC10a STIMULATION",
        brain,
        lc10_r,
        dna02_l,
        dna02_r
    )


if __name__ == "__main__":
    main()