import numpy as np

from brain.fly import FlyAgentBrain


STEPS = 200
TRIALS = 5
DRIVE = 0.8


def count_dna02(
    brain,
    stim_cells,
    target_neuron,
    seed
):
    brain.brain.reset(seed=seed)

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

        if np.any(fired == target_neuron):
            spikes += 1

    return spikes


def run_trials(
    label,
    brain,
    stim_cells,
    target_neuron
):

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    results = []

    for trial in range(TRIALS):

        # Same seeds between conditions,
        # so noise is comparable.
        seed = 1000 + trial

        spikes = count_dna02(
            brain,
            stim_cells,
            target_neuron,
            seed
        )

        results.append(spikes)

        print(
            f"trial {trial + 1:02d}: "
            f"DNa02 spikes = {spikes}"
        )

    mean = np.mean(results)

    print(
        f"mean = {mean:.2f}"
    )

    return results


def main():

    print("================================")
    print("   INTERNAL WEIGHT MUTATION TEST")
    print("================================")

    brain = FlyAgentBrain()

    core = brain.brain

    if core.device != "cuda":
        raise RuntimeError(
            "This test currently expects CUDA."
        )

    lc10_left = np.asarray(
        core.cells(
            ["LC10a"],
            side="L"
        )
    )

    dna02_left = int(
        core.cells(
            ["DNa02"],
            side="L"
        )[0]
    )

    print()
    print(
        "LC10a L neurons:",
        len(lc10_left)
    )

    print(
        "DNa02 L:",
        dna02_left
    )

    # ---------------------------------
    # Locate all synapses INTO DNa02_L
    # ---------------------------------

    W = core._W

    start = int(
        W.indptr[dna02_left].item()
    )

    end = int(
        W.indptr[dna02_left + 1].item()
    )

    incoming_count = (
        end - start
    )

    print()
    print(
        "Incoming synapses to DNa02_L:",
        incoming_count
    )

    # Save the REAL original weights.
    original_weights = (
        W.data[start:end].copy()
    )

    original_cpu = (
        original_weights.get()
    )

    print(
        "Weight min:",
        original_cpu.min()
    )

    print(
        "Weight max:",
        original_cpu.max()
    )

    print(
        "Mean abs weight:",
        np.mean(
            np.abs(original_cpu)
        )
    )

    # =================================
    # 1. ORIGINAL CONNECTOME
    # =================================

    baseline = run_trials(
        "ORIGINAL CONNECTOME",
        brain,
        lc10_left,
        dna02_left
    )

    # =================================
    # 2. WEAKEN REAL SYNAPSES
    # =================================

    print()
    print(
        "Weakening DNa02_L input "
        "synapses to 10%..."
    )

    W.data[start:end] *= 0.10

    weakened = run_trials(
        "DNa02_L INPUTS AT 10%",
        brain,
        lc10_left,
        dna02_left
    )

    # =================================
    # 3. RESTORE CONNECTOME
    # =================================

    print()
    print(
        "Restoring original weights..."
    )

    W.data[start:end] = (
        original_weights
    )

    restored = run_trials(
        "RESTORED CONNECTOME",
        brain,
        lc10_left,
        dna02_left
    )

    # =================================
    # RESULTS
    # =================================

    print()
    print("================================")
    print("             RESULTS")
    print("================================")

    print(
        "Original mean:",
        np.mean(baseline)
    )

    print(
        "Weakened mean:",
        np.mean(weakened)
    )

    print(
        "Restored mean:",
        np.mean(restored)
    )

    print()

    difference = (
        np.mean(baseline)
        - np.mean(weakened)
    )

    print(
        "Spike reduction:",
        difference
    )


if __name__ == "__main__":
    main()