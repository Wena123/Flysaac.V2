import numpy as np

from brain.fly import FlyAgentBrain


def main():

    print("================================")
    print("       MaleCNS group list")
    print("================================")

    brain = FlyAgentBrain()

    groups = brain.groups

    print()
    print("Number of groups:", len(groups))
    print()

    print(
        f"{'GROUP':35s} "
        f"{'SIZE':>8s} "
        f"FIRST NEURON IDS"
    )

    print("-" * 80)

    for name in sorted(groups.keys()):

        neurons = np.asarray(
            groups[name]
        ).reshape(-1)

        preview = neurons[:8]

        preview_text = ", ".join(
            str(int(x))
            for x in preview
        )

        if len(neurons) > 8:
            preview_text += ", ..."

        print(
            f"{name:35s} "
            f"{len(neurons):8d} "
            f"{preview_text}"
        )


if __name__ == "__main__":
    main()