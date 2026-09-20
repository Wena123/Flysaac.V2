import numpy as np

from brain.fly import FlyAgentBrain
from brain.decoder import ActionDecoder
from environment.arena2d import Arena2D

from learning.plasticity import RewardPlasticity


LC10_DRIVE = 0.8

TRAIN_EPISODES = 80
MAX_STEPS = 200

DECODER_WINDOW = 3


def stimulate(
    brain,
    lc10_left,
    lc10_right,
    delta
):

    if delta < 0:

        return brain.brain.step(
            inject=[
                (
                    lc10_left,
                    LC10_DRIVE
                )
            ]
        )

    if delta > 0:

        return brain.brain.step(
            inject=[
                (
                    lc10_right,
                    LC10_DRIVE
                )
            ]
        )

    return brain.brain.step()


def measure_pathway(
    brain,
    stim_cells,
    target_neuron
):

    results = []

    for trial in range(5):

        brain.brain.reset(
            seed=1000 + trial
        )

        count = 0

        for _ in range(200):

            fired = brain.brain.step(
                inject=[
                    (
                        stim_cells,
                        LC10_DRIVE
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
                count += 1

        results.append(count)

    return results


def run_episode(
    brain,
    decoder,
    world,
    lc10_left,
    lc10_right,
    target_x,
    seed,
    plasticity=None,
    learning=False
):

    world.reset()

    world.player_x = 50
    world.target_x = target_x

    brain.brain.reset(
        seed=seed
    )

    decoder.reset()

    if plasticity is not None:
        plasticity.reset_traces()

    total_reward = 0.0

    for step in range(MAX_STEPS):

        state = world.state()

        delta = (
            state["target_x"]
            - state["player_x"]
        )

        fired = stimulate(
            brain,
            lc10_left,
            lc10_right,
            delta
        )

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        if learning:

            plasticity.observe(
                fired
            )

        action = decoder.decode(
            fired
        )

        state, reward, done = world.step(
            action
        )

        total_reward += reward

        if learning:

            plasticity.apply_reward(
                reward,
                action
            )

        if done:
            break

    success = (
        state["player_x"]
        == state["target_x"]
    )

    return {
        "success": success,
        "final": state["player_x"],
        "steps": step + 1,
        "reward": total_reward,
    }


def evaluate(
    label,
    brain,
    decoder,
    world,
    lc10_left,
    lc10_right
):

    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    targets = [
        15, 20, 25,
        30, 35, 40,
        60, 65, 70,
        75, 80, 85
    ]

    successes = 0

    for i, target in enumerate(targets):

        result = run_episode(
            brain,
            decoder,
            world,
            lc10_left,
            lc10_right,
            target_x=target,
            seed=5000 + i,
            learning=False
        )

        if result["success"]:
            successes += 1

        print(
            f"target={target:03d} | "
            f"final={result['final']:03d} | "
            f"steps={result['steps']:03d} | "
            f"{'SUCCESS' if result['success'] else 'FAIL'}"
        )

    print()
    print(
        f"Successes: "
        f"{successes}/{len(targets)}"
    )

    return successes


def main():

    print("================================")
    print("   MALECNS REWARD LEARNING TEST")
    print("================================")

    brain = FlyAgentBrain()

    world = Arena2D()

    decoder = ActionDecoder(
        brain,
        window=DECODER_WINDOW
    )

    lc10_left = np.asarray(
        brain.brain.cells(
            ["LC10a"],
            side="L"
        )
    )

    lc10_right = np.asarray(
        brain.brain.cells(
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

    plasticity = RewardPlasticity(
        brain,
        learning_rate=0.0
    )

    # --------------------------------
    # ORIGINAL BRAIN
    # --------------------------------

    print()
    print("Original pathway response")

    original_left = measure_pathway(
        brain,
        lc10_left,
        dna02_left
    )

    original_right = measure_pathway(
        brain,
        lc10_right,
        dna02_right
    )

    print(
        "LEFT:",
        original_left
    )

    print(
        "RIGHT:",
        original_right
    )

    # --------------------------------
    # DAMAGE THE PATHWAYS
    # --------------------------------

    plasticity.impair(
        factor=0.75
    )

    impaired_left = measure_pathway(
        brain,
        lc10_left,
        dna02_left
    )

    impaired_right = measure_pathway(
        brain,
        lc10_right,
        dna02_right
    )

    print()
    print("After impairment")

    print(
        "LEFT:",
        impaired_left
    )

    print(
        "RIGHT:",
        impaired_right
    )

    evaluate(
        "BEFORE LEARNING",
        brain,
        decoder,
        world,
        lc10_left,
        lc10_right
    )

    # --------------------------------
    # LEARNING
    # --------------------------------

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    rng = np.random.default_rng(
        12345
    )

    training_successes = 0

    for episode in range(
        TRAIN_EPISODES
    ):

        if episode % 2 == 0:

            # LEFT target
            target = int(
                rng.integers(
                    10,
                    40
                )
            )

        else:

            # RIGHT target
            target = int(
                rng.integers(
                    61,
                    90
                )
            )

        result = run_episode(
            brain,
            decoder,
            world,
            lc10_left,
            lc10_right,
            target_x=target,
            seed=10000 + episode,
            plasticity=plasticity,
            learning=True
        )

        if result["success"]:
            training_successes += 1

        print(
            f"episode "
            f"{episode + 1:02d}/{TRAIN_EPISODES} | "
            f"target={target:03d} | "
            f"final={result['final']:03d} | "
            f"reward={result['reward']:+.1f} | "
            f"Lmult="
            f"{plasticity.mean_multiplier('left'):.3f} | "
            f"Rmult="
            f"{plasticity.mean_multiplier('right'):.3f} | "
            f"{'SUCCESS' if result['success'] else 'FAIL'}"
        )

    # --------------------------------
    # AFTER LEARNING
    # --------------------------------

    learned_left = measure_pathway(
        brain,
        lc10_left,
        dna02_left
    )

    learned_right = measure_pathway(
        brain,
        lc10_right,
        dna02_right
    )

    print()
    print("================================")
    print("      PATHWAY COMPARISON")
    print("================================")

    print()
    print("LEFT")
    print(
        "original:",
        original_left
    )
    print(
        "damaged: ",
        impaired_left
    )
    print(
        "learned: ",
        learned_left
    )

    print()
    print("RIGHT")
    print(
        "original:",
        original_right
    )
    print(
        "damaged: ",
        impaired_right
    )
    print(
        "learned: ",
        learned_right
    )

    print()

    print(
        "Final LEFT multiplier:",
        plasticity.mean_multiplier(
            "left"
        )
    )

    print(
        "Final RIGHT multiplier:",
        plasticity.mean_multiplier(
            "right"
        )
    )

    evaluate(
        "AFTER LEARNING",
        brain,
        decoder,
        world,
        lc10_left,
        lc10_right
    )


if __name__ == "__main__":
    main()