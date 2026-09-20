from brain.fly import FlyAgentBrain
from brain.decoder import ActionDecoder
from environment.world import SimpleWorld
from vision.encoder import VisionEncoder


EPISODES = 50
MAX_STEPS = 200


def main():
    print("================================")
    print("       FlyAgent evaluation")
    print("================================")

    brain = FlyAgentBrain()
    decoder = ActionDecoder(brain)
    vision = VisionEncoder(brain)
    world = SimpleWorld()

    successes = 0

    left_targets = 0
    right_targets = 0

    correct_moves = 0
    wrong_moves = 0

    total_left_actions = 0
    total_right_actions = 0
    total_none_actions = 0

    for episode in range(EPISODES):

        state = world.reset()
        brain.reset()
        decoder.reset()

        start_x = state["player_x"]
        target_x = state["target_x"]

        if target_x < start_x:
            target_side = "left"
            left_targets += 1
        else:
            target_side = "right"
            right_targets += 1

        episode_correct = 0
        episode_wrong = 0

        for step in range(MAX_STEPS):

            state = world.state()

            eye_drive = vision.encode(state)

            fired = brain.step(
                eye_drive
            )

            action = decoder.decode(
                fired
            )

            if action == "left":
                total_left_actions += 1

                if target_x < state["player_x"]:
                    correct_moves += 1
                    episode_correct += 1
                else:
                    wrong_moves += 1
                    episode_wrong += 1

            elif action == "right":
                total_right_actions += 1

                if target_x > state["player_x"]:
                    correct_moves += 1
                    episode_correct += 1
                else:
                    wrong_moves += 1
                    episode_wrong += 1

            else:
                total_none_actions += 1

            state, reward, done = world.step(
                action
            )

            if done:
                break

        success = state["player_x"] == target_x

        if success:
            successes += 1

        print(
            f"Episode {episode + 1:02d} | "
            f"target={target_x:03d} "
            f"side={target_side:5s} | "
            f"final={state['player_x']:03d} | "
            f"correct={episode_correct:03d} "
            f"wrong={episode_wrong:03d} | "
            f"{'SUCCESS' if success else 'FAIL'}"
        )

    directional_moves = correct_moves + wrong_moves

    print("\n================================")
    print("             RESULTS")
    print("================================")

    print(f"Episodes:       {EPISODES}")
    print(f"Successes:      {successes}/{EPISODES}")

    print()
    print(f"Left targets:   {left_targets}")
    print(f"Right targets:  {right_targets}")

    print()
    print(f"Left actions:   {total_left_actions}")
    print(f"Right actions:  {total_right_actions}")
    print(f"None actions:   {total_none_actions}")

    print()
    print(f"Correct moves:  {correct_moves}")
    print(f"Wrong moves:    {wrong_moves}")

    if directional_moves > 0:
        accuracy = (
            correct_moves /
            directional_moves
        ) * 100

        print(
            f"Direction accuracy: "
            f"{accuracy:.1f}%"
        )


if __name__ == "__main__":
    main()