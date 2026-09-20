import time

from isaac.environment import IsaacEnvironment


def main():

    env = IsaacEnvironment()

    try:
        # Capture one frame.
        frame = env.observe()

        print(
            "Frame shape:",
            frame.shape
        )

        print(
            "Testing movement + shooting in 2 seconds..."
        )

        time.sleep(2)

        env.controls.focus()

        # Hold LEFT + SHOOT UP together.
        env.act(
            held_actions={
                "move_left",
                "shoot_up",
            }
        )

        time.sleep(1)

        # Release continuous controls.
        env.act(
            held_actions=set()
        )

        print(
            "Movement/shooting test finished."
        )

        # I would NOT test bomb automatically yet,
        # because it will actually consume a bomb.
        #
        # When you want to test it:
        #
        # env.act(
        #     tap_actions=[
        #         "bomb"
        #     ]
        # )

    finally:

        env.stop()


if __name__ == "__main__":
    main()