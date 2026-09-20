import time

from isaac.reward_reader import (
    IsaacRewardReader
)


def main():

    print("==============================")
    print("     ISAAC REWARD TEST")
    print("==============================")

    reader = IsaacRewardReader()

    print()
    print(
        "Play Isaac normally."
    )

    print(
        "Kill enemies, clear a room,"
    )

    print(
        "and take damage once."
    )

    print()
    print(
        "CTRL+C stops the test."
    )

    print()

    try:

        while True:

            reward, events = (
                reader.poll()
            )

            for event in events:

                print(
                    f"{event['name']:12s} | "
                    f"reward={event['reward']:+.1f} | "
                    f"extra={event['extra']}"
                )

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print(
            "Reward test stopped."
        )

    finally:

        reader.close()


if __name__ == "__main__":
    main()