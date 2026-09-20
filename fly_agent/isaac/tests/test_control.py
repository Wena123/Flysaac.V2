import sys
import time

from isaac.controls import IsaacControls


def main():

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m isaac.test_control ACTION"
        )

        print()
        print(
            "Available actions:"
        )

        for action in (
            IsaacControls.ACTIONS
        ):
            print(
                " ",
                action
            )

        return

    action = sys.argv[1]

    controls = IsaacControls()

    if action not in controls.ACTIONS:

        raise ValueError(
            f"Unknown action: {action}"
        )

    print(
        f"Testing: {action}"
    )

    print(
        "Isaac will receive the key "
        "in 2 seconds..."
    )

    time.sleep(2)

    controls.focus()

    time.sleep(0.3)

    try:

        controls.tap(
            action,
            duration=0.20
        )

    finally:

        controls.release_all()

    print(
        "Done."
    )


if __name__ == "__main__":
    main()