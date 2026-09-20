import time

from isaac.capture import IsaacCapture
from isaac.perception import IsaacPerception


def main():

    print("==============================")
    print("     ISAAC PERCEPTION TEST")
    print("==============================")

    capture = IsaacCapture()

    perception = IsaacPerception(
        motion_threshold=12
    )

    print()
    print(
        "Isaac will be focused in 3 seconds."
    )

    print(
        "After that, move around, shoot, "
        "or enter a room with enemies."
    )

    print()
    print(
        "Press CTRL+C in the terminal "
        "when you want to stop."
    )

    print()

    time.sleep(3)

    capture.window.focus()

    time.sleep(1)

    perception.reset()

    frame_number = 0

    try:

        while True:

            frame = capture.frame()

            features = (
                perception.process(
                    frame
                )
            )

            frame_number += 1

            # Don't flood terminal quite as badly.
            if frame_number % 3 == 0:

                print(
                    " | ".join(
                        f"{name}="
                        f"{value:.3f}"
                        for name, value
                        in features.items()
                    )
                )

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print(
            "Perception test stopped."
        )


if __name__ == "__main__":
    main()