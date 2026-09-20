import cv2

from isaac.capture import IsaacCapture


def main():

    capture = IsaacCapture()

    frame = capture.frame()

    print(
        "Captured frame:",
        frame.shape
    )

    cv2.imwrite(
        "isaac_capture_test.png",
        frame
    )

    print(
        "Saved:"
        "\nisaac_capture_test.png"
    )


if __name__ == "__main__":
    main()