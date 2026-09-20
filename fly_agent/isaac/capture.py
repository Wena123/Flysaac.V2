import time

import mss
import numpy as np

from isaac.window import (
    IsaacWindow
)


class IsaacCapture:

    def __init__(
        self,
    ):

        self.window = (
            IsaacWindow()
        )

        self.sct = (
            mss.mss()
        )

    # =========================================================
    # CAPTURE
    # =========================================================

    def frame(
        self,
    ):

        last_error = None

        # Try a few times because Windows can take
        # a moment to restore a minimized game window.

        for attempt in range(
            3
        ):

            try:

                rect = (
                    self.window
                    .client_rect()
                )

                if (
                    rect["width"] <= 0
                    or rect["height"] <= 0
                ):

                    raise RuntimeError(
                        "Isaac capture region "
                        "has zero size."
                    )

                shot = self.sct.grab(
                    rect
                )

                frame = np.asarray(
                    shot
                )

                # MSS gives BGRA.
                # Keep BGR because OpenCV uses BGR.

                frame = (
                    frame[
                        :,
                        :,
                        :3
                    ].copy()
                )

                if (
                    frame.shape[0] <= 0
                    or frame.shape[1] <= 0
                ):

                    raise RuntimeError(
                        "Captured Isaac frame "
                        "was empty."
                    )

                return frame

            except Exception as error:

                last_error = error

                print(
                    "Isaac capture failed "
                    f"(attempt {attempt + 1}/3):",
                    error
                )

                print(
                    "Restoring Isaac window..."
                )

                self.window.restore()

                time.sleep(
                    0.5
                )

        raise RuntimeError(
            "Could not capture Isaac after "
            "3 attempts."
        ) from last_error