import time

from isaac.capture import IsaacCapture
from isaac.controls import IsaacControls


class IsaacEnvironment:

    def __init__(self):

        self.capture = IsaacCapture()
        self.controls = IsaacControls()

        self.last_frame_time = time.perf_counter()

    def reset_controls(self):
        """
        Release everything the agent may be holding.
        """
        self.controls.release_all()

    def observe(self):
        """
        Get current Isaac screen.
        """
        return self.capture.frame()

    def act(
        self,
        held_actions=None,
        tap_actions=None,
    ):
        """
        held_actions:
            movement/shooting actions that stay pressed.

        tap_actions:
            one-shot actions such as E/Q/Space.
        """

        if held_actions is None:
            held_actions = set()

        if tap_actions is None:
            tap_actions = []

        self.controls.set_held(
            held_actions
        )

        for action in tap_actions:
            self.controls.tap(
                action,
                duration=0.04
            )

    def stop(self):
        self.controls.release_all()

    