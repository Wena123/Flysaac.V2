import numpy as np


class IsaacVisionBridge:
    """
    Converts the 3x3 Isaac motion representation
    into activity for MaleCNS visual receptors.

    This is only a sensory encoding layer.
    It does NOT choose actions.
    """

    def __init__(
        self,
        brain,
        gain=3.0,
        max_drive=0.8,
    ):
        self.brain = brain

        self.gain = float(gain)
        self.max_drive = float(max_drive)

        self.visual_count = len(
            brain.brain.visual
        )

        self.azimuth = np.asarray(
            brain.brain.azimuth,
            dtype=np.float32,
        )

        # Some FlyBrain versions may also expose
        # elevation for the visual receptors.
        elevation = getattr(
            brain.brain,
            "elevation",
            None,
        )

        if elevation is not None:

            elevation = np.asarray(
                elevation,
                dtype=np.float32,
            )

            if len(elevation) == self.visual_count:
                self.elevation = elevation
            else:
                self.elevation = None

        else:
            self.elevation = None

        print(
            "Visual bridge:",
            self.visual_count,
            "receptors"
        )

        print(
            "Elevation available:",
            self.elevation is not None
        )

    def encode(
        self,
        features,
    ):
        """
        Returns an array of shape (6006,).
        """

        drive = np.zeros(
            self.visual_count,
            dtype=np.float32,
        )

        # ---------------------------------
        # HORIZONTAL POSITION
        # ---------------------------------

        left_mask = (
            self.azimuth < -0.33
        )

        center_mask = (
            (self.azimuth >= -0.33)
            & (self.azimuth <= 0.33)
        )

        right_mask = (
            self.azimuth > 0.33
        )

        # If elevation exists, use the full
        # 3x3 representation.
        if self.elevation is not None:

            top_mask = (
                self.elevation > 0.33
            )

            middle_mask = (
                (self.elevation >= -0.33)
                & (self.elevation <= 0.33)
            )

            bottom_mask = (
                self.elevation < -0.33
            )

            regions = {
                "top_left":
                    top_mask & left_mask,

                "top":
                    top_mask & center_mask,

                "top_right":
                    top_mask & right_mask,

                "left":
                    middle_mask & left_mask,

                "center":
                    middle_mask & center_mask,

                "right":
                    middle_mask & right_mask,

                "bottom_left":
                    bottom_mask & left_mask,

                "bottom":
                    bottom_mask & center_mask,

                "bottom_right":
                    bottom_mask & right_mask,
            }

        else:

            # Fallback:
            # combine vertical information and preserve
            # only left / center / right.
            left_value = max(
                features["top_left"],
                features["left"],
                features["bottom_left"],
            )

            center_value = max(
                features["top"],
                features["center"],
                features["bottom"],
            )

            right_value = max(
                features["top_right"],
                features["right"],
                features["bottom_right"],
            )

            regions = {
                "left": left_mask,
                "center": center_mask,
                "right": right_mask,
            }

            features = {
                "left": left_value,
                "center": center_value,
                "right": right_value,
            }

        for name, mask in regions.items():

            value = float(
                features[name]
            )

            value *= self.gain

            value = np.clip(
                value,
                0.0,
                self.max_drive,
            )

            drive[mask] = value

        return drive