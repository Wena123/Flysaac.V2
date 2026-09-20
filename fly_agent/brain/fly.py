import numpy as np
from flybrain import FlyBrain


class FlyAgentBrain:

    def __init__(self):
        print("Loading MaleCNS...")

        self.brain = FlyBrain(
            device="cuda",
            sensory_input=True,
        )

        print(f"Neurons: {self.brain.n}")
        print(f"Visual receptors: {len(self.brain.visual)}")
        print(f"Device: {self.brain.device}")

        self.visual_count = len(self.brain.visual)

    def reset(self):
        self.brain.reset()

    def step(self, eye_drive):

        eye_drive = np.asarray(
            eye_drive,
            dtype=np.float32,
        )

        if eye_drive.shape != (self.visual_count,):
            raise ValueError(
                f"Expected eye_drive shape "
                f"({self.visual_count},), "
                f"got {eye_drive.shape}"
            )

        return self.brain.step(
            eye_drive=eye_drive
        )

    @property
    def groups(self):
        return self.brain.groups