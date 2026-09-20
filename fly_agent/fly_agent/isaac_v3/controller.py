from .retina import FlyRetina
from .sensory import HybridVisualEncoder
from .trace import DescendingTrace


class HybridIsaacController:
    """One frame -> fly retina -> frozen MaleCNS -> DN trace -> decoder."""

    def __init__(self, fly_agent_brain, decoder=None):
        self.fly_agent_brain = fly_agent_brain
        self.retina = FlyRetina()
        self.encoder = HybridVisualEncoder(fly_agent_brain)
        self.trace = DescendingTrace(fly_agent_brain)
        self.decoder = decoder
        self.last_retina = None
        self.last_fired = None
        self.last_dn = None
        self.last_probs = None

    @property
    def dn_size(self):
        return self.trace.size

    def reset(self, seed=12345):
        self.fly_agent_brain.brain.reset(seed=int(seed))
        self.retina.reset()
        self.trace.reset()
        self.last_retina = None
        self.last_fired = None
        self.last_dn = None
        self.last_probs = None

    def step_brain(self, frame_bgr, grid_features=None, tactical=None, dopamine=None):
        retina = self.retina.process(frame_bgr)
        fired = self.encoder.step(retina, grid_features=grid_features, tactical=tactical, dopamine=dopamine)
        dn = self.trace.update(fired)
        self.last_retina = retina
        self.last_fired = fired
        self.last_dn = dn
        return dn

    def step(self, frame_bgr):
        dn = self.step_brain(frame_bgr)
        if self.decoder is None:
            return set(), None, dn
        actions, probs = self.decoder.predict_actions(dn)
        self.last_probs = probs
        return actions, probs, dn
