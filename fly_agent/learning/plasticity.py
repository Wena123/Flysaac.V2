import numpy as np
import cupy as cp


class RewardPlasticity:
    """
    Reward-modulated plasticity on real MaleCNS synapses.

    For this experiment we modify excitatory synapses
    entering the two DNa02 steering neurons.

    learned_weight = original_weight * multiplier
    """

    def __init__(
        self,
        brain,
        learning_rate=0.01,
        pre_decay=0.90,
        eligibility_decay=0.95,
        min_multiplier=0.05,
        max_multiplier=1.5,
        reward_baseline_rate=0.01,
        homeostasis=0.0,
    ):
        self.brain = brain
        self.core = brain.brain
        self.W = self.core._W

        self.learning_rate = learning_rate
        self.pre_decay = pre_decay
        self.eligibility_decay = eligibility_decay

        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

        # NEW: reward prediction system
        self.reward_baseline_rate = reward_baseline_rate
        self.homeostasis = homeostasis

        self.reward_baseline = {
            "left": 0.0,
            "right": 0.0,
        }

        self.sides = {}

        self._register_side(
            "left",
            int(brain.groups["steer_L"][0])
        )

        self._register_side(
            "right",
            int(brain.groups["steer_R"][0])
        )

    def _register_side(self, name, post_neuron):

        start = int(
            self.W.indptr[post_neuron].item()
        )

        end = int(
            self.W.indptr[post_neuron + 1].item()
        )

        pre_ids = cp.asnumpy(
            self.W.indices[start:end]
        ).astype(np.int64)

        base_weights = cp.asnumpy(
            self.W.data[start:end]
        ).astype(np.float32)

        # Only positive/excitatory synapses
        # are plastic in this first experiment.
        plastic_mask = base_weights > 0

        self.sides[name] = {
            "post": post_neuron,
            "start": start,
            "end": end,

            "pre_ids": pre_ids,
            "base": base_weights,

            "plastic": plastic_mask,

            "multiplier": np.ones(
                len(base_weights),
                dtype=np.float32
            ),

            "pre_trace": np.zeros(
                len(base_weights),
                dtype=np.float32
            ),

            "eligibility": np.zeros(
                len(base_weights),
                dtype=np.float32
            ),
        }

        print(
            f"{name}: "
            f"{len(base_weights)} incoming synapses, "
            f"{plastic_mask.sum()} plastic excitatory synapses"
        )

    def reset_traces(self):

        for side in self.sides.values():

            side["pre_trace"].fill(0.0)
            side["eligibility"].fill(0.0)

    def impair(self, factor=0.60):
        """
        Artificially weaken the steering pathways.
        """

        print(
            f"\nImpairing plastic synapses "
            f"to {factor:.2f}x original strength..."
        )

        for side in self.sides.values():

            mask = side["plastic"]

            side["multiplier"][mask] = factor

            self._apply_side(side)

    def observe(self, fired):
        """
        Update eligibility traces from actual CNS activity.
        """

        fired = np.asarray(
            fired,
            dtype=np.int64
        )

        for side in self.sides.values():

            pre_trace = side["pre_trace"]
            eligibility = side["eligibility"]

            # Presynaptic activity fades over time.
            pre_trace *= self.pre_decay

            if fired.size:

                active = np.isin(
                    side["pre_ids"],
                    fired
                )

                pre_trace[active] = 1.0

            # Old eligibility fades.
            eligibility *= self.eligibility_decay

            # When DNa02 fires, recently active
            # incoming synapses become eligible.
            if (
                fired.size
                and np.any(
                    fired == side["post"]
                )
            ):

                eligibility += pre_trace

            np.clip(
                eligibility,
                0.0,
                5.0,
                out=eligibility
            )

            # Non-plastic synapses stay untouched.
            eligibility[
                ~side["plastic"]
            ] = 0.0

    def apply_reward(self, reward, action):
        """
        Reward prediction error controls learning.

        Unexpected reward:
            strong learning

        Expected reward:
            weak learning

        Homeostasis slowly pulls weights
        back toward the original connectome.
        """

        if action not in self.sides:
            return

        side = self.sides[action]
        mask = side["plastic"]

        # +1, 0, or -1
        signal = float(
            np.clip(
                reward,
                -1.0,
                1.0
            )
        )

        # -----------------------------
        # REWARD PREDICTION ERROR
        # -----------------------------

        expected = self.reward_baseline[action]

        prediction_error = (
            signal - expected
        )

        # Update expected reward.
        self.reward_baseline[action] += (
            self.reward_baseline_rate
            * prediction_error
        )

        # -----------------------------
        # THREE-FACTOR PLASTICITY
        # -----------------------------

        change = (
            self.learning_rate
            * prediction_error
            * side["eligibility"]
        )

        side["multiplier"][mask] += (
            change[mask]
        )

        # -----------------------------
        # HOMEOSTASIS
        # -----------------------------
        #
        # Slowly move weights toward
        # original multiplier = 1.0

        side["multiplier"][mask] += (
            self.homeostasis
            * (
                1.0
                - side["multiplier"][mask]
            )
        )

        # Safety limits.
        np.clip(
            side["multiplier"],
            self.min_multiplier,
            self.max_multiplier,
            out=side["multiplier"]
        )

        # Apply new weights to real MaleCNS matrix.
        self._apply_side(side)

    def _apply_side(self, side):

        new_weights = side["base"].copy()

        mask = side["plastic"]

        new_weights[mask] = (
            side["base"][mask]
            * side["multiplier"][mask]
        )

        self.W.data[
            side["start"]:side["end"]
        ] = cp.asarray(
            new_weights,
            dtype=self.W.data.dtype
        )

    def save_checkpoint(self, path):
        """
        Save learned synaptic state to disk.
        """

        np.savez(
            path,

            left_multiplier=
                self.sides["left"]["multiplier"],

            right_multiplier=
                self.sides["right"]["multiplier"],

            left_reward_baseline=
                self.reward_baseline["left"],

            right_reward_baseline=
                self.reward_baseline["right"],
        )

        print()
        print(
            f"Brain checkpoint saved to:"
            f"\n{path}"
        )


    def load_checkpoint(self, path):
        """
        Load learned synaptic state and apply it
        back into the real MaleCNS synaptic matrix.
        """

        data = np.load(path)

        left = data[
            "left_multiplier"
        ].astype(np.float32)

        right = data[
            "right_multiplier"
        ].astype(np.float32)

        if len(left) != len(
            self.sides["left"]["multiplier"]
        ):
            raise ValueError(
                "LEFT checkpoint size does not "
                "match this MaleCNS model."
            )

        if len(right) != len(
            self.sides["right"]["multiplier"]
        ):
            raise ValueError(
                "RIGHT checkpoint size does not "
                "match this MaleCNS model."
            )

        self.sides[
            "left"
        ]["multiplier"][:] = left

        self.sides[
            "right"
        ]["multiplier"][:] = right

        self.reward_baseline["left"] = float(
            data["left_reward_baseline"]
        )

        self.reward_baseline["right"] = float(
            data["right_reward_baseline"]
        )

        # Actually put learned weights back into MaleCNS.
        self._apply_side(
            self.sides["left"]
        )

        self._apply_side(
            self.sides["right"]
        )

        print()
        print(
            f"Brain checkpoint loaded from:"
            f"\n{path}"
        )

        print(
            "LEFT multiplier:",
            self.mean_multiplier("left")
        )

        print(
            "RIGHT multiplier:",
            self.mean_multiplier("right")
        )

    def mean_multiplier(self, name):

        side = self.sides[name]
        mask = side["plastic"]

        return float(
            side["multiplier"][mask].mean()
        )