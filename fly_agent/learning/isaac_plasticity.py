from pathlib import Path
import math
import time

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


# ============================================================
# ISAAC ACTION -> MALECNS OUTPUT POPULATION
# ============================================================

ACTION_GROUPS = {
    "move_left": "steer_L",
    "move_right": "steer_R",

    "move_up": "forward_L",
    "move_down": "forward_R",

    "shoot_left": "punch_L",
    "shoot_right": "punch_R",

    "shoot_up": "kick_L",
    "shoot_down": "kick_R",

    "bomb": "escape_L",
    "active_item": "escape_R",

    "consumable": "backward_L",
}


# ============================================================
# TEMPORAL CREDIT WINDOWS
# ============================================================
#
# How long an action remains causally relevant after execution.
#
# Movement consequences are usually immediate.
# Shooting takes longer to produce kills.
# Bombs/items can have significantly delayed consequences.
# ============================================================

ACTION_TAU = {
    "move_left": 1.5,
    "move_right": 1.5,
    "move_up": 1.5,
    "move_down": 1.5,

    "shoot_left": 3.0,
    "shoot_right": 3.0,
    "shoot_up": 3.0,
    "shoot_down": 3.0,

    "bomb": 6.0,
    "active_item": 5.0,
    "consumable": 5.0,
}


class IsaacMotorPlasticity:
    """
    Reward-modulated plasticity acting on REAL MaleCNS synapses.

    Two reward modes exist:

    1. apply_reward(...)
       General Isaac reward.

       Recent ACTUAL executed actions receive credit according
       to their age.

       Example:
           enemy dies
           shoot_right 0.2 sec ago -> strong credit
           move_down   0.8 sec ago -> some credit
           bomb        4.0 sec ago -> possible delayed credit

    2. apply_action_reward(...)
       Known action-specific consequence.

       Example:
           move_left + wall + no displacement
               ->
           punish ONLY recently active move_left synapses.

    No external neural policy is used.
    The altered values are the actual MaleCNS synaptic weights.
    """

    def __init__(
        self,
        brain,
        core=None,
        learning_rate=0.001,
        pre_decay=0.90,
        eligibility_decay=0.99,
        reward_scale=10.0,
        min_multiplier=0.25,
        max_multiplier=1.75,
        post_reward_decay=0.50,
    ):

        self.brain = brain

        self.learning_rate = float(
            learning_rate
        )

        self.pre_decay = float(
            pre_decay
        )

        self.eligibility_decay = float(
            eligibility_decay
        )

        self.reward_scale = float(
            reward_scale
        )

        self.min_multiplier = float(
            min_multiplier
        )

        self.max_multiplier = float(
            max_multiplier
        )

        self.post_reward_decay = float(
            post_reward_decay
        )

        # ----------------------------------------------------
        # CORE / REAL CONNECTOME MATRIX
        # ----------------------------------------------------

        if core is None:

            try:
                core = brain.brain.core

            except Exception as error:

                raise RuntimeError(
                    "MaleCNS core was not supplied and "
                    "could not be discovered."
                ) from error

        self.core = core

        if not hasattr(
            self.core,
            "_W",
        ):

            raise RuntimeError(
                "Could not find MaleCNS synaptic matrix core._W"
            )

        self.W = (
            self.core._W
        )

        self.neuron_count = int(
            self.W.shape[1]
        )

        # Reusable boolean array.
        #
        # This is much faster than calling np.isin for every
        # group on every brain step.

        self._fired_mask = np.zeros(
            self.neuron_count,
            dtype=np.bool_,
        )

        # ----------------------------------------------------
        # GPU / CPU MATRIX TYPE
        # ----------------------------------------------------

        self._gpu_weights = (
            cp is not None
            and (
                type(
                    self.W.data
                )
                .__module__
                .startswith(
                    "cupy"
                )
            )
        )

        # ----------------------------------------------------
        # ROW STORAGE
        # ----------------------------------------------------

        self.rows = []

        self.rows_by_action = {
            action: []

            for action in ACTION_GROUPS
        }

        # Last REAL keyboard action time.

        self.last_action_time = {
            action: None

            for action in ACTION_GROUPS
        }

        self._register_rows()

        self.reset_traces()

        self._print_summary()

    # ========================================================
    # ARRAY HELPERS
    # ========================================================

    @staticmethod
    def _numpy(
        value,
    ):

        if hasattr(
            value,
            "get",
        ):

            return np.asarray(
                value.get()
            )

        return np.asarray(
            value
        )

    @staticmethod
    def _scalar_int(
        value,
    ):

        if hasattr(
            value,
            "item",
        ):

            return int(
                value.item()
            )

        return int(
            value
        )

    # ========================================================
    # REGISTER REAL SYNAPSES
    # ========================================================

    def _register_rows(
        self,
    ):

        groups = (
            self.brain.groups
        )

        for (
            action,
            group_name,
        ) in ACTION_GROUPS.items():

            if group_name not in groups:

                raise KeyError(
                    f"MaleCNS group missing: {group_name}"
                )

            neurons = np.asarray(
                groups[
                    group_name
                ],
                dtype=np.int64,
            ).reshape(-1)

            for neuron in neurons:

                neuron = int(
                    neuron
                )

                row_start = (
                    self._scalar_int(
                        self.W.indptr[
                            neuron
                        ]
                    )
                )

                row_end = (
                    self._scalar_int(
                        self.W.indptr[
                            neuron + 1
                        ]
                    )
                )

                if (
                    row_end
                    <= row_start
                ):

                    continue

                absolute_positions = np.arange(
                    row_start,
                    row_end,
                    dtype=np.int64,
                )

                row_weights = (
                    self._numpy(
                        self.W.data[
                            row_start:
                            row_end
                        ]
                    )
                    .astype(
                        np.float32
                    )
                )

                pre_ids = (
                    self._numpy(
                        self.W.indices[
                            row_start:
                            row_end
                        ]
                    )
                    .astype(
                        np.int64
                    )
                )

                # We currently plasticize positive incoming
                # synapses, as in our proven steering experiment.

                positive = (
                    row_weights > 0.0
                )

                if not np.any(
                    positive
                ):

                    continue

                positions = (
                    absolute_positions[
                        positive
                    ]
                )

                baseline = (
                    row_weights[
                        positive
                    ].copy()
                )

                presynaptic = (
                    pre_ids[
                        positive
                    ].copy()
                )

                count = len(
                    positions
                )

                row = {
                    "action":
                        action,

                    "neuron":
                        neuron,

                    "positions":
                        positions,

                    "pre_ids":
                        presynaptic,

                    # The synaptic state present when this
                    # plasticity object is created becomes the
                    # reference weight.
                    "baseline":
                        baseline,

                    "multiplier":
                        np.ones(
                            count,
                            dtype=np.float32,
                        ),

                    # Recent presynaptic context.
                    "pre_trace":
                        np.zeros(
                            count,
                            dtype=np.float32,
                        ),

                    # Pre/post eligibility.
                    "eligibility":
                        np.zeros(
                            count,
                            dtype=np.float32,
                        ),
                }

                self.rows.append(
                    row
                )

                self.rows_by_action[
                    action
                ].append(
                    row
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    def _print_summary(
        self,
    ):

        print()
        print(
            "================================"
        )

        print(
            " ISAAC MOTOR PLASTICITY V3"
        )

        print(
            " TEMPORAL ACTION CREDIT"
        )

        print(
            "================================"
        )

        total = 0

        groups = (
            self.brain.groups
        )

        for (
            action,
            group_name,
        ) in ACTION_GROUPS.items():

            neuron_count = len(
                np.asarray(
                    groups[
                        group_name
                    ]
                ).reshape(-1)
            )

            synapses = sum(
                len(
                    row[
                        "positions"
                    ]
                )

                for row
                in self.rows_by_action[
                    action
                ]
            )

            total += (
                synapses
            )

            print(
                f"{action:12s} | "
                f"neurons="
                f"{neuron_count:2d} | "
                f"plastic synapses="
                f"{synapses}"
            )

        print(
            "--------------------------------"
        )

        print(
            "Total plastic Isaac synapses:",
            total
        )

        print()

    # ========================================================
    # WRITE MULTIPLIERS INTO REAL MALECNS MATRIX
    # ========================================================

    def _write_row(
        self,
        row,
    ):

        values = (
            row[
                "baseline"
            ]
            * row[
                "multiplier"
            ]
        ).astype(
            np.float32
        )

        positions = (
            row[
                "positions"
            ]
        )

        if self._gpu_weights:

            self.W.data[
                cp.asarray(
                    positions
                )
            ] = cp.asarray(
                values
            )

        else:

            self.W.data[
                positions
            ] = values

    # ========================================================
    # NEURAL ACTIVITY
    # ========================================================

    def observe(
        self,
        fired,
    ):

        fired = np.asarray(
            fired,
            dtype=np.int64,
        ).reshape(-1)

        mask = (
            self._fired_mask
        )

        mask.fill(
            False
        )

        valid = fired[
            (
                fired >= 0
            )
            & (
                fired
                < self.neuron_count
            )
        ]

        mask[
            valid
        ] = True

        for row in self.rows:

            pre_trace = (
                row[
                    "pre_trace"
                ]
            )

            eligibility = (
                row[
                    "eligibility"
                ]
            )

            # -----------------------------------------------
            # PRESYNAPTIC CONTEXT
            # -----------------------------------------------

            pre_trace *= (
                self.pre_decay
            )

            pre_active = (
                mask[
                    row[
                        "pre_ids"
                    ]
                ]
            )

            # Recently fired presynaptic cells are maximally
            # represented in the context trace.

            pre_trace[
                pre_active
            ] = 1.0

            # -----------------------------------------------
            # ELIGIBILITY
            # -----------------------------------------------

            eligibility *= (
                self.eligibility_decay
            )

            post_active = bool(
                mask[
                    row[
                        "neuron"
                    ]
                ]
            )

            if post_active:

                eligibility += (
                    pre_trace
                )

                # Prevent unlimited accumulation.

                np.clip(
                    eligibility,
                    0.0,
                    5.0,
                    out=eligibility,
                )

    # ========================================================
    # ACTUAL EXECUTED ACTIONS
    # ========================================================

    def observe_actions(
        self,
        held_actions,
        tap_actions=None,
        now=None,
    ):
        """
        Record what actually reached Isaac's keyboard.

        This is deliberately AFTER motor.decode().

        Random spontaneous spikes that fail to produce a key
        press therefore receive no action credit.
        """

        if now is None:

            now = (
                time.perf_counter()
            )

        actions = set(
            held_actions
        )

        if tap_actions:

            actions.update(
                tap_actions
            )

        for action in actions:

            if (
                action
                in self.last_action_time
            ):

                self.last_action_time[
                    action
                ] = float(
                    now
                )

    # ========================================================
    # ACTION RECENCY
    # ========================================================

    def action_credit(
        self,
        action,
        now=None,
    ):

        if now is None:

            now = (
                time.perf_counter()
            )

        last = (
            self.last_action_time.get(
                action
            )
        )

        if last is None:

            return 0.0

        age = max(
            0.0,
            float(
                now
            )
            - float(
                last
            ),
        )

        tau = float(
            ACTION_TAU.get(
                action,
                2.0,
            )
        )

        return float(
            math.exp(
                -age / tau
            )
        )

    # ========================================================
    # GENERAL REWARD
    # ========================================================

    def apply_reward(
        self,
        reward,
        now=None,
    ):
        """
        General game reward.

        Only recently executed actions receive meaningful credit.

        Reward does NOT blindly modify all eleven action systems.
        """

        reward = float(
            reward
        )

        if reward == 0.0:
            return

        if now is None:

            now = (
                time.perf_counter()
            )

        signal = float(
            np.clip(
                reward
                / self.reward_scale,
                -1.0,
                1.0,
            )
        )

        changed = False

        for (
            action,
            rows,
        ) in self.rows_by_action.items():

            credit = (
                self.action_credit(
                    action,
                    now=now,
                )
            )

            # Ancient actions receive no update.

            if credit < 0.01:
                continue

            for row in rows:

                eligibility = (
                    row[
                        "eligibility"
                    ]
                )

                if not np.any(
                    eligibility
                ):
                    continue

                change = (
                    self.learning_rate
                    * signal
                    * credit
                    * eligibility
                )

                row[
                    "multiplier"
                ] += change.astype(
                    np.float32
                )

                np.clip(
                    row[
                        "multiplier"
                    ],
                    self.min_multiplier,
                    self.max_multiplier,
                    out=row[
                        "multiplier"
                    ],
                )

                self._write_row(
                    row
                )

                changed = True

        # Reward consumes some accumulated causal evidence.

        for row in self.rows:

            row[
                "eligibility"
            ] *= (
                self.post_reward_decay
            )

        return changed

    # ========================================================
    # TARGETED ACTION REWARD
    # ========================================================

    def apply_action_reward(
        self,
        action,
        reward,
        now=None,
        contextual=True,
    ):
        """
        Reward or punish ONE action only.

        Used for consequences where the causal action is known.

        WALL_STUCK example:

            wall left
              +
            move_left executed
              +
            no displacement
              ↓
            only move_left synapses are punished

        contextual=True means recently active presynaptic
        connections within that action population receive the
        largest update.

        Thus we are learning approximately:

            "move_left in THIS sensory context is bad"

        rather than:

            "move_left is globally bad".
        """

        if (
            action
            not in self.rows_by_action
        ):

            return False

        reward = float(
            reward
        )

        if reward == 0.0:
            return False

        if now is None:

            now = (
                time.perf_counter()
            )

        signal = float(
            np.clip(
                reward
                / self.reward_scale,
                -1.0,
                1.0,
            )
        )

        changed = False

        for row in self.rows_by_action[
            action
        ]:

            eligibility = (
                row[
                    "eligibility"
                ]
            )

            if contextual:

                # Recent presynaptic activity acts as a
                # context gate.

                context = np.clip(
                    row[
                        "pre_trace"
                    ],
                    0.0,
                    1.0,
                )

                local_eligibility = (
                    eligibility
                    * context
                )

            else:

                local_eligibility = (
                    eligibility
                )

            if not np.any(
                local_eligibility
            ):

                continue

            change = (
                self.learning_rate
                * signal
                * local_eligibility
            )

            row[
                "multiplier"
            ] += change.astype(
                np.float32
            )

            np.clip(
                row[
                    "multiplier"
                ],
                self.min_multiplier,
                self.max_multiplier,
                out=row[
                    "multiplier"
                ],
            )

            self._write_row(
                row
            )

            changed = True

            # Only consume eligibility for the known causal
            # pathway, not unrelated actions.

            row[
                "eligibility"
            ] *= (
                self.post_reward_decay
            )

        return changed

    # ========================================================
    # TEMPORARY TRACE RESET
    # ========================================================

    def reset_traces(
        self,
    ):

        for row in self.rows:

            row[
                "pre_trace"
            ].fill(
                0.0
            )

            row[
                "eligibility"
            ].fill(
                0.0
            )

        for action in self.last_action_time:

            self.last_action_time[
                action
            ] = None

    # ========================================================
    # SUMMARY
    # ========================================================

    def multiplier_summary(
        self,
    ):

        result = {}

        for (
            action,
            rows,
        ) in self.rows_by_action.items():

            arrays = [
                row[
                    "multiplier"
                ]

                for row in rows

                if len(
                    row[
                        "multiplier"
                    ]
                ) > 0
            ]

            if not arrays:

                result[
                    action
                ] = 1.0

                continue

            result[
                action
            ] = float(
                np.mean(
                    np.concatenate(
                        arrays
                    )
                )
            )

        return result

    # ========================================================
    # CHECKPOINT SAVE
    # ========================================================

    def save_checkpoint(
        self,
        path,
    ):

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "format_version":
                np.asarray(
                    [3],
                    dtype=np.int32,
                )
        }

        for row in self.rows:

            action = (
                row[
                    "action"
                ]
            )

            neuron = (
                row[
                    "neuron"
                ]
            )

            # Simple key also makes backwards/forwards inspection
            # easy with np.load.

            key = (
                f"{action}_{neuron}"
            )

            payload[
                key
            ] = (
                row[
                    "multiplier"
                ]
                .astype(
                    np.float32
                )
            )

        np.savez_compressed(
            path,
            **payload,
        )

        print()
        print(
            "Isaac brain checkpoint saved:"
        )

        print(
            path
        )

    # ========================================================
    # FLEXIBLE OLD CHECKPOINT LOOKUP
    # ========================================================

    @staticmethod
    def _candidate_keys(
        action,
        neuron,
        local_index,
    ):

        return [
            f"{action}_{neuron}",
            f"{action}__{neuron}",
            f"{action}_neuron_{neuron}",
            f"{action}__neuron_{neuron}",
            f"multiplier_{action}_{neuron}",
            f"{action}_{local_index}",
        ]

    # ========================================================
    # CHECKPOINT LOAD
    # ========================================================

    def load_checkpoint(
        self,
        path,
    ):

        path = Path(
            path
        )

        if not path.exists():

            print(
                "Isaac checkpoint not found:"
            )

            print(
                path
            )

            return False

        data = np.load(
            path,
            allow_pickle=False,
        )

        loaded_rows = 0
        missing_actions = set()

        for (
            action,
            rows,
        ) in self.rows_by_action.items():

            # ------------------------------------------------
            # PER-NEURON KEYS
            # ------------------------------------------------

            for (
                local_index,
                row,
            ) in enumerate(
                rows
            ):

                neuron = (
                    row[
                        "neuron"
                    ]
                )

                source = None

                for candidate in (
                    self._candidate_keys(
                        action,
                        neuron,
                        local_index,
                    )
                ):

                    if candidate in data:

                        source = np.asarray(
                            data[
                                candidate
                            ],
                            dtype=np.float32,
                        ).reshape(-1)

                        break

                # ------------------------------------------------
                # FUZZY LEGACY KEY
                # ------------------------------------------------

                if source is None:

                    neuron_text = str(
                        neuron
                    )

                    for key in (
                        data.files
                    ):

                        if (
                            action in key
                            and neuron_text
                            in key
                        ):

                            candidate = np.asarray(
                                data[
                                    key
                                ]
                            )

                            if np.issubdtype(
                                candidate.dtype,
                                np.number,
                            ):

                                source = (
                                    candidate
                                    .astype(
                                        np.float32
                                    )
                                    .reshape(-1)
                                )

                                break

                if source is None:

                    missing_actions.add(
                        action
                    )

                    continue

                destination = (
                    row[
                        "multiplier"
                    ]
                )

                if source.size == 1:

                    destination.fill(
                        float(
                            source[0]
                        )
                    )

                elif (
                    source.size
                    == destination.size
                ):

                    destination[:] = (
                        source
                    )

                else:

                    # Different checkpoint topology.
                    # Leave this row at 1 rather than corrupt it.

                    continue

                np.clip(
                    destination,
                    self.min_multiplier,
                    self.max_multiplier,
                    out=destination,
                )

                self._write_row(
                    row
                )

                loaded_rows += 1

        print()
        print(
            "Isaac brain checkpoint loaded:"
        )

        print(
            path
        )

        if missing_actions:

            # Useful for loading checkpoints created before
            # bomb/Q/Space channels existed.

            print(
                "Channels without compatible old data "
                "remain at their current multiplier:"
            )

            print(
                ", ".join(
                    sorted(
                        missing_actions
                    )
                )
            )

        return (
            loaded_rows > 0
        )