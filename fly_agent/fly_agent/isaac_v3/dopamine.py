import time
from pathlib import Path

import numpy as np

from . import config
from . import dopamine_settings as dset
from .door_memory import DoorRewardMemory
from . import door_memory_settings as door_settings


class DopamineSystem:
    """Reward-modulated state for FlyIsaac V3.6.

    Two intentionally separate mechanisms are bundled here:

    1. Neural neuromodulatory input
       Positive reward preferentially excites PAM-like dopaminergic cells and
       negative reward preferentially excites PPL-like dopaminergic cells when
       those labels exist in MaleCNS. The fixed connectome then propagates that
       activity normally.

    2. Small dopamine-gated policy memory
       A bounded residual bias over movement/shoot classes is updated from
       reward using an eligibility trace. It never overwrites the supervised
       decoder and can be disabled independently.

    This is a computational dopamine-like mechanism, not a claim that the full
    biological dopamine/plasticity chemistry of Drosophila is reproduced.
    """

    def __init__(
        self,
        fly_agent_brain=None,
        checkpoint=None,
        enable_policy_plasticity=True,
    ):
        self.fly_agent_brain = fly_agent_brain
        self.brain = getattr(fly_agent_brain, "brain", fly_agent_brain)
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.enable_policy_plasticity = bool(enable_policy_plasticity)

        self.positive = 0.0
        self.negative = 0.0
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.reward_events = 0

        self.move_bias = np.zeros(len(config.MOVEMENT_CLASSES), np.float32)
        self.shoot_bias = np.zeros(len(config.SHOOT_CLASSES), np.float32)
        self.move_elig = np.zeros_like(self.move_bias)
        self.shoot_elig = np.zeros_like(self.shoot_bias)

        self.pam_cells = np.empty(0, np.int64)
        self.ppl_cells = np.empty(0, np.int64)
        self.dan_cells = np.empty(0, np.int64)
        self.door_memory = DoorRewardMemory()
        self._discover_dopamine_cells()
        self._last_save = time.perf_counter()

        if self.checkpoint is not None and self.checkpoint.exists():
            self.load(self.checkpoint)

        print(
            "Dopamine settings: isaac_v3/dopamine_settings.py | "
            f"custom events={'ON' if dset.USE_CUSTOM_EVENT_VALUES else 'OFF'}"
        )
        print(
            "Door reward memory: "
            f"NEW_ROOM={'ON' if door_settings.GATE_NEW_ROOM_BY_DOOR else 'OFF'} | "
            f"ROOM_CLEAR={'ON' if door_settings.GATE_ROOM_CLEAR_ONCE_PER_ROOM else 'OFF'}"
        )

    @property
    def value(self):
        return float(self.positive - self.negative)

    @staticmethod
    def _take_evenly(cells, limit):
        cells = np.asarray(cells, dtype=np.int64).reshape(-1)
        limit = max(1, int(limit))
        if len(cells) <= limit:
            return cells
        idx = np.linspace(0, len(cells) - 1, limit).round().astype(np.int64)
        return cells[idx]

    def _discover_dopamine_cells(self):
        brain = self.brain
        if brain is None:
            return
        try:
            ct = np.asarray(getattr(brain, "cell_type"), dtype=str)
        except Exception:
            return
        if ct.ndim != 1 or ct.size == 0:
            return
        try:
            sc = np.asarray(getattr(brain, "superclass"), dtype=str)
            if sc.shape != ct.shape:
                sc = np.array([""] * len(ct), dtype=str)
        except Exception:
            sc = np.array([""] * len(ct), dtype=str)

        upper = np.char.upper(ct.astype(str))
        lower_all = np.char.lower(np.char.add(np.char.add(ct.astype(str), "|"), sc.astype(str)))
        pam = np.char.startswith(upper, "PAM")
        ppl = np.char.startswith(upper, "PPL")
        general = (
            np.char.find(lower_all, "dopamin") >= 0
        ) | (
            np.char.find(lower_all, " dopamine") >= 0
        ) | (
            np.char.find(upper, "DAN") >= 0
        ) | pam | ppl

        self.pam_cells = self._take_evenly(
            np.flatnonzero(pam), dset.DOPAMINE_MAX_CELLS_PER_VALENCE
        )
        self.ppl_cells = self._take_evenly(
            np.flatnonzero(ppl), dset.DOPAMINE_MAX_CELLS_PER_VALENCE
        )
        self.dan_cells = np.flatnonzero(general).astype(np.int64)

        print(
            "FlyIsaac V3.6 dopamine neurons | "
            f"PAM={len(self.pam_cells)} PPL={len(self.ppl_cells)} "
            f"DAN-like total={len(self.dan_cells)}"
        )
        if len(self.pam_cells) == 0 or len(self.ppl_cells) == 0:
            print(
                "  note: missing PAM/PPL labels -> that valence will keep its "
                "computational dopamine state but will not inject a guessed cell set."
            )

    def reset_transient(self):
        self.positive = 0.0
        self.negative = 0.0
        self.last_reward = 0.0
        self.move_elig.fill(0.0)
        self.shoot_elig.fill(0.0)

    def reset_run_memory(self):
        """Clear door/room anti-farming memory for a genuinely new run."""
        self.door_memory.reset_run()

    def observe_room(self, features, now=None):
        """Feed FLYROOM metadata and resolve queued NEW_ROOM/ROOM_CLEAR events."""
        info = self.door_memory.observe(features, now=now)
        allowed, _blocked = self.door_memory.flush(now=now)
        if allowed:
            self._observe_events_direct(allowed)
        return info

    def step(self):
        self.positive *= float(dset.DOPAMINE_DECAY)
        self.negative *= float(dset.DOPAMINE_DECAY)
        if self.positive < 1e-5:
            self.positive = 0.0
        if self.negative < 1e-5:
            self.negative = 0.0
        return self.value

    def apply_reward(self, reward):
        reward = float(reward)
        if not np.isfinite(reward) or reward == 0.0:
            return self.value
        normalized = float(np.tanh(reward / float(dset.DOPAMINE_REWARD_SCALE)))
        self.last_reward = reward
        self.total_reward += reward
        self.reward_events += 1
        if normalized > 0.0:
            self.positive = min(
                1.5,
                self.positive + normalized * float(dset.DOPAMINE_POSITIVE_GAIN),
            )
        else:
            self.negative = min(
                1.5,
                self.negative + (-normalized) * float(dset.DOPAMINE_NEGATIVE_GAIN),
            )

        if self.enable_policy_plasticity:
            lr = float(dset.DOPAMINE_POLICY_LR)
            clip = float(dset.DOPAMINE_POLICY_BIAS_CLIP)
            self.move_bias += np.float32(lr * normalized) * self.move_elig
            self.shoot_bias += np.float32(lr * normalized) * self.shoot_elig
            np.clip(self.move_bias, -clip, clip, out=self.move_bias)
            np.clip(self.shoot_bias, -clip, clip, out=self.shoot_bias)
            # Once reward is delivered, reduce stale credit but do not erase it.
            post = np.float32(dset.DOPAMINE_POST_REWARD_ELIGIBILITY)
            self.move_elig *= post
            self.shoot_elig *= post
        return self.value

    @staticmethod
    def _event_dopamine_value(event):
        """Return the dopamine value configured for one Isaac event."""
        try:
            raw_reward = float(event.get("reward", 0.0))
        except Exception:
            raw_reward = 0.0
        if not np.isfinite(raw_reward):
            raw_reward = 0.0

        name = str(event.get("name", "")).strip().upper()
        value = raw_reward
        custom = False

        if bool(dset.USE_CUSTOM_EVENT_VALUES) and name in dset.EVENT_DOPAMINE:
            configured = dset.EVENT_DOPAMINE[name]
            custom = True
            if configured is None:
                value = raw_reward
            else:
                try:
                    value = float(configured)
                except Exception:
                    value = 0.0
        elif not bool(dset.USE_ORIGINAL_REWARD_FOR_UNLISTED_EVENTS):
            value = 0.0

        value *= float(dset.GLOBAL_EVENT_MULTIPLIER)
        if value > 0.0:
            value *= float(dset.POSITIVE_EVENT_MULTIPLIER)
        elif value < 0.0:
            value *= float(dset.NEGATIVE_EVENT_MULTIPLIER)

        if not np.isfinite(value):
            value = 0.0
        return name, raw_reward, float(value), custom

    def _observe_events_direct(self, events):
        total = 0.0
        for event in events or []:
            name, raw_reward, reward, custom = self._event_dopamine_value(event)
            if bool(dset.PRINT_DOPAMINE_EVENTS) and reward != 0.0:
                source = "custom" if custom else "raw"
                print(
                    f"DOPAMINE EVENT | {name or '?':18s} | "
                    f"raw={raw_reward:+.2f} -> DA={reward:+.2f} ({source})"
                )
            total += reward
        if total != 0.0:
            self.apply_reward(total)
        return total

    def observe_events(self, events, now=None):
        """Observe game reward events with anti-farming door/room gating.

        NEW_ROOM is delayed briefly until FLYROOM tells us which door was
        crossed.  The first A<->B crossing can pay; B->A or later repetitions
        through the same connection cannot pay dopamine again in that run.
        """
        now = time.perf_counter() if now is None else float(now)
        immediate = []

        for event in events or []:
            name = str(event.get("name", "")).strip().upper()

            # A real new run invalidates the previous run's map memory.
            if name == "RUN_START":
                self.reset_run_memory()

            if (
                (name == "NEW_ROOM" and door_settings.GATE_NEW_ROOM_BY_DOOR)
                or (name == "ROOM_CLEAR" and door_settings.GATE_ROOM_CLEAR_ONCE_PER_ROOM)
            ):
                self.door_memory.queue(event, now=now)
            else:
                immediate.append(event)

        # Gated events are intentionally resolved by observe_room(), which is
        # called immediately after the current FLYROOM/grid poll. This avoids
        # matching a fresh NEW_ROOM event to the previous doorway transition.
        return self._observe_events_direct(immediate)

    def make_injections(self):
        inject = []
        gain = float(dset.DOPAMINE_INJECTION_GAIN)
        max_drive = float(dset.DOPAMINE_MAX_DRIVE)
        if self.positive > 1e-4 and len(self.pam_cells):
            drive = min(max_drive, gain * float(self.positive))
            if drive > 0.0:
                inject.append((self.pam_cells, drive))
        if self.negative > 1e-4 and len(self.ppl_cells):
            drive = min(max_drive, gain * float(self.negative))
            if drive > 0.0:
                inject.append((self.ppl_cells, drive))
        return inject

    @staticmethod
    def _renormalize(probs, bias):
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        b = np.asarray(bias, dtype=np.float32).reshape(-1)
        if len(p) != len(b):
            return p
        logits = np.log(np.clip(p, 1e-7, 1.0))
        logits = logits + np.float32(dset.DOPAMINE_POLICY_STRENGTH) * b
        logits -= float(np.max(logits))
        q = np.exp(logits).astype(np.float32)
        denom = float(q.sum())
        return q / denom if denom > 0.0 else p

    def modulate_decision(self, info):
        """Apply bounded learned residual biases and return (actions, new_info)."""
        if not self.enable_policy_plasticity:
            move_probs = np.asarray(info["move_probs"], np.float32)
            shoot_probs = np.asarray(info["shoot_probs"], np.float32)
        else:
            move_probs = self._renormalize(info["move_probs"], self.move_bias)
            shoot_probs = self._renormalize(info["shoot_probs"], self.shoot_bias)

        mi = int(np.argmax(move_probs))
        si = int(np.argmax(shoot_probs))
        move_name = config.MOVEMENT_CLASSES[mi]
        shoot_name = config.SHOOT_CLASSES[si]
        actions = set(config.MOVEMENT_ACTIONS[move_name]) | set(config.SHOOT_ACTIONS[shoot_name])

        new_info = dict(info)
        new_info.update(
            move_probs=move_probs,
            shoot_probs=shoot_probs,
            move_name=move_name,
            shoot_name=shoot_name,
            move_confidence=float(move_probs[mi]),
            shoot_confidence=float(shoot_probs[si]),
            dopamine=float(self.value),
        )
        return actions, new_info

    def observe_decision(self, info):
        if not self.enable_policy_plasticity:
            return
        decay = np.float32(dset.DOPAMINE_ELIGIBILITY_DECAY)
        self.move_elig *= decay
        self.shoot_elig *= decay

        mp = np.asarray(info["move_probs"], np.float32).reshape(-1)
        sp = np.asarray(info["shoot_probs"], np.float32).reshape(-1)
        if len(mp) == len(self.move_elig):
            mi = int(np.argmax(mp))
            grad = -mp.copy()
            grad[mi] += 1.0
            self.move_elig += grad
        if len(sp) == len(self.shoot_elig):
            si = int(np.argmax(sp))
            grad = -sp.copy()
            grad[si] += 1.0
            self.shoot_elig += grad

    def save(self, path=None):
        path = Path(path) if path is not None else self.checkpoint
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=np.asarray([36], np.int32),
            move_bias=self.move_bias.astype(np.float32),
            shoot_bias=self.shoot_bias.astype(np.float32),
            total_reward=np.asarray([self.total_reward], np.float32),
            reward_events=np.asarray([self.reward_events], np.int64),
        )
        self._last_save = time.perf_counter()
        return path

    def load(self, path=None):
        path = Path(path) if path is not None else self.checkpoint
        if path is None or not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                mb = np.asarray(data["move_bias"], np.float32)
                sb = np.asarray(data["shoot_bias"], np.float32)
                if mb.shape == self.move_bias.shape:
                    self.move_bias[:] = mb
                if sb.shape == self.shoot_bias.shape:
                    self.shoot_bias[:] = sb
                if "total_reward" in data:
                    self.total_reward = float(data["total_reward"][0])
                if "reward_events" in data:
                    self.reward_events = int(data["reward_events"][0])
            print("Dopamine policy memory loaded:", path)
            return True
        except Exception as exc:
            print("Dopamine policy memory load warning:", exc)
            return False

    def maybe_autosave(self):
        if not self.enable_policy_plasticity or self.checkpoint is None:
            return
        if time.perf_counter() - self._last_save >= float(dset.DOPAMINE_AUTOSAVE_SECONDS):
            self.save()

    def summary(self):
        out = {
            "dopamine": float(self.value),
            "positive": float(self.positive),
            "negative": float(self.negative),
            "last_reward": float(self.last_reward),
            "move_bias_peak": float(np.max(np.abs(self.move_bias))),
            "shoot_bias_peak": float(np.max(np.abs(self.shoot_bias))),
            "pam_cells": int(len(self.pam_cells)),
            "ppl_cells": int(len(self.ppl_cells)),
        }
        out.update(self.door_memory.summary())
        return out
