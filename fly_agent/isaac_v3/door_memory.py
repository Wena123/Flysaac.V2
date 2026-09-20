import time
from collections import deque

from . import door_memory_settings as settings


class DoorRewardMemory:
    """Per-run memory for anti-farming room/door dopamine credit.

    The old room bridge already exposes:
      stage, stage_type, room_index, previous_room

    A physical traversal is represented as an UNDIRECTED edge between two room
    identities. Therefore A->B and B->A share one door key and can only produce
    one NEW_ROOM dopamine reward per run.

    ROOM_CLEAR is separately de-duplicated once per room, which prevents a
    cleared room with a very large configured reward from becoming another
    re-entry exploit.
    """

    def __init__(self, event_window=None):
        self.event_window = float(
            settings.EVENT_MATCH_WINDOW_SECONDS if event_window is None else event_window
        )
        self.pending = deque()
        self.reset_run()

    @staticmethod
    def _value(features, key, default=None):
        value = (features or {}).get(key, default)
        try:
            if hasattr(value, 'item'):
                value = value.item()
        except Exception:
            pass
        return value

    @classmethod
    def room_token(cls, features, room_key='room_index'):
        try:
            room = int(cls._value(features, room_key, -1))
        except Exception:
            return None
        if room < 0:
            return None
        try:
            stage = int(cls._value(features, 'stage', 0))
        except Exception:
            stage = 0
        try:
            stage_type = int(cls._value(features, 'stage_type', 0))
        except Exception:
            stage_type = 0
        return (stage, stage_type, room)

    @staticmethod
    def _door_key(a, b):
        if a is None or b is None or a == b:
            return None
        # Tokens are integer tuples, so lexical sorting is stable.
        return tuple(sorted((tuple(a), tuple(b))))

    @staticmethod
    def _fmt_room(token):
        if token is None:
            return '?'
        return f'{token[0]}:{token[1]}:{token[2]}'

    @classmethod
    def _fmt_door(cls, key):
        if key is None:
            return '?'
        return f'{cls._fmt_room(key[0])} <-> {cls._fmt_room(key[1])}'

    def reset_run(self):
        self.seen_doors = set()
        self.cleared_rooms = set()
        self.current_room = None
        self.last_transition_signature = None
        self.last_door_key = None
        self.last_transition_new = False
        self.last_transition_at = None
        self.last_transition_consumed = True
        self.just_changed = False
        self.pending = deque()
        self.transitions = 0
        self.repeat_transitions = 0
        self.allowed_new_room_rewards = 0
        self.blocked_new_room_rewards = 0
        self.allowed_room_clear_rewards = 0
        self.blocked_room_clear_rewards = 0

    def queue(self, event, now=None):
        now = time.perf_counter() if now is None else float(now)
        self.pending.append((now, dict(event)))

    def observe(self, features, now=None):
        """Observe the latest room state and return transition audit info."""
        now = time.perf_counter() if now is None else float(now)
        current = self.room_token(features, 'room_index')
        previous = self.room_token(features, 'previous_room')

        # previous_room belongs to the same floor metadata as current room.
        if current is not None:
            self.current_room = current

        changed = False
        self.just_changed = False
        new_door = False
        door_key = None

        if current is not None and previous is not None and current != previous:
            signature = (previous, current)
            if signature != self.last_transition_signature:
                changed = True
                self.just_changed = True
                self.last_transition_signature = signature
                door_key = self._door_key(previous, current)
                self.last_door_key = door_key
                self.last_transition_at = now
                self.last_transition_consumed = False
                self.transitions += 1
                if door_key is not None and door_key not in self.seen_doors:
                    self.seen_doors.add(door_key)
                    new_door = True
                    self.last_transition_new = True
                else:
                    self.repeat_transitions += 1
                    self.last_transition_new = False

                if settings.PRINT_DOOR_MEMORY:
                    print(
                        f"DOOR MEMORY | {'NEW' if new_door else 'REPEAT':6s} | "
                        f"{self._fmt_door(door_key)} | remembered={len(self.seen_doors)}"
                    )

        return {
            'current_room': current,
            'previous_room': previous,
            'door_key': door_key if changed else self.last_door_key,
            'changed': changed,
            'new_door': new_door if changed else False,
            'seen_doors': len(self.seen_doors),
            'cleared_rooms': len(self.cleared_rooms),
        }

    def _match_new_room(self, event_time, now):
        """Return True/False when resolvable, None while waiting for FLYROOM."""
        if not settings.GATE_NEW_ROOM_BY_DOOR:
            return True

        if self.last_transition_at is not None:
            event_time = float(event_time)
            transition_time = float(self.last_transition_at)

            # Best case: this observe() call discovered the transition that the
            # queued NEW_ROOM event belongs to. Reward event may be a few frames
            # earlier or later than FLYROOM, so use the full matching window.
            if self.just_changed and abs(event_time - transition_time) <= self.event_window:
                if self.last_transition_consumed:
                    self.blocked_new_room_rewards += 1
                    return False
                self.last_transition_consumed = True
                allowed = bool(self.last_transition_new)
                if allowed:
                    self.allowed_new_room_rewards += 1
                else:
                    self.blocked_new_room_rewards += 1
                if settings.PRINT_DOOR_MEMORY:
                    print(
                        f"DOOR MEMORY | NEW_ROOM dopamine "
                        f"{'ALLOWED' if allowed else 'BLOCKED'} | "
                        f"{self._fmt_door(self.last_door_key)}"
                    )
                return allowed

            # FLYROOM can occasionally arrive one loop BEFORE the reward log.
            # Only accept a very short positive lag here. A long lag is more
            # likely to be the NEXT doorway event and must wait for its own
            # transition instead of being matched to the previous one.
            lag = event_time - transition_time
            max_lag = float(settings.POST_TRANSITION_EVENT_LAG_SECONDS)
            if 0.0 <= lag <= max_lag:
                if self.last_transition_consumed:
                    self.blocked_new_room_rewards += 1
                    if settings.PRINT_DOOR_MEMORY:
                        print(
                            f"DOOR MEMORY | duplicate NEW_ROOM dopamine BLOCKED | "
                            f"{self._fmt_door(self.last_door_key)}"
                        )
                    return False
                self.last_transition_consumed = True
                allowed = bool(self.last_transition_new)
                if allowed:
                    self.allowed_new_room_rewards += 1
                else:
                    self.blocked_new_room_rewards += 1
                if settings.PRINT_DOOR_MEMORY:
                    print(
                        f"DOOR MEMORY | NEW_ROOM dopamine "
                        f"{'ALLOWED' if allowed else 'BLOCKED'} | "
                        f"{self._fmt_door(self.last_door_key)}"
                    )
                return allowed

        # Event may have arrived before its FLYROOM transition. Keep it queued.
        if (float(now) - float(event_time)) <= self.event_window:
            return None

        if settings.PRINT_DOOR_MEMORY:
            print("DOOR MEMORY | NEW_ROOM dopamine BLOCKED | no matching room transition")
        self.blocked_new_room_rewards += 1
        return False if settings.FAIL_CLOSED_IF_ROOM_STATE_MISSING else True

    def _match_room_clear(self, event_time, now):
        if not settings.GATE_ROOM_CLEAR_ONCE_PER_ROOM:
            return True
        room = self.current_room
        if room is None:
            if (float(now) - float(event_time)) <= self.event_window:
                return None
            return False if settings.FAIL_CLOSED_IF_ROOM_STATE_MISSING else True

        if room in self.cleared_rooms:
            self.blocked_room_clear_rewards += 1
            if settings.PRINT_DOOR_MEMORY:
                print(
                    f"ROOM MEMORY | ROOM_CLEAR dopamine BLOCKED | "
                    f"already rewarded {self._fmt_room(room)}"
                )
            return False

        self.cleared_rooms.add(room)
        self.allowed_room_clear_rewards += 1
        if settings.PRINT_DOOR_MEMORY:
            print(
                f"ROOM MEMORY | ROOM_CLEAR dopamine ALLOWED | "
                f"first clear {self._fmt_room(room)}"
            )
        return True

    def flush(self, now=None):
        """Return (allowed_events, blocked_events), preserving unresolved ones."""
        now = time.perf_counter() if now is None else float(now)
        allowed = []
        blocked = []
        keep = deque()

        while self.pending:
            event_time, event = self.pending.popleft()
            name = str(event.get('name', '')).strip().upper()
            if name == 'NEW_ROOM':
                decision = self._match_new_room(event_time, now)
            elif name == 'ROOM_CLEAR':
                decision = self._match_room_clear(event_time, now)
            else:
                decision = True

            if decision is None:
                keep.append((event_time, event))
            elif decision:
                allowed.append(event)
            else:
                blocked.append(event)

        self.pending = keep
        return allowed, blocked

    def summary(self):
        return {
            'doors_seen': int(len(self.seen_doors)),
            'rooms_cleared_rewarded': int(len(self.cleared_rooms)),
            'door_transitions': int(self.transitions),
            'repeat_door_transitions': int(self.repeat_transitions),
            'new_room_allowed': int(self.allowed_new_room_rewards),
            'new_room_blocked': int(self.blocked_new_room_rewards),
            'room_clear_allowed': int(self.allowed_room_clear_rewards),
            'room_clear_blocked': int(self.blocked_room_clear_rewards),
        }
