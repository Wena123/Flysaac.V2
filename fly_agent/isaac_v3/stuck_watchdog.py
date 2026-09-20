import time

try:
    import pydirectinput
except Exception:  # allows synthetic tests on non-Windows systems
    pydirectinput = None

from . import stuck_settings as settings


class RoomStuckWatchdog:
    """Detects an unchanged Isaac room identity for a configurable duration.

    It only uses room identity metadata already exposed by the old room-state
    bridge. It does NOT use pixel similarity, so standing still for a moment or
    fighting in the same visual area does not reset the timer.
    """

    ROOM_KEYS = (
        "room_index",
        "current_room_index",
        "room_id",
        "room_grid_index",
    )

    FLOOR_KEYS = (
        "floor_index",
        "floor",
        "stage",
    )

    STAGE_TYPE_KEYS = (
        "stage_type",
        "stageType",
    )

    def __init__(self, timeout_seconds=None, status_every_seconds=None):
        self.timeout_seconds = float(
            settings.ROOM_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self.status_every_seconds = float(
            settings.STATUS_EVERY_SECONDS if status_every_seconds is None else status_every_seconds
        )
        self.room_token = None
        self.room_entered_at = None
        self.last_status_at = None
        self.latched = False
        self.missing_room_warned = False

    @staticmethod
    def _first(features, keys):
        for key in keys:
            if key not in features:
                continue
            value = features.get(key)
            if value is None:
                continue
            try:
                if hasattr(value, "item"):
                    value = value.item()
            except Exception:
                pass
            if isinstance(value, float) and not (value == value):
                continue
            return value
        return None

    def room_identity(self, features):
        f = dict(features or {})
        room = self._first(f, self.ROOM_KEYS)
        if room is None:
            return None
        floor = self._first(f, self.FLOOR_KEYS)
        stage_type = self._first(f, self.STAGE_TYPE_KEYS)
        # floor/stage context prevents room index reuse on a later floor from
        # being mistaken for one continuous 5-minute stay.
        return (floor, stage_type, room)

    def reset(self, now=None):
        self.room_token = None
        self.room_entered_at = None
        self.last_status_at = None
        self.latched = False
        self.missing_room_warned = False

    def observe(self, features, now=None):
        """Return (triggered, info_dict)."""
        now = time.perf_counter() if now is None else float(now)
        token = self.room_identity(features)
        if token is None:
            return False, {
                "room": None,
                "elapsed": 0.0,
                "remaining": self.timeout_seconds,
                "changed": False,
                "status_due": False,
            }

        changed = token != self.room_token
        if changed or self.room_entered_at is None:
            self.room_token = token
            self.room_entered_at = now
            self.last_status_at = now
            self.latched = False

        elapsed = max(0.0, now - float(self.room_entered_at))
        remaining = max(0.0, self.timeout_seconds - elapsed)
        status_due = False
        if self.status_every_seconds > 0.0 and self.last_status_at is not None:
            if (now - self.last_status_at) >= self.status_every_seconds:
                status_due = True
                self.last_status_at = now

        triggered = (not self.latched) and elapsed >= self.timeout_seconds
        if triggered:
            self.latched = True

        return triggered, {
            "room": token,
            "elapsed": elapsed,
            "remaining": remaining,
            "changed": changed,
            "status_due": status_due,
        }


def hold_isaac_restart_key(env, seconds=None, key=None, key_api=None):
    """Release agent controls, hold Isaac's restart key, always release it."""
    seconds = float(settings.RESTART_HOLD_SECONDS if seconds is None else seconds)
    key = str(settings.RESTART_KEY if key is None else key)
    api = pydirectinput if key_api is None else key_api
    if api is None:
        raise RuntimeError("pydirectinput is unavailable; cannot hold Isaac restart key.")

    try:
        env.reset_controls()
    except Exception:
        pass

    api.keyDown(key)
    try:
        time.sleep(max(0.0, seconds))
    finally:
        api.keyUp(key)
