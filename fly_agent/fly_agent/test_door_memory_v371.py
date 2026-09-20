import numpy as np
from isaac_v3.dopamine import DopamineSystem
from isaac_v3 import dopamine_settings as dset


def f(room, prev, stage=1, stage_type=0):
    return {"stage": stage, "stage_type": stage_type, "room_index": room, "previous_room": prev}

new_room_da = float(dset.EVENT_DOPAMINE.get("NEW_ROOM", 0.0) or 0.0)
room_clear_da = float(dset.EVENT_DOPAMINE.get("ROOM_CLEAR", 0.0) or 0.0)

d = DopamineSystem(None, checkpoint=None, enable_policy_plasticity=False)
d.observe_room(f(84, -1), now=0.0)

# Event first, FLYROOM second: first A<->B is rewarded.
d.observe_events([{"name":"NEW_ROOM","reward":999.0}], now=1.00)
assert d.total_reward == 0.0
d.observe_room(f(85, 84), now=1.05)
assert abs(d.total_reward - new_room_da) < 1e-6

# Same physical door in reverse is blocked.
d.observe_events([{"name":"NEW_ROOM","reward":999.0}], now=2.00)
d.observe_room(f(84, 85), now=2.05)
assert abs(d.total_reward - new_room_da) < 1e-6

# And same direction again is blocked.
d.observe_events([{"name":"NEW_ROOM","reward":999.0}], now=3.00)
d.observe_room(f(85, 84), now=3.05)
assert abs(d.total_reward - new_room_da) < 1e-6

# A genuinely different connection can pay once.
d.observe_events([{"name":"NEW_ROOM","reward":999.0}], now=4.00)
d.observe_room(f(86, 85), now=4.05)
assert abs(d.total_reward - 2.0 * new_room_da) < 1e-6

# ROOM_CLEAR is protected once per room too.
d.observe_events([{"name":"ROOM_CLEAR","reward":999.0}], now=5.00)
d.observe_room(f(86, 85), now=5.01)
expected = 2.0 * new_room_da + room_clear_da
assert abs(d.total_reward - expected) < 1e-6

d.observe_events([{"name":"ROOM_CLEAR","reward":999.0}], now=6.00)
d.observe_room(f(86, 85), now=6.01)
assert abs(d.total_reward - expected) < 1e-6

# Run-map reset must NOT erase learned policy biases.
d.move_bias[:] = 0.123
d.reset_run_memory()
assert np.allclose(d.move_bias, 0.123)

print("DOOR MEMORY V3.7.1 TEST: OK")
print("A<->B first crossing reward : YES")
print("B->A same door farming      : BLOCKED")
print("A->B repeat farming         : BLOCKED")
print("new door reward             : YES")
print("ROOM_CLEAR duplicate        : BLOCKED")
print("dopamine numeric settings   : READ ONLY / NOT MODIFIED")
