# ============================================================
# FLYISAAC V3.7.1 — DOOR / ROOM REWARD MEMORY
# ============================================================
#
# This file does NOT change dopamine amounts. It only prevents reward farming
# by remembering which room connection (door) has already paid NEW_ROOM reward.
#
# A door is treated as the undirected connection:
#     floor + room A <-> room B
# so A->B and B->A are THE SAME door.
# ============================================================

# NEW_ROOM dopamine is paid only the first time a door/connection is crossed
# during the current run.
GATE_NEW_ROOM_BY_DOOR = True

# Protects large ROOM_CLEAR rewards too: a room can pay ROOM_CLEAR dopamine
# only once per run. This does not change the numeric ROOM_CLEAR value.
GATE_ROOM_CLEAR_ONCE_PER_ROOM = True

# Reward event and FLYROOM state can arrive a few frames apart. Keep a short
# association window so they still match safely.
EVENT_MATCH_WINDOW_SECONDS = 2.0

# If FLYROOM transition arrives before the reward event, only associate an
# event that follows very shortly afterwards. This prevents the next doorway
# event from being accidentally matched to the previous transition.
POST_TRANSITION_EVENT_LAG_SECONDS = 0.35

# If room metadata is unavailable, suppress NEW_ROOM/ROOM_CLEAR rather than
# accidentally making an infinite dopamine farm.
FAIL_CLOSED_IF_ROOM_STATE_MISSING = True

# Console audit lines such as:
# DOOR MEMORY | NEW ... | dopamine ALLOWED
# DOOR MEMORY | REPEAT ... | dopamine BLOCKED
PRINT_DOOR_MEMORY = True
