"""
EASY REWARD CONFIG FOR FLYBRAIN / ISAAC

CHANGE THE NUMBERS HERE.
You normally do NOT need to edit train_isaac.py.

Positive number  = reward
Negative number  = punishment
0.0              = ignore that event

Examples:
    PLAYER_HIT = -10.0
    NEW_ROOM   = +15.0
"""

# ============================================================
# MAIN ISAAC EVENTS
# ============================================================

EVENT_REWARDS = {

    # -------------------------
    # DAMAGE / DEATH
    # -------------------------
    "PLAYER_HIT": -8.0,
    "PLAYER_DEATH": -40.0,

    # -------------------------
    # EXPLORATION
    # -------------------------
    "NEW_ROOM": +20.0,
    "NEW_FLOOR": +100.0,

    # Special rooms.
    "TREASURE_ROOM": +40.0,
    "DEVIL_ROOM": +40.0,
    "ANGEL_ROOM": +40.0,

    # -------------------------
    # COMBAT / PROGRESS
    # -------------------------
    "ENEMY_KILL": +5.0,
    "ROOM_CLEAR": +10.0,
    "BOSS_KILL": +40.0,
    "RUN_WIN": +100000.0,

    # -------------------------
    # PICKUPS / ITEMS
    # -------------------------
    "ITEM_PICKUP": +6.0,

    "COIN": +0.5,
    "COIN_PICKUP": +0.5,

    "KEY": +1.0,
    "KEY_PICKUP": +1.0,

    "BOMB_PICKUP": +1.0,

    "HEAL": +1.0,

    # This is per half-heart of max-health increase in the bridge.
    "MAX_HEALTH": +2.0,
}


# ============================================================
# SHOOTING
# ============================================================

# Correct shot that hits an enemy.
SHOT_HIT = +2.0

# Shot fired while enemies exist, but the shot misses.
SHOT_MISS = -0.10

# Shooting when there are no enemies.
# IMPORTANT: shooting is still ALLOWED.
# This only changes the punishment.
NO_ENEMY_SHOT = -0.20


# ============================================================
# DODGING / PROJECTILES
# ============================================================

# Moving TOWARD a nearby hostile projectile.
PROJECTILE_TOWARD = -0.20

# Additional punishment applied specifically to a movement action
# that was active at the exact moment Isaac was hit.
HIT_MOVEMENT = -1.50

# How close a hostile projectile has to be before the
# PROJECTILE_TOWARD punishment can happen.
PROJECTILE_RADIUS = 180.0

# Direction sensitivity for "moving toward projectile".
PROJECTILE_DOT_THRESHOLD = 0.15

# Minimum seconds between projectile-toward penalties.
PROJECTILE_COOLDOWN = 0.20

# Maximum accumulated projectile-toward punishment in one episode.
PROJECTILE_EPISODE_CAP = -1.50


# ============================================================
# EMPTY-ROOM SHOOTING TIMING
# ============================================================

NO_ENEMY_SHOT_COOLDOWN = 0.35


# ============================================================
# WALL / COLLISION
# ============================================================

# Positive reward for escaping after getting stuck against a wall.
WALL_ESCAPE_REWARD = +0.10


# ============================================================
# ROOM CAMPING / STALL
# ============================================================

# Seconds of NO meaningful progress before camping punishment starts.
ROOM_STALL_GRACE_SECONDS = 120.0

# How often a new camping penalty is given.
ROOM_STALL_TICK_SECONDS = 5.0

# First camping penalty.
ROOM_STALL_BASE_PENALTY = 0.10

# Extra punishment added every tick.
#
# Example:
#   base=0.10
#   ramp=0.05
#
# gives:
#   -0.10
#   -0.15
#   -0.20
#   -0.25
#   ...
ROOM_STALL_RAMP_PER_TICK = 0.05

# Maximum punishment for ONE camping tick.
ROOM_STALL_MAX_PENALTY_PER_TICK = 1.00

# Maximum total camping punishment before entering a different room
# or otherwise resetting the room-stall episode.
ROOM_STALL_MAX_ROOM_PENALTY = 30.0


# ============================================================
# HELPER
# ============================================================

def reward_for(
    event_name,
    fallback,
):
    """
    Return configured reward if one exists.
    Otherwise preserve the reward supplied by the Lua bridge.
    """

    return float(
        EVENT_REWARDS.get(
            event_name,
            fallback,
        )
    )
