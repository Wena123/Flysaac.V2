"""Easy settings for the autonomous room-stuck watchdog.

Edit ONLY this file if you want to change the 5 minute / 3 second behavior.
"""

# Master switch.
ENABLED = True

# If the fly remains in the same detected Isaac room for this long,
# the watchdog considers the run stuck.
ROOM_TIMEOUT_SECONDS = 5.0 * 60.0

# Isaac's hold-to-reset key.
RESTART_KEY = "r"

# Hold R for this long. The user requested 3 seconds.
RESTART_HOLD_SECONDS = 3.0

# Give Isaac/log bridges a little time to settle after releasing R.
POST_RESTART_WAIT_SECONDS = 2.0

# Print a progress line at most this often while staying in one room.
# Set to 0 to disable periodic room timer messages.
STATUS_EVERY_SECONDS = 60.0
