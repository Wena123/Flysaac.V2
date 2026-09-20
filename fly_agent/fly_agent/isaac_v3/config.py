"""Central configuration for the FlyIsaac V3.4 hybrid pipeline."""

RETINA_ROWS = 60
RETINA_COLS = 60

# Retina dynamics
RETINA_NAKA_N = 2.0
RETINA_NAKA_SIGMA = 0.35
RETINA_ADAPT_ALPHA = 0.94
RETINA_CONTRAST_GAIN = 1.35
RETINA_EDGE_GAIN = 1.25

# Sensory injection
LC10_TILES_Y = 6
LC10_TILES_X_PER_SIDE = 10
LC10_TOP_K_TILES_PER_SIDE = 12
LC10_GAIN = 1.55
LC10_MAX_DRIVE = 0.40
LC10_MIN_ACTIVITY = 0.045

LPLC1_GAIN = 0.22
LPLC2_GAIN = 0.50
LC4_GAIN = 0.62

# Descending-neuron trace
DN_TRACE_DECAY = 0.92
DN_TRACE_CLIP = 6.0

# Decoder
DECODER_FEATURES = 768
DECODER_HIDDEN = 256
DECODER_SEED = 2026

# V3.2 temporal representation.
# Demo/DAgger datasets were sampled at 10 Hz, so 4 frames cover ~300 ms of
# history plus the current sample.
TEMPORAL_CONTEXT_FRAMES = 4
TEMPORAL_SAMPLE_HZ = 10.0
LAG_SWEEP_MIN = -5
LAG_SWEEP_MAX = 5
LAG_PROBE_EPOCHS = 3
LAG_PROBE_HIDDEN = 64
LAG_PROBE_FEATURES = 256
ALLOW_NEGATIVE_LAG_DEFAULT = False

MOVEMENT_CLASSES = (
    "none",
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)

SHOOT_CLASSES = (
    "none",
    "up",
    "down",
    "left",
    "right",
)

MOVEMENT_ACTIONS = {
    "none": set(),
    "up": {"move_up"},
    "down": {"move_down"},
    "left": {"move_left"},
    "right": {"move_right"},
    "up_left": {"move_up", "move_left"},
    "up_right": {"move_up", "move_right"},
    "down_left": {"move_down", "move_left"},
    "down_right": {"move_down", "move_right"},
}

SHOOT_ACTIONS = {
    "none": set(),
    "up": {"shoot_up"},
    "down": {"shoot_down"},
    "left": {"shoot_left"},
    "right": {"shoot_right"},
}

# Recorder format: unchanged for compatibility with all V3/V3.1 datasets.
ACTIONS = (
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "shoot_up",
    "shoot_down",
    "shoot_left",
    "shoot_right",
)

BRAIN_HZ = 45.0
DEMO_HZ = 10.0
AUTOSAVE_SAMPLES = 500
DAGGER_WEIGHT = 2.0
VALIDATION_FRACTION = 0.20
VALIDATION_GUARD_SAMPLES = 20
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4

# Runtime / game automation
START_FOCUS_DELAY = 3.0
START_ESCAPE_DELAY = 0.50
RESTART_COOLDOWN = 1.25

# V3.3 multimodal diagnostics / decoder comparison
V33_VISUAL_TILES_Y = 6
V33_VISUAL_TILES_X = 10
V33_INPUT_MODES = ("dn", "visual", "fusion")
V33_BEST_MODE_FILE = "isaac_decoder_v33_best_mode.txt"


# V3.4 richer multiscale vision
V34_VISUAL_FINE_TILES_Y = 10
V34_VISUAL_FINE_TILES_X = 16
V34_VISUAL_COARSE_TILES_Y = 6
V34_VISUAL_COARSE_TILES_X = 10
V34_INPUT_MODES = ("dn", "visual", "fusion")
V34_BEST_MODE_FILE = "isaac_decoder_v34_best_mode.txt"

# Live neural activity monitor
NEURON_MONITOR_HZ = 12.0
NEURON_MONITOR_HISTORY = 120
NEURON_MONITOR_RASTER_BINS = 96

# Robust automatic restart
RESTART_ATTEMPTS = 3
RESTART_RETRY_DELAY = 0.85


# ============================================================
# V3.5 GRID ASSIST
# ============================================================
# Fixed decoder grid. Old room/combat bridges may expose variable native room
# shapes; GridFeatureExtractor resamples them to this common representation.
V35_GRID_ROWS = 12
V35_GRID_COLS = 20
V35_GRID_CHANNELS = (
    "collision_grid",
    "near_collision_grid",
    "room_valid_mask",
    "door_grid",
    "unexplored_door_grid",
    "locked_door_grid",
    "enemy_grid",
    "self_grid",
    # Optional old-world channels. Missing channels are encoded as zeros.
    "pickup_grid",
    "item_grid",
    "hazard_grid",
    "trapdoor_grid",
)
V35_GRID_SCALARS = (
    "collision_up", "collision_down", "collision_left", "collision_right",
    "enemy_player_x", "enemy_player_y",
    "enemy_count", "enemies_alive",
    "projectile_count", "projectile_distance", "projectile_dx", "projectile_dy",
    "rooms_visited", "rooms_known",
)
V35_INPUT_MODES = ("visual", "grid", "visual_grid", "visual_grid_dn")
V35_BEST_MODE_FILE = "isaac_decoder_v35_best_mode.txt"
V35_BEST_ASSISTED_MODE_FILE = "isaac_decoder_v35_best_assisted_mode.txt"

# Branch decoder compression. Each sensory stream gets its own learned branch
# before fusion, preventing the large DN/grid vectors from simply swamping vision.
V35_VISUAL_BRANCH_HIDDEN = 128
V35_GRID_BRANCH_HIDDEN = 96
V35_DN_BRANCH_HIDDEN = 96
V35_SHARED_HIDDEN = 192
V35_VISUAL_FEATURES = 768
V35_GRID_FEATURES = 640
V35_DN_FEATURES = 512


# ============================================================
# V3.6 DOPAMINE + GRID -> MALECNS SENSORY PATHWAY
# ============================================================
# Grid information is privileged Isaac bridge data. It is injected into the
# visual projection pathway ONLY for assisted runs/recordings; --pure-vision
# keeps this pathway off so the benchmark remains interpretable.
V36_BEST_MODE_FILE = "isaac_decoder_v36_best_mode.txt"
V36_BEST_ASSISTED_MODE_FILE = "isaac_decoder_v36_best_assisted_mode.txt"

# Weak top-k grid injection into LC10a. We deliberately keep this lower than
# pixel-retina drive so the bridge augments rather than replaces real pixels.
V36_GRID_BRAIN_GAIN = 0.42
V36_GRID_BRAIN_MAX_DRIVE = 0.16
V36_GRID_BRAIN_MIN_ACTIVITY = 0.10
V36_GRID_BRAIN_TOP_K_PER_SIDE = 8
V36_GRID_BRAIN_WEIGHTS = {
    "enemy_grid": 1.00,
    "hazard_grid": 0.88,
    "unexplored_door_grid": 0.70,
    "locked_door_grid": 0.62,
    "trapdoor_grid": 0.58,
    "door_grid": 0.48,
    "item_grid": 0.38,
    "pickup_grid": 0.32,
    "self_grid": 0.28,
    "near_collision_grid": 0.18,
    "collision_grid": 0.10,
}

# Phasic reward signal. This is a dopamine-LIKE computational interface:
# reward drives identified PAM/PPL(-like) MaleCNS neurons when those labels are
# available, plus a very small online action-bias memory gated by eligibility.
DOPAMINE_DECAY = 0.965
DOPAMINE_REWARD_SCALE = 5.0
DOPAMINE_POSITIVE_GAIN = 1.00
DOPAMINE_NEGATIVE_GAIN = 1.00
DOPAMINE_INJECTION_GAIN = 0.13
DOPAMINE_MAX_DRIVE = 0.14
DOPAMINE_MAX_CELLS_PER_VALENCE = 128
DOPAMINE_ELIGIBILITY_DECAY = 0.94
DOPAMINE_POLICY_LR = 0.012
DOPAMINE_POLICY_BIAS_CLIP = 0.30
DOPAMINE_POLICY_STRENGTH = 0.65
DOPAMINE_AUTOSAVE_SECONDS = 30.0
DOPAMINE_POLICY_FILE = "isaac_dopamine_policy_v36.npz"


# ============================================================
# V3.7 FAST SENSORY / TACTICAL GRID / PROJECTILE PREDICTION
# ============================================================
V37_BEST_MODE_FILE = "isaac_decoder_v37_best_mode.txt"
V37_BEST_ASSISTED_MODE_FILE = "isaac_decoder_v37_best_assisted_mode.txt"

# The learned decoder still receives the old fixed 12x20 grid for backwards
# compatibility.  This NEW compact grid is exclusively a neural sensory map.
V37_TACTICAL_GRID_ROWS = 7
V37_TACTICAL_GRID_COLS = 14
V37_TACTICAL_CHANNEL_WEIGHTS = {
    "enemy_grid": 1.00,
    "projectile_danger_grid": 1.00,
    "hazard_grid": 0.82,
    "unexplored_door_grid": 0.62,
    "locked_door_grid": 0.55,
    "trapdoor_grid": 0.52,
    "door_grid": 0.42,
    "item_grid": 0.34,
    "pickup_grid": 0.28,
    "self_grid": 0.24,
    "near_collision_grid": 0.17,
    "collision_grid": 0.10,
}
V37_TACTICAL_LC10_GAIN = 0.48
V37_TACTICAL_LC10_MAX_DRIVE = 0.18
V37_TACTICAL_LC10_MIN_ACTIVITY = 0.075
V37_TACTICAL_LC10_TOP_K_PER_SIDE = 9

# Hostile projectile danger also feeds collision/looming-like pathways.
V37_PROJECTILE_LPLC2_GAIN = 0.38
V37_PROJECTILE_LC4_GAIN = 0.52
V37_PROJECTILE_MAX_EXTRA_DRIVE = 0.22
