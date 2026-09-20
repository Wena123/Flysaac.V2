import time
from pathlib import Path
import numpy as np

import pydirectinput
import win32gui

from brain.fly import FlyAgentBrain

from learning.plasticity import (
    RewardPlasticity,
)

from learning.isaac_plasticity import (
    IsaacMotorPlasticity,
)

from isaac.environment import (
    IsaacEnvironment,
)

from isaac.async_vision import (
    AsyncIsaacVision,
)

from isaac.lc10_vision import (
    IsaacLC10Vision,
)

from isaac.motor_map import (
    IsaacMotorMap,
)

from isaac.neural_explorer import (
    NeuralExplorer,
)

from isaac.reward_reader import (
    IsaacRewardReader,
)

from isaac.dashboard import (
    IsaacDashboard,
)

from isaac.restart import (
    restart_after_death,
)

from isaac.wall_penalty import (
    WallCollisionPenalty,
)

from isaac.combat_state import (
    IsaacCombatState,
)

from isaac.room_state import (
    IsaacRoomState,
)

from isaac.world_state import (
    IsaacWorldState,
)

from isaac.combat_shaping import (
    CombatShaping,
)

from isaac.sensory_debugger import (
    SensoryGridDebugger,
)

from isaac import rewards_config as reward_cfg


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINT_DIR = Path(
    "checkpoints"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "isaac_brain_v3_latest.npz"
)

OLD_V3_CHECKPOINT = Path(
    "isaac_brain_v3.npz"
)

BASE_CHECKPOINT = Path(
    "learned_brain.npz"
)

AUTOSAVE_SECONDS = 30.0
GENERATION_SECONDS = 600.0


# ============================================================
# VISION
# ============================================================

VISION_TARGET_HZ = 30.0

VISION_PROCESS_SCALE = 0.5

VISION_BUFFER_FRAMES = 12


# ============================================================
# DASHBOARD / PERF
# ============================================================

DASHBOARD_INTERVAL = 0.10

PERF_PRINT_STEPS = 50


# ============================================================
# TRAINING
# ============================================================

RESUME_ISAAC_TRAINING = True

FRESH_V3_BRANCH = True

ENABLE_TAPS = True


# ============================================================
# ROOM STALL / CAMPING PENALTY
# ============================================================

class RoomStallPenalty:
    """
    Penalize staying in one room without meaningful progress.

    No action is blocked and Python never chooses a direction.
    This only supplies a negative learning consequence for stagnation.
    """

    def __init__(
        self,
        grace_seconds=120.0,
        tick_seconds=5.0,
        base_penalty=0.10,
        ramp_per_tick=0.05,
        max_penalty_per_tick=1.00,
        max_room_penalty=30.0,
    ):

        self.grace_seconds = float(grace_seconds)
        self.tick_seconds = float(tick_seconds)
        self.base_penalty = float(base_penalty)
        self.ramp_per_tick = float(ramp_per_tick)
        self.max_penalty_per_tick = float(max_penalty_per_tick)
        self.max_room_penalty = float(max_room_penalty)

        self.reset()

    def reset(self, now=None):

        if now is None:
            now = time.perf_counter()

        self.current_room_index = None
        self.room_enter_time = float(now)
        self.last_progress_time = float(now)
        self.last_penalty_time = None
        self.room_penalty_total = 0.0
        self.penalty_tick = 0

    def observe_room(
        self,
        room_index,
        now=None,
    ):

        if now is None:
            now = time.perf_counter()

        now = float(now)

        try:
            room_index = int(room_index)
        except (TypeError, ValueError):
            return False

        if room_index < 0:
            return False

        if (
            self.current_room_index is None
            or room_index != self.current_room_index
        ):

            self.current_room_index = room_index
            self.room_enter_time = now
            self.last_progress_time = now
            self.last_penalty_time = None
            self.room_penalty_total = 0.0
            self.penalty_tick = 0

            return True

        return False

    def note_progress(
        self,
        room_index=None,
        reason="progress",
        now=None,
    ):

        if now is None:
            now = time.perf_counter()

        now = float(now)

        if room_index is not None:
            self.observe_room(
                room_index,
                now=now,
            )

        self.last_progress_time = now
        self.last_penalty_time = None
        self.penalty_tick = 0

    def update(
        self,
        room_index,
        now=None,
    ):

        if now is None:
            now = time.perf_counter()

        now = float(now)

        self.observe_room(
            room_index,
            now=now,
        )

        if self.current_room_index is None:
            return None

        stagnant_for = (
            now
            - self.last_progress_time
        )

        if stagnant_for < self.grace_seconds:
            return None

        if self.room_penalty_total >= self.max_room_penalty:
            return None

        if (
            self.last_penalty_time is not None
            and (
                now
                - self.last_penalty_time
            ) < self.tick_seconds
        ):
            return None

        penalty = (
            self.base_penalty
            + self.penalty_tick
            * self.ramp_per_tick
        )

        penalty = min(
            penalty,
            self.max_penalty_per_tick,
        )

        remaining = (
            self.max_room_penalty
            - self.room_penalty_total
        )

        penalty = min(
            penalty,
            remaining,
        )

        if penalty <= 0.0:
            return None

        self.penalty_tick += 1
        self.last_penalty_time = now
        self.room_penalty_total += penalty

        return {
            "name": "ROOM_STALL",
            "reward": -float(penalty),
            "action": None,
            "extra": (
                f"room={self.current_room_index} "
                f"stagnant={stagnant_for:.1f}s "
                f"tick={self.penalty_tick} "
                f"room_total=-{self.room_penalty_total:.2f}"
            ),
        }


ROOM_PROGRESS_EVENTS = {
    "NEW_ROOM",
    "NEW_FLOOR",

    "ENEMY_KILL",
    "BOSS_KILL",
    "ROOM_CLEAR",

    "ITEM_PICKUP",

    "COIN",
    "COIN_PICKUP",

    "KEY",
    "KEY_PICKUP",

    "BOMB_PICKUP",

    "HEAL",
    "MAX_HEALTH",

    "TREASURE_ROOM",
    "DEVIL_ROOM",
    "ANGEL_ROOM",
}


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def ensure_checkpoint_dir():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def find_resume_checkpoint():

    # Fresh V3 branch:
    # - only resume V3 files
    # - NEVER load old V2 Isaac checkpoints
    # - if no V3 checkpoint exists, training starts from learned_brain.npz
    #   and creates a brand-new isaac_brain_v3_latest.npz

    if (
        RESUME_ISAAC_TRAINING
        and LATEST_CHECKPOINT.exists()
    ):

        return (
            LATEST_CHECKPOINT
        )

    if (
        RESUME_ISAAC_TRAINING
        and OLD_V3_CHECKPOINT.exists()
    ):

        return (
            OLD_V3_CHECKPOINT
        )

    return None


def next_generation_path():

    ensure_checkpoint_dir()

    highest = 0

    for path in (
        CHECKPOINT_DIR.glob(
            "isaac_brain_v3_gen_*.npz"
        )
    ):

        try:

            number = int(
                path.stem.split(
                    "_"
                )[-1]
            )

            highest = max(
                highest,
                number,
            )

        except ValueError:

            continue

    return (
        CHECKPOINT_DIR
        / (
            "isaac_brain_v3_gen_"
            f"{highest + 1:03d}.npz"
        )
    )


def save_latest(
    plasticity,
):

    ensure_checkpoint_dir()

    plasticity.save_checkpoint(
        LATEST_CHECKPOINT
    )


def save_generation(
    plasticity,
):

    path = (
        next_generation_path()
    )

    plasticity.save_checkpoint(
        path
    )

    print()
    print(
        "GENERATION CHECKPOINT:"
    )

    print(
        path
    )

    return path


# ============================================================
# WINDOW FOCUS
# ============================================================

def focus_isaac(
    env,
):

    window = (
        env.controls.window
    )

    print()
    print(
        "Trying to focus Isaac..."
    )

    window.focus()

    time.sleep(
        0.5
    )

    if (
        win32gui.GetForegroundWindow()
        == window.hwnd
    ):

        print(
            "Isaac successfully focused."
        )

        return True

    rect = (
        window.client_rect()
    )

    pydirectinput.click(
        (
            rect["left"]
            + rect["width"] // 2
        ),
        (
            rect["top"]
            + rect["height"] // 2
        ),
    )

    time.sleep(
        0.7
    )

    return (
        win32gui.GetForegroundWindow()
        == window.hwnd
    )


def press_escape():

    pydirectinput.keyDown(
        "esc"
    )

    time.sleep(
        0.15
    )

    pydirectinput.keyUp(
        "esc"
    )


# ============================================================
# PRINT WEIGHTS
# ============================================================

def print_weights(
    plasticity,
):

    summary = (
        plasticity
        .multiplier_summary()
    )

    text = " | ".join(
        (
            f"{action}="
            f"{value:.4f}"
        )

        for action, value
        in summary.items()
    )

    print(
        "WEIGHTS | "
        + text
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_checkpoint_dir()

    print(
        "================================"
    )

    print(
        "   MALECNS ISAAC V10 / FRESH BRAIN V3"
    )

    print(
        "================================"
    )

    print()
    print(
        "Learning: ON"
    )

    print(
        "Temporal action credit: ON"
    )

    print(
        "Context-specific wall learning: ON"
    )

    print(
        "Neural exploration: ON (V4 reduced/learning-visible)"
    )

    print(
        "Projectile vision + future trajectory: ON"
    )

    print(
        "Stronger shot credit: HIT +0.80 / MISS -0.20 / EMPTY -0.20"
    )

    print(
        "Dodge learning: hostile tears visible + toward penalty + hit credit"
    )

    print(
        "Reward config: isaac/rewards_config.py"
    )

    print(
        "Brain branch: FRESH isaac_brain_v3 (old V2 checkpoints ignored)"
    )

    print(
        "Rewards | "
        f"hit={reward_cfg.EVENT_REWARDS.get('PLAYER_HIT')} | "
        f"death={reward_cfg.EVENT_REWARDS.get('PLAYER_DEATH')} | "
        f"new_room={reward_cfg.EVENT_REWARDS.get('NEW_ROOM')} | "
        f"shot_hit={reward_cfg.SHOT_HIT:+.2f} | "
        f"shot_miss={reward_cfg.SHOT_MISS:+.2f}"
    )

    print(
        "Room anti-camping: ON | "
        f"grace={reward_cfg.ROOM_STALL_GRACE_SECONDS:.0f}s | "
        f"tick={reward_cfg.ROOM_STALL_TICK_SECONDS:.0f}s"
    )

    print(
        "Full sensory audit: ON | self/enemies/doors/items/tears/etc."
    )

    print(
        "11 Isaac actions: ON"
    )

    print(
        "Async visual retina: ON"
    )

    print(
        "Room collision sensing: ON"
    )

    print(
        "Dynamic Isaac room grids: ON"
    )

    print(
        "LC10 EXPERIMENTAL retina: 12x20 (240 spatial positions)"
    )

    print(
        "Vision target:",
        VISION_TARGET_HZ,
        "Hz"
    )

    print(
        "CTRL+C to stop."
    )

    # ========================================================
    # ISAAC
    # ========================================================

    env = (
        IsaacEnvironment()
    )

    rewards = (
        IsaacRewardReader()
    )

    room_state = (
        IsaacRoomState()
    )

    world_state = (
        IsaacWorldState()
    )

    combat = (
        IsaacCombatState()
    )

    combat_shaping = (
        CombatShaping(
            # Random empty-room shooting was far too cheap.
            # Shooting is still ALLOWED; it is only punished more clearly.
            no_enemy_penalty=(
                reward_cfg.NO_ENEMY_SHOT
            ),

            no_enemy_cooldown=(
                reward_cfg.NO_ENEMY_SHOT_COOLDOWN
            ),

            projectile_penalty=(
                reward_cfg.PROJECTILE_TOWARD
            ),

            projectile_radius=(
                reward_cfg.PROJECTILE_RADIUS
            ),

            projectile_dot_threshold=(
                reward_cfg.PROJECTILE_DOT_THRESHOLD
            ),

            projectile_cooldown=(
                reward_cfg.PROJECTILE_COOLDOWN
            ),

            projectile_episode_cap=(
                reward_cfg.PROJECTILE_EPISODE_CAP
            ),
        )
    )

    dashboard = (
        IsaacDashboard()
    )

    dashboard_enabled = True

    # ========================================================
    # ASYNC SENSOR STREAM
    # ========================================================

    vision_stream = (
        AsyncIsaacVision(
            target_hz=(
                VISION_TARGET_HZ
            ),

            motion_threshold=18,

            process_scale=(
                VISION_PROCESS_SCALE
            ),

            max_buffer_frames=(
                VISION_BUFFER_FRAMES
            ),
        )
    )

    print()
    print(
        "Starting async retina..."
    )

    vision_stream.start()

    # ========================================================
    # MALECNS
    # ========================================================

    brain = (
        FlyAgentBrain()
    )

    # ========================================================
    # BASE BRAIN
    # ========================================================

    baseline_loader = (
        RewardPlasticity(
            brain,
            learning_rate=0.0,
            homeostasis=0.0,
        )
    )

    baseline_loader.load_checkpoint(
        BASE_CHECKPOINT
    )

    # ========================================================
    # V3 ISAAC PLASTICITY
    # ========================================================

    plasticity = (
        IsaacMotorPlasticity(
            brain,

            core=(
                baseline_loader.core
            ),

            learning_rate=0.001,

            pre_decay=0.90,

            eligibility_decay=0.99,

            reward_scale=10.0,

            min_multiplier=0.25,

            max_multiplier=1.75,

            post_reward_decay=0.50,
        )
    )

    resume_path = (
        find_resume_checkpoint()
    )

    if resume_path is not None:

        print()
        print(
            "Resuming Isaac brain from:"
        )

        print(
            resume_path
        )

        plasticity.load_checkpoint(
            resume_path
        )

    # ========================================================
    # LC10
    # ========================================================

    lc10_vision = (
        IsaacLC10Vision(
            brain,

            gain=1.5,

            max_drive=0.40,

            minimum_activity=0.07,

            # 12x20 has more useful spatial detail; allow more simultaneous
            # strong locations without flooding the whole LC10 population.
            top_k_spatial=10,

            color_gain=1.0,

            color_max_drive=0.20,

            color_minimum=0.08,

            color_top_k=3,

            enemy_visual_strength=0.85,

            map_visual_strength=0.22,

            self_visual_strength=0.55,
        )
    )

    # ========================================================
    # SENSORY GRID DEBUGGER
    # ========================================================

    sensory_debugger = (
        SensoryGridDebugger(
            print_every_steps=50,
        )
    )

    # Pure adapter test: this does NOT step MaleCNS and does not
    # change learned weights. It only proves synthetic map/enemy/self
    # positions can be translated into real LC10 neuron injections.
    sensory_debugger.run_adapter_self_test(
        lc10_vision
    )

    # ========================================================
    # MOTOR
    # ========================================================

    motor = (
        IsaacMotorMap(
            brain,

            window=10,

            enable_taps=(
                ENABLE_TAPS
            ),

            tap_cooldown_steps=20,
        )
    )

    # ========================================================
    # EXPLORATION
    # ========================================================

    explorer = (
        NeuralExplorer(
            brain,

            # V4: exploration remains neural, but is no longer almost
            # continuous. Old settings produced ~2 random bursts/second,
            # which could visibly mask learned behavior for hours.
            drive=0.75,

            min_wait_steps=8,

            max_wait_steps=25,

            min_burst_steps=4,

            max_burst_steps=8,

            tap_burst_steps=2,

            tap_probability=0.05,

            seed=2026,
        )
    )

    # ========================================================
    # WALL CONSEQUENCE DETECTOR
    # ========================================================

    wall_penalty = (
        WallCollisionPenalty(
            collision_threshold=0.55,

            escape_reward=(
                reward_cfg.WALL_ESCAPE_REWARD
            ),

            escape_window=2.0,
        )
    )

    # ========================================================
    # ROOM ANTI-CAMPING
    # ========================================================

    room_stall = (
        RoomStallPenalty(
            grace_seconds=(
                reward_cfg.ROOM_STALL_GRACE_SECONDS
            ),

            tick_seconds=(
                reward_cfg.ROOM_STALL_TICK_SECONDS
            ),

            base_penalty=(
                reward_cfg.ROOM_STALL_BASE_PENALTY
            ),

            ramp_per_tick=(
                reward_cfg.ROOM_STALL_RAMP_PER_TICK
            ),

            max_penalty_per_tick=(
                reward_cfg.ROOM_STALL_MAX_PENALTY_PER_TICK
            ),

            max_room_penalty=(
                reward_cfg.ROOM_STALL_MAX_ROOM_PENALTY
            ),
        )
    )

    # ========================================================
    # TEMPORARY STATE RESET
    # ========================================================

    brain.brain.reset(
        seed=12345
    )

    plasticity.reset_traces()

    motor.reset()

    explorer.reset()

    wall_penalty.reset()

    room_stall.reset()

    env.reset_controls()

    print()
    print(
        "Starting multipliers:"
    )

    print_weights(
        plasticity
    )

    # ========================================================
    # INITIAL VISUAL SAMPLE
    # ========================================================

    vision_sequence = -1

    print()
    print(
        "Waiting for first visual observation..."
    )

    (
        frame,
        features,
        vision_sequence,
        frames_combined,
        vision_stats,
    ) = vision_stream.consume(
        after_sequence=(
            vision_sequence
        ),
        timeout=10.0,
    )

    # Native/dynamic room geometry first.
    room_state.poll()

    features = (
        room_state.enrich_features(
            features
        )
    )

    world_state.poll()

    features = (
        world_state.enrich_features(
            features
        )
    )

    # Combat grids then adopt the exact current room-map shape.
    combat_events = (
        combat.poll()
    )

    features = (
        combat.enrich_features(
            features
        )
    )

    print(
        "Async retina ready."
    )

    dashboard.update(
        frame=frame,
        features=features,

        left_motion=0.0,
        right_motion=0.0,

        left_drive=0.0,
        right_drive=0.0,

        held=set(),

        explore_action=None,

        recent_events=[],

        total_reward=0.0,

        multipliers=(
            plasticity
            .multiplier_summary()
        ),

        fired_count=0,

        step=0,
        elapsed=0.0,
    )

    # ========================================================
    # FOCUS GAME
    # ========================================================

    print()
    print(
        "Isaac will be focused in 3 seconds..."
    )

    time.sleep(
        3.0
    )

    if not focus_isaac(
        env
    ):

        vision_stream.stop()

        env.stop()

        rewards.close()

        room_state.close()

        world_state.close()

        combat.close()

        dashboard.close()

        return

    time.sleep(
        1.0
    )

    print(
        "Sending ESC..."
    )

    press_escape()

    time.sleep(
        0.5
    )

    vision_stream.reset()

    # ========================================================
    # START
    # ========================================================

    print()
    print(
        "================================"
    )

    print(
        "      TRAINING STARTED"
    )

    print(
        "================================"
    )

    start_time = (
        time.perf_counter()
    )

    last_autosave = (
        start_time
    )

    last_generation = (
        start_time
    )

    last_dashboard_update = 0.0

    step = 0

    total_reward = 0.0

    reward_events = 0

    recent_events = []

    exploration_bursts = 0

    death_count = 0

    run_count = 1

    last_held = set()

    restart_requested = False

    final_generation = None

    # ========================================================
    # PERF
    # ========================================================

    perf_steps = 0

    perf_start = (
        time.perf_counter()
    )

    perf_brain_ms = 0.0
    perf_learning_ms = 0.0
    perf_motor_ms = 0.0
    perf_wait_ms = 0.0

    perf_visual_frames = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    try:

        while True:

            now = (
                time.perf_counter()
            )

            elapsed = (
                now
                - start_time
            )

            # =================================================
            # FOCUS PAUSE
            # =================================================

            if (
                win32gui.GetForegroundWindow()
                != env.controls.window.hwnd
            ):

                env.reset_controls()

                last_held = set()

                wall_penalty.reset()

                combat_shaping.reset()

                print()
                print(
                    "Isaac lost focus."
                )

                print(
                    "Training paused."
                )

                while (
                    win32gui.GetForegroundWindow()
                    != env.controls.window.hwnd
                ):

                    time.sleep(
                        0.25
                    )

                print(
                    "Isaac focused again."
                )

                motor.reset()

                wall_penalty.reset()

                room_stall.note_progress(
                    room_index=(
                        world_state.room_index
                    ),
                    reason="focus_resume",
                    now=time.perf_counter(),
                )

                plasticity.reset_traces()

                vision_stream.reset()

                continue

            # =================================================
            # NEW SENSORY INPUT
            # =================================================

            t0 = (
                time.perf_counter()
            )

            (
                frame,
                features,
                vision_sequence,
                frames_combined,
                vision_stats,
            ) = vision_stream.consume(
                after_sequence=(
                    vision_sequence
                ),

                timeout=2.0,
            )

            visual_wait_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            # =================================================
            # DYNAMIC ROOM + COMBAT SOURCE GRIDS
            # =================================================
            #
            # ROOM FIRST:
            #   collision_grid gets the native current-room shape:
            #   7x13 / 14x13 / 7x26 / 14x26 / narrow variants.
            #
            # COMBAT SECOND:
            #   enemy_grid + self_grid copy that exact source shape.
            #
            # LC10 THEN independently resizes the whole source room
            # to its fixed 12x20 retina.
            # =================================================

            room_state.poll()

            features = (
                room_state.enrich_features(
                    features
                )
            )

            world_state.poll()

            features = (
                world_state.enrich_features(
                    features
                )
            )

            room_stall.observe_room(
                world_state.room_index,
                now=time.perf_counter(),
            )

            combat_events = (
                combat.poll()
            )

            features = (
                combat.enrich_features(
                    features
                )
            )

            # =================================================
            # NEURAL EXPLORATION
            # =================================================

            (
                extra_inject,
                explore_action,
                started_burst,
            ) = explorer.step()

            if started_burst:

                exploration_bursts += 1

                print(
                    f"EXPLORE "
                    f"step={step:07d} | "
                    f"{explore_action}"
                )

            # =================================================
            # MALECNS
            # =================================================

            t0 = (
                time.perf_counter()
            )

            # ==========================================
            # DEBUG RAW ENEMY PIPELINE
            # ==========================================
            #
            # Print periodically instead of every brain step.
            # ==========================================

            if step % 25 == 0:

                print(
                    "\n========== RAW ENEMY DEBUG =========="
                )

                print(
                    "enemy_count:",
                    features.get(
                        "enemy_count",
                        "<MISSING>",
                    ),
                )

                print(
                    "enemies_alive:",
                    features.get(
                        "enemies_alive",
                        "<MISSING>",
                    ),
                )

                print(
                    "enemy_grid exists:",
                    "enemy_grid" in features,
                )

                enemy_debug_grid = (
                    features.get(
                        "enemy_grid"
                    )
                )

                if isinstance(
                    enemy_debug_grid,
                    np.ndarray,
                ):

                    print(
                        "enemy_grid shape:",
                        enemy_debug_grid.shape,
                    )

                    print(
                        "enemy_grid max:",
                        float(
                            np.max(
                                enemy_debug_grid
                            )
                        ),
                    )

                    print(
                        "enemy_grid nonzero:",
                        int(
                            np.count_nonzero(
                                enemy_debug_grid
                            )
                        ),
                    )

                else:

                    print(
                        "enemy_grid value:",
                        enemy_debug_grid,
                    )

                print(
                    "projectiles:",
                    features.get(
                        "projectile_count",
                        "<MISSING>",
                    ),
                    "nearest:",
                    features.get(
                        "projectile_distance",
                        "<MISSING>",
                    ),
                )

                print(
                    "enemy-related keys:",
                    [
                        key
                        for key in features.keys()
                        if (
                            "enemy" in key.lower()
                            or "projectile" in key.lower()
                        )
                    ],
                )

                print(
                    "=====================================\n"
                )

            fired = (

                lc10_vision.step(
                    features,

                    extra_inject=(
                        extra_inject
                    ),
                )
            )

            # =================================================
            # PROVE WHICH REAL GRIDS REACHED LC10 THIS STEP
            # =================================================

            sensory_debugger.update(
                features,
                lc10_vision,
                step,
                fired=fired,
            )

            brain_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            # =================================================
            # SYNAPTIC ACTIVITY TRACE
            # =================================================

            t0 = (
                time.perf_counter()
            )

            plasticity.observe(
                fired
            )

            learning_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            # =================================================
            # MOTOR DECODING
            # =================================================

            t0 = (
                time.perf_counter()
            )

            (
                held,
                taps,
                totals,
            ) = motor.decode(
                fired
            )

            requested_held = set(
                held
            )

            action_now = (
                time.perf_counter()
            )

            # Combat state/events were already polled immediately
            # after the new visual observation, before LC10.

            # ============================================================
            # EMPTY-ROOM SHOOTING SHAPING
            # ============================================================
            #
            # filter_shooting() no longer blocks any shot.
            # It returns the requested actions unchanged and only creates
            # NO_ENEMY_SHOT reward events when appropriate.
            # ============================================================

            (
                current_held,
                no_enemy_events,
            ) = combat_shaping.filter_shooting(
                requested_held,
                enemies_alive=(
                    combat.enemies_alive
                ),
                now=action_now,
            )

            # ============================================================
            # NO-ENEMY SHOOT ATTEMPTS
            # ============================================================

            for event in no_enemy_events:

                reward_events += 1

                total_reward += float(
                    event["reward"]
                )

                recent_events.append(
                    event.copy()
                )

                recent_events = (
                    recent_events[-10:]
                )

                print(
                    "EVENT  "
                    f"{event['name']:18s} | "
                    f"reward="
                    f"{event['reward']:+.2f} | "
                    f"action={event['action']} | "
                    f"{event['extra']}"
                )

                plasticity.apply_action_reward(
                    action=(
                        event["action"]
                    ),

                    reward=(
                        event["reward"]
                    ),

                    now=action_now,

                    contextual=True,
                )


            # ============================================================
            # SHOT HIT / MISS
            # ============================================================

            for event in combat_events:

                event_name = (
                    event[
                        "name"
                    ]
                )

                event_reward = float(
                    event["reward"]
                )

                # V4 directional combat credit.
                # These are intentionally much clearer than the old
                # +0.35 / -0.05 ratio.
                if event_name == "SHOT_HIT":
                    event_reward = float(
                        reward_cfg.SHOT_HIT
                    )

                elif event_name == "SHOT_MISS":
                    event_reward = float(
                        reward_cfg.SHOT_MISS
                    )

                action = (
                    event["action"]
                )

                if event_name == "SHOT_HIT":

                    room_stall.note_progress(
                        room_index=(
                            world_state.room_index
                        ),
                        reason=event_name,
                        now=action_now,
                    )

                reward_events += 1

                total_reward += (
                    event_reward
                )

                recent_events.append(
                    event.copy()
                )

                recent_events = (
                    recent_events[-10:]
                )

                print(
                    "EVENT  "
                    f"{event_name:18s} | "
                    f"reward="
                    f"{event_reward:+.2f} | "
                    f"action={action} | "
                    f"{event['extra']}"
                )

                # Exact shooting direction is known.
                #
                # contextual=False because a miss may only become known
                # after the tear has travelled for a while.

                plasticity.apply_action_reward(
                    action=action,
                    reward=event_reward,
                    now=action_now,
                    contextual=False,
                )


            # ============================================================
            # MOVING TOWARD NEARBY ENEMY PROJECTILE
            # ============================================================

            projectile_events = (
                combat_shaping
                .projectile_events(
                    current_held,
                    combat,
                    now=action_now,
                )
            )

            for event in projectile_events:

                event_reward = float(
                    event["reward"]
                )

                reward_events += 1

                total_reward += (
                    event_reward
                )

                recent_events.append(
                    event.copy()
                )

                recent_events = (
                    recent_events[-10:]
                )

                print(
                    "EVENT  "
                    f"{event['name']:18s} | "
                    f"reward="
                    f"{event_reward:+.2f} | "
                    f"action={event['action']} | "
                    f"{event['extra']}"
                )

                plasticity.apply_action_reward(
                    action=(
                        event["action"]
                    ),

                    reward=event_reward,

                    now=action_now,

                    contextual=True,
                )

            # ============================================================
            # REAL GAME INPUT
            # ============================================================

            env.act(
                held_actions=(
                    current_held
                ),

                tap_actions=(
                    taps
                ),
            )

            # Only actions that ACTUALLY reached Isaac get normal
            # action-time credit. Record this ONCE.

            plasticity.observe_actions(
                current_held,
                taps,
                now=action_now,
            )

            motor_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            # =================================================
            # ACTION LOG
            # =================================================

            if (
                current_held
                != last_held
            ):

                if current_held:

                    print(
                        f"ACTION "
                        f"step={step:07d} | "
                        f"{sorted(current_held)}"
                    )

                last_held = (
                    current_held.copy()
                )

            if taps:

                print(
                    f"TAP    "
                    f"step={step:07d} | "
                    f"{taps}"
                )

            # =================================================
            # CONTEXT-SPECIFIC WALL LEARNING
            # =================================================

            wall_events = (
                wall_penalty.update(
                    current_held,
                    features,
                    now=(
                        action_now
                    ),
                )
            )

            for event in wall_events:

                event_reward = float(
                    event[
                        "reward"
                    ]
                )

                action = (
                    event[
                        "action"
                    ]
                )

                reward_events += 1

                total_reward += (
                    event_reward
                )

                recent_events.append(
                    event.copy()
                )

                recent_events = (
                    recent_events[
                        -10:
                    ]
                )

                print(
                    "EVENT  "
                    f"{event['name']:18s} | "
                    f"reward="
                    f"{event_reward:+.2f} | "
                    f"action={action} | "
                    f"{event['extra']}"
                )

                # ---------------------------------------------
                # THIS IS THE IMPORTANT CHANGE
                # ---------------------------------------------
                #
                # Do NOT:
                #
                #     plasticity.apply_reward(...)
                #
                # Instead modify only the causal movement
                # pathway and only its context-active synapses.
                # ---------------------------------------------

                plasticity.apply_action_reward(
                    action=action,
                    reward=event_reward,
                    now=action_now,
                    contextual=True,
                )

                print_weights(
                    plasticity
                )

            # =================================================
            # REAL ISAAC EVENTS
            # =================================================

            (
                aggregate_reward,
                events,
            ) = rewards.poll()

            if events:

                event_now = (
                    time.perf_counter()
                )

                for event in events:

                    event_name = (
                        event[
                            "name"
                        ]
                    )

                    event_reward = float(
                        event[
                            "reward"
                        ]
                    )

                    # Easy editable reward table.
                    #
                    # If this event exists in rewards_config.py, its value
                    # overrides the Lua bridge reward.
                    #
                    # Events not listed in rewards_config.py keep the
                    # bridge-provided value.
                    event_reward = (
                        reward_cfg.reward_for(
                            event_name,
                            event_reward,
                        )
                    )

                    event = event.copy()

                    event[
                        "reward"
                    ] = event_reward

                    if (
                        event_name
                        in ROOM_PROGRESS_EVENTS
                    ):

                        room_stall.note_progress(
                            room_index=(
                                world_state.room_index
                            ),
                            reason=event_name,
                            now=event_now,
                        )

                    # Give extra, action-specific credit to movement that was
                    # being executed at the moment Isaac was hit. This does
                    # not choose a dodge direction; it punishes the causal
                    # movement pathway in the current sensory context.
                    if event_name == "PLAYER_HIT":

                        for move_action in (
                            "move_left",
                            "move_right",
                            "move_up",
                            "move_down",
                        ):

                            if move_action not in current_held:
                                continue

                            plasticity.apply_action_reward(
                                action=move_action,
                                reward=(
                                    reward_cfg.HIT_MOVEMENT
                                ),
                                now=event_now,
                                contextual=True,
                            )

                            print(
                                "EVENT  HIT_MOVEMENT       | "
                                f"reward={reward_cfg.HIT_MOVEMENT:+.2f} | "
                                f"action={move_action}"
                            )

                    print(
                        "EVENT  "
                        f"{event_name:18s} | "
                        f"reward="
                        f"{event_reward:+.2f} | "
                        f"{event['extra']}"
                    )

                    reward_events += 1

                    recent_events.append(
                        event.copy()
                    )

                    recent_events = (
                        recent_events[
                            -10:
                        ]
                    )

                    if (
                        event_name
                        == "PLAYER_DEATH"
                    ):

                        restart_requested = (
                            True
                        )

                    elif (
                        event_name
                        == "RUN_WIN"
                    ):

                        print()
                        print(
                            "***** RUN WON *****"
                        )

                    # -----------------------------------------
                    # APPLY EACH EVENT SEPARATELY
                    # -----------------------------------------

                    if event_reward != 0.0:

                        total_reward += (
                            event_reward
                        )

                        plasticity.apply_reward(
                            event_reward,
                            now=event_now,
                        )

                if (
                    any(
                        float(
                            event[
                                "reward"
                            ]
                        )
                        != 0.0

                        for event in events
                    )
                ):

                    print_weights(
                        plasticity
                    )

                    print()

            # Fallback in case reward_reader ever returns a
            # nonzero total without individual event entries.

            elif (
                aggregate_reward
                != 0.0
            ):

                total_reward += float(
                    aggregate_reward
                )

                plasticity.apply_reward(
                    aggregate_reward,
                    now=(
                        time.perf_counter()
                    ),
                )

            # =================================================
            # ROOM STALL / CAMPING PENALTY
            # =================================================
            #
            # First 120 seconds without meaningful progress are free.
            # Then a penalty arrives every 5 seconds:
            #
            # -0.10, -0.15, -0.20 ... up to -1.00 per tick.
            #
            # The ramp resets when:
            #   - another room is entered
            #   - an enemy is hit/killed
            #   - the room is cleared
            #   - an item/resource is collected
            #
            # No action is blocked or selected here.
            # =================================================

            stall_now = (
                time.perf_counter()
            )

            stall_event = (
                room_stall.update(
                    room_index=(
                        world_state.room_index
                    ),
                    now=stall_now,
                )
            )

            if stall_event is not None:

                stall_reward = float(
                    stall_event[
                        "reward"
                    ]
                )

                reward_events += 1

                total_reward += (
                    stall_reward
                )

                recent_events.append(
                    stall_event.copy()
                )

                recent_events = (
                    recent_events[
                        -10:
                    ]
                )

                print(
                    "EVENT  "
                    f"{stall_event['name']:18s} | "
                    f"reward="
                    f"{stall_reward:+.2f} | "
                    f"{stall_event['extra']}"
                )

                plasticity.apply_reward(
                    stall_reward,
                    now=stall_now,
                )

                print_weights(
                    plasticity
                )

            # =================================================
            # DASHBOARD
            # =================================================

            dashboard_now = (
                time.perf_counter()
            )

            if (
                dashboard_enabled
                and (
                    dashboard_now
                    - last_dashboard_update
                )
                >= DASHBOARD_INTERVAL
            ):

                dashboard_open = (
                    dashboard.update(
                        frame=frame,

                        features=features,

                        left_motion=(
                            lc10_vision
                            .last_left_activity
                        ),

                        right_motion=(
                            lc10_vision
                            .last_right_activity
                        ),

                        left_drive=(
                            lc10_vision
                            .last_left_drive
                        ),

                        right_drive=(
                            lc10_vision
                            .last_right_drive
                        ),

                        held=(
                            current_held
                            | set(
                                taps
                            )
                        ),

                        explore_action=(
                            explore_action
                        ),

                        recent_events=(
                            recent_events
                        ),

                        total_reward=(
                            total_reward
                        ),

                        multipliers=(
                            plasticity
                            .multiplier_summary()
                        ),

                        fired_count=(
                            len(
                                fired
                            )
                        ),

                        step=step,

                        elapsed=elapsed,
                    )
                )

                last_dashboard_update = (
                    dashboard_now
                )

                if not dashboard_open:

                    dashboard_enabled = (
                        False
                    )

                    print()
                    print(
                        "Dashboard closed."
                    )

                    print(
                        "Training continues."
                    )

            # =================================================
            # DEATH / RESTART
            # =================================================

            if restart_requested:

                death_count += 1

                print()
                print(
                    "================================"
                )

                print(
                    f" DEATH #{death_count}"
                )

                print(
                    f" FINISHED RUN #{run_count}"
                )

                print(
                    "================================"
                )

                save_latest(
                    plasticity
                )

                (
                    restart_success,
                    restart_events,
                ) = restart_after_death(
                    env,
                    rewards,
                )

                if restart_success:

                    for restart_event in (
                        restart_events
                    ):

                        recent_events.append(
                            restart_event
                        )

                    recent_events = (
                        recent_events[
                            -10:
                        ]
                    )

                # Learned real synaptic weights remain.
                #
                # Only temporary activity/traces reset.

                brain.brain.reset(
                    seed=(
                        12345
                        + death_count
                    )
                )

                plasticity.reset_traces()

                motor.reset()

                explorer.reset()

                wall_penalty.reset()

                room_stall.reset()

                # A new Isaac run reuses the same stage / stage_type and
                # room indices. If the previous run's room history remains,
                # doors are falsely shown as VISITED in the dashboard.
                room_state.reset_run_history()

                # Remove every cached sensory source from the dead run.
                world_state.reset()
                combat.reset()

                # Also blank adapter/debug caches immediately so the
                # dashboard cannot display phantom doors for even one frame.
                lc10_vision.reset_dynamic_state()
                sensory_debugger.reset()

                combat_shaping.reset()

                env.reset_controls()

                vision_stream.reset()

                last_held = set()

                restart_requested = (
                    False
                )

                run_count += 1

                last_autosave = (
                    time.perf_counter()
                )

                print()
                print(
                    "================================"
                )

                print(
                    f" STARTING RUN #{run_count}"
                )

                print(
                    " Learned synapses preserved."
                )

                print(
                    "================================"
                )

                continue

            # =================================================
            # CHECKPOINTS
            # =================================================

            checkpoint_now = (
                time.perf_counter()
            )

            if (
                checkpoint_now
                - last_autosave
                >= AUTOSAVE_SECONDS
            ):

                save_latest(
                    plasticity
                )

                last_autosave = (
                    checkpoint_now
                )

                print(
                    "Latest checkpoint autosaved."
                )

            if (
                checkpoint_now
                - last_generation
                >= GENERATION_SECONDS
            ):

                save_generation(
                    plasticity
                )

                last_generation = (
                    checkpoint_now
                )

            # =================================================
            # PERFORMANCE
            # =================================================

            perf_steps += 1

            perf_brain_ms += (
                brain_ms
            )

            perf_learning_ms += (
                learning_ms
            )

            perf_motor_ms += (
                motor_ms
            )

            perf_wait_ms += (
                visual_wait_ms
            )

            perf_visual_frames += (
                frames_combined
            )

            if (
                perf_steps
                >= PERF_PRINT_STEPS
            ):

                perf_now = (
                    time.perf_counter()
                )

                duration = (
                    perf_now
                    - perf_start
                )

                actual_hz = (
                    perf_steps
                    / duration
                    if duration > 0
                    else 0.0
                )

                frames_per_brain = (
                    perf_visual_frames
                    / perf_steps
                )

                print()
                print(
                    "PERF | "
                    f"VISION="
                    f"{vision_stats['vision_hz']:.1f} Hz"
                    " | "
                    f"BRAIN="
                    f"{actual_hz:.1f} Hz"
                    " | "
                    f"frames/brain="
                    f"{frames_per_brain:.2f}"
                )

                print(
                    "     | "
                    f"capture="
                    f"{vision_stats['capture_ms']:.1f}ms"
                    " | "
                    f"retina="
                    f"{vision_stats['perception_ms']:.1f}ms"
                    " | "
                    f"brain="
                    f"{perf_brain_ms / perf_steps:.1f}ms"
                )

                print(
                    "     | "
                    f"vision-wait="
                    f"{perf_wait_ms / perf_steps:.1f}ms"
                    " | "
                    f"learning="
                    f"{perf_learning_ms / perf_steps:.1f}ms"
                    " | "
                    f"motor="
                    f"{perf_motor_ms / perf_steps:.1f}ms"
                )

                print()

                perf_steps = 0

                perf_start = (
                    perf_now
                )

                perf_brain_ms = 0.0
                perf_learning_ms = 0.0
                perf_motor_ms = 0.0
                perf_wait_ms = 0.0

                perf_visual_frames = 0

            step += 1

    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print()
        print(
            "================================"
        )

        print(
            "       CTRL+C RECEIVED"
        )

        print(
            "================================"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        env.reset_controls()

        print()
        print(
            "Saving latest brain..."
        )

        save_latest(
            plasticity
        )

        print()
        print(
            "Saving final generation..."
        )

        final_generation = (
            save_generation(
                plasticity
            )
        )

        vision_stream.stop()

        env.stop()

        rewards.close()

        room_state.close()

        world_state.close()

        combat.close()

        dashboard.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    final_elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print(
        "================================"
    )

    print(
        "       TRAINING STOPPED"
    )

    print(
        "================================"
    )

    print(
        "Runtime:",
        f"{final_elapsed:.1f}",
        "seconds"
    )

    print(
        "MaleCNS steps:",
        step
    )

    if final_elapsed > 0:

        print(
            "Average MaleCNS rate:",
            f"{step / final_elapsed:.2f}",
            "Hz"
        )

    print(
        "Runs:",
        run_count
    )

    print(
        "Deaths:",
        death_count
    )

    print(
        "Exploration bursts:",
        exploration_bursts
    )

    print(
        "Reward events:",
        reward_events
    )

    print(
        "Total reward:",
        f"{total_reward:+.2f}"
    )

    print()
    print(
        "Final multipliers:"
    )

    print_weights(
        plasticity
    )

    print()
    print(
        "Latest checkpoint:"
    )

    print(
        LATEST_CHECKPOINT
    )

    print(
        "Final generation:"
    )

    print(
        final_generation
    )


if __name__ == "__main__":

    main()