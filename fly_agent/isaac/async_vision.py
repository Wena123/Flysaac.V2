import threading
import time

from collections import deque

import numpy as np

from isaac.capture import IsaacCapture
from isaac.perception import IsaacPerception
from isaac.room_state import IsaacRoomState


class AsyncIsaacVision:
    """
    Independent high-frequency Isaac sensory stream.

    Combines:

        camera vision
        motion
        edges
        colour
        collision proximity
        doors
        unexplored-door memory

    No pathfinding or external policy is used.
    """

    MAX_ARRAY_KEYS = {
        "motion_grid",
        "salience_grid",
        "color_contrast_grid",
        "red_grid",
        "green_grid",
        "blue_grid",

        # Room sensory channels
        "near_collision_grid",
        "door_grid",
        "unexplored_door_grid",
        "locked_door_grid",
    }

    MAX_SCALAR_KEYS = {
        "top_left",
        "top",
        "top_right",

        "left",
        "center",
        "right",

        "bottom_left",
        "bottom",
        "bottom_right",

        "global_motion",
    }

    def __init__(
        self,
        target_hz=30.0,
        motion_threshold=18,
        process_scale=0.5,
        max_buffer_frames=12,
    ):

        self.target_hz = float(
            target_hz
        )

        self.motion_threshold = int(
            motion_threshold
        )

        self.process_scale = float(
            process_scale
        )

        self.max_buffer_frames = int(
            max_buffer_frames
        )

        self._thread = None

        self._stop_event = (
            threading.Event()
        )

        self._lock = (
            threading.Lock()
        )

        self._condition = (
            threading.Condition(
                self._lock
            )
        )

        self._buffer = deque(
            maxlen=(
                self.max_buffer_frames
            )
        )

        self._latest_frame = None

        self._sequence = 0

        self._error = None

        self._reset_requested = False

        self._vision_hz = 0.0

        self._capture_ms = 0.0

        self._perception_ms = 0.0

        self._frames_total = 0

        self._dropped_frames = 0

        self._fps_window_start = (
            time.perf_counter()
        )

        self._fps_window_frames = 0

    # =========================================================
    # START
    # =========================================================

    def start(
        self,
    ):

        if (
            self._thread is not None
            and self._thread.is_alive()
        ):

            return

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._worker,

            name=(
                "IsaacVision"
            ),

            daemon=True,
        )

        self._thread.start()

    # =========================================================
    # STOP
    # =========================================================

    def stop(
        self,
    ):

        self._stop_event.set()

        with self._condition:

            self._condition.notify_all()

        if self._thread is not None:

            self._thread.join(
                timeout=3.0
            )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
    ):

        with self._condition:

            self._buffer.clear()

            self._reset_requested = (
                True
            )

            self._condition.notify_all()

    # =========================================================
    # COPY
    # =========================================================

    @staticmethod
    def _copy_value(
        value,
    ):

        if isinstance(
            value,
            np.ndarray,
        ):

            return value.copy()

        return value

    # =========================================================
    # TEMPORAL MERGE
    # =========================================================

    def _merge_features(
        self,
        samples,
    ):

        if not samples:

            return None

        first_features = (
            samples[0][1]
        )

        merged = {}

        for key, value in (
            first_features.items()
        ):

            merged[
                key
            ] = self._copy_value(
                value
            )

        for _, features in (
            samples[1:]
        ):

            for key, value in (
                features.items()
            ):

                if (
                    key
                    in self.MAX_ARRAY_KEYS
                ):

                    if (
                        key in merged
                        and isinstance(
                            value,
                            np.ndarray,
                        )
                        and isinstance(
                            merged[key],
                            np.ndarray,
                        )
                    ):

                        merged[
                            key
                        ] = np.maximum(
                            merged[key],
                            value,
                        )

                    else:

                        merged[
                            key
                        ] = self._copy_value(
                            value
                        )

                elif (
                    key
                    in self.MAX_SCALAR_KEYS
                ):

                    old_value = float(
                        merged.get(
                            key,
                            0.0,
                        )
                    )

                    new_value = float(
                        value
                    )

                    merged[
                        key
                    ] = max(
                        old_value,
                        new_value,
                    )

                else:

                    # Static/current state uses newest sample.

                    merged[
                        key
                    ] = self._copy_value(
                        value
                    )

        if (
            "salience_grid"
            in merged
        ):

            merged[
                "grid"
            ] = merged[
                "salience_grid"
            ]

        return merged

    # =========================================================
    # CONSUME
    # =========================================================

    def consume(
        self,
        after_sequence=-1,
        timeout=2.0,
    ):

        deadline = (
            time.perf_counter()
            + timeout
        )

        with self._condition:

            while True:

                if (
                    self._error
                    is not None
                ):

                    raise RuntimeError(
                        "Async Isaac vision "
                        "thread failed."
                    ) from self._error

                has_new_frame = (
                    self._sequence
                    > after_sequence
                )

                if (
                    has_new_frame
                    and len(
                        self._buffer
                    ) > 0
                ):

                    break

                remaining = (
                    deadline
                    - time.perf_counter()
                )

                if remaining <= 0:

                    raise TimeoutError(
                        "Timed out waiting for "
                        "new Isaac vision."
                    )

                self._condition.wait(
                    timeout=remaining
                )

            samples = list(
                self._buffer
            )

            self._buffer.clear()

            latest_frame = (
                self._latest_frame
            )

            sequence = (
                self._sequence
            )

            stats = {
                "vision_hz":
                    self._vision_hz,

                "capture_ms":
                    self._capture_ms,

                "perception_ms":
                    self._perception_ms,

                "frames_total":
                    self._frames_total,

                "dropped_frames":
                    self._dropped_frames,
            }

        merged_features = (
            self._merge_features(
                samples
            )
        )

        return (
            latest_frame,
            merged_features,
            sequence,
            len(samples),
            stats,
        )

    # =========================================================
    # WORKER
    # =========================================================

    def _worker(
        self,
    ):

        capture = None

        room_state = None

        try:

            # MSS must be created inside the same thread
            # in which it is used.

            capture = (
                IsaacCapture()
            )

            perception = (
                IsaacPerception(
                    motion_threshold=(
                        self.motion_threshold
                    ),

                    process_scale=(
                        self.process_scale
                    ),
                )
            )

            room_state = (
                IsaacRoomState()
            )

            perception.reset()

            interval = (
                1.0
                / self.target_hz
            )

            while (
                not self._stop_event.is_set()
            ):

                iteration_start = (
                    time.perf_counter()
                )

                # =============================================
                # RESET
                # =============================================

                with self._lock:

                    do_reset = (
                        self._reset_requested
                    )

                    self._reset_requested = (
                        False
                    )

                if do_reset:

                    perception.reset()

                # =============================================
                # CAMERA
                # =============================================

                t0 = (
                    time.perf_counter()
                )

                frame = (
                    capture.frame()
                )

                capture_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

                # =============================================
                # VISUAL RETINA
                # =============================================

                t0 = (
                    time.perf_counter()
                )

                features = (
                    perception.process(
                        frame
                    )
                )

                # =============================================
                # COLLISION + ROOM MEMORY
                # =============================================

                features = (
                    room_state
                    .enrich_features(
                        features
                    )
                )

                perception_ms = (
                    time.perf_counter()
                    - t0
                ) * 1000.0

                # =============================================
                # BUFFER
                # =============================================

                now = (
                    time.perf_counter()
                )

                with self._condition:

                    if (
                        self._frames_total
                        == 0
                    ):

                        self._capture_ms = (
                            capture_ms
                        )

                        self._perception_ms = (
                            perception_ms
                        )

                    else:

                        alpha = 0.10

                        self._capture_ms = (
                            (
                                1.0 - alpha
                            )
                            * self._capture_ms
                            + alpha
                            * capture_ms
                        )

                        self._perception_ms = (
                            (
                                1.0 - alpha
                            )
                            * self._perception_ms
                            + alpha
                            * perception_ms
                        )

                    self._sequence += 1

                    self._frames_total += 1

                    self._fps_window_frames += 1

                    if (
                        len(
                            self._buffer
                        )
                        >= self.max_buffer_frames
                    ):

                        self._dropped_frames += 1

                    self._buffer.append(
                        (
                            self._sequence,
                            features,
                        )
                    )

                    self._latest_frame = (
                        frame
                    )

                    fps_elapsed = (
                        now
                        - self._fps_window_start
                    )

                    if (
                        fps_elapsed
                        >= 1.0
                    ):

                        self._vision_hz = (
                            self._fps_window_frames
                            / fps_elapsed
                        )

                        self._fps_window_frames = 0

                        self._fps_window_start = (
                            now
                        )

                    self._condition.notify_all()

                # =============================================
                # TARGET RATE
                # =============================================

                spent = (
                    time.perf_counter()
                    - iteration_start
                )

                remaining = (
                    interval
                    - spent
                )

                if remaining > 0:

                    self._stop_event.wait(
                        remaining
                    )

        except Exception as error:

            with self._condition:

                self._error = error

                self._condition.notify_all()

        finally:

            if (
                room_state is not None
            ):

                try:

                    room_state.close()

                except Exception:

                    pass

            if (
                capture is not None
            ):

                try:

                    capture.sct.close()

                except Exception:

                    pass