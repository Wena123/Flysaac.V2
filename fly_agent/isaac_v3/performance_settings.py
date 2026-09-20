# FlyIsaac V3.7 performance tuning
# Safe place to adjust responsiveness without touching the runtime code.

VISION_CAPTURE_HZ = 60.0
BRAIN_HZ = 45.0
VISION_PROCESS_SCALE = 0.50
VISION_BUFFER_FRAMES = 3
DROP_STALE_FRAMES = True
MONITOR_HZ = 12.0
PERF_PRINT_SECONDS = 5.0

# Dataset sampling intentionally stays at 10 Hz.  The brain may run faster,
# but training examples remain comparable to the old V3.4/V3.5/V3.6 datasets.
DEMO_HZ = 10.0
