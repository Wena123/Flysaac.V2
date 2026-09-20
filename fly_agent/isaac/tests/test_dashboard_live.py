import time

from isaac.environment import IsaacEnvironment
from isaac.perception import IsaacPerception
from isaac.dashboard import IsaacDashboard


def main():

    print("==============================")
    print("     DASHBOARD LIVE TEST")
    print("==============================")
    print()
    print("Isaac only needs to be OPEN.")
    print("You do NOT need to focus Isaac.")
    print()

    # -----------------------------------------
    # Isaac screen capture
    # -----------------------------------------

    env = IsaacEnvironment()

    perception = IsaacPerception(
        motion_threshold=12
    )

    perception.reset()

    # -----------------------------------------
    # Dashboard
    # -----------------------------------------

    dashboard = IsaacDashboard()

    start_time = time.perf_counter()

    step = 0

    multipliers = {
        "move_left": 1.0,
        "move_right": 1.0,
        "move_up": 1.0,
        "move_down": 1.0,
        "shoot_left": 1.0,
        "shoot_right": 1.0,
        "shoot_up": 1.0,
        "shoot_down": 1.0,
    }

    try:

        while True:

            # =====================================
            # CAPTURE REAL ISAAC SCREEN
            # =====================================

            frame = env.observe()

            if frame is None:

                print(
                    "ERROR: env.observe() returned None"
                )

                time.sleep(
                    0.5
                )

                continue

            # =====================================
            # PERCEPTION
            # =====================================

            features = perception.process(
                frame
            )

            # =====================================
            # SAME VALUES LC10 VISION WOULD USE
            # =====================================

            left_motion = max(
                features.get(
                    "top_left",
                    0.0,
                ),
                features.get(
                    "left",
                    0.0,
                ),
                features.get(
                    "bottom_left",
                    0.0,
                ),
            )

            right_motion = max(
                features.get(
                    "top_right",
                    0.0,
                ),
                features.get(
                    "right",
                    0.0,
                ),
                features.get(
                    "bottom_right",
                    0.0,
                ),
            )

            left_drive = min(
                left_motion * 4.0,
                0.8,
            )

            right_drive = min(
                right_motion * 4.0,
                0.8,
            )

            # =====================================
            # DEBUG TERMINAL OUTPUT
            # =====================================

            if step % 20 == 0:

                print(
                    f"FRAME {step:05d} | "
                    f"shape={frame.shape} | "
                    f"L={left_motion:.3f} | "
                    f"R={right_motion:.3f}"
                )

            # =====================================
            # DASHBOARD
            # =====================================

            dashboard_open = dashboard.update(
                frame=frame,

                features=features,

                left_motion=left_motion,
                right_motion=right_motion,

                left_drive=left_drive,
                right_drive=right_drive,

                held=set(),

                explore_action=None,

                recent_events=[],

                total_reward=0.0,

                multipliers=multipliers,

                fired_count=0,

                step=step,

                elapsed=(
                    time.perf_counter()
                    - start_time
                ),
            )

            if not dashboard_open:

                print(
                    "Dashboard closed."
                )

                break

            step += 1

            time.sleep(
                0.05
            )

    except KeyboardInterrupt:

        print()
        print(
            "Stopped."
        )

    finally:

        env.stop()

        dashboard.close()


if __name__ == "__main__":
    main()