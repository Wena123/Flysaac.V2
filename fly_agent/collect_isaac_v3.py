import argparse
import time
from datetime import datetime
from pathlib import Path

from brain.fly import FlyAgentBrain
from isaac.async_vision import AsyncIsaacVision
from isaac.environment import IsaacEnvironment
from isaac.reward_reader import IsaacRewardReader
from isaac_v3 import config
from isaac_v3.controller import HybridIsaacController
from isaac_v3.dopamine import DopamineSystem
from isaac_v3.grid_features import IsaacGridBridge
from isaac_v3.tactical_grid import ProjectileBridgeV37, TacticalGridV37
from isaac_v3 import performance_settings as perfset
from isaac_v3.multimodal import V35DemoBuffer
from isaac_v3.runtime import (
    FastVisionLoop, isaac_focused, pause_until_isaac_focused, prepare_game_window,
    robust_restart_after_death, safe_shutdown,
)
from isaac_v3.teacher import read_teacher_labels
from isaac_v3.visual_features import RetinaFeatureExtractor

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "datasets"


def main():
    p = argparse.ArgumentParser(description="Record FlyIsaac V3.6 DN + rich vision + grid with dopamine-modulated MaleCNS.")
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--demo-hz", type=float, default=config.DEMO_HZ)
    p.add_argument("--no-start-escape", action="store_true")
    p.add_argument("--no-auto-restart", action="store_true")
    p.add_argument("--no-dopamine", action="store_true")
    p.add_argument("--no-brain-grid", action="store_true")
    args = p.parse_args()

    env = IsaacEnvironment(); rewards = IsaacRewardReader()
    vision = AsyncIsaacVision(target_hz=perfset.VISION_CAPTURE_HZ, motion_threshold=18, process_scale=perfset.VISION_PROCESS_SCALE, max_buffer_frames=perfset.VISION_BUFFER_FRAMES); vision.start()
    brain = FlyAgentBrain(); controller = HybridIsaacController(brain); visual_extractor = RetinaFeatureExtractor(); grid_bridge = IsaacGridBridge(); projectile_bridge = ProjectileBridgeV37(); tactical_grid = TacticalGridV37()
    dopamine = None if args.no_dopamine else DopamineSystem(brain, checkpoint=None, enable_policy_plasticity=False)
    controller.reset(seed=12345); loop = FastVisionLoop(vision, brain_hz=perfset.BRAIN_HZ, drop_stale=perfset.DROP_STALE_FRAMES); buffer = V35DemoBuffer()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); output = DATASET_DIR / f"isaac_v37_demo_{stamp}.npz"

    print("\n=============================================")
    print(" FLYISAAC V3.7 - FAST TACTICAL SENSORY DEMO")
    print("=============================================")
    print(f"Records DN={controller.dn_size}, VIS={visual_extractor.dim}, GRID={grid_bridge.dim}.")
    print("GRID -> MaleCNS LC10:", "OFF" if args.no_brain_grid else "ON")
    print("dopamine -> PAM/PPL-like MaleCNS:", "OFF" if dopamine is None else "ON")
    print("Human plays; no online policy plasticity is used while recording demonstrations.")
    print("CTRL+C stops and saves. Alt-Tab pauses. Auto-restart:", "OFF" if args.no_auto_restart else "ON")
    print("Output:", output)

    if not prepare_game_window(env, delay=config.START_FOCUS_DELAY, send_escape=not args.no_start_escape):
        safe_shutdown(env, vision, rewards); grid_bridge.close(); return
    try: vision.reset()
    except Exception: pass
    loop.reset(); deadline = time.perf_counter() + max(.01, args.minutes) * 60
    interval = 1 / max(1, args.demo_hz); next_sample = time.perf_counter(); last_saved = 0; brain_steps = 0

    try:
        while time.perf_counter() < deadline:
            try: _, events = rewards.poll()
            except Exception: events = []
            if dopamine is not None: dopamine.observe_events(events)

            if any(e.get("name") == "PLAYER_DEATH" for e in events):
                if args.no_auto_restart:
                    controller.reset(seed=12345 + brain_steps); grid_bridge.reset_run(); projectile_bridge.reset_run()
                    if dopamine is not None:
                        dopamine.reset_transient()
                        dopamine.reset_run_memory()
                    time.sleep(.5); continue
                ok, _ = robust_restart_after_death(env, rewards, attempts=config.RESTART_ATTEMPTS, retry_delay=config.RESTART_RETRY_DELAY)
                controller.reset(seed=12345 + brain_steps); grid_bridge.reset_run(); projectile_bridge.reset_run()
                if dopamine is not None:
                    dopamine.reset_transient()
                    dopamine.reset_run_memory()
                try: vision.reset()
                except Exception: pass
                loop.reset(); next_sample = time.perf_counter(); time.sleep(config.RESTART_COOLDOWN if ok else 1.0); continue

            if not isaac_focused(env):
                paused = pause_until_isaac_focused(env, label="FlyIsaac V3.6 recording paused. No samples are being recorded.")
                deadline += paused; controller.retina.reset(); controller.trace.reset()
                try: vision.reset()
                except Exception: pass
                loop.reset(); next_sample = time.perf_counter(); continue

            try: frame, _, _, _ = loop.next_frame(timeout=2)
            except Exception as exc:
                print("Vision wait:", exc); time.sleep(.05); continue

            grid_features, grid = grid_bridge.poll()

            # FLYCOMBAT SHOT_HIT / SHOT_MISS are emitted by the combat Lua
            # bridge, not by the legacy FLYAI reward stream. V3.7 used to
            # discard these events inside IsaacGridBridge, so dopamine never
            # saw successful hits. Drain and deliver them exactly once here.
            combat_events = grid_bridge.drain_events()
            if dopamine is not None and combat_events:
                dopamine.observe_events(combat_events)

            if dopamine is not None:
                dopamine.observe_room(grid_features)
            projectile_state = projectile_bridge.poll()
            tactical = tactical_grid.build(grid_features, projectile_state)
            if dopamine is not None: dopamine.step()
            dn = controller.step_brain(
                frame,
                grid_features=(None if args.no_brain_grid else grid_features),
                tactical=(None if args.no_brain_grid else tactical),
                dopamine=dopamine,
            )
            visual = visual_extractor.extract(controller.last_retina); brain_steps += 1
            now = time.perf_counter()
            if now < next_sample: continue
            labels, held = read_teacher_labels(); buffer.append(dn, visual, grid, labels); next_sample = now + interval
            if len(buffer) % 50 == 0:
                d = "OFF" if dopamine is None else f"{dopamine.value:+.3f}"
                print(f"samples={len(buffer):6d} | brain_steps={brain_steps:7d} | DA={d} | teacher={sorted(held) if held else ['NONE']}")
            if len(buffer) - last_saved >= config.AUTOSAVE_SAMPLES:
                buffer.save(output, metadata="FlyIsaac V3.7: DN uses 7x14 tactical grid + predicted hostile projectiles; decoder VIS/legacy GRID unchanged")
                last_saved = len(buffer)
    except KeyboardInterrupt:
        print("\nStopping V3.7 recorder...")
    finally:
        if len(buffer):
            buffer.save(output, metadata="FlyIsaac V3.7: DN uses 7x14 tactical grid + predicted hostile projectiles; decoder VIS/legacy GRID unchanged")
            print("Train: python train_decoder_v3.py")
        else:
            print("No samples recorded.")
        projectile_bridge.close(); grid_bridge.close(); safe_shutdown(env, vision, rewards)


if __name__ == "__main__":
    main()
