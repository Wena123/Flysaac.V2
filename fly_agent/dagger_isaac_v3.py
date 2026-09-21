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
from isaac_v3.decoder_v35 import BranchDecoderV35
from isaac_v3.dopamine import DopamineSystem
from isaac_v3.grid_features import IsaacGridBridge
from isaac_v3.tactical_grid import ProjectileBridgeV37, TacticalGridV37
from isaac_v3 import performance_settings as perfset
from isaac_v3.multimodal import V35DemoBuffer, source_vector, mode_base_dim
from isaac_v3.neuron_monitor import NeuronMonitor
from isaac_v3.runtime import (
    FastVisionLoop, isaac_focused, pause_until_isaac_focused, prepare_game_window,
    robust_restart_after_death, safe_shutdown,
)
from isaac_v3.teacher import ToggleKey, read_teacher_labels
from isaac_v3.temporal import TemporalDNContext
from isaac_v3.visual_features import RetinaFeatureExtractor

HERE = Path(__file__).resolve().parent
CK = HERE / "checkpoints"
DATA = HERE / "datasets"
MODEL = CK / "isaac_decoder_v37_best.npz"
MODE_FILE = CK / config.V37_BEST_MODE_FILE
DOPA_FILE = CK / config.DOPAMINE_POLICY_FILE


def mode_file():
    m = MODE_FILE.read_text(encoding="utf-8").strip().lower() if MODE_FILE.exists() else ""
    if m not in config.V35_INPUT_MODES:
        raise RuntimeError("Train V3.6 first")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--mode", choices=config.V35_INPUT_MODES, default=None)
    p.add_argument("--demo-hz", type=float, default=config.DEMO_HZ)
    p.add_argument("--no-start-escape", action="store_true")
    p.add_argument("--no-auto-restart", action="store_true")
    p.add_argument("--no-neuron-monitor", action="store_true")
    p.add_argument("--neuron-projection", choices=("xy", "xz", "yz"), default="xz")
    p.add_argument("--no-dopamine", action="store_true")
    p.add_argument("--no-dopamine-plasticity", action="store_true")
    p.add_argument("--no-brain-grid", action="store_true")
    args = p.parse_args()

    mode = args.mode or mode_file(); decoder = BranchDecoderV35.load(args.model)
    if decoder.mode != mode: raise RuntimeError(f"Model mode={decoder.mode}, requested {mode}")
    env = IsaacEnvironment(); rewards = IsaacRewardReader()
    vision = AsyncIsaacVision(target_hz=perfset.VISION_CAPTURE_HZ, motion_threshold=18, process_scale=perfset.VISION_PROCESS_SCALE, max_buffer_frames=perfset.VISION_BUFFER_FRAMES); vision.start()
    brain = FlyAgentBrain(); controller = HybridIsaacController(brain); vis = RetinaFeatureExtractor(); grid_bridge = IsaacGridBridge(); projectile_bridge = ProjectileBridgeV37(); tactical_grid = TacticalGridV37()
    dopamine = None if args.no_dopamine else DopamineSystem(brain, checkpoint=DOPA_FILE, enable_policy_plasticity=not args.no_dopamine_plasticity)
    controller.reset(seed=12345); loop = FastVisionLoop(vision, brain_hz=perfset.BRAIN_HZ, drop_stale=perfset.DROP_STALE_FRAMES)
    base = mode_base_dim(mode, controller.dn_size, vis.dim, grid_bridge.dim); temporal = TemporalDNContext(base, decoder.context_frames, decoder.sample_hz)
    buffer = V35DemoBuffer(); toggle = ToggleKey(); teacher = False; next_sample = 0.; interval = 1 / max(1, args.demo_hz)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); output = DATA / f"isaac_v37_dagger_{stamp}.npz"
    monitor = NeuronMonitor(update_hz=perfset.MONITOR_HZ, history=config.NEURON_MONITOR_HISTORY, raster_bins=config.NEURON_MONITOR_RASTER_BINS, enabled=not args.no_neuron_monitor, brain=brain.brain, dn_cells=controller.trace.cells, projection=args.neuron_projection); monitor.open()
    brain_grid_on = not args.no_brain_grid

    print("\n================================")
    print("     FLYISAAC V3.7 DAgger")
    print("================================")
    print("agent mode:", mode.upper())
    print("F8 toggles HUMAN CORRECTION. Corrections save DN + VISUAL + GRID.")
    print("GRID -> MaleCNS LC10:", "ON" if brain_grid_on else "OFF")
    print("dopamine:", "OFF" if dopamine is None else "ON")
    print("Alt-Tab pauses. Auto-restart:", "OFF" if args.no_auto_restart else "ON")

    if not prepare_game_window(env, delay=config.START_FOCUS_DELAY, send_escape=not args.no_start_escape):
        monitor.close(); projectile_bridge.close(); grid_bridge.close(); safe_shutdown(env, vision, rewards); return
    try: vision.reset()
    except Exception: pass
    loop.reset(); step = 0

    try:
        while True:
            try: _, events = rewards.poll()
            except Exception: events = []
            if dopamine is not None: dopamine.observe_events(events)

            if any(e.get("name") == "PLAYER_DEATH" for e in events):
                env.reset_controls(); temporal.reset(); monitor.reset()
                if args.no_auto_restart:
                    if dopamine is not None:
                        dopamine.reset_transient()
                        dopamine.reset_run_memory()
                    time.sleep(.5); continue
                ok, _ = robust_restart_after_death(env, rewards, attempts=config.RESTART_ATTEMPTS, retry_delay=config.RESTART_RETRY_DELAY)
                controller.reset(seed=12345 + step + len(buffer)); grid_bridge.reset_run(); projectile_bridge.reset_run()
                if dopamine is not None:
                    dopamine.reset_transient()
                    dopamine.reset_run_memory()
                try: vision.reset()
                except Exception: pass
                loop.reset(); next_sample = 0.; time.sleep(config.RESTART_COOLDOWN if ok else 1.0); continue

            if not isaac_focused(env):
                pause_until_isaac_focused(env, label="FlyIsaac V3.6 DAgger paused.")
                temporal.reset(); controller.retina.reset(); controller.trace.reset(); monitor.reset()
                if dopamine is not None: dopamine.move_elig.fill(0); dopamine.shoot_elig.fill(0)
                try: vision.reset()
                except Exception: pass
                loop.reset(); next_sample = 0. if teacher else next_sample; continue

            if toggle.pressed():
                teacher = not teacher; env.reset_controls(); next_sample = 0. if teacher else next_sample; temporal.reset()
                if dopamine is not None: dopamine.move_elig.fill(0); dopamine.shoot_elig.fill(0)
                print("\nMODE:", "HUMAN CORRECTION" if teacher else "AGENT")
                if teacher: print("Human keyboard capture ACTIVE (WASD + arrows).")
                time.sleep(.12)

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
            dn = controller.step_brain(frame, grid_features=(grid_features if brain_grid_on else None), tactical=(tactical if brain_grid_on else None), dopamine=dopamine)
            visual = vis.extract(controller.last_retina); raw = source_vector(mode, dn, visual, grid); ctx = temporal.update(raw); now = time.perf_counter()
            grid_field = controller.encoder.last_grid_field if brain_grid_on else None
            perf = loop.perf_snapshot()
            dopa_value = dopamine.value if dopamine is not None else None

            if teacher:
                monitor.update(dn=dn, fired=controller.last_fired, retina=controller.last_retina, actions=set(), mode=mode + " / HUMAN", step=step, grid_field=grid_field, dopamine=dopa_value, tactical=(tactical if brain_grid_on else None), perf=perf)
                # IMPORTANT V3.3.1 fix preserved: do NOT reset controls here.
                if now >= next_sample:
                    labels, held = read_teacher_labels(); buffer.append(dn, visual, grid, labels); next_sample = now + interval
                    if len(buffer) % 25 == 0: print(f"corrections={len(buffer):5d} | teacher={sorted(held) if held else ['NONE']}")
            else:
                if ctx is None:
                    env.reset_controls(); continue
                actions, info = decoder.predict_actions(ctx)
                if dopamine is not None:
                    actions, info = dopamine.modulate_decision(info); dopamine.observe_decision(info); dopamine.maybe_autosave()
                env.act(held_actions=actions, tap_actions=[]); step += 1
                monitor.update(
                    dn=dn, fired=controller.last_fired, retina=controller.last_retina, actions=actions, mode=mode + " / AGENT", step=step,
                    move_info=(info["move_name"], info["move_confidence"]), shoot_info=(info["shoot_name"], info["shoot_confidence"]),
                    move_probs=info["move_probs"], shoot_probs=info["shoot_probs"], grid_field=grid_field, dopamine=dopa_value, tactical=(tactical if brain_grid_on else None), perf=perf,
                )
                if step % 75 == 0: print(f"agent step {step} | DA={dopa_value if dopa_value is not None else 'OFF'} | {sorted(actions) if actions else ['NONE']}")
    except KeyboardInterrupt:
        print("\nStopping V3.7 DAgger...")
    finally:
        if len(buffer):
            buffer.save(output, metadata="FlyIsaac V3.7 DAgger: tactical 7x14 + projectile-prediction DN")
            print("Retrain: python train_decoder_v3.py")
        else:
            print("No corrections recorded.")
        if dopamine is not None and dopamine.enable_policy_plasticity: dopamine.save()
        monitor.close(); projectile_bridge.close(); grid_bridge.close(); safe_shutdown(env, vision, rewards)


if __name__ == "__main__":
    main()
