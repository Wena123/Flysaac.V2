import argparse
import time
from pathlib import Path

import numpy as np

from brain.fly import FlyAgentBrain
from isaac.async_vision import AsyncIsaacVision
from isaac.environment import IsaacEnvironment
from isaac.reward_reader import IsaacRewardReader
from isaac_v3 import config
from isaac_v3.controller import HybridIsaacController
from isaac_v3.decoder_v35 import BranchDecoderV35
try:
    from isaac_v3.decoder import ActionDecoder as LegacyV34Decoder
except Exception:
    LegacyV34Decoder = None
from isaac_v3.dopamine import DopamineSystem
from isaac_v3.grid_features import IsaacGridBridge
from isaac_v3.tactical_grid import ProjectileBridgeV37, TacticalGridV37
from isaac_v3 import performance_settings as perfset
from isaac_v3.multimodal import source_vector, mode_base_dim
from isaac_v3.neuron_monitor import NeuronMonitor
from isaac_v3.stuck_watchdog import RoomStuckWatchdog, hold_isaac_restart_key
from isaac_v3 import stuck_settings
from isaac_v3.runtime import (
    FastVisionLoop, focus_isaac, isaac_focused, pause_until_isaac_focused,
    prepare_game_window, robust_restart_after_death, safe_shutdown,
)
from isaac_v3.temporal import TemporalDNContext
from isaac_v3.visual_features import RetinaFeatureExtractor

HERE = Path(__file__).resolve().parent
CK = HERE / "checkpoints"
V37_BEST = CK / "isaac_decoder_v37_best.npz"
V37_BEST_MODE = CK / config.V37_BEST_MODE_FILE
V37_ASSIST = CK / "isaac_decoder_v37_best_assisted.npz"
V37_ASSIST_MODE = CK / config.V37_BEST_ASSISTED_MODE_FILE
V36_BEST = CK / "isaac_decoder_v36_best.npz"
V36_BEST_MODE = CK / config.V36_BEST_MODE_FILE
V36_ASSIST = CK / "isaac_decoder_v36_best_assisted.npz"
V36_ASSIST_MODE = CK / config.V36_BEST_ASSISTED_MODE_FILE
V35_BEST = CK / "isaac_decoder_v35_best.npz"
V35_BEST_MODE = CK / config.V35_BEST_MODE_FILE
V35_ASSIST = CK / "isaac_decoder_v35_best_assisted.npz"
V35_ASSIST_MODE = CK / config.V35_BEST_ASSISTED_MODE_FILE
DOPA_FILE = CK / config.DOPAMINE_POLICY_FILE


def read_mode(path):
    m = path.read_text(encoding="utf-8").strip().lower() if path.exists() else ""
    if m not in config.V35_INPUT_MODES:
        raise RuntimeError(f"Missing/invalid mode file: {path}. Run training.")
    return m


def find_v34_visual_model():
    direct = CK / "isaac_decoder_v34_visual.npz"
    if direct.exists():
        return direct
    mode_file = CK / getattr(config, "V34_BEST_MODE_FILE", "isaac_decoder_v34_best_mode.txt")
    best = CK / "isaac_decoder_v34_best.npz"
    if mode_file.exists() and best.exists() and mode_file.read_text(encoding="utf-8").strip().lower() == "visual":
        return best
    return None


def decoder_version(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(data["version"][0]) if "version" in data else 0
    except Exception:
        return 0


def load_any_decoder(path, mode):
    version = decoder_version(path)
    if version == 35:
        dec = BranchDecoderV35.load(path)
        if dec.mode != mode:
            raise RuntimeError(f"Model is {dec.mode}, requested {mode}")
        return dec, False
    if version == 34 and mode == "visual" and LegacyV34Decoder is not None:
        return LegacyV34Decoder.load(path), True
    raise RuntimeError(f"Unsupported decoder format v{version} for mode {mode}: {path}")


def choose_model(args):
    if args.pure_vision:
        mode = "visual"
        candidates = [
            args.model,
            CK / "isaac_decoder_v37_visual.npz",
            CK / "isaac_decoder_v36_visual.npz",
            CK / "isaac_decoder_v35_visual.npz",
            find_v34_visual_model(),
        ]
        return mode, next((p for p in candidates if p is not None and p.exists()), None)

    if args.grid_assist:
        if V37_ASSIST.exists() and V37_ASSIST_MODE.exists():
            return read_mode(V37_ASSIST_MODE), args.model or V37_ASSIST
        if V36_ASSIST.exists() and V36_ASSIST_MODE.exists():
            return read_mode(V36_ASSIST_MODE), args.model or V36_ASSIST
        if V35_ASSIST.exists() and V35_ASSIST_MODE.exists():
            return read_mode(V35_ASSIST_MODE), args.model or V35_ASSIST
        return None, None

    if args.mode:
        mode = args.mode
        candidates = [args.model, CK / f"isaac_decoder_v37_{mode}.npz", CK / f"isaac_decoder_v36_{mode}.npz", CK / f"isaac_decoder_v35_{mode}.npz"]
        if mode == "visual":
            candidates.append(find_v34_visual_model())
        return mode, next((p for p in candidates if p is not None and p.exists()), None)

    if V37_BEST.exists() and V37_BEST_MODE.exists():
        return read_mode(V37_BEST_MODE), args.model or V37_BEST
    if V36_BEST.exists() and V36_BEST_MODE.exists():
        return read_mode(V36_BEST_MODE), args.model or V36_BEST
    if V35_BEST.exists() and V35_BEST_MODE.exists():
        return read_mode(V35_BEST_MODE), args.model or V35_BEST
    return "visual", args.model or find_v34_visual_model()


def main():
    p = argparse.ArgumentParser(description="FlyIsaac V3.6: grid-to-brain + dopamine neuromodulation.")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--mode", choices=config.V35_INPUT_MODES, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pure-vision", action="store_true")
    g.add_argument("--grid-assist", action="store_true")
    p.add_argument("--no-brain-grid", action="store_true", help="Do not inject assisted grid into MaleCNS LC10.")
    p.add_argument("--no-dopamine", action="store_true", help="Disable dopamine neural drive and policy plasticity.")
    p.add_argument("--no-dopamine-plasticity", action="store_true", help="Keep DAN neural signal, but disable online action-bias learning.")
    p.add_argument("--no-start-escape", action="store_true")
    p.add_argument("--no-auto-restart", action="store_true")
    p.add_argument("--restart-cooldown", type=float, default=config.RESTART_COOLDOWN)
    p.add_argument("--no-neuron-monitor", action="store_true")
    p.add_argument("--neuron-projection", choices=("xy", "xz", "yz"), default="xz")
    p.add_argument("--no-stuck-watchdog", action="store_true", help="Disable 5-minute same-room R-reset watchdog.")
    args = p.parse_args()

    mode, model = choose_model(args)
    if mode is None or model is None or not model.exists():
        raise SystemExit(
            "No compatible trained decoder found. GRID modes need V3.6/V3.5 grid data + training; "
            "PURE VISUAL can reuse your V3.4 visual checkpoint."
        )

    decoder, legacy_v34 = load_any_decoder(model, mode)
    env = IsaacEnvironment()
    rewards = IsaacRewardReader()
    vision = AsyncIsaacVision(target_hz=perfset.VISION_CAPTURE_HZ, motion_threshold=18, process_scale=perfset.VISION_PROCESS_SCALE, max_buffer_frames=perfset.VISION_BUFFER_FRAMES)
    vision.start()
    brain = FlyAgentBrain()
    controller = HybridIsaacController(brain)
    vis = RetinaFeatureExtractor()
    grid_bridge = IsaacGridBridge(); projectile_bridge = ProjectileBridgeV37(); tactical_grid = TacticalGridV37()
    dopamine = None if args.no_dopamine else DopamineSystem(
        brain,
        checkpoint=DOPA_FILE,
        enable_policy_plasticity=not args.no_dopamine_plasticity,
    )
    controller.reset(seed=12345)
    loop = FastVisionLoop(vision, brain_hz=perfset.BRAIN_HZ, drop_stale=perfset.DROP_STALE_FRAMES)

    base = mode_base_dim(mode, controller.dn_size, vis.dim, grid_bridge.dim)
    expected_base = (decoder.full_input_dim // decoder.context_frames) if legacy_v34 else decoder.base_input_dim
    if base != expected_base:
        raise RuntimeError(f"Input mismatch: runtime {base}, model {expected_base}")
    temporal = TemporalDNContext(dn_dim=base, frames=decoder.context_frames, sample_hz=decoder.sample_hz)
    monitor = NeuronMonitor(
        update_hz=perfset.MONITOR_HZ,
        history=config.NEURON_MONITOR_HISTORY,
        raster_bins=config.NEURON_MONITOR_RASTER_BINS,
        enabled=not args.no_neuron_monitor,
        brain=brain.brain,
        dn_cells=controller.trace.cells,
        projection=args.neuron_projection,
    )
    monitor.open()
    stuck_watchdog = RoomStuckWatchdog()
    stuck_enabled = bool(stuck_settings.ENABLED) and (not args.no_stuck_watchdog)
    stuck_resets = 0

    brain_grid_on = (mode != "visual") and (not args.no_brain_grid)
    print("\n================================")
    print("      FLYISAAC V3.7 PLAY")
    print("================================")
    print("input mode:", mode.upper())
    print("decoder:", "V3.4 LEGACY VISUAL (reused)" if legacy_v34 else "V3.5 branch decoder")
    print("model:", model.name)
    print("7x14 tactical grid -> MaleCNS:", "ON" if brain_grid_on else "OFF")
    print(f"capture target={perfset.VISION_CAPTURE_HZ:.0f}Hz | MaleCNS target={perfset.BRAIN_HZ:.0f}Hz | stale-frame drop={perfset.DROP_STALE_FRAMES}")
    print("dopamine neural signal:", "OFF" if dopamine is None else "ON")
    print("dopamine policy plasticity:", "OFF" if dopamine is None or not dopamine.enable_policy_plasticity else "ON")
    print("dopamine memory:", DOPA_FILE.name)
    print("door reward anti-farm memory: ON (A<->B pays NEW_ROOM dopamine once per run)")
    print(f"VIS={vis.dim} GRID={grid_bridge.dim} DN={controller.dn_size}")
    print(f"context={decoder.context_frames} @ {decoder.sample_hz:.1f}Hz lag={decoder.lag_samples:+d}")
    print("auto restart:", "OFF" if args.no_auto_restart else "ON")
    print("focus pause: ON")
    print("anatomical neuron monitor:", "OFF" if args.no_neuron_monitor else "ON")
    print("same-room stuck watchdog:", "OFF" if not stuck_enabled else f"ON ({stuck_watchdog.timeout_seconds/60.0:.1f} min -> hold {stuck_settings.RESTART_KEY.upper()} {stuck_settings.RESTART_HOLD_SECONDS:.1f}s)")
    print("CTRL+C to stop")

    if not prepare_game_window(env, delay=config.START_FOCUS_DELAY, send_escape=not args.no_start_escape):
        monitor.close(); projectile_bridge.close(); grid_bridge.close(); safe_shutdown(env, vision, rewards); return
    try: vision.reset()
    except Exception: pass
    loop.reset()
    step = 0; runs = 1; deaths = 0; warm = False

    try:
        while True:
            try: _, events = rewards.poll()
            except Exception: events = []
            if dopamine is not None:
                dopamine.observe_events(events)

            if any(e.get("name") == "PLAYER_DEATH" for e in events):
                env.reset_controls(); deaths += 1
                print(f"\nDEATH #{deaths} | RUN #{runs}")
                if args.no_auto_restart:
                    temporal.reset(); controller.reset(seed=12345 + deaths)
                    if dopamine is not None:
                        dopamine.reset_transient()
                        dopamine.reset_run_memory()
                    time.sleep(.5); continue
                ok, _ = robust_restart_after_death(
                    env, rewards, attempts=config.RESTART_ATTEMPTS, retry_delay=config.RESTART_RETRY_DELAY
                )
                controller.reset(seed=12345 + deaths); temporal.reset(); grid_bridge.reset_run(); projectile_bridge.reset_run(); monitor.reset(); stuck_watchdog.reset(); warm = False
                if dopamine is not None:
                    dopamine.reset_transient()
                    dopamine.reset_run_memory()
                try: vision.reset()
                except Exception: pass
                loop.reset()
                if ok:
                    runs += 1; time.sleep(args.restart_cooldown)
                    if not isaac_focused(env): focus_isaac(env)
                else:
                    time.sleep(1)
                continue

            if not isaac_focused(env):
                pause_until_isaac_focused(env, label="FlyIsaac V3.6 PLAY paused.")
                temporal.reset(); controller.retina.reset(); controller.trace.reset(); monitor.reset(); stuck_watchdog.reset()
                if dopamine is not None: dopamine.move_elig.fill(0); dopamine.shoot_elig.fill(0)
                try: vision.reset()
                except Exception: pass
                loop.reset(); warm = False; continue

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

            # --------------------------------------------------------
            # SAME-ROOM STUCK WATCHDOG
            # --------------------------------------------------------
            if stuck_enabled:
                stuck_now, stuck_info = stuck_watchdog.observe(grid_features)
                if stuck_info["changed"]:
                    print(f"ROOM WATCHDOG | entered room={stuck_info['room']} | timer reset")
                elif stuck_info["status_due"]:
                    print(
                        f"ROOM WATCHDOG | room={stuck_info['room']} | "
                        f"same room {stuck_info['elapsed']:.0f}s / {stuck_watchdog.timeout_seconds:.0f}s"
                    )

                if stuck_now:
                    stuck_resets += 1
                    env.reset_controls()
                    print()
                    print("================================")
                    print("     ROOM STUCK WATCHDOG")
                    print("================================")
                    print(f"Same room for {stuck_info['elapsed']:.1f}s.")
                    print(
                        f"Holding {stuck_settings.RESTART_KEY.upper()} for "
                        f"{stuck_settings.RESTART_HOLD_SECONDS:.1f}s..."
                    )
                    try:
                        hold_isaac_restart_key(
                            env,
                            seconds=stuck_settings.RESTART_HOLD_SECONDS,
                            key=stuck_settings.RESTART_KEY,
                        )
                    except Exception as exc:
                        print("ROOM WATCHDOG restart warning:", exc)

                    # Reset only transient neural/runtime state. Learned decoder
                    # and saved dopamine policy memory are deliberately preserved.
                    controller.reset(seed=12345 + deaths + 10000 + stuck_resets)
                    temporal.reset()
                    grid_bridge.reset_run()
                    projectile_bridge.reset_run()
                    monitor.reset()
                    stuck_watchdog.reset()
                    warm = False
                    if dopamine is not None:
                        dopamine.reset_transient()
                        dopamine.reset_run_memory()
                    try:
                        vision.reset()
                    except Exception:
                        pass
                    loop.reset()
                    runs += 1
                    print("MaleCNS transient state reset. Learned decoder/dopamine memory preserved.")
                    print(f"Waiting {stuck_settings.POST_RESTART_WAIT_SECONDS:.1f}s for new run...")
                    time.sleep(max(0.0, float(stuck_settings.POST_RESTART_WAIT_SECONDS)))
                    if not isaac_focused(env):
                        focus_isaac(env)
                    continue

            if dopamine is not None: dopamine.step()
            dn = controller.step_brain(
                frame,
                grid_features=(grid_features if brain_grid_on else None),
                tactical=(tactical if brain_grid_on else None),
                dopamine=dopamine,
            )
            visual = vis.extract(controller.last_retina)
            raw = source_vector(mode, dn, visual, grid)
            ctx = temporal.update(raw)
            grid_field = controller.encoder.last_grid_field if brain_grid_on else None
            perf = loop.perf_snapshot()
            dopa_value = dopamine.value if dopamine is not None else None

            if ctx is None:
                env.reset_controls()
                monitor.update(
                    dn=dn, fired=controller.last_fired, retina=controller.last_retina,
                    actions=set(), mode=mode, run=runs, deaths=deaths, step=step,
                    grid_field=grid_field, dopamine=dopa_value, tactical=(tactical if brain_grid_on else None), perf=perf,
                )
                if not warm:
                    print(f"Temporal warm-up: {decoder.context_frames} snapshots..."); warm = True
                continue

            actions, info = decoder.predict_actions(ctx)
            if dopamine is not None:
                actions, info = dopamine.modulate_decision(info)
                dopamine.observe_decision(info)
                dopamine.maybe_autosave()
            env.act(held_actions=actions, tap_actions=[])
            step += 1
            monitor.update(
                dn=dn, fired=controller.last_fired, retina=controller.last_retina,
                actions=actions, mode=mode, run=runs, deaths=deaths, step=step,
                move_info=(info["move_name"], info["move_confidence"]),
                shoot_info=(info["shoot_name"], info["shoot_confidence"]),
                move_probs=info["move_probs"], shoot_probs=info["shoot_probs"],
                grid_field=grid_field, dopamine=dopa_value, tactical=(tactical if brain_grid_on else None), perf=perf,
            )
            if step % 50 == 0:
                d = f"{dopa_value:+.3f}" if dopa_value is not None else "OFF"
                print(
                    f"step={step:07d} | run={runs} deaths={deaths} | {mode} | "
                    f"DA={d} | gridBrain={int(brain_grid_on)} | proj={tactical.get('projectile_count',0)} | "
                    f"cap={perf.get('capture_fps',0):.1f} brain={perf.get('brain_fps',0):.1f} | "
                    f"actions={sorted(actions) if actions else ['NONE']} | "
                    f"move={info['move_name']}:{info['move_confidence']:.2f} "
                    f"shoot={info['shoot_name']}:{info['shoot_confidence']:.2f}"
                )
    except KeyboardInterrupt:
        print("\nStopping FlyIsaac V3.7...")
    finally:
        if dopamine is not None and dopamine.enable_policy_plasticity:
            dopamine.save()
            print("Dopamine memory saved:", DOPA_FILE)
        print(f"Session: runs={runs} deaths={deaths} stuck_resets={stuck_resets} steps={step}")
        monitor.close(); projectile_bridge.close(); grid_bridge.close(); safe_shutdown(env, vision, rewards)


if __name__ == "__main__":
    main()
