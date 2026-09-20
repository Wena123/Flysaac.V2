import time
import pydirectinput
import win32gui


class VisionLoop:
    def __init__(self, vision_stream):
        self.vision_stream = vision_stream
        self.sequence = -1
    def reset(self): self.sequence = -1
    def next_frame(self, timeout=2.0):
        result = self.vision_stream.consume(after_sequence=self.sequence, timeout=float(timeout))
        frame, features, self.sequence, frames_combined, stats = result
        return frame, features, frames_combined, stats


class FastVisionLoop:
    """Consume a faster capture stream while stepping MaleCNS at a bounded rate.

    Frames that arrive between brain ticks are consumed but intentionally not
    stepped through the brain.  This keeps latency low instead of building a
    queue of old frames.
    """
    def __init__(self, vision_stream, brain_hz=45.0, drop_stale=True):
        self.vision_stream = vision_stream
        self.brain_hz = max(1.0, float(brain_hz))
        self.period = 1.0/self.brain_hz
        self.drop_stale = bool(drop_stale)
        self.sequence = -1
        self.next_due = 0.0
        self.started = time.perf_counter()
        self.capture_frames = 0
        self.brain_frames = 0
        self.dropped_frames = 0
        self._last_seq = -1
    def reset(self):
        self.sequence = -1; self.next_due = 0.0
        self.started = time.perf_counter(); self.capture_frames = 0; self.brain_frames = 0; self.dropped_frames = 0; self._last_seq = -1
    def next_frame(self, timeout=2.0):
        deadline = time.perf_counter() + float(timeout)
        latest = None
        while True:
            remain = max(.001, deadline-time.perf_counter())
            result = self.vision_stream.consume(after_sequence=self.sequence, timeout=remain)
            frame, features, seq, frames_combined, stats = result
            if self._last_seq >= 0:
                delta = max(1, int(seq)-int(self._last_seq))
                self.capture_frames += delta
                self.dropped_frames += max(0, delta-1)
            else:
                self.capture_frames += 1
            self._last_seq = int(seq); self.sequence = int(seq)
            latest = (frame, features, frames_combined, stats)
            now = time.perf_counter()
            if self.next_due <= 0.0:
                self.next_due = now
            if now >= self.next_due or not self.drop_stale:
                self.brain_frames += 1
                self.next_due = max(self.next_due + self.period, now + .15*self.period)
                return latest
            # Continue consuming fresh frames until the next brain tick.  At a
            # 60 Hz source / 45 Hz brain this normally drops only ~1 frame.
            if time.perf_counter() >= deadline:
                self.brain_frames += 1
                self.next_due = time.perf_counter() + self.period
                return latest
    def perf_snapshot(self):
        dt = max(1e-6, time.perf_counter()-self.started)
        return dict(capture_fps=self.capture_frames/dt, brain_fps=self.brain_frames/dt, dropped_frames=int(self.dropped_frames))


def isaac_focused(env):
    try: return win32gui.GetForegroundWindow() == env.controls.window.hwnd
    except Exception: return True

def focus_isaac(env):
    window = env.controls.window; print("\nTrying to focus Isaac...")
    try: window.focus()
    except Exception as exc: print("window.focus warning:", exc)
    time.sleep(.5)
    try:
        if win32gui.GetForegroundWindow() == window.hwnd:
            print("Isaac successfully focused."); return True
    except Exception: return True
    try:
        rect = window.client_rect(); pydirectinput.click(rect["left"]+rect["width"]//2, rect["top"]+rect["height"]//2); time.sleep(.7)
        ok = win32gui.GetForegroundWindow() == window.hwnd; print("Isaac focus after click:", "OK" if ok else "FAILED"); return bool(ok)
    except Exception as exc: print("Isaac focus failed:", exc); return False

def press_escape():
    pydirectinput.keyDown("esc"); time.sleep(.15); pydirectinput.keyUp("esc")

def prepare_game_window(env, delay=3.0, send_escape=True):
    print(f"\nIsaac will be focused in {float(delay):.1f} seconds..."); time.sleep(max(0.0,float(delay)))
    if not focus_isaac(env): return False
    if send_escape: time.sleep(.5); print("Sending ESC..."); press_escape(); time.sleep(.5)
    return True

def wait_for_focus(env, message="Focus the Isaac window..."):
    printed=False
    while not isaac_focused(env):
        try: env.reset_controls()
        except Exception: pass
        if not printed: print(message); printed=True
        time.sleep(.20)
    if printed: print("Isaac focused. Continuing.")

def pause_until_isaac_focused(env, label="FlyIsaac paused."):
    if isaac_focused(env): return 0.0
    try: env.reset_controls()
    except Exception: pass
    print("\nIsaac lost focus."); print(label); started=time.perf_counter()
    while not isaac_focused(env): time.sleep(.25)
    paused=time.perf_counter()-started; print("Isaac focused again."); print(f"Resuming after {paused:.1f}s pause."); return float(paused)

def robust_restart_after_death(env, rewards, attempts=3, retry_delay=.85):
    from isaac.restart import restart_after_death
    attempts=max(1,int(attempts)); last=[]
    try: env.reset_controls()
    except Exception: pass
    for attempt in range(1,attempts+1):
        if not isaac_focused(env): focus_isaac(env)
        print(f"Auto-restart attempt {attempt}/{attempts}...")
        try: success, events = restart_after_death(env,rewards); last=events or []
        except Exception as exc: success=False; print("restart_after_death error:",exc)
        if success: print("Auto-restart: SUCCESS"); return True,last
        if attempt<attempts: print("Auto-restart attempt failed; retrying..."); time.sleep(max(0,float(retry_delay)))
    print("Auto-restart: FAILED after all attempts"); return False,last

def safe_shutdown(env=None, vision_stream=None, rewards=None):
    if env is not None:
        try: env.reset_controls()
        except Exception: pass
    if vision_stream is not None:
        try: vision_stream.stop()
        except Exception: pass
    if rewards is not None:
        try: rewards.close()
        except Exception: pass
    if env is not None:
        try: env.stop()
        except Exception: pass
