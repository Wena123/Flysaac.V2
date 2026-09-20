from pathlib import Path
import time
import numpy as np

from . import config


class ProjectileBridgeV37:
    """Tail FLYCOMBATV37 lines emitted by the optional V3.7 Isaac Lua mod.

    The old FLYCOMBAT reader is deliberately left untouched.  This bridge adds
    detailed hostile projectile position, velocity and hitbox information while
    the legacy 12x20 decoder grid remains backwards compatible.
    """

    PREFIX = "FLYCOMBATV37|STATE|"

    def __init__(self, log_path=None):
        if log_path is None:
            log_path = Path.home()/"Documents"/"My Games"/"Binding of Isaac Repentance+"/"log.txt"
        self.log_path = Path(log_path)
        self.file = None
        self.position = 0
        self.has_state = False
        self.room_index = -1
        self.player_x = 0.5
        self.player_y = 0.5
        self.player_rx = 0.018
        self.player_ry = 0.030
        self.projectiles = []
        self.last_update = 0.0
        self._open()

    def _open(self):
        if not self.log_path.exists():
            return
        self.file = open(self.log_path, "r", encoding="utf-8", errors="ignore")
        self.file.seek(0, 2)
        self.position = self.file.tell()

    @staticmethod
    def _f(x, default=0.0):
        try:
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    @classmethod
    def parse_state_payload(cls, payload):
        parts = payload.strip().split("|")
        if len(parts) < 7:
            return None
        try:
            room = int(float(parts[0]))
        except Exception:
            room = -1
        px = float(np.clip(cls._f(parts[1], .5), 0, 1))
        py = float(np.clip(cls._f(parts[2], .5), 0, 1))
        prx = max(0.0, cls._f(parts[3], .018))
        pry = max(0.0, cls._f(parts[4], .030))
        try:
            count = max(0, int(float(parts[5])))
        except Exception:
            count = 0
        text = parts[6] if len(parts) >= 7 else "-"
        projectiles = []
        if text and text != "-":
            for item in text.split(";"):
                p = item.split(",")
                if len(p) < 6:
                    continue
                x, y = np.clip([cls._f(p[0]), cls._f(p[1])], 0, 1)
                vx, vy = cls._f(p[2]), cls._f(p[3])  # normalized room units / second
                rx, ry = max(0.0, cls._f(p[4])), max(0.0, cls._f(p[5]))
                projectiles.append(dict(x=float(x), y=float(y), vx=vx, vy=vy, rx=rx, ry=ry))
        if count and len(projectiles) > count:
            projectiles = projectiles[:count]
        return dict(room_index=room, player_x=px, player_y=py, player_rx=prx, player_ry=pry, projectiles=projectiles)

    def poll(self):
        if self.file is None:
            self._open()
            if self.file is None:
                return self.snapshot()
        try:
            size = self.log_path.stat().st_size
            if size < self.position:
                self.file.close(); self.file = None; self._open()
                return self.snapshot()
        except OSError:
            return self.snapshot()
        self.file.seek(self.position)
        latest = None
        while True:
            line = self.file.readline()
            if not line:
                break
            marker = line.find(self.PREFIX)
            if marker < 0:
                continue
            payload = line[marker + len(self.PREFIX):].strip()
            parsed = self.parse_state_payload(payload)
            if parsed is not None:
                latest = parsed
        self.position = self.file.tell()
        if latest is not None:
            self.room_index = latest["room_index"]
            self.player_x = latest["player_x"]; self.player_y = latest["player_y"]
            self.player_rx = latest["player_rx"]; self.player_ry = latest["player_ry"]
            self.projectiles = latest["projectiles"]
            self.has_state = True
            self.last_update = time.perf_counter()
        return self.snapshot()

    def snapshot(self):
        return dict(
            has_state=bool(self.has_state), room_index=int(self.room_index),
            player_x=float(self.player_x), player_y=float(self.player_y),
            player_rx=float(self.player_rx), player_ry=float(self.player_ry),
            projectiles=[dict(p) for p in self.projectiles],
            age=(time.perf_counter() - self.last_update if self.last_update else float("inf")),
        )

    def reset_run(self):
        self.has_state = False
        self.projectiles = []
        self.last_update = 0.0

    def close(self):
        if self.file is not None:
            try: self.file.close()
            except Exception: pass
        self.file = None


class TacticalGridV37:
    ROWS = 7
    COLS = 14
    FUTURE_TIMES = (0.15, 0.30, 0.50)

    def __init__(self):
        self.rows = int(config.V37_TACTICAL_GRID_ROWS)
        self.cols = int(config.V37_TACTICAL_GRID_COLS)
        self.last = None
        yy = (np.arange(self.rows, dtype=np.float32) + .5) / self.rows
        xx = (np.arange(self.cols, dtype=np.float32) + .5) / self.cols
        self._xx, self._yy = np.meshgrid(xx, yy)

    def _resize(self, value):
        if not isinstance(value, np.ndarray) or value.ndim != 2 or value.size == 0:
            return np.zeros((self.rows, self.cols), np.float32)
        a = np.asarray(value, np.float32)
        rr = np.clip(np.round((np.arange(self.rows)+.5)*a.shape[0]/self.rows-.5).astype(int), 0, a.shape[0]-1)
        cc = np.clip(np.round((np.arange(self.cols)+.5)*a.shape[1]/self.cols-.5).astype(int), 0, a.shape[1]-1)
        return np.clip(np.nan_to_num(a[rr][:, cc]), 0, 1).astype(np.float32)

    def _ellipse(self, x, y, rx, ry, strength=1.0):
        rx = max(float(rx), .50/self.cols)
        ry = max(float(ry), .50/self.rows)
        d = ((self._xx-float(x))/rx)**2 + ((self._yy-float(y))/ry)**2
        # Soft edge gives sub-cell precision even on a compact 7x14 neural grid.
        return (float(strength) * np.exp(-1.75*d)).astype(np.float32)

    def _projectile_map(self, projectiles, t):
        g = np.zeros((self.rows, self.cols), np.float32)
        for p in projectiles:
            x = p["x"] + p["vx"]*t
            y = p["y"] + p["vy"]*t
            if x < -.08 or x > 1.08 or y < -.08 or y > 1.08:
                continue
            # Slight uncertainty growth with prediction horizon.
            rx = p["rx"] + .010 + .018*t
            ry = p["ry"] + .010 + .018*t
            g = np.maximum(g, self._ellipse(x, y, rx, ry, 1.0))
        return np.clip(g, 0, 1)

    @staticmethod
    def _tti(projectile, px, py, prx, pry):
        x, y = projectile["x"], projectile["y"]
        vx, vy = projectile["vx"], projectile["vy"]
        vv = vx*vx + vy*vy
        if vv < 1e-8:
            return None
        t = ((px-x)*vx + (py-y)*vy) / vv
        if t < 0.0 or t > .80:
            return None
        cx, cy = x + vx*t, y + vy*t
        # ellipse-ish conservative collision radius in normalized room units
        rx = prx + projectile["rx"] + .012
        ry = pry + projectile["ry"] + .012
        q = ((cx-px)/max(rx,1e-5))**2 + ((cy-py)/max(ry,1e-5))**2
        return float(t) if q <= 1.0 else None

    def build(self, legacy_features=None, projectile_state=None):
        f = dict(legacy_features or {})
        ps = dict(projectile_state or {})
        channels = {}
        for name in (
            "collision_grid", "near_collision_grid", "room_valid_mask", "door_grid",
            "unexplored_door_grid", "locked_door_grid", "enemy_grid", "self_grid",
            "pickup_grid", "item_grid", "hazard_grid", "trapdoor_grid",
        ):
            channels[name] = self._resize(f.get(name))

        projectiles = list(ps.get("projectiles") or [])
        now = self._projectile_map(projectiles, 0.0)
        f150 = self._projectile_map(projectiles, .15)
        f300 = self._projectile_map(projectiles, .30)
        f500 = self._projectile_map(projectiles, .50)

        # Dense trajectory corridor: near-future danger weighs more strongly.
        danger = np.maximum.reduce([
            now,
            .92*f150,
            .72*f300,
            .50*f500,
        ]).astype(np.float32)
        for t in np.arange(.05, .56, .05):
            corridor = self._projectile_map(projectiles, float(t))
            danger = np.maximum(danger, np.float32(max(.35, 1.0 - 1.15*t))*corridor)
        danger = np.clip(danger, 0, 1)

        channels["projectile_now_grid"] = now
        channels["projectile_future_150_grid"] = f150
        channels["projectile_future_300_grid"] = f300
        channels["projectile_future_500_grid"] = f500
        channels["projectile_danger_grid"] = danger

        weights = config.V37_TACTICAL_CHANNEL_WEIGHTS
        neural = np.zeros((self.rows, self.cols), np.float32)
        for name, weight in weights.items():
            a = channels.get(name)
            if a is not None:
                neural = np.maximum(neural, np.float32(weight)*a)
        neural = np.clip(neural, 0, 1)

        px = float(ps.get("player_x", f.get("enemy_player_x", .5) or .5))
        py = float(ps.get("player_y", f.get("enemy_player_y", .5) or .5))
        prx = float(ps.get("player_rx", f.get("player_hitbox_rx", .018) or .018))
        pry = float(ps.get("player_ry", f.get("player_hitbox_ry", .030) or .030))
        ttis = [self._tti(p, px, py, prx, pry) for p in projectiles]
        ttis = [t for t in ttis if t is not None]
        min_tti = min(ttis) if ttis else None
        mid = self.cols//2
        left = float(danger[:, :mid].max()) if danger.size else 0.0
        right = float(danger[:, mid:].max()) if danger.size else 0.0

        out = dict(
            rows=self.rows, cols=self.cols, channels=channels, neural_field=neural,
            projectile_count=len(projectiles), min_tti=min_tti,
            danger_left=left, danger_right=right,
            projectile_bridge_ok=bool(ps.get("has_state", False)),
            projectile_bridge_age=float(ps.get("age", float("inf"))),
        )
        self.last = out
        return out
