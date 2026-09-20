
import time
from collections import deque

import cv2
import numpy as np

from . import config


class NeuronMonitor:
    """Larger, richer anatomical MaleCNS activity monitor.

    Uses FlyBrain.positions from brain.npz (real soma / soma-tract positions),
    so the main view is an anatomical point-cloud projection rather than a fake
    neuron index grid.
    """

    AXES = {
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }

    def __init__(
        self,
        title="FlyIsaac Anatomical Neural Monitor",
        width=1800,
        height=1050,
        history=140,
        raster_bins=144,
        update_hz=10.0,
        enabled=True,
        brain=None,
        dn_cells=None,
        projection="xz",
    ):
        self.title = str(title)
        self.width = int(width)
        self.height = int(height)
        self.history = max(10, int(history))
        self.raster_bins = max(8, int(raster_bins))
        self.update_interval = 1.0 / max(1.0, float(update_hz))
        self.enabled = bool(enabled)
        self._opened = False
        self._last_update = 0.0
        self._raster = deque(maxlen=self.history)

        projection = str(projection).lower()
        if projection not in self.AXES:
            projection = "xz"
        self.projection = projection

        self.brain = None
        self.positions = None
        self.cell_type = None
        self.superclass = None
        self.dn_cells = None
        self._valid = None
        self._xy_norm = None
        self._spike_glow = None
        self._anatomy_cache = {}
        self._activity_groups = {}

        if brain is not None:
            self.attach_brain(brain, dn_cells=dn_cells)

    def attach_brain(self, brain, dn_cells=None):
        self.brain = brain
        pos = getattr(brain, "positions", None)
        if pos is None:
            self.positions = None
            return

        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim != 2 or pos.shape[1] < 3:
            self.positions = None
            return

        self.positions = pos[:, :3].copy()
        self.cell_type = np.asarray(getattr(brain, "cell_type", np.array([""] * len(pos))), dtype=str)
        self.superclass = np.asarray(getattr(brain, "superclass", np.array([""] * len(pos))), dtype=str)
        upper = np.char.upper(self.cell_type.astype(str))
        self._activity_groups = {
            "LC10": np.flatnonzero(np.char.startswith(upper, "LC10")),
            "LPLC1": np.flatnonzero(np.char.startswith(upper, "LPLC1")),
            "LPLC2": np.flatnonzero(np.char.startswith(upper, "LPLC2")),
            "LC4": np.flatnonzero(np.char.startswith(upper, "LC4")),
            "PAM": np.flatnonzero(np.char.startswith(upper, "PAM")),
            "PPL": np.flatnonzero(np.char.startswith(upper, "PPL")),
        }

        if dn_cells is None:
            try:
                dn_cells = brain.cells(["descending_neuron"])
            except Exception:
                dn_cells = np.empty(0, dtype=np.int64)
        self.dn_cells = np.asarray(dn_cells, dtype=np.int64).reshape(-1)

        self._prepare_projection()
        self._spike_glow = np.zeros(len(self.positions), dtype=np.float32)
        self._anatomy_cache.clear()

    def _prepare_projection(self):
        if self.positions is None:
            return
        a, b = self.AXES[self.projection]
        p = self.positions[:, [a, b]]
        valid = np.isfinite(p).all(axis=1)
        self._valid = valid
        self._xy_norm = np.full((len(p), 2), np.nan, dtype=np.float32)
        if not np.any(valid):
            return
        q = p[valid]
        lo = np.nanpercentile(q, 0.5, axis=0)
        hi = np.nanpercentile(q, 99.5, axis=0)
        span = np.maximum(hi - lo, 1.0)
        n = np.clip((p - lo) / span, 0.0, 1.0)
        n[:, 1] = 1.0 - n[:, 1]
        self._xy_norm[valid] = n[valid]

    def set_projection(self, projection):
        projection = str(projection).lower()
        if projection not in self.AXES or projection == self.projection:
            return
        self.projection = projection
        self._prepare_projection()
        self._anatomy_cache.clear()

    def open(self):
        if not self.enabled or self._opened:
            return
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.width, self.height)
        self._opened = True
        blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._label(blank, "FlyIsaac anatomical monitor - waiting for MaleCNS activity", 30, 52, 0.90, 2)
        cv2.imshow(self.title, blank)
        cv2.waitKey(1)

    @staticmethod
    def _label(canvas, text, x, y, scale=0.48, thickness=1, color=(235, 235, 235)):
        cv2.putText(canvas, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)

    @staticmethod
    def _panel(canvas, x, y, w, h, title=None, title_scale=0.50):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (52, 56, 66), 1)
        if title:
            cv2.rectangle(canvas, (x, y - 24), (x + min(w, 260), y - 2), (24, 26, 34), -1)
            NeuronMonitor._label(canvas, title, x + 8, y - 8, title_scale, 1, (220, 225, 235))

    @staticmethod
    def _retina_panel(field, width, height):
        f = np.asarray(field, dtype=np.float32)
        u8 = np.asarray(np.clip(f * 255.0, 0, 255), dtype=np.uint8)
        img = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
        img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (38, 40, 46), 1)
        return img

    def _panel_coords(self, indices, x, y, w, h, projection=None):
        if self.positions is None:
            return np.empty((0, 2), np.int32), np.empty(0, np.int64)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        idx = idx[(idx >= 0) & (idx < len(self.positions))]
        if idx.size == 0:
            return np.empty((0, 2), np.int32), idx
        proj = self.projection if projection is None else projection
        axes = self.AXES.get(proj, self.AXES[self.projection])
        p = self.positions[idx][:, axes]
        ok = np.isfinite(p).all(axis=1)
        idx = idx[ok]
        p = p[ok]
        if idx.size == 0:
            return np.empty((0, 2), np.int32), idx
        ref = self.positions[:, axes]
        lo = np.nanpercentile(ref, 0.5, axis=0)
        hi = np.nanpercentile(ref, 99.5, axis=0)
        span = np.maximum(hi - lo, 1.0)
        uv = np.clip((p - lo) / span, 0.0, 1.0)
        uv[:, 1] = 1.0 - uv[:, 1]
        px = x + 8 + uv[:, 0] * max(1, w - 16)
        py = y + 8 + uv[:, 1] * max(1, h - 16)
        return np.column_stack([px, py]).astype(np.int32), idx

    def _anatomy_background(self, w, h, projection=None):
        proj = self.projection if projection is None else projection
        key = (int(w), int(h), proj)
        cached = self._anatomy_cache.get(key)
        if cached is not None:
            return cached.copy()
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        vign = 1.0 - 0.55 * ((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
        bg[:, :, 0] = np.asarray(18 + 10 * yy + 8 * vign, dtype=np.uint8)
        bg[:, :, 1] = np.asarray(14 + 8 * yy + 6 * vign, dtype=np.uint8)
        bg[:, :, 2] = np.asarray(20 + 16 * yy + 10 * vign, dtype=np.uint8)
        if self.positions is None:
            self._label(bg, "No MaleCNS anatomical positions available", 20, 40, 0.58, 1)
            self._anatomy_cache[key] = bg
            return bg.copy()
        pts_all, _ = self._panel_coords(np.arange(len(self.positions)), 0, 0, w, h, projection=proj)
        if pts_all.size == 0:
            self._label(bg, "No valid anatomical positions", 20, 40, 0.58, 1)
            self._anatomy_cache[key] = bg
            return bg.copy()
        hist = np.zeros((h, w), dtype=np.float32)
        sel = pts_all.astype(np.float32)[::max(1, len(pts_all) // 45000)]
        for px, py in sel.astype(np.int32):
            if 0 <= px < w and 0 <= py < h:
                hist[py, px] += 1.0
        hist = cv2.GaussianBlur(hist, (0, 0), 8.0)
        if hist.max() > 0:
            hist /= hist.max()
        cloud = cv2.applyColorMap(np.asarray(np.clip(hist * 255.0, 0, 255), dtype=np.uint8), cv2.COLORMAP_BONE)
        bg = cv2.addWeighted(bg, 0.82, cloud, 0.28, 0.0)
        for lvl, col in [(0.15, (60, 72, 96)), (0.32, (76, 88, 116)), (0.55, (108, 126, 164))]:
            mask = np.asarray(hist >= lvl, dtype=np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bg, contours, -1, col, 1, cv2.LINE_AA)
        stride = max(1, len(pts_all) // 30000)
        for px, py in pts_all[::stride]:
            cv2.circle(bg, (int(px), int(py)), 1, (78, 82, 92), -1, cv2.LINE_AA)
        cv2.rectangle(bg, (0, 0), (w - 1, h - 1), (52, 56, 66), 1)
        self._anatomy_cache[key] = bg
        return bg.copy()

    def _compress_dn(self, dn):
        dn = np.asarray(dn, dtype=np.float32).reshape(-1)
        edges = np.linspace(0, len(dn), self.raster_bins + 1, dtype=int)
        out = np.zeros(self.raster_bins, dtype=np.float32)
        for i in range(self.raster_bins):
            chunk = dn[edges[i]:edges[i + 1]]
            out[i] = float(chunk.max()) if chunk.size else 0.0
        return out

    @staticmethod
    def _bar(canvas, x, y, w, label, value, chosen=False):
        value = float(np.clip(value, 0.0, 1.0))
        cv2.rectangle(canvas, (x, y), (x + w, y + 14), (55, 58, 65), 1)
        fill = int(round((w - 2) * value))
        if fill > 0:
            color = (255, 245, 196) if chosen else (110, 140, 210)
            cv2.rectangle(canvas, (x + 1, y + 1), (x + 1 + fill, y + 13), color, -1)
        NeuronMonitor._label(canvas, f"{label:<11s} {value:4.2f}", x + w + 8, y + 12, 0.40, 1, (248, 248, 248) if chosen else (172, 178, 188))

    def _update_spike_glow(self, fired):
        if self._spike_glow is None:
            return np.empty(0, dtype=np.int64)
        self._spike_glow *= np.float32(0.78)
        if fired is None:
            return np.empty(0, dtype=np.int64)
        arr = fired.get() if hasattr(fired, "get") else fired
        arr = np.asarray(arr)
        if arr.dtype == np.bool_:
            arr = np.flatnonzero(arr)
        arr = arr.astype(np.int64, copy=False).reshape(-1)
        if self.positions is not None and arr.size:
            arr = np.mod(arr, len(self.positions))
            arr = arr[(arr >= 0) & (arr < len(self.positions))]
            self._spike_glow[arr] = 1.0
        return arr

    def _draw_anatomy_activity(self, panel, fired, dn):
        h, w = panel.shape[:2]
        fired_idx = self._update_spike_glow(fired)
        if self._spike_glow is not None:
            active = np.flatnonzero(self._spike_glow >= 0.14)
            if active.size:
                if active.size > 9000:
                    order = np.argpartition(self._spike_glow[active], -9000)[-9000:]
                    active = active[order]
                pts, mapped = self._panel_coords(active, 0, 0, w, h)
                vals = self._spike_glow[mapped] if mapped.size else np.empty(0)
                glow = np.zeros_like(panel)
                dots = np.zeros_like(panel)
                for (px, py), v in zip(pts, vals):
                    radius = 3 if v < 0.45 else (5 if v < 0.8 else 7)
                    cv2.circle(glow, (int(px), int(py)), radius, (22, 48, 110), -1, cv2.LINE_AA)
                    cv2.circle(dots, (int(px), int(py)), 1 + int(v > 0.55), (255, 245, 235), -1, cv2.LINE_AA)
                glow = cv2.GaussianBlur(glow, (0, 0), 5.0)
                panel[:] = cv2.addWeighted(panel, 1.0, glow, 0.70, 0.0)
                panel[:] = cv2.addWeighted(panel, 1.0, dots, 1.0, 0.0)
        dn_highlighted = 0
        if self.dn_cells is not None and dn is not None:
            dn = np.asarray(dn, dtype=np.float32).reshape(-1)
            n = min(len(dn), len(self.dn_cells))
            local = np.flatnonzero(dn[:n] >= 0.12)
            dn_highlighted = int(local.size)
            if local.size:
                if local.size > 1200:
                    top = np.argpartition(dn[local], -1200)[-1200:]
                    local = local[top]
                full_idx = self.dn_cells[local]
                pts, _ = self._panel_coords(full_idx, 0, 0, w, h)
                for (px, py), v in zip(pts, dn[local]):
                    v = float(np.clip(v, 0.0, 1.0))
                    cv2.circle(panel, (int(px), int(py)), 2 + int(v > 0.55), (55, 240, 255), 1, cv2.LINE_AA)
                    if v > 0.45:
                        cv2.circle(panel, (int(px), int(py)), 1, (255, 255, 255), -1, cv2.LINE_AA)
        return int(len(fired_idx)), dn_highlighted

    def _projection_thumb(self, projection, fired=None, dn=None, w=138, h=112):
        thumb = self._anatomy_background(w, h, projection=projection)
        if fired is not None:
            arr = fired.get() if hasattr(fired, 'get') else fired
            arr = np.asarray(arr)
            if arr.dtype == np.bool_:
                arr = np.flatnonzero(arr)
            if arr.size:
                pts, _ = self._panel_coords(arr, 0, 0, w, h, projection=projection)
                for px, py in pts[:3000]:
                    cv2.circle(thumb, (int(px), int(py)), 1, (255, 245, 232), -1, cv2.LINE_AA)
        if self.dn_cells is not None and dn is not None:
            dn = np.asarray(dn, dtype=np.float32).reshape(-1)
            n = min(len(dn), len(self.dn_cells))
            local = np.flatnonzero(dn[:n] >= 0.22)
            if local.size:
                local = local[:400]
                pts, _ = self._panel_coords(self.dn_cells[local], 0, 0, w, h, projection=projection)
                for px, py in pts:
                    cv2.circle(thumb, (int(px), int(py)), 1, (55, 240, 255), -1, cv2.LINE_AA)
        return thumb


    def _draw_special_groups(self, panel, fired):
        if fired is None or not self._activity_groups:
            return {}
        arr = fired.get() if hasattr(fired, "get") else fired
        arr = np.asarray(arr)
        if arr.dtype == np.bool_:
            arr = np.flatnonzero(arr)
        arr = arr.astype(np.int64, copy=False).reshape(-1)
        if arr.size == 0:
            return {}
        colors = {
            "LC10": (90, 235, 120),
            "LPLC1": (80, 190, 255),
            "LPLC2": (40, 145, 255),
            "LC4": (70, 80, 255),
            "PAM": (80, 245, 245),
            "PPL": (220, 90, 220),
        }
        counts = {}
        h, w = panel.shape[:2]
        for name, group in self._activity_groups.items():
            if group is None or len(group) == 0:
                counts[name] = 0
                continue
            hit = arr[np.isin(arr, group, assume_unique=False)]
            counts[name] = int(len(hit))
            if len(hit) == 0:
                continue
            pts, _ = self._panel_coords(hit, 0, 0, w, h)
            color = colors[name]
            for px, py in pts[:1400]:
                cv2.circle(panel, (int(px), int(py)), 4, color, 1, cv2.LINE_AA)
        return counts
    def render(self, dn, fired=None, retina=None, actions=None, mode="?", run=1, deaths=0, step=0, move_info=None, shoot_info=None, move_probs=None, shoot_probs=None, grid_field=None, dopamine=None, tactical=None, perf=None):
        dn = np.asarray(dn, dtype=np.float32).reshape(-1)
        compressed = self._compress_dn(dn)
        self._raster.append(compressed)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :, 0] = 10
        canvas[:, :, 1] = 10
        canvas[:, :, 2] = 14
        self._label(canvas, "FlyIsaac V3.7 | MaleCNS Anatomical Activity 2.0", 22, 34, 0.92, 2)
        self._label(canvas, "real soma / soma-tract positions | bright glow = recent spikes | cyan = descending neuron activity", 22, 60, 0.47, 1, (172, 178, 188))
        ax, ay, aw, ah = 20, 86, 1120, 740
        right_x = 1160
        right_w = self.width - right_x - 20
        self._panel(canvas, ax, ay, aw, ah, title=f"MAIN VIEW ({self.projection.upper()})", title_scale=0.56)
        anatomy = self._anatomy_background(aw, ah)
        fired_count, dn_high = self._draw_anatomy_activity(anatomy, fired, dn)
        group_counts = self._draw_special_groups(anatomy, fired)
        canvas[ay:ay + ah, ax:ax + aw] = anatomy
        self._label(canvas, f"fired this step: {fired_count}", ax + 12, ay + 24, 0.46, 1, (232, 238, 245))
        self._label(canvas, f"DN highlighted: {dn_high}", ax + 205, ay + 24, 0.46, 1, (232, 238, 245))
        if group_counts:
            legend = " | ".join(f"{k}:{group_counts.get(k,0)}" for k in ("LC10","LPLC1","LPLC2","LC4","PAM","PPL"))
            self._label(canvas, legend, ax + 390, ay + 24, 0.36, 1, (210, 215, 225))
        ph = 132
        if retina is not None:
            self._panel(canvas, right_x, 86, right_w, ph, title="RETINA / TARGET")
            self._panel(canvas, right_x, 252, right_w, ph, title="RETINA / MOTION")
            canvas[86:86 + ph, right_x:right_x + right_w] = self._retina_panel(retina.target, right_w, ph)
            canvas[252:252 + ph, right_x:right_x + right_w] = self._retina_panel(retina.motion, right_w, ph)
        if tactical is not None and tactical.get("neural_field") is not None:
            self._panel(canvas, right_x, 418, right_w, 100, title="TACTICAL GRID 7x14 -> MALECNS")
            canvas[418:518, right_x:right_x + right_w] = self._retina_panel(tactical["neural_field"], right_w, 100)
            ch = tactical.get("channels", {})
            names = [("projectile_now_grid", "NOW"), ("projectile_future_150_grid", "+150"), ("projectile_future_300_grid", "+300"), ("projectile_future_500_grid", "+500")]
            pw = max(1, (right_w - 12)//4)
            py0 = 548
            for ii, (key, label) in enumerate(names):
                a = ch.get(key)
                if a is None:
                    continue
                xx = right_x + ii*(pw+3)
                canvas[py0:py0+72, xx:xx+pw] = self._retina_panel(a, pw, 72)
                self._label(canvas, label, xx+3, py0+15, 0.34, 1, (245,245,245))
        elif grid_field is not None:
            self._panel(canvas, right_x, 418, right_w, 100, title="GRID -> MALECNS LC10")
            gf = np.asarray(grid_field, dtype=np.float32)
            if gf.ndim == 2 and gf.size:
                canvas[418:518, right_x:right_x + right_w] = self._retina_panel(gf, right_w, 100)
        self._panel(canvas, right_x, 650, right_w, 176, title="STATUS / PERFORMANCE")
        active_dn = int(np.count_nonzero(dn >= 0.15))
        peak = float(dn.max()) if dn.size else 0.0
        mean = float(dn.mean()) if dn.size else 0.0
        actions_text = ", ".join(sorted(actions or [])) or "NONE"
        status = [f"mode: {str(mode).upper()}", f"run {run} | deaths {deaths} | step {step}", f"DN active: {active_dn}/{len(dn)}", f"DN mean/peak: {mean:.3f} / {peak:.3f}", f"action: {actions_text}"]
        if retina is not None:
            status.append(f"loom L/R: {retina.looming_left:.2f} / {retina.looming_right:.2f}")
        if dopamine is not None:
            status.append(f"dopamine: {float(dopamine):+.3f}")
        if tactical is not None:
            tti = tactical.get("min_tti")
            status.append(f"hostile projectiles: {tactical.get('projectile_count',0)} | min TTI: {'-' if tti is None else f'{tti*1000:.0f} ms'}")
            status.append(f"projectile bridge: {'OK' if tactical.get('projectile_bridge_ok') else 'WAITING'}")
        if perf:
            status.append(f"capture {perf.get('capture_fps',0):.1f} FPS | brain {perf.get('brain_fps',0):.1f} FPS | drop {perf.get('dropped_frames',0)}")
        sy = 675
        for i, line in enumerate(status[:8]):
            self._label(canvas, line, right_x + 10, sy + i * 22, 0.44, 1)
        thumb_w, thumb_h = 112, 78
        thumbs = ['xy', 'xz', 'yz']
        tx = right_x + right_w - (thumb_w + 10) * 3 + 10
        ty = 742
        for pidx, proj in enumerate(thumbs):
            thumb = self._projection_thumb(proj, fired=fired, dn=dn, w=thumb_w, h=thumb_h)
            x = tx + pidx * (thumb_w + 6)
            canvas[ty:ty + thumb_h, x:x + thumb_w] = thumb
            cv2.rectangle(canvas, (x, ty), (x + thumb_w, ty + thumb_h), (64, 68, 78), 1)
            self._label(canvas, proj.upper(), x + 6, ty + 16, 0.42, 1, (235, 235, 240))
            if proj == self.projection:
                cv2.rectangle(canvas, (x - 1, ty - 1), (x + thumb_w + 1, ty + thumb_h + 1), (70, 190, 255), 1)
        raster_x, raster_y, raster_w, raster_h = 20, 858, 560, 170
        self._panel(canvas, raster_x, raster_y, raster_w, raster_h, title="DESCENDING ACTIVITY OVER TIME")
        raster = np.zeros((self.history, self.raster_bins), dtype=np.float32)
        if self._raster:
            rows = np.stack(list(self._raster))
            raster[-len(rows):] = rows
        raster_img = cv2.applyColorMap(np.asarray(np.clip(raster * 255.0, 0, 255), dtype=np.uint8), cv2.COLORMAP_TURBO)
        canvas[raster_y:raster_y + raster_h, raster_x:raster_x + raster_w] = cv2.resize(raster_img, (raster_w, raster_h), interpolation=cv2.INTER_NEAREST)
        bx = 606
        by = 858
        self._panel(canvas, bx, by, 540, 170, title="MOVEMENT DECODER")
        move_probs = np.asarray(move_probs, dtype=np.float32).reshape(-1) if move_probs is not None else np.zeros(len(config.MOVEMENT_CLASSES), np.float32)
        chosen_move = int(np.argmax(move_probs)) if move_probs.size else -1
        for i, name in enumerate(config.MOVEMENT_CLASSES):
            col = 0 if i < 5 else 1
            row = i if i < 5 else i - 5
            x = bx + 12 + col * 245
            y = by + 18 + row * 24
            self._bar(canvas, x, y, 102, name, float(move_probs[i]) if i < len(move_probs) else 0.0, chosen=(i == chosen_move))
        sx = 1170
        sy2 = 858
        self._panel(canvas, sx, sy2, self.width - sx - 20, 170, title="SHOOT DECODER")
        shoot_probs = np.asarray(shoot_probs, dtype=np.float32).reshape(-1) if shoot_probs is not None else np.zeros(len(config.SHOOT_CLASSES), np.float32)
        chosen_shoot = int(np.argmax(shoot_probs)) if shoot_probs.size else -1
        for i, name in enumerate(config.SHOOT_CLASSES):
            y = sy2 + 18 + i * 24
            self._bar(canvas, sx + 12, y, 96, name, float(shoot_probs[i]) if i < len(shoot_probs) else 0.0, chosen=(i == chosen_shoot))
        return canvas

    def update(self, **kwargs):
        if not self.enabled:
            return
        if not self._opened:
            self.open()
        now = time.perf_counter()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now
        cv2.imshow(self.title, self.render(**kwargs))
        cv2.waitKey(1)

    def reset(self):
        self._raster.clear()
        if self._spike_glow is not None:
            self._spike_glow.fill(0.0)

    def close(self):
        if not self._opened:
            return
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except Exception:
            pass
        self._opened = False
