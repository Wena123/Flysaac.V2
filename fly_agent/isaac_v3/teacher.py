import ctypes

import numpy as np

from . import config


try:
    import win32api
    import win32con
except Exception:  # Allows importing/testing non-Windows parts elsewhere.
    win32api = None
    win32con = None


_KEY_MAP = {
    "move_up": 0x57,      # W
    "move_down": 0x53,    # S
    "move_left": 0x41,    # A
    "move_right": 0x44,   # D
    "shoot_up": 0x26,     # Arrow Up
    "shoot_down": 0x28,   # Arrow Down
    "shoot_left": 0x25,   # Arrow Left
    "shoot_right": 0x27,  # Arrow Right
}

VK_F8 = 0x77


def _down(vk):
    if win32api is not None:
        return bool(win32api.GetAsyncKeyState(vk) & 0x8000)
    # Fallback for Windows without pywin32 import resolution.
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def read_teacher_labels():
    labels = np.zeros(len(config.ACTIONS), dtype=np.float32)
    held = set()
    for i, name in enumerate(config.ACTIONS):
        if _down(_KEY_MAP[name]):
            labels[i] = 1.0
            held.add(name)
    return labels, held


class ToggleKey:
    def __init__(self, vk=VK_F8):
        self.vk = int(vk)
        self._was_down = False

    def pressed(self):
        now = _down(self.vk)
        edge = now and not self._was_down
        self._was_down = now
        return edge
