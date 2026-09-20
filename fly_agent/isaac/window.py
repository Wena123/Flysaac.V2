# pyright: reportMissingModuleSource=false

import time

import win32con
import win32gui


class IsaacWindow:

    def __init__(
        self,
        title_contains="Binding of Isaac",
    ):

        self.title_contains = (
            title_contains
        )

        self.hwnd = (
            self._find_window()
        )

        if self.hwnd is None:

            raise RuntimeError(
                "Could not find Binding of Isaac window."
            )

        print(
            "Isaac window:",
            win32gui.GetWindowText(
                self.hwnd
            )
        )

    # =========================================================
    # FIND WINDOW
    # =========================================================

    def _find_window(
        self,
    ):

        matches = []

        def callback(
            hwnd,
            _,
        ):

            if not win32gui.IsWindowVisible(
                hwnd
            ):
                return

            title = (
                win32gui.GetWindowText(
                    hwnd
                )
            )

            if (
                self.title_contains.lower()
                in title.lower()
            ):

                matches.append(
                    hwnd
                )

        win32gui.EnumWindows(
            callback,
            None,
        )

        if not matches:
            return None

        return matches[0]

    # =========================================================
    # CHECK WINDOW
    # =========================================================

    def is_minimized(
        self,
    ):

        return bool(
            win32gui.IsIconic(
                self.hwnd
            )
        )

    # =========================================================
    # RESTORE
    # =========================================================

    def restore(
        self,
    ):

        if self.is_minimized():

            print(
                "Isaac is minimized - restoring..."
            )

            win32gui.ShowWindow(
                self.hwnd,
                win32con.SW_RESTORE,
            )

            time.sleep(
                0.5
            )

        else:

            win32gui.ShowWindow(
                self.hwnd,
                win32con.SW_SHOW,
            )

    # =========================================================
    # FOCUS
    # =========================================================

    def focus(
        self,
    ):

        self.restore()

        try:

            win32gui.BringWindowToTop(
                self.hwnd
            )

        except Exception:
            pass

        try:

            win32gui.SetForegroundWindow(
                self.hwnd
            )

        except Exception:
            # Windows sometimes refuses this call
            # depending on foreground-lock rules.
            pass

        time.sleep(
            0.2
        )

    # =========================================================
    # CLIENT RECT
    # =========================================================

    def client_rect(
        self,
    ):

        # A minimized Windows window can report
        # (-32000, -32000) and a zero-sized client area.
        # Restore before asking for capture coordinates.

        self.restore()

        left, top, right, bottom = (
            win32gui.GetClientRect(
                self.hwnd
            )
        )

        width = (
            right - left
        )

        height = (
            bottom - top
        )

        screen_left, screen_top = (
            win32gui.ClientToScreen(
                self.hwnd,
                (0, 0),
            )
        )

        if (
            width <= 0
            or height <= 0
        ):

            raise RuntimeError(
                "Isaac client area has invalid size: "
                f"{width}x{height}"
            )

        return {
            "left": int(
                screen_left
            ),

            "top": int(
                screen_top
            ),

            "width": int(
                width
            ),

            "height": int(
                height
            ),
        }