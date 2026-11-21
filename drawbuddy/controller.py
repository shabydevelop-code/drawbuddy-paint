"""Window activation, coordinate conversion, and mouse/keyboard interactions."""
import ctypes
import math
import time
from ctypes import wintypes
import pyautogui as pg

pg.FAILSAFE = True
pg.PAUSE = 0.12
u = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        u.SetProcessDPIAware()
    except Exception:
        pass

class DrawingCancelled(Exception):
    """User requested immediate cancellation with the F8 key."""


def check_cancel():
    # F8 is polled globally; unlike Esc, it does not cancel Paint's active tool.
    if u.GetAsyncKeyState(0x77) & 0x8000:
        raise DrawingCancelled('Drawing stopped by user (F8)')


CALLBACK = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
u.EnumWindows.argtypes = [CALLBACK, wintypes.LPARAM]
u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
u.GetForegroundWindow.restype = wintypes.HWND


def window_title(hwnd):
    if not hwnd:
        return '(none)'
    n = u.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_paint():
    found = []

    @CALLBACK
    def visit(hwnd, _):
        if u.IsWindowVisible(hwnd):
            title = window_title(hwnd)
            if title == 'Paint' or title.endswith(' - Paint'):
                found.append(hwnd)
        return True

    u.EnumWindows(visit, 0)
    return found[0] if found else None


class PaintController:
    def __init__(self, drawing_speed=1.0):
        self.drawing_speed = drawing_speed
        self.hwnd = find_paint()
        if not self.hwnd:
            raise RuntimeError('Open classic Paint (Home tab) with a blank canvas.')
        u.ShowWindow(self.hwnd, 3)
        time.sleep(0.4)
        rect = wintypes.RECT()
        if not u.GetWindowRect(self.hwnd, ctypes.byref(rect)):
            raise RuntimeError('Cannot read Paint window rectangle.')
        self.left, self.top = max(0, rect.left), max(0, rect.top)
        if pg.size().width < 1000 or pg.size().height < 650:
            raise RuntimeError('Display must be at least 1000x650.')

    def point(self, x, y):
        return self.left + x, self.top + y

    def activate(self):
        if u.GetForegroundWindow() == self.hwnd:
            return
        u.ShowWindow(self.hwnd, 3)
        u.SetForegroundWindow(self.hwnd)
        time.sleep(0.35)
        if u.GetForegroundWindow() != self.hwnd:
            pg.click(*self.point(560, 12))
            time.sleep(0.35)
        if u.GetForegroundWindow() != self.hwnd:
            raise RuntimeError('Paint could not be activated; foreground: ' + window_title(u.GetForegroundWindow()))

    def refresh_display(self):
        """Force Paint to repaint before screenshots without changing the canvas."""
        self.activate()
        # Repaint the window and its children. Unlike File > New, this does
        # not modify the document or open a save confirmation.
        RDW_INVALIDATE = 0x0001
        RDW_UPDATENOW = 0x0100
        RDW_ALLCHILDREN = 0x0080
        u.RedrawWindow(self.hwnd, None, None,
                       RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN)
        time.sleep(0.25)
        # Hide/show via minimize/restore also refreshes the compositor and
        # Paint's canvas view if a direct invalidation was insufficient.
        u.ShowWindow(self.hwnd, 6)  # SW_MINIMIZE
        time.sleep(0.25)
        u.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        time.sleep(0.35)
        self.activate()
        u.RedrawWindow(self.hwnd, None, None,
                       RDW_INVALIDATE | RDW_UPDATENOW | RDW_ALLCHILDREN)
        time.sleep(0.4)
        print('Paint display refreshed before capture.')

    def click(self, x, y, button='left'):
        self.activate()
        absolute = self.point(x, y)
        print(f'  ACTION click button={button} relative=({x},{y}) absolute={absolute}')
        pg.click(*absolute, button=button)
        time.sleep(0.3)

    def click_element(self, element, button='left'):
        print(f'  ELEMENT {element.name}: {element.description}')
        self.click(element.x, element.y, button=button)

    def drag(self, x1, y1, x2, y2):
        self.activate()
        print(f'  ACTION drag relative=({x1},{y1})->({x2},{y2})')
        pg.moveTo(*self.point(x1, y1), duration=0.25)
        pg.dragTo(*self.point(x2, y2), duration=0.9, button='left')
        time.sleep(0.4)

    def validate_canvas_points(self, points):
        """Fail before mouseDown if a Pencil path reaches Paint's ribbon or screen edge."""
        screen = pg.size()
        for x, y in points:
            ax, ay = self.point(x, y)
            if y < 148 or x < 5 or ax >= screen.width - 5 or ay >= screen.height - 35:
                raise ValueError(
                    f'Pencil point ({x},{y}) is outside the safe Paint canvas; '
                    'drawing aborted before pressing the mouse button.'
                )

    @staticmethod
    def stroke_delay(previous, current, following, speed, base_duration):
        """Target event spacing based on local path geometry.

        Tiny facial features and sharp turns need more Paint sampling time.
        Long straight runs can move much faster. The minimum interval is
        deliberately independent of extreme speed multipliers.
        """
        if following is None:
            return max(base_duration / speed, 0.002)
        ax, ay = current[0] - previous[0], current[1] - previous[1]
        bx, by = following[0] - current[0], following[1] - current[1]
        alen, blen = math.hypot(ax, ay), math.hypot(bx, by)
        if alen < 0.01 or blen < 0.01:
            return max(base_duration / speed, 0.002)
        cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / (alen * blen)))
        turn = 1.0 - cosine
        if turn > 0.25:
            floor = 0.012
        elif turn > 0.055 or (alen <= 2 and blen <= 2):
            floor = 0.008
        else:
            floor = 0.005
        return max(base_duration / speed, floor)

    @staticmethod
    def wait_until(deadline):
        """Use a monotonic clock rather than inaccurate sub-ms sleep calls."""
        while True:
            check_cancel()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            if remaining > 0.003:
                time.sleep(remaining - 0.001)

    def stroke(self, points, duration=0.008):
        """Pace every mouse event; Esc cancels and releases the mouse."""
        check_cancel()
        if len(points) < 2:
            raise ValueError('Stroke requires at least two points')
        self.validate_canvas_points(points)
        self.activate()
        print(f'  ACTION stroke points={len(points)} speed={self.drawing_speed:g}x '
              'geometry_pacing=on')
        pg.moveTo(*self.point(*points[0]), duration=0)
        old_pause, old_min = pg.PAUSE, pg.MINIMUM_DURATION
        pg.PAUSE, pg.MINIMUM_DURATION = 0, 0
        try:
            pg.mouseDown(button='left')
            try:
                for i in range(1, len(points)):
                    check_cancel()
                    x, y = points[i]
                    pg.moveTo(*self.point(x, y), duration=0)
                    following = points[i + 1] if i + 1 < len(points) else None
                    delay = self.stroke_delay(
                        points[i - 1], points[i], following,
                        self.drawing_speed, duration)
                    self.wait_until(time.perf_counter() + delay)
            finally:
                pg.mouseUp(button='left')
        finally:
            pg.PAUSE, pg.MINIMUM_DURATION = old_pause, old_min
        time.sleep(0.01)

    def raster_runs(self, runs):
        """Draw isolated raster spans with paced Paint mouse events.

        Paint can lose rapid input transitions under load, leaving random
        connector lines. Keep a settled button-up state before repositioning,
        then use a nonzero-duration drag and a settled button-up afterward.
        """
        if not runs:
            return
        screen = pg.size()
        for x1, x2, y in runs:
            if x1 > x2:
                raise ValueError("Invalid raster run")
            for x in (x1, x2):
                ax, ay = self.point(x, y)
                if y < 148 or x < 5 or ax >= screen.width - 5 or ay >= screen.height - 35:
                    raise ValueError(f"Raster point ({x},{y}) outside safe Paint canvas")
        self.activate()
        old_pause, old_min = pg.PAUSE, pg.MINIMUM_DURATION
        pg.PAUSE, pg.MINIMUM_DURATION = 0, 0
        release_dwell = 0.035
        press_dwell = 0.012
        try:
            for i, (x1, x2, y) in enumerate(runs):
                check_cancel()
                pg.mouseUp(button='left')
                self.wait_until(time.perf_counter() + release_dwell)
                pg.moveTo(*self.point(x1, y), duration=0)
                self.wait_until(time.perf_counter() + press_dwell)
                pg.mouseDown(button='left')
                try:
                    self.wait_until(time.perf_counter() + press_dwell)
                    if x1 != x2:
                        # The old zero-duration jump could be interpreted as
                        # a spurious connector when Paint was under load.
                        drag_seconds = min(0.12, max(0.035, (x2 - x1) * 0.0003))
                        pg.moveTo(*self.point(x2, y), duration=drag_seconds)
                    self.wait_until(time.perf_counter() + press_dwell)
                finally:
                    pg.mouseUp(button='left')
                    self.wait_until(time.perf_counter() + release_dwell)
                if (i + 1) % 500 == 0:
                    print(f"  raster progress: {i + 1}/{len(runs)} spans")
        finally:
            pg.mouseUp(button='left')
            pg.PAUSE, pg.MINIMUM_DURATION = old_pause, old_min

    def press(self, key):
        self.activate()
        print(f'  ACTION key={key}')
        if '+' in key:
            pg.hotkey(*(part.strip() for part in key.split('+')))
        else:
            pg.press(key)
        time.sleep(0.35)

    def pixel(self, x, y):
        value = pg.screenshot().getpixel(self.point(x, y))[:3]
        print(f'  OBSERVE pixel relative=({x},{y}) rgb={value}')
        return value

    def screenshot(self, path):
        pg.screenshot().save(path)
        print('Screenshot:', path)
