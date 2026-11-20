"""Window activation, coordinate conversion, and mouse/keyboard interactions."""
import ctypes
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

    def stroke(self, points, duration=0.008):
        """Continuous Paint Pencil stroke. Speed changes stroke timing only, not UI waits."""
        if len(points) < 2:
            raise ValueError('Stroke requires at least two points')
        self.validate_canvas_points(points)
        self.activate()
        step_duration = max(0.006, duration / self.drawing_speed)
        print(f'  ACTION stroke points={len(points)} speed={self.drawing_speed:g}x '
              f'seconds_per_segment={step_duration:.5f}')
        pg.moveTo(*self.point(*points[0]), duration=0.25)
        # PyAutoGUI's global PAUSE and MINIMUM_DURATION otherwise override
        # per-segment timing; change them only inside this stroke and restore.
        old_pause, old_min = pg.PAUSE, pg.MINIMUM_DURATION
        pg.PAUSE, pg.MINIMUM_DURATION = 0, 0
        try:
            pg.mouseDown(button='left')
            try:
                for x, y in points[1:]:
                    # PyAutoGUI may divide by zero for tiny positive durations.
                    # Move immediately while holding the button; pace separately.
                    pg.moveTo(*self.point(x, y), duration=0)
                    if step_duration > 0:
                        time.sleep(step_duration)
            finally:
                pg.mouseUp(button='left')
        finally:
            pg.PAUSE, pg.MINIMUM_DURATION = old_pause, old_min
        time.sleep(0.4)

    def press(self, key):
        self.activate()
        print(f'  ACTION key={key}')
        pg.press(key)
        time.sleep(0.35)

    def pixel(self, x, y):
        value = pg.screenshot().getpixel(self.point(x, y))[:3]
        print(f'  OBSERVE pixel relative=({x},{y}) rgb={value}')
        return value

    def screenshot(self, path):
        pg.screenshot().save(path)
        print('Screenshot:', path)
