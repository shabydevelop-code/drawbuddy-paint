"""Paint UI components and canvas operations, independent of lesson logic."""
import time
from .elements import ELEMENTS

RGB = {'orange': (255, 127, 39), 'brown': (185, 122, 87), 'black': (0, 0, 0), 'white': (255, 255, 255)}


def close_to(actual, expected, tolerance=30):
    return all(abs(int(a) - int(b)) <= tolerance for a, b in zip(actual, expected))


class PaintComponent:
    def __init__(self, controller, name):
        self.controller = controller
        self.element = ELEMENTS[name]

    def click(self, button='left'):
        self.controller.click_element(self.element, button)

    def read_pixel(self):
        return self.controller.pixel(self.element.x, self.element.y)


class PaintColors:
    def __init__(self, controller):
        self.controller = controller
        self.primary = PaintComponent(controller, 'color_1')
        self.secondary = PaintComponent(controller, 'color_2')
        self.swatches = {name: PaintComponent(controller, name) for name in RGB}
        self.current_primary = None

    def select_rgb(self, rgb):
        """Select a custom Paint Color 1 via the Windows Edit Colors dialog.

        Requires pywinauto; fail closed if dialog controls cannot be identified.
        """
        import time
        from pywinauto import Desktop
        from pywinauto.keyboard import send_keys
        rgb = tuple(int(v) for v in rgb)
        if len(rgb) != 3 or any(not 0 <= v <= 255 for v in rgb):
            raise ValueError(f"Invalid RGB color: {rgb}")
        if self.current_primary == rgb:
            return
        # Edit Colors is to the right of the built-in Paint palette.
        # UIA lookup avoids guessing an uncalibrated ribbon coordinate.
        self.controller.activate()
        window = Desktop(backend="uia").window(handle=self.controller.hwnd)
        buttons = window.descendants(control_type="Button")
        edit = next((b for b in buttons
                     if "edit colors" in (b.window_text() or "").lower()
                     or "ערוך צבעים" in (b.window_text() or "")), None)
        if edit is None:
            raise RuntimeError("Paint Edit Colors button not found via UI Automation; color drawing stopped")
        edit.click_input()
        time.sleep(0.35)
        # The Edit Colors dialog title differs across Paint versions and
        # locales. Prefer a visible top-level window owned by Paint, then
        # fall back to the localized title. Never pick an unrelated dialog.
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        GW_OWNER = 4
        dialog = None
        deadline = time.monotonic() + 6.0
        observed = []
        while time.monotonic() < deadline and dialog is None:
            desktop = Desktop(backend="uia")
            observed = []
            for candidate in desktop.windows():
                try:
                    handle = candidate.handle
                    title = candidate.window_text() or ""
                    owner = user32.GetWindow(handle, GW_OWNER)
                    # Dialog can be a child-owned popup or a top-level
                    # owned window, with a translated or empty title.
                    owned = owner == self.controller.hwnd
                    if not owned and owner:
                        owned = user32.GetAncestor(owner, 2) == self.controller.hwnd
                    observed.append((handle, title, owned))
                    if owned and candidate.is_visible() and handle != self.controller.hwnd:
                        dialog = candidate
                        break
                except Exception:
                    continue
            if dialog is None:
                time.sleep(0.2)
        if dialog is None:
            titles = [repr(title) for _, title, owned in observed if owned]
            raise RuntimeError(
                "Paint Edit Colors dialog opened but could not be identified "
                "as a visible Paint-owned window; owned windows: "
                + (", ".join(titles) if titles else "(none)")
                + ". Color drawing stopped.")
        # Common Windows Paint dialog exposes Red/Green/Blue edit boxes.
        edits = dialog.descendants(control_type="Edit")
        if len(edits) < 3:
            raise RuntimeError("Paint RGB controls not found; color drawing stopped")
        # Identify RGB fields by adjacent text labels rather than arbitrary indexes.
        labels = dialog.descendants(control_type="Text")
        targets = {}
        for label in labels:
            name = (label.window_text() or "").strip().lower().rstrip(":")
            if name in ("red", "green", "blue", "אדום", "ירוק", "כחול"):
                box = label.rectangle()
                neighbors = [e for e in edits if abs(e.rectangle().top - box.top) < 18
                             and e.rectangle().left > box.right]
                if neighbors:
                    key = {"red": "red", "אדום": "red", "green": "green",
                           "ירוק": "green", "blue": "blue", "כחול": "blue"}[name]
                    targets[key] = min(neighbors, key=lambda e: e.rectangle().left)
        if len(targets) != 3:
            raise RuntimeError("Cannot reliably map Paint RGB inputs; color drawing stopped")
        for key, value in zip(("red", "green", "blue"), rgb):
            targets[key].set_edit_text(str(value))
        ok = next((b for b in dialog.descendants(control_type="Button")
                   if (b.window_text() or "").strip().lower() in ("ok", "אישור")), None)
        if ok is None:
            raise RuntimeError("Paint RGB dialog OK button not found")
        ok.click_input()
        self.current_primary = rgb
        print(f"  COLOR custom RGB={rgb}")

    def select_primary(self, name):
        """Select the calibrated palette swatch without treating the ribbon
        Color 1 label pixel as the actual ink color.

        The former read at (689,70) consistently returned black even after
        selecting orange. It is not a validated color-preview coordinate.
        """
        if name not in self.swatches:
            raise ValueError(f'Unknown color: {name}')
        if name == self.current_primary:
            return True
        self.primary.click(button='left')
        self.swatches[name].click(button='left')
        time.sleep(0.25)
        self.current_primary = name
        print(f'  COLOR selected swatch={name} expected_rgb={RGB[name]} '
              '(ribbon preview pixel not used for verification)')
        return True


class PaintTools:
    def __init__(self, controller):
        self.rounded_rectangle = PaintComponent(controller, 'rounded_rectangle')
        self.ellipse = PaintComponent(controller, 'ellipse')
        self.bucket = PaintComponent(controller, 'bucket')
        self.pencil = PaintComponent(controller, 'pencil')
        self.current = None

    def select(self, name):
        """Choose a Paint tool only when it differs from the current selection."""
        if name not in ('pencil', 'bucket', 'ellipse', 'rounded_rectangle'):
            raise ValueError(f'Unsupported Paint tool: {name}')
        if self.current == name:
            print(f'  TOOL unchanged: {name}; skipping ribbon selection')
            return
        getattr(self, name).click()
        self.current = name


class PaintStrokeSize:
    """Paint ribbon Size selector; UI coordinates belong to the element registry."""

    def __init__(self, controller):
        self.menu = PaintComponent(controller, 'size_menu')
        self.options = {n: PaintComponent(controller, f'size_{n}') for n in (1, 2, 3, 4)}
        self.current = None

    def select(self, level):
        if level not in self.options:
            raise ValueError('Paint stroke size must be 1, 2, 3 or 4')
        if level != self.current:
            self.menu.click()
            self.options[level].click()
            self.current = level


class PaintCanvas:
    def __init__(self, controller):
        self.controller = controller

    def draw_shape(self, bounds):
        self.controller.drag(*bounds)

    def commit(self):
        self.controller.press('enter')

    def fill(self, x, y):
        self.controller.click(x, y, button='left')

    def stroke(self, points, duration=0.008):
        self.controller.stroke(points, duration=duration)

    def sample(self, x, y):
        return self.controller.pixel(x, y)
