"""Paint UI components and canvas operations, independent of lesson logic."""
import time
from .elements import ELEMENTS

RGB = {'orange': (255, 127, 39), 'brown': (185, 122, 87), 'black': (0, 0, 0)}


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

    def select_primary(self, name):
        """Synchronize the real Paint Color 1 with the requested palette swatch.

        Never trust the cached color when Paint may have changed independently.
        """
        if name not in self.swatches:
            raise ValueError(f'Unknown color: {name}')
        expected = RGB[name]
        actual = self.primary.read_pixel()
        if close_to(actual, expected, tolerance=12):
            self.current_primary = name
            print(f'  COLOR verified existing: {name} RGB={actual}')
            return True
        for attempt in range(1, 4):
            self.primary.click(button='left')
            self.swatches[name].click(button='left')
            time.sleep(0.25)
            actual = self.primary.read_pixel()
            print(f'  COLOR select attempt={attempt} requested={name} Color1={actual}')
            if close_to(actual, expected, tolerance=12):
                self.current_primary = name
                return True
        self.current_primary = None
        raise RuntimeError(
            f'Paint Color 1 mismatch after 3 attempts: requested={name} '
            f'expected={expected}, actual={actual}')


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
