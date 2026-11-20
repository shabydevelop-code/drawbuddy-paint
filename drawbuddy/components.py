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

    def select_primary(self, name):
        """Select Color 1; preview is diagnostic, not proof of failed selection.

        The ribbon preview pixel can be black even when the swatch click worked.
        The caller verifies the *actual canvas fill* before continuing.
        """
        if name not in self.swatches:
            raise ValueError(f'Unknown color: {name}')
        self.primary.click(button='left')
        self.swatches[name].click(button='left')
        time.sleep(0.3)
        actual = self.primary.read_pixel()
        secondary = self.secondary.read_pixel()
        verified = close_to(actual, RGB[name])
        print(f'  COLOR requested={name} Color1={actual} Color2={secondary} '
              f'preview_match={verified} (preview is informational)')
        if not verified:
            print('  NOTE: Color 1 preview differs; continuing to verify actual canvas fill.')
        return verified


class PaintTools:
    def __init__(self, controller):
        self.rounded_rectangle = PaintComponent(controller, 'rounded_rectangle')
        self.ellipse = PaintComponent(controller, 'ellipse')
        self.bucket = PaintComponent(controller, 'bucket')
        self.pencil = PaintComponent(controller, 'pencil')


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
