"""Central registry of Paint ribbon controls, calibrated to the supplied screenshot."""
from dataclasses import dataclass

@dataclass(frozen=True)
class UIElement:
    name: str
    x: int
    y: int
    description: str = ''

ELEMENTS = {
    'color_1': UIElement('color_1', 689, 70, 'Primary color selector'),
    'color_2': UIElement('color_2', 731, 70, 'Secondary color selector'),
    'black': UIElement('black', 762, 62, 'Black palette swatch'),
    'orange': UIElement('orange', 851, 62, 'Orange palette swatch'),
    'brown': UIElement('brown', 807, 84, 'Brown palette swatch'),
    'rounded_rectangle': UIElement('rounded_rectangle', 460, 64, 'Rounded Rectangle'),
    'ellipse': UIElement('ellipse', 420, 64, 'Ellipse'),
    'pencil': UIElement('pencil', 244, 70, 'Pencil tool for continuous strokes'),
    'size_menu': UIElement('size_menu', 636, 91, 'Paint Size dropdown'),
    'size_1': UIElement('size_1', 674, 159, 'Size option 1'),
    'size_2': UIElement('size_2', 674, 179, 'Size option 2'),
    'size_3': UIElement('size_3', 674, 219, 'Size option 3'),
    'size_4': UIElement('size_4', 674, 259, 'Size option 4'),
    'bucket': UIElement('bucket', 268, 70, 'Fill with color'),
}
