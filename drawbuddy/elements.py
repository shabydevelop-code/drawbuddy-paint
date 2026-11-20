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
    'bucket': UIElement('bucket', 268, 70, 'Fill with color'),
}
