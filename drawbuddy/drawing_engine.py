"""Execute declarative drawing lessons through the existing Paint OOP components."""
import json
import math
from pathlib import Path
from .controller import PaintController
from .components import PaintColors, PaintTools, PaintCanvas, RGB


class DrawingEngine:
    SUPPORTED_SHAPES = frozenset(('ellipse', 'rounded_rectangle'))
    SUPPORTED_TOOLS = SUPPORTED_SHAPES | {'stroke', 'closed_path'}

    def __init__(self, controller=None, speed=1.0):
        self.paint = controller if controller is not None else PaintController(drawing_speed=speed)
        self.colors = PaintColors(self.paint)
        self.tools = PaintTools(self.paint)
        self.canvas = PaintCanvas(self.paint)

    @classmethod
    def load(cls, path):
        """Load and validate a drawing definition before any Paint interaction."""
        path = Path(path)
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not isinstance(data, dict) or not isinstance(data.get('steps'), list) or not data['steps']:
            raise ValueError(f'{path}: expected a nonempty steps array')
        for index, step in enumerate(data['steps'], 1):
            if not isinstance(step, dict):
                raise ValueError(f'Step {index} must be an object')
            if not isinstance(step.get('name'), str) or not step['name']:
                raise ValueError(f'Step {index}: missing name')
            if step.get('tool') not in cls.SUPPORTED_TOOLS:
                raise ValueError(f'Step {index}: unsupported tool {step.get("tool")!r}')
            if step['tool'] in ('stroke', 'closed_path'):
                if step.get('color') not in RGB:
                    raise ValueError(f'Step {index}: unsupported stroke color')
                points = step.get('points')
                if (not isinstance(points, list) or len(points) < 2 or
                    any(not isinstance(pt, list) or len(pt) != 2 or
                        any(type(v) is not int or v < 0 for v in pt) for pt in points)):
                    raise ValueError(f'Step {index}: stroke needs at least two [x,y] points')
                if type(step.get('smooth', False)) is not bool:
                    raise ValueError(f'Step {index}: smooth must be boolean')
                if type(step.get('width', 1)) is not int or not 1 <= step.get('width', 1) <= 3:
                    raise ValueError(f'Step {index}: width must be 1, 2 or 3')
                if step['tool'] == 'closed_path':
                    if step.get('fill') not in RGB:
                        raise ValueError(f'Step {index}: unsupported fill color')
                    fill_point = step.get('fill_point')
                    if (not isinstance(fill_point, list) or len(fill_point) != 2 or
                        any(type(v) is not int or v < 0 for v in fill_point)):
                        raise ValueError(f'Step {index}: closed_path requires fill_point [x,y]')
                    if len(set(map(tuple, points))) < 3:
                        raise ValueError(f'Step {index}: closed_path needs at least three distinct points')
                    if 'outside_point' in step and (not isinstance(step['outside_point'], list) or len(step['outside_point']) != 2 or any(type(v) is not int or v < 0 for v in step['outside_point'])):
                        raise ValueError(f'Step {index}: outside_point must be [x,y]')
                    if not cls.point_in_polygon(fill_point, points):
                        raise ValueError(f'Step {index}: fill_point must be inside closed path')
                continue
            for field in ('outline', 'fill'):
                if step.get(field) not in RGB:
                    raise ValueError(f'Step {index}: unsupported {field} color {step.get(field)!r}')
            bounds = step.get('bounds')
            point = step.get('fill_point')
            if (not isinstance(bounds, list) or len(bounds) != 4 or
                    any(type(v) is not int for v in bounds) or
                    bounds[0] >= bounds[2] or bounds[1] >= bounds[3]):
                raise ValueError(f'Step {index}: invalid bounds')
            if (not isinstance(point, list) or len(point) != 2 or
                    any(type(v) is not int for v in point) or
                    not (bounds[0] < point[0] < bounds[2] and bounds[1] < point[1] < bounds[3])):
                raise ValueError(f'Step {index}: fill_point must be inside bounds')
        return data

    @staticmethod
    def point_in_polygon(point, vertices):
        """Ray casting validation for a fill seed (not a guarantee against self-crossing)."""
        x, y = point
        inside = False
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            x1, y1 = a
            x2, y2 = b
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < crossing:
                    inside = not inside
        return inside

    @staticmethod
    def color_matches(actual, expected, tolerance=38):
        return all(abs(int(actual[i]) - expected[i]) <= tolerance for i in range(3))

    @staticmethod
    def interpolate(points, spacing=3.0):
        """Catmull-Rom interpolation for smooth mouse motion (not a bitmap)."""
        result = [tuple(points[0])]
        for i in range(len(points) - 1):
            p0 = points[max(0, i - 1)]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[min(len(points) - 1, i + 2)]
            dist = math.dist(p1, p2)
            count = max(2, math.ceil(dist / spacing))
            for j in range(1, count + 1):
                t = j / count
                def component(k):
                    return round(0.5 * ((2*p1[k]) + (-p0[k]+p2[k])*t +
                        (2*p0[k]-5*p1[k]+4*p2[k]-p3[k])*t*t +
                        (-p0[k]+3*p1[k]-3*p2[k]+p3[k])*t*t*t))
                point = (component(0), component(1))
                if point != result[-1]:
                    result.append(point)
        return result

    def execute(self, drawing, diagnostic_path):
        steps = drawing['steps']
        print(f'DrawBuddy: {drawing.get("name", "drawing")} ({len(steps)} steps).')
        print('Use a NEW blank Paint canvas; do not touch mouse/keyboard.')
        try:
            for index, step in enumerate(steps, 1):
                name, tool = step['name'], step['tool']
                if tool in ('stroke', 'closed_path'):
                    print(f'\n{index}/{len(steps)} {name}: {tool}, outline={step["color"]}')
                    self.tools.pencil.click()
                    self.colors.select_primary(step['color'])
                    points = list(step['points'])
                    if tool == 'closed_path' and points[-1] != points[0]:
                        points.append(points[0])
                    if step.get('smooth', False):
                        points = self.interpolate(points)
                    self.canvas.stroke(points)
                    width = step.get('width', 1)
                    # Pencil has a fixed calibrated size. Build a thicker line
                    # with small perpendicular passes using the same mouse tool.
                    # Avoid changing Paint's ribbon-size setting.
                    if width > 1:
                        for offset in ((1,) if width == 2 else (-1, 1)):
                            shifted = []
                            for i, (px, py) in enumerate(points):
                                prev = points[max(0, i - 1)]
                                nxt = points[min(len(points) - 1, i + 1)]
                                dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
                                length = math.hypot(dx, dy)
                                if length == 0:
                                    shifted.append((px, py))
                                else:
                                    shifted.append((round(px - offset * dy / length),
                                                    round(py + offset * dx / length)))
                            self.canvas.stroke(shifted)
                    print(f'  DONE outline: {len(points)} mouse positions, width={width}')
                    if tool == 'closed_path':
                        self.tools.bucket.click()
                        self.colors.select_primary(step['fill'])
                        x, y = step['fill_point']
                        # Confirm the seed is not already the intended color:
                        # otherwise an unclosed path could go undetected.
                        before = self.canvas.sample(x, y)
                        if self.color_matches(before, RGB[step['fill']]):
                            # Layered drawings may overlap an area already filled
                            # with the same color. Recoloring cannot verify a new
                            # boundary, so preserve the existing fill and continue.
                            print(f'  NOTE {name}: fill seed already {step["fill"]}; '
                                  'skipping redundant bucket click for overlapping layer.')
                            actual = before
                        else:
                            self.canvas.fill(x, y)
                            actual = self.canvas.sample(x, y)
                        if not self.color_matches(actual, RGB[step['fill']]):
                            raise RuntimeError(f'{name}: fill failed, got {actual}, expected {RGB[step["fill"]]}')
                        # Check an exterior pixel is still white to detect bucket leakage.
                        if 'outside_point' in step:
                            ox, oy = step['outside_point']
                            outside = self.canvas.sample(ox, oy)
                            if not self.color_matches(outside, (255, 255, 255), tolerance=20):
                                raise RuntimeError(f'{name}: possible fill leak at outside_point: {outside}')
                        print(f'  PASS closed_path fill: RGB={actual}')
                    continue
                outline, fill = step['outline'], step['fill']
                bounds, point = step['bounds'], step['fill_point']
                print(f'\n{index}/{len(steps)} {name}: {tool}, outline={outline}, fill={fill}')
                # Preserve the last working sequence: tool -> outline -> draw ->
                # Enter -> bucket -> fill color -> fill -> verify canvas pixel.
                getattr(self.tools, tool).click()
                print(f'  OUTLINE requested={outline}')
                self.colors.select_primary(outline)
                self.canvas.draw_shape(bounds)
                self.canvas.commit()
                self.tools.bucket.click()
                print(f'  FILL requested={fill}')
                self.colors.select_primary(fill)
                self.canvas.fill(*point)
                actual = self.canvas.sample(*point)
                if not self.color_matches(actual, RGB[fill]):
                    raise RuntimeError(
                        f'{name} fill verification failed: got {actual}, '
                        f'expected {RGB[fill]}; stopped for diagnosis.'
                    )
                print(f'  PASS {name}: RGB={actual}')
            print('\nDONE: drawing completed.')
        finally:
            self.paint.screenshot(diagnostic_path)
