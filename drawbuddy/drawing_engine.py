"""Execute declarative drawing lessons through the existing Paint OOP components."""
import json
import math
from pathlib import Path
from .controller import PaintController
from .components import PaintColors, PaintTools, PaintCanvas, PaintStrokeSize, RGB


class DrawingEngine:
    SUPPORTED_SHAPES = frozenset(('ellipse', 'rounded_rectangle'))
    SUPPORTED_TOOLS = SUPPORTED_SHAPES | {'stroke', 'closed_path', 'raster_runs', 'brush_paths', 'binary_region'}

    def __init__(self, controller=None, speed=1.0):
        self.paint = controller if controller is not None else PaintController(drawing_speed=speed)
        self.colors = PaintColors(self.paint)
        self.tools = PaintTools(self.paint)
        self.canvas = PaintCanvas(self.paint)
        self.stroke_size = PaintStrokeSize(self.paint)

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
            if step['tool'] == 'binary_region':
                if step.get('color') not in ('black', 'white'):
                    raise ValueError(f'Step {index}: invalid binary region color')
                pts = step.get('points')
                seed = step.get('fill_point')
                if not isinstance(pts, list) or len(pts) < 3 or any(
                    not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or v < 0 for v in p)
                    for p in pts):
                    raise ValueError(f'Step {index}: invalid region contour')
                if not isinstance(seed, list) or len(seed) != 2 or any(type(v) is not int or v < 0 for v in seed):
                    raise ValueError(f'Step {index}: invalid region seed')
                continue
            if step['tool'] == 'brush_paths':
                if step.get('color') not in RGB or step.get('width') not in (1, 2, 3, 4):
                    raise ValueError(f'Step {index}: invalid brush color or size')
                paths = step.get('paths')
                if not isinstance(paths, list) or not paths or any(
                    not isinstance(path, list) or len(path) < 2 or
                    any(not isinstance(p, list) or len(p) != 2 or
                        any(type(v) is not int or v < 0 for v in p) for p in path)
                    for path in paths):
                    raise ValueError(f'Step {index}: brush_paths needs nonempty paths')
                continue
            if step['tool'] == 'raster_runs':
                if step.get('color') not in RGB:
                    raise ValueError(f'Step {index}: unsupported raster color')
                runs = step.get('runs')
                if not isinstance(runs, list) or not runs or any(
                    not isinstance(run, list) or len(run) != 3 or
                    any(type(v) is not int or v < 0 for v in run) or run[0] > run[1]
                    for run in runs):
                    raise ValueError(f'Step {index}: raster_runs requires [x_start,x_end,y] rows')
                continue
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
                if type(step.get('width', 1)) is not int or not 1 <= step.get('width', 1) <= 4:
                    raise ValueError(f'Step {index}: width must be 1, 2, 3 or 4')
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

    @staticmethod
    def verify_fill_boundary(step, thickness=1):
        """Check the mapped contour is raster-closed and seed is safely interior.

        A 1px Paint pencil line can leak diagonally; check connectivity on
        a rasterized contour before clicking the bucket.
        """
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise ValueError("Fill boundary validation requires opencv-python and numpy") from exc
        points = step["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs)-3, min(ys)-3
        width, height = max(xs)-x0+4, max(ys)-y0+4
        if width < 8 or height < 8:
            raise ValueError(f'{step["name"]}: mapped fill contour is too small')
        mask = np.zeros((height, width), dtype=np.uint8)
        contour = np.array([[x-x0, y-y0] for x, y in points], dtype=np.int32)
        cv2.polylines(mask, [contour], True, 255, thickness, lineType=cv2.LINE_8)
        seed = (step["fill_point"][0]-x0, step["fill_point"][1]-y0)
        if not (1 <= seed[0] < width-1 and 1 <= seed[1] < height-1):
            raise ValueError(f'{step["name"]}: fill seed outside mapped contour')
        if mask[seed[1], seed[0]]:
            raise ValueError(f'{step["name"]}: fill seed falls on outline')
        free = (mask == 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
        region = labels[seed[1], seed[0]]
        if region == 0 or region == labels[0, 0]:
            raise ValueError(f'{step["name"]}: fill seed leaks to exterior')
        if stats[region, cv2.CC_STAT_AREA] < 12:
            raise ValueError(f'{step["name"]}: fill interior too small')
        return True

    def verify_pencil_color(self, color):
        """Verify actual Paint canvas ink, not the unreliable ribbon preview.

        The probe is drawn outside the mapped artwork and undone immediately.
        Never continue with a mismatched swatch.
        """
        probe_x, probe_y = 860, 555
        expected = RGB[color]
        for attempt in range(2):
            before = self.canvas.sample(probe_x, probe_y)
            if not self.color_matches(before, (255, 255, 255), tolerance=12):
                raise RuntimeError("Color probe area is not blank; use a new white Paint canvas")
            self.canvas.stroke([[probe_x, probe_y], [probe_x+3, probe_y]])
            try:
                actual = self.canvas.sample(probe_x+1, probe_y)
            finally:
                self.paint.press('ctrl+z')
            restored = self.canvas.sample(probe_x+1, probe_y)
            if not self.color_matches(restored, (255, 255, 255), tolerance=12):
                raise RuntimeError("Color probe undo did not restore the white canvas")
            if self.color_matches(actual, expected, tolerance=12):
                print(f"  VERIFIED pencil color={color} RGB={actual}")
                return
            print(f"  COLOR PROBE mismatch: requested={expected}, drawn={actual}")
            self.colors.current_primary = None
            if attempt == 0:
                self.colors.select_primary(color)
        raise RuntimeError(
            f"Paint is not drawing requested color {color} {expected}; "
            "stopped before drawing the region")

    def execute(self, drawing, diagnostic_path):
        steps = drawing['steps']
        print(f'DrawBuddy: {drawing.get("name", "drawing")} ({len(steps)} steps).')
        print('Use a NEW blank Paint canvas; do not touch mouse/keyboard.')
        try:
            for index, step in enumerate(steps, 1):
                name, tool = step['name'], step['tool']
                if tool == 'binary_region':
                    print(f'\\n{index}/{len(steps)} {name}: filled {step["color"]} region')
                    # Never bucket-fill without checking the physical Paint
                    # outline. A leak can otherwise paint the entire canvas.
                    self.tools.select('pencil')
                    self.colors.select_primary(step['color'])
                    self.stroke_size.select(2)
                    if step['color'] == 'white':
                        # Verify the real ink on the canvas, not just ribbon state.
                        # This catches a miscalibrated white swatch before any
                        # white contour or bucket can damage the drawing.
                        self.verify_pencil_color('white')
                    points = step['points'] + [step['points'][0]]
                    self.paint.validate_canvas_points(points)
                    self.canvas.stroke(points)
                    x, y = step['fill_point']
                    # The contour must be closed in the rasterized geometry.
                    try:
                        self.verify_fill_boundary({"name": name, "points": points,
                                                   "fill_point": [x, y]}, thickness=2)
                    except ValueError as exc:
                        raise RuntimeError(f'{name}: unsafe binary contour: {exc}') from exc
                    guard = (300, 148)
                    before_guard = self.canvas.sample(*guard)
                    if not self.color_matches(before_guard, (255, 255, 255), tolerance=12):
                        raise RuntimeError(f'{name}: white background guard is not white: {before_guard}')
                    before_seed = self.canvas.sample(x, y)
                    if self.color_matches(before_seed, RGB[step['color']], tolerance=12):
                        print(f'  SKIP bucket: seed already {step["color"]}')
                        continue
                    self.tools.select('bucket')
                    # The Paint ribbon may reset Color 1 when tools change.
                    # Force a real swatch click before each bucket operation.
                    self.colors.current_primary = None
                    self.colors.select_primary(step['color'])
                    self.canvas.fill(x, y)
                    actual = self.canvas.sample(x, y)
                    after_guard = self.canvas.sample(*guard)
                    if not self.color_matches(after_guard, before_guard, tolerance=12):
                        self.paint.press('ctrl+z')
                        raise RuntimeError(f'{name}: bucket leaked outside contour; fill undone')
                    if not self.color_matches(actual, RGB[step['color']], tolerance=20):
                        self.paint.press('ctrl+z')
                        raise RuntimeError(f'{name}: bucket fill verification failed: {actual}; fill undone')
                    continue
                if tool == 'brush_paths':
                    print(f'\\n{index}/{len(steps)} {name}: {len(step["paths"])} brush paths')
                    self.tools.select('pencil')
                    self.colors.select_primary(step['color'])
                    self.stroke_size.select(step['width'])
                    self.colors.select_primary(step['color'])
                    for path in step['paths']:
                        self.canvas.stroke(path)
                    continue
                if tool == 'raster_runs':
                    print(f'\\n{index}/{len(steps)} {name}: {len(step["runs"])} raster spans, color={step["color"]}')
                    self.tools.select('pencil')
                    self.colors.select_primary(step['color'])
                    self.stroke_size.select(1)
                    self.colors.select_primary(step['color'])
                    for x1, x2, y in step['runs']:
                        if x1 == x2:
                            self.paint.click(x1, y)
                        else:
                            self.canvas.stroke([[x1, y], [x2, y]])
                    continue
                if tool in ('stroke', 'closed_path'):
                    if tool == 'closed_path':
                        try:
                            self.verify_fill_boundary(step)
                        except ValueError as exc:
                            print(f'  SKIP unsafe fill {name}: {exc}')
                            continue
                    print(f'\n{index}/{len(steps)} {name}: {tool}, outline={step["color"]}')
                    self.tools.select('pencil')
                    self.colors.select_primary(step['color'])
                    if tool == 'closed_path' and not step.get('skip_color_probe', False):
                        self.verify_pencil_color(step['color'])
                    points = list(step['points'])
                    if tool == 'closed_path' and points[-1] != points[0]:
                        points.append(points[0])
                    if step.get('smooth', False):
                        points = self.interpolate(points)
                    width = step.get('width', 1)
                    self.stroke_size.select(width)
                    self.canvas.stroke(points)
                    print(f'  DONE outline: {len(points)} mouse positions, Paint size={width}')
                    if tool == 'closed_path':
                        self.tools.select('bucket')
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
                            raise RuntimeError(
                                f'{name}: fill failed, got {actual}, expected {RGB[step["fill"]]}; '
                                'Paint may have selected the wrong swatch or the boundary leaked. '
                                'Stopped before further drawing.')
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
                self.tools.select(tool)
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
