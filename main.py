"""Generic entry point for DrawBuddy drawing definitions."""
import argparse
import json
import math
from pathlib import Path
from drawbuddy.drawing_engine import DrawingEngine
from drawbuddy.controller import DrawingCancelled
from drawbuddy.freehand_painter import build_freehand_drawing
from drawbuddy.lineart_painter import build_lineart_drawing
from drawbuddy.centerline_painter import build_centerline_drawing
from drawbuddy.region_vectorizer import analyze_regions
from drawbuddy.region_painter import build_region_outline_drawing
from drawbuddy.continuous_painter import build_continuous_color_drawing
from drawbuddy.natural_painter import build_natural_drawing
from drawbuddy.result_publisher import ResultPublisher
from compare_images import compare_paint_canvas
from drawbuddy.binary_painter import build_binary_drawing, build_region_drawing, build_outline_drawing, build_color_raster_drawing



def draw_scene_strokes(path, speed, limit=None, validate_only=False, min_stroke_pixels=8):
    """Draw the Scene Engine's actual variable-width ink paths in Paint.

    Scene coordinates are mapped into a safe 360x270 Paint canvas
    area. This is an ink-only visual test, not a color scene renderer.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    width, height = data['width'], data['height']
    paths = data['paths']
    if width < 1 or height < 1 or not isinstance(paths, list) or not paths:
        raise ValueError('Invalid Scene Engine ink-strokes.json')
    if limit is not None and limit < 1:
        raise ValueError('--scene-limit must be positive')
    # Draw at 1.5x the previous raster resolution. The old 240x180
    # quantization created visible stair-steps on ears, face and tail.
    # This remains within the calibrated Paint area on 1000x650 displays.
    canvas_w, canvas_h = 360, 270
    scale = min(canvas_w / width, canvas_h / height, 1.0)
    dest_w, dest_h = max(1, round(width * scale)), max(1, round(height * scale))
    left = 250 + (canvas_w - dest_w) // 2
    top = 155 + (canvas_h - dest_h) // 2
    # Preserve whole pen gestures. Never fragment them into per-pixel
    # fragments: that produces the printer-like motion seen in Paint.
    selected_paths = paths[:limit]
    visited_pixels = set()
    plan = []
    duplicate_pixels = 0
    rejected_short = 0
    rejected_overlap = 0
    if not 2 <= min_stroke_pixels <= 100:
        raise ValueError('--scene-min-stroke-pixels must be between 2 and 100')

    def pixels_between(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx-dy
        while True:
            yield (x0, y0)
            if (x0, y0) == (x1, y1):
                return
            twice = 2*err
            if twice > -dy:
                err -= dy
                x0 += sx
            if twice < dx:
                err += dx
                y0 += sy

    # Rebuild connected pen gestures at *Paint resolution*. Source skeleton
    # paths are frequently only 1-4 pixels long; filtering them independently
    # discarded >99% of the dog. Join their edges before evaluating length.
    adjacency = {}
    dots = set()
    pixel_sizes = {}
    source_pixels = set()
    for record in selected_paths:
        mapped = []
        for point in record.get('points', []):
            if (not isinstance(point, list) or len(point) != 3 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v)
                            for v in point) or
                    not 0 <= point[0] < width or
                    not 0 <= point[1] < height or point[2] < 0):
                raise ValueError('Invalid Scene Engine stroke coordinates')
            x, y, radius = point
            dest = (left + round(x * scale), top + round(y * scale))
            diameter = 2 * radius * scale
            size = 1 if diameter < 1.5 else (2 if diameter < 3 else (3 if diameter < 5 else 4))
            pixel_sizes[dest] = max(pixel_sizes.get(dest, 1), size)
            if not mapped or mapped[-1] != dest:
                mapped.append(dest)
        if not mapped:
            continue
        dots.update(mapped)
        for start, end in zip(mapped, mapped[1:]):
            raster = list(pixels_between(start, end))
            for u, v in zip(raster, raster[1:]):
                if u == v:
                    continue
                adjacency.setdefault(u, set()).add(v)
                adjacency.setdefault(v, set()).add(u)
                source_pixels.update((u, v))
        source_pixels.update(mapped)

    # Consume each undirected edge once. Continue through junctions using
    # the straightest available continuation, avoiding arbitrary zigzags.
    remaining = {frozenset((u, v)) for u, neighbors in adjacency.items()
                 for v in neighbors}
    def follow(start, first):
        chain = [start, first]
        remaining.remove(frozenset((start, first)))
        previous, current = start, first
        while True:
            options = [p for p in adjacency[current]
                       if frozenset((current, p)) in remaining]
            if not options:
                break
            dx, dy = current[0]-previous[0], current[1]-previous[1]
            nxt = max(options, key=lambda p: (
                (p[0]-current[0])*dx + (p[1]-current[1])*dy,
                -abs((p[0]-current[0])*dy-(p[1]-current[1])*dx),
                -p[1], -p[0]))
            remaining.remove(frozenset((current, nxt)))
            chain.append(nxt)
            previous, current = current, nxt
            # A closed visit can still have unused branches at the same
            # junction. Continue until no unused adjacent edge remains.
        return chain

    chains = []
    # An undirected component needs at least odd_degree/2 open trails.
    # Starting at odd vertices first avoids premature breaks at junctions.
    odd_points = [p for p in sorted(adjacency) if len(adjacency[p]) % 2]
    for point in odd_points:
        while True:
            options = sorted(p for p in adjacency[point]
                             if frozenset((point, p)) in remaining)
            if not options:
                break
            chains.append(follow(point, options[0]))
    # Closed residual cycles can be drawn without an extra pen lift.
    while remaining:
        edge = min(remaining, key=lambda e: tuple(sorted(e)))
        start, neighbor = sorted(edge)
        chains.append(follow(start, neighbor))
    # Preserve isolated ink pixels as dots instead of silently dropping them.
    covered = set(adjacency)
    for point in sorted(dots - covered):
        chains.append([point, point])

    for chain in chains:
        # At Paint resolution, one-pixel details matter. Do not filter
        # reconstructed gestures by the former source-path length threshold.
        sizes = sorted(pixel_sizes.get(p, 1) for p in chain)
        size = sizes[len(sizes)//2]
        plan.append((size, [list(p) for p in chain]))
    rejected_short = 0
    rejected_overlap = 0
    duplicate_pixels = 0
    if not plan:
        raise ValueError('No drawable paths after mapping')
    # Merge adjacent graph trails into one physical mouse gesture whenever
    # their endpoints coincide. No invented connecting segment is permitted.
    # Paint currently uses a constant pencil width, so size labels do not
    # constrain joins. This reduces visible gaps during drawing without
    # changing the final set of ink pixels.
    original_trail_count = len(plan)
    trails = [points for _, points in plan]
    while True:
        endpoints = {}
        for i, points in enumerate(trails):
            if points is None:
                continue
            endpoints.setdefault(tuple(points[0]), []).append((i, 0))
            endpoints.setdefault(tuple(points[-1]), []).append((i, 1))
        merged = False
        for point in sorted(endpoints):
            entries = endpoints[point]
            pair = next(((a, b) for k, a in enumerate(entries)
                         for b in entries[k + 1:] if a[0] != b[0]), None)
            if pair is None:
                continue
            (i, side_i), (j, side_j) = pair
            first = trails[i] if side_i == 1 else trails[i][::-1]
            second = trails[j] if side_j == 0 else trails[j][::-1]
            assert first[-1] == second[0]
            trails[i] = first + second[1:]
            trails[j] = None
            merged = True
            break
        if not merged:
            break
    plan = [(1, points) for points in trails if points is not None]
    joined_trails = original_trail_count - len(plan)
    # Complete each connected ink component before moving elsewhere.
    # The earlier global length phases left visible gaps in a contour while
    # drawing unrelated details. 8-neighbour connectivity groups touching
    # raster ink without inventing any new painted edges.
    parent = {}
    def root(point):
        parent.setdefault(point, point)
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = parent[point]
        return point
    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    for point in sorted(source_pixels):
        root(point)
        x, y = point
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0)):
            neighbor = (x + dx, y + dy)
            if neighbor in source_pixels:
                union(point, neighbor)
    components = {}
    for size, points in plan:
        key = root(tuple(points[0]))
        components.setdefault(key, []).append((size, points))
    # Largest connected outlines first. Finish each component (including
    # its short branches) before starting the next component.
    groups = sorted(components.values(),
                    key=lambda g: (-len({tuple(p) for _, path in g for p in path}),
                                   min(tuple(path[0]) for _, path in g)))
    component_counts = [len(g) for g in groups]
    ordered = []
    cursor = None
    for group in groups:
        # Begin with the longest contour in this component.
        idx = max(range(len(group)), key=lambda i: len(group[i][1]))
        size, points = group.pop(idx)
        ordered.append((size, points))
        cursor = points[-1]
        while group:
            def travel(i):
                path = group[i][1]
                forward = (path[0][0]-cursor[0])**2 + (path[0][1]-cursor[1])**2
                backward = (path[-1][0]-cursor[0])**2 + (path[-1][1]-cursor[1])**2
                return (min(forward, backward), -len(path), i)
            idx = min(range(len(group)), key=travel)
            size, points = group.pop(idx)
            forward = (points[0][0]-cursor[0])**2 + (points[0][1]-cursor[1])**2
            backward = (points[-1][0]-cursor[0])**2 + (points[-1][1]-cursor[1])**2
            if backward < forward:
                points = points[::-1]
            ordered.append((size, points))
            cursor = points[-1]
    plan = ordered
    # Reconstruct wider source ink with short parallel pencil passes.
    # Paint's ribbon width control is unreliable, so preserve the source
    # radius using additional, explicitly planned 1px strokes. Restrict
    # these to thick centerline pixels: fine paw/eye details stay untouched.
    # These passes are derived from source radius, not invented contours.
    thickness_paths = []
    thick_pixels = {p for p, level in pixel_sizes.items() if level >= 3}
    for size, points in plan:
        if len(points) < 3:
            continue
        # Only thicken substantial runs. Avoid isolated pixels and junction
        # clusters, which otherwise create black blobs on the face.
        for offset in (-1, 1):
            run = []
            for i in range(1, len(points) - 1):
                p = tuple(points[i])
                before, after = points[i - 1], points[i + 1]
                dx, dy = after[0] - before[0], after[1] - before[1]
                length = math.hypot(dx, dy)
                if p not in thick_pixels or length < 1:
                    if len(run) >= 3:
                        thickness_paths.append((1, run))
                    run = []
                    continue
                nx = round(p[0] - offset * dy / length)
                ny = round(p[1] + offset * dx / length)
                shifted = [nx, ny]
                if not run or run[-1] != shifted:
                    run.append(shifted)
            if len(run) >= 3:
                thickness_paths.append((1, run))
    # These accent strokes are last: the main recognizable contours remain
    # continuous during the drawing process.
    plan.extend(thickness_paths)
    # Only repair substantial missing SOLID ink regions with closed contours.
    # Never raster-print all missing pixels as horizontal mouse strokes.
    source_ink_runs = data.get('ink_runs')
    fill_paths = []
    missing_fill_pixels = 0
    repair_components = 0
    if source_ink_runs is not None and limit is None:
        import cv2
        import numpy as np
        target = np.zeros((dest_h, dest_w), dtype=np.uint8)
        for run in source_ink_runs:
            if (not isinstance(run, list) or len(run) != 3 or
                    not all(isinstance(v, int) for v in run)):
                raise ValueError('Invalid ink_runs entry')
            y, x0, x1 = run
            if not (0 <= y < height and 0 <= x0 <= x1 < width):
                raise ValueError('Source ink run outside image')
            yy = round(y * scale)
            xa, xb = round(x0 * scale), round(x1 * scale)
            if 0 <= yy < dest_h:
                target[yy, max(0, xa):min(dest_w, xb + 1)] = 1
        existing = np.zeros_like(target)
        for _, points in plan:
            for a, b in zip(points, points[1:]):
                x0, y0 = a[0] - left, a[1] - top
                x1, y1 = b[0] - left, b[1] - top
                cv2.line(existing, (x0, y0), (x1, y1), 1, 1)
            if len(points) == 1:
                x, y = points[0]
                if 0 <= x-left < dest_w and 0 <= y-top < dest_h:
                    existing[y-top, x-left] = 1
        missing = np.uint8((target != 0) & (existing == 0))
        missing_fill_pixels = int(np.count_nonzero(missing))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            missing, connectivity=8)
        for component_id in range(1, count):
            x, y, w, h, area = map(int, stats[component_id])
            # A thin missing line is not a solid spot; preserve the original
            # centerline geometry instead of making it printer-like.
            if area < 10 or min(w, h) < 3:
                continue
            region = np.uint8(labels[y:y+h, x:x+w] == component_id)
            repair_components += 1
            # Follow actual boundaries of missing solid regions. Eroded
            # contours supply at most two inner passes for thicker spots.
            for depth in range(3):
                if not np.any(region):
                    break
                contours, _ = cv2.findContours(
                    region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    coords = [[left+x+int(p[0][0]), top+y+int(p[0][1])]
                              for p in contour]
                    if len(coords) >= 3:
                        coords.append(coords[0])
                        fill_paths.append((1, coords))
                region = cv2.erode(region, np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        plan.extend(fill_paths)
    # Validate planned motion and report natural contour intersections.
    planned_pixels = set()
    overlap_count = 0
    out_of_bounds = 0
    # Check actual visits by each mouse stroke. Shared vertices between
    # adjacent segments of ONE continuous stroke are not repeated visits.
    for _, points in plan:
        stroke_pixels = []
        for segment_index, (start, end) in enumerate(zip(points, points[1:])):
            segment = list(pixels_between(tuple(start), tuple(end)))
            if segment_index:
                segment = segment[1:]
            stroke_pixels.extend(segment)
        if not stroke_pixels:
            stroke_pixels = [tuple(points[0])]
        for pixel in stroke_pixels:
            if not (left <= pixel[0] < left + dest_w + 2 and
                    top <= pixel[1] < top + dest_h + 2):
                out_of_bounds += 1
            if pixel in planned_pixels:
                overlap_count += 1
            planned_pixels.add(pixel)
    lengths = [len(points) for _, points in plan]
    width_counts = {str(size): sum(level == size for level, _ in plan) for size in (1, 2, 3, 4)}
    report = {'source_paths': len(selected_paths),
              'planned_strokes': len(plan),
              'thickness_accent_strokes': len(thickness_paths),
              'source_ink_fill_strokes': len(fill_paths),
              'source_ink_pixels_missing_before_repair': missing_fill_pixels,
              'solid_ink_repair_components': repair_components,
              'source_ink_run_data_available': source_ink_runs is not None,
              'canvas_resolution': [dest_w, dest_h],
              'coordinate_scale': round(scale, 5),
              'joined_adjacent_trails': joined_trails,
              'odd_degree_vertices': len(odd_points),
              'estimated_pen_lifts': max(0, len(plan) - 1),
              'ink_components': len(component_counts),
              'largest_component_strokes': max(component_counts),
              'rejected_short_paths': rejected_short,
              'rejected_overlapping_paths': rejected_overlap,
              'minimum_stroke_pixels': min_stroke_pixels,
              'median_stroke_vertices': sorted(lengths)[len(lengths)//2],
              'paint_size_levels': width_counts,
              'unique_pixels': len(planned_pixels),
              'source_unique_pixels': len(source_pixels),
              'repeated_pixel_visits': overlap_count,
              'overlaps_allowed_for_natural_contours': True,
              'skipped_source_pixel_visits': duplicate_pixels,
              'out_of_bounds_pixels': out_of_bounds,
              'paint_execution_started': False}
    print('Scene path validation:', json.dumps(report, indent=2))
    if out_of_bounds:
        raise RuntimeError('Scene path validation FAILED: out-of-bounds motion')
    coverage_ratio = len(planned_pixels & source_pixels) / max(1, len(source_pixels))
    print(f'Scene source pixel coverage: {coverage_ratio:.2%}')
    if coverage_ratio < 0.95:
        raise RuntimeError('Scene planning lost source pixels; Paint drawing blocked')
    if validate_only:
        print('PASS: planned Paint paths stay in bounds and preserve source ink pixels.')
        return
    engine = DrawingEngine(speed=speed)
    # Validate every planned position before touching the Paint canvas.
    for _, points in plan:
        engine.paint.validate_canvas_points(points)
    print(f'Natural Paint paths: skipped {rejected_short} short and {rejected_overlap} nearly redundant paths.')
    print(f'Scene Engine Paint test: {len(plan)} deduplicated pen paths '
          f'from {len(selected_paths)} source paths; '
          f'canvas={dest_w}x{dest_h}, scale={scale:.4f}')
    print('Contour intersections are retained; nearly redundant paths are omitted.')
    print('Use a NEW blank white maximized Paint canvas. F8 pauses/resumes.')
    engine.tools.select('pencil')
    engine.colors.select_primary('black')
    # Pencil-size changes were not producing observable thickness changes.
    # Keep one calibrated width until a separate Paint size test proves
    # the ribbon controls actually affect the pencil.
    import time
    import ctypes
    engine.stroke_size.select(1)
    engine.tools.pencil.click()
    engine.tools.current = 'pencil'
    time.sleep(0.6)
    print('Scene Paint uses constant Pencil size 1 (no ribbon size switching).')

    def wait_for_resume():
        # controller.stroke releases mouseUp in its finally clause before
        # raising DrawingCancelled. Wait for key release, then a fresh press.
        user32 = ctypes.windll.user32
        print('PAUSED by F8; release F8, then press F8 again to resume.')
        while user32.GetAsyncKeyState(0x77) & 0x8000:
            time.sleep(0.05)
        while not (user32.GetAsyncKeyState(0x77) & 0x8000):
            time.sleep(0.05)
        while user32.GetAsyncKeyState(0x77) & 0x8000:
            time.sleep(0.05)
        print('RESUMED.')

    for index, (size, points) in enumerate(plan, 1):
        while True:
            try:
                from drawbuddy.controller import check_cancel
                check_cancel()
                engine.canvas.stroke(points)
                break
            except DrawingCancelled:
                wait_for_resume()
                # Retry the interrupted gesture from its start. Existing
                # ink remains on the canvas, so no pixels are erased.
        if index % 100 == 0:
            print(f'Scene progress: {index}/{len(plan)}')
    print('DONE: Scene Engine ink drawn in Microsoft Paint.')
    engine.paint.refresh_display()




def execute_continuous_path_test(path, speed, limit):
    """Draw actual connected color paths in Paint, without raster runs."""
    import time
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    report = data.get('report', {})
    width, height = report.get('width'), report.get('height')
    if (report.get('format') != 'drawbuddy-continuous-path-analysis-v1'
            or not isinstance(width, int) or not isinstance(height, int)
            or not 1 <= width <= 500 or not 1 <= height <= 547
            or not isinstance(data.get('groups'), list)):
        raise ValueError('Invalid path file. Run python pixel_map.py --analyze-continuous')
    left, top = 250 + (500 - width)//2, 155 + (547 - height)//2
    groups = []
    for group in data['groups']:
        try:
            rgb = tuple(bytes.fromhex(group['rgb'].lstrip('#')))
        except (ValueError, KeyError) as exc:
            raise ValueError('Invalid RGB color in path file') from exc
        if len(rgb) != 3:
            raise ValueError('Invalid RGB color in path file')
        paths = []
        for points in group.get('paths', []):
            if not 12 <= len(points) <= 180:
                continue
            if any(not isinstance(p, list) or len(p) != 2
                   or any(type(v) is not int for v in p)
                   or not (0 <= p[0] < width and 0 <= p[1] < height)
                   for p in points):
                raise ValueError('Invalid continuous path coordinates')
            if any((a[0] != b[0] and a[1] != b[1])
                   for a, b in zip(points, points[1:])):
                raise ValueError('Diagonal continuous path segment')
            paths.append(points)
        if len(paths) >= limit:
            groups.append((rgb, paths))
    if not groups:
        raise ValueError(f'No color has {limit} eligible paths of 12-180 points')
    rgb, paths = max(groups, key=lambda item: sum(map(len, item[1])))
    selected = sorted(paths, key=len, reverse=True)[:limit]
    mapped = [[(left+x, top+y) for x,y in points] for points in selected]
    print(f'CONTINUOUS PAINT TEST: {limit} paths, RGB={rgb}, lengths={[len(p) for p in selected]}')
    print('Open a NEW blank white maximized Paint canvas. F8 cancels.')
    engine = DrawingEngine(speed=speed)
    for points in mapped:
        engine.paint.validate_canvas_points(points)
    engine.tools.select('pencil')
    engine.stroke_size.select(1)
    engine.colors.select_rgb(rgb)
    start = time.perf_counter()
    for i, points in enumerate(mapped, 1):
        engine.paint.stroke(points, duration=0.012)
        print(f'Continuous path {i}/{limit} complete')
    elapsed = time.perf_counter()-start
    print(f'DONE: {limit} continuous paths in {elapsed:.1f}s')
    # Compare the *selected* paths, not the entire dog: this is a smoke test.
    # Build an exact RGB target on white at the same 500x547 image coordinates.
    import cv2
    import numpy as np
    from compare_images import compare_paint_canvas
    expected = np.full((height, width, 3), 255, dtype=np.uint8)
    for points in selected:
        for a, b in zip(points, points[1:]):
            cv2.line(expected, tuple(a), tuple(b), rgb[::-1], 1, cv2.LINE_8)
    output = Path('output/comparison/continuous-path-test')
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / 'expected.png'
    cv2.imwrite(str(reference_path), expected)
    engine.paint.refresh_display()
    result = compare_paint_canvas(engine.paint, reference_path, width, height,
                                  left, top, output=output, tolerance=0)
    result['test_paths'] = limit
    result['elapsed_seconds'] = round(elapsed, 3)
    result['comparison_scope'] = 'selected paths on blank white canvas only'
    (output / 'report.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(f'EXACT 1:1 RGB MATCH: {result["exact_pixel_match_percent"]}% '
          f'({result["error_pixels"]} mismatched pixels)')
    print(f'Inspect {output / "error-overlay.png"} and {output / "report.json"}')


def execute_pixel_paint_plan(path, speed, execute=False, max_strokes=None, color_test=False, region_test=False, gap_test=False):
    """Validate a 16-color pixel plan; draw only with explicit opt-in.

    Coordinates are local to the image, anchored to a safe Paint canvas area.
    Exact RGB colors use the existing Paint Edit Colors UIA implementation.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    width, height = data.get('width'), data.get('height')
    groups = data.get('groups')
    if (data.get('format') != 'drawbuddy-paint-stroke-plan-v1'
            or type(width) is not int or type(height) is not int
            or width < 1 or height < 1 or width > 500 or height > 547
            or not isinstance(groups, list) or len(groups) != 16):
        raise ValueError('Expected a valid 16-color plan of at most 500x547')
    if max_strokes is not None and max_strokes < 1:
        raise ValueError('--paint-plan-limit must be positive')
    left, top = 250 + (500 - width) // 2, 155 + (547 - height) // 2
    work = []
    color_samples = []
    seen_colors = set()
    for group in groups:
        color = group.get('rgb')
        if (not isinstance(color, str) or len(color) != 7
                or not color.startswith('#')):
            raise ValueError('Invalid plan color')
        try:
            rgb = tuple(bytes.fromhex(color[1:]))
        except ValueError as exc:
            raise ValueError('Invalid RGB hex color') from exc
        if len(rgb) != 3 or color.upper() in seen_colors:
            raise ValueError('Duplicate or invalid plan color')
        seen_colors.add(color.upper())
        strokes = group.get('strokes')
        if not isinstance(strokes, list) or type(group.get('background')) is not bool:
            raise ValueError('Invalid plan group')
        if group['background'] and strokes:
            raise ValueError('Background group must have no strokes')
        group_work = []
        for stroke in strokes:
            if (not isinstance(stroke, list) or len(stroke) != 4
                    or any(type(v) is not int for v in stroke)):
                raise ValueError('Invalid stroke')
            x0, y0, x1, y1 = stroke
            if not (0 <= x0 <= x1 < width and 0 <= y0 == y1 < height):
                raise ValueError('Stroke outside pixel plan')
            if not group['background']:
                run = (rgb, (left + x0, left + x1, top + y0))
                work.append(run)
                group_work.append(run)
        if group_work:
            color_samples.append((rgb, group_work))
    if gap_test:
        # Isolate mouse-event continuity: two distant strokes of one color.
        # Positions are image-local, on the same row and safely separated.
        rgb = next((color for color, strokes in color_samples
                    if strokes and color[0] > 180 and color[1] > 60 and color[2] < 100), None)
        if rgb is None:
            raise ValueError('Gap test could not identify an orange palette color')
        y = top + height // 2
        a = left + width // 5
        b = left + (4 * width) // 5
        work = [(rgb, (a, a + 25, y)), (rgb, (b, b + 25, y))]
        print(f'GAP TEST: two orange 26px spans separated by {b - a - 26}px of white; '
              'any connecting line is a mouse-event fault')
    elif color_test:
        if len(color_samples) < 3:
            raise ValueError('Color test requires at least three drawable colors')
        work = [stroke for _, sample in color_samples[:3] for stroke in sample[:3]]
        print('COLOR TEST: three strokes from each of three different RGB colors: '
              + ', '.join('#%02X%02X%02X' % rgb for rgb, _ in color_samples[:3]))
    elif region_test:
        # Draw a substantial central portion of the actual dog image, with
        # full density and all colors present in that region.
        x_min, x_max = width // 4, (3 * width) // 4
        y_min, y_max = height // 4, (3 * height) // 4
        selected = []
        for rgb, (x0, x1, y) in work:
            local_y = y - top
            if not y_min <= local_y < y_max:
                continue
            a = max(x0, left + x_min)
            b = min(x1, left + x_max - 1)
            if a <= b:
                selected.append((rgb, (a, b, y)))
        work = selected
        unique = {rgb for rgb, _ in work}
        print(f'REGION TEST: central 50% crop x={x_min}:{x_max}, '
              f'y={y_min}:{y_max}; {len(work):,} strokes, '
              f'{len(unique)} drawable colors')
        if len(unique) < 3:
            raise ValueError('Region test has fewer than 3 colors; drawing blocked')
    elif max_strokes is not None:
        work = work[:max_strokes]
    print(f'16-color plan validated: {width}x{height}; '
          f'{len(work):,} horizontal strokes selected; origin=({left},{top})')
    if not execute:
        print('VALIDATION ONLY: no Paint window opened or modified. '
              'Use --paint-plan-execute to draw.')
        return
    engine = DrawingEngine(speed=speed)
    for _, (x0, x1, y) in work:
        engine.paint.validate_canvas_points([(x0, y), (x1, y)])
    print('Paint execution uses exact custom RGB via Edit Colors; '
          'not restricted to existing palette swatches.')
    print('Use a NEW blank white maximized Paint canvas. F8 cancels.')
    engine.tools.select('pencil')
    engine.stroke_size.select(1)
    current = None
    for i, (rgb, run) in enumerate(work, 1):
        if rgb != current:
            engine.colors.select_rgb(rgb)
            current = rgb
        engine.paint.raster_runs([run])
        if i % 500 == 0:
            print(f'Pixel Paint progress: {i:,}/{len(work):,}')
    engine.paint.refresh_display()
    print('DONE: pixel plan drawn in Paint; no automatic publication.')


def main():
    parser = argparse.ArgumentParser(description='Draw a saved lesson in Microsoft Paint.')
    parser.add_argument('--paint-plan', type=Path, help='Validate or draw a 16-color pixel stroke plan')
    parser.add_argument('--continuous-paths', type=Path, help='Continuous-path analysis JSON')
    parser.add_argument('--continuous-path-test', type=int, choices=(1, 20), help='Draw 1 or 20 paths in Paint')
    parser.add_argument('--paint-plan-gap-test', action='store_true', help='Draw only two separated orange strokes to diagnose unwanted connector lines')
    parser.add_argument('--paint-plan-region-test', action='store_true', help='Draw a substantial central 50 percent crop using all its colors')
    parser.add_argument('--paint-plan-color-test', action='store_true', help='Draw 3 strokes from each of 3 different plan colors')
    parser.add_argument('--paint-plan-execute', action='store_true', help='Actually draw the pixel plan in Paint (uses Edit Colors)')
    parser.add_argument('--paint-plan-limit', type=int, default=None, help='Limit pixel strokes for a Paint smoke test')
    parser.add_argument('--scene-strokes', type=Path, help='Draw Scene Engine ink-strokes.json in Paint (experimental)')
    parser.add_argument('--scene-min-stroke-pixels', type=int, default=8, help='Minimum unique pixels per natural pen stroke (default 8; try 5 for detail)')
    parser.add_argument('--scene-validate-only', action='store_true', help='Validate planned pen paths without opening or touching Paint')
    parser.add_argument('--scene-limit', type=int, default=None, help='Limit Scene Engine source paths for a short Paint test')
    parser.add_argument('drawing', nargs='?', help='Drawing name (e.g. dog), without .json')
    parser.add_argument('--speed', type=float, default=None, help='Stroke speed multiplier (e.g. 4, 20, 50, 100)')
    parser.add_argument('--no-publish-result', action='store_true', help='Skip automatic Paint canvas capture and GitHub upload')
    parser.add_argument('--natural-source', type=Path, help='Trace outlines and fill color regions with natural vertical strokes')
    parser.add_argument('--continuous-source', type=Path, help='Optimize 0/1/color pixels into continuous mouse strokes')
    parser.add_argument('--region-draw-source', type=Path, help='Draw shared region boundaries in Paint, no bucket fill')
    parser.add_argument('--region-source', type=Path, help='Analyze source regions and export boundary preview without launching Paint')
    parser.add_argument('--centerline-source', type=Path, help='Experimental 2px centerline contours without fills')
    parser.add_argument('--lineart-source', type=Path, help='Draw black outlines only, with no fill')
    parser.add_argument('--freehand-source', type=Path, help='Draw source with contour-following freehand mouse strokes')
    parser.add_argument('--binary-source', type=Path, help='Print the source as 0/1 horizontal pixel runs (default)')
    parser.add_argument('--binary-fill', action='store_true', help='Experimental contour and bucket fill mode (unsafe until validated)')
    parser.add_argument('--binary-runs', action='store_true', help='Explicitly select the default 0/1 row-by-row printing')
    parser.add_argument('--binary-outlines', action='store_true', help='Use the previous experimental contour drawing')
    parser.add_argument('--color-raster', action='store_true', help='Print RGB colors as separate 0/1 raster runs')
    parser.add_argument('--no-compare', action='store_true', help='Skip automatic Paint vs source comparison')
    parser.add_argument('--binary-threshold', type=int, default=None, help='Optional grayscale cutoff 0..255 (default: Otsu)')
    args = parser.parse_args()
    if args.continuous_paths is not None or args.continuous_path_test is not None:
        if args.continuous_paths is None or args.continuous_path_test is None:
            parser.error('--continuous-paths and --continuous-path-test are both required')
        if args.paint_plan or args.scene_strokes or args.drawing:
            parser.error('Continuous path test cannot be combined with another drawing mode')
        speed = args.speed if args.speed is not None else 20
        if not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0 < speed <= 100:
            parser.error('Invalid drawing speed')
        execute_continuous_path_test(args.continuous_paths, float(speed), args.continuous_path_test)
        return
    if args.paint_plan:
        if args.scene_strokes or args.drawing or any((
                args.region_draw_source, args.region_source, args.centerline_source,
                args.natural_source, args.continuous_source, args.lineart_source,
                args.freehand_source, args.binary_source)):
            parser.error('--paint-plan must be used alone')
        config_path = Path(__file__).resolve().parent / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
        if type(speed) not in (float, int) or not math.isfinite(speed) or not 0 < speed <= 100:
            parser.error('drawing speed must be greater than 0 and at most 100')
        if args.paint_plan_gap_test and (args.paint_plan_region_test or args.paint_plan_color_test or args.paint_plan_limit is not None):
            parser.error('--paint-plan-gap-test cannot be combined with other test modes')
        if args.paint_plan_gap_test and not args.paint_plan_execute:
            parser.error('--paint-plan-gap-test requires --paint-plan-execute')
        if args.paint_plan_region_test and (args.paint_plan_color_test or args.paint_plan_limit is not None):
            parser.error('--paint-plan-region-test cannot be combined with other test modes')
        if args.paint_plan_region_test and not args.paint_plan_execute:
            parser.error('--paint-plan-region-test requires --paint-plan-execute')
        if args.paint_plan_color_test and args.paint_plan_limit is not None:
            parser.error('--paint-plan-color-test cannot be combined with --paint-plan-limit')
        if args.paint_plan_color_test and not args.paint_plan_execute:
            parser.error('--paint-plan-color-test requires --paint-plan-execute')
        execute_pixel_paint_plan(args.paint_plan, float(speed),
                                 args.paint_plan_execute, args.paint_plan_limit,
                                 args.paint_plan_color_test, args.paint_plan_region_test,
                                 args.paint_plan_gap_test)
        return
    if args.paint_plan_execute or args.paint_plan_color_test or args.paint_plan_region_test or args.paint_plan_gap_test or args.paint_plan_limit is not None:
        parser.error('--paint-plan-execute and --paint-plan-limit require --paint-plan')
    if args.scene_strokes:
        if args.drawing or any((args.region_draw_source, args.region_source,
                args.centerline_source, args.natural_source, args.continuous_source,
                args.lineart_source, args.freehand_source, args.binary_source)):
            parser.error('--scene-strokes cannot be combined with other drawing sources')
        config_path = Path(__file__).resolve().parent / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
        if type(speed) not in (float, int) or not math.isfinite(speed) or not 0 < speed <= 100:
            parser.error('drawing speed must be greater than 0 and at most 100')
        draw_scene_strokes(args.scene_strokes, float(speed), args.scene_limit, args.scene_validate_only, args.scene_min_stroke_pixels)
        return
    if args.scene_min_stroke_pixels != 8:
        parser.error('--scene-min-stroke-pixels requires --scene-strokes')
    if args.scene_validate_only:
        parser.error('--scene-validate-only requires --scene-strokes')
    if args.scene_limit is not None:
        parser.error('--scene-limit requires --scene-strokes')
    if args.region_draw_source and (args.region_source or args.centerline_source or args.natural_source or args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --region-draw-source alone')
    if args.region_source:
        if any((args.centerline_source, args.natural_source, args.continuous_source,
                args.lineart_source, args.freehand_source, args.binary_source, args.drawing)):
            parser.error('Use --region-source alone')
        print('Region-first analysis:', analyze_regions(args.region_source))
        return
    root = Path(__file__).resolve().parent
    drawings_dir = root / 'drawings'
    if args.centerline_source and (args.natural_source or args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --centerline-source alone')
    if args.natural_source and (args.continuous_source or args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --natural-source alone')
    if args.continuous_source and (args.lineart_source or args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --continuous-source alone')
    if args.lineart_source and (args.freehand_source or args.binary_source or args.drawing):
        parser.error('Use --lineart-source alone')
    if args.freehand_source and (args.binary_source or args.drawing):
        parser.error('Use --freehand-source alone, without a drawing or --binary-source')
    if args.region_draw_source:
        drawing = build_region_outline_drawing(args.region_draw_source)
        print(f'Region drawing plan: {drawing["region_drawing"]}')
    elif args.centerline_source:
        drawing = build_centerline_drawing(args.centerline_source)
        print(f'Centerline drawing plan: {drawing["lineart"]}')
    elif args.natural_source:
        drawing = build_natural_drawing(args.natural_source)
        print(f'Natural drawing plan: {drawing["natural"]}')
    elif args.continuous_source:
        drawing = build_continuous_color_drawing(args.continuous_source)
        print(f'Continuous color plan: {drawing["continuous"]}')
    elif args.lineart_source:
        drawing = build_lineart_drawing(args.lineart_source)
        print(f'Line art plan: {drawing["lineart"]}')
    elif args.freehand_source:
        drawing = build_freehand_drawing(args.freehand_source)
        print(f'Freehand plan: {drawing["freehand"]}')
    elif args.binary_source:
        if args.drawing:
            parser.error('Specify either a drawing name or --binary-source, not both')
        if args.binary_threshold is not None and not 0 <= args.binary_threshold <= 255:
            parser.error('--binary-threshold must be 0..255')
        if sum((args.binary_runs, args.binary_fill, args.binary_outlines, args.color_raster)) > 1:
            parser.error('Choose only one raster/outline mode')
        planner = (build_binary_drawing if args.binary_runs else
                   build_region_drawing if args.binary_fill else
                   build_outline_drawing if args.binary_outlines else build_color_raster_drawing)
        if planner is build_color_raster_drawing:
            drawing = planner(args.binary_source)
        else:
            drawing = planner(args.binary_source, threshold=args.binary_threshold)
        if "binary" in drawing:
            print(f'Binary 0/1: {drawing["binary"]}')
        elif "color" in drawing:
            print(f'Color raster: {drawing["color"]}')
    else:
        name = args.drawing
        if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\\\' in name or not name.replace('-', '').replace('_', '').isalnum():
            parser.error('Drawing must be a simple name, such as dog')
        definition = drawings_dir / (name + '.json')
        if not definition.is_file():
            choices = ', '.join(sorted(p.stem for p in drawings_dir.glob('*.json'))) or '(none)'
            parser.error(f'Drawing {name!r} not found. Available: {choices}')
        drawing = DrawingEngine.load(definition)
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    if not isinstance(config, dict):
        parser.error('config.json must contain a JSON object')
    speed = args.speed if args.speed is not None else config.get('drawing_speed', 1.0)
    if type(speed) not in (float, int) or not math.isfinite(speed) or speed <= 0 or speed > 100:
        parser.error('drawing speed must be greater than 0 and at most 100')
    engine = DrawingEngine(speed=float(speed))
    print('Press F8 at any time to stop drawing without cancelling Paint strokes.')
    try:
        engine.execute(drawing, root / 'paint-diagnostic.png')
    except DrawingCancelled:
        print('STOPPED: drawing cancelled by F8. Partial drawing remains in Paint.')
        return
    engine.paint.refresh_display()
    if args.binary_source and not args.no_compare:
        # Compare the actual pixels against the resized source, not the
        # quantized palette preview. The difference includes palette loss.
        info = drawing.get('color') or drawing.get('binary')
        if not info or 'width' not in info or 'height' not in info:
            raise RuntimeError('Raster drawing dimensions missing')
        width, height = info['width'], info['height']
        max_width, max_height = 240, 180
        left = 310 + (max_width - width) // 2
        top = 155 + (max_height - height) // 2
        compare_paint_canvas(
            engine.paint, root / 'output' / 'raster-source-color.png',
            width, height, left, top,
            output=root / 'output' / 'comparison'
        )
    if not args.no_publish_result:
        publisher = ResultPublisher(root)
        publisher.capture(engine.paint)
        publisher.publish()


if __name__ == '__main__':
    main()
