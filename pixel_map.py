"""Build a complete RGB pixel map from an image, without tracing or Paint.

Every map cell owns one RGB color (including white). The source is fitted
inside the requested resolution without stretching; padding is pure white.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np



# Standard 16-color Windows/Paint-style palette. This is an explicit
# approximation of available basic swatches, not a UI calibration.
PAINT_16_RGB = np.array([
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
], dtype=np.uint8)


def export_paint_palette_preview(rgb, output, grid_scale):
    """Map each source pixel to the nearest of 16 fixed RGB swatches."""
    palette = PAINT_16_RGB.astype(np.float32)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    # Chunk to avoid large temporary arrays on bigger images.
    ids = np.empty(len(pixels), dtype=np.uint8)
    for start in range(0, len(pixels), 20000):
        part = pixels[start:start + 20000]
        squared = np.sum((part[:, None, :] - palette[None, :, :]) ** 2, axis=2)
        ids[start:start + len(part)] = np.argmin(squared, axis=1)
    mapped = PAINT_16_RGB[ids].reshape(rgb.shape)
    base = output.with_name(output.stem + '-paint16')
    cv2.imwrite(str(base.with_suffix('.png')), cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR))
    if grid_scale > 1:
        zoom = cv2.resize(mapped, (mapped.shape[1] * grid_scale,
                                  mapped.shape[0] * grid_scale),
                          interpolation=cv2.INTER_NEAREST)
        zoom[::grid_scale, :, :] = 195
        zoom[:, ::grid_scale, :] = 195
        cv2.imwrite(str(base.with_name(base.name + '-grid.png')),
                    cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
    counts = np.bincount(ids, minlength=len(PAINT_16_RGB))
    report = {
        'palette': 'standard-16-rgb-approximation-not-calibrated-paint-ui',
        'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
        'mean_absolute_rgb_error': round(float(np.mean(
            np.abs(rgb.astype(np.float32) - mapped.astype(np.float32)))), 3),
        'colors': [
            {'rgb': '#%02X%02X%02X' % tuple(int(c) for c in color),
             'pixels': int(count)}
            for color, count in zip(PAINT_16_RGB, counts)
        ],
    }
    base.with_suffix('.json').write_text(json.dumps(report, indent=2),
                                         encoding='utf-8')
    print('PAINT 16 preview (fixed standard swatches; not UI-calibrated):')
    print(f"  used colors: {sum(c > 0 for c in counts)}/16; mean RGB error: {report['mean_absolute_rgb_error']}")
    for entry in report['colors']:
        if entry['pixels']:
            print(f"  {entry['rgb']}: {entry['pixels']:,} pixels")
    print(f'  Preview: {base.with_suffix(".png")}')
    print(f'  Report: {base.with_suffix(".json")}')



def export_seven_color_preview(rgb, output, grid_scale):
    """Find seven representative colors in the resized source, without guessing."""
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(
        pixels, 7, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.1),
        8, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    ids = labels.reshape(-1)
    counts = np.bincount(ids, minlength=7)
    order = np.argsort(-counts)
    centers = centers[order]
    remap = np.empty(7, dtype=np.uint8)
    remap[order] = np.arange(7, dtype=np.uint8)
    ids = remap[ids]
    mapped = centers[ids].reshape(rgb.shape)
    base = output.with_name(output.stem + '-palette7')
    cv2.imwrite(str(base.with_suffix('.png')), cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR))
    if grid_scale > 1:
        zoom = cv2.resize(mapped, (rgb.shape[1] * grid_scale,
                                  rgb.shape[0] * grid_scale),
                          interpolation=cv2.INTER_NEAREST)
        zoom[::grid_scale, :, :] = 195
        zoom[:, ::grid_scale, :] = 195
        cv2.imwrite(str(base.with_name(base.name + '-grid.png')),
                    cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
    entries = [
        {'rgb': '#%02X%02X%02X' % tuple(int(c) for c in color),
         'pixels': int(np.count_nonzero(ids == index))}
        for index, color in enumerate(centers)
    ]
    report = {
        'method': 'opencv-kmeans-rgb-7-on-resized-image',
        'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
        'mean_absolute_rgb_error': round(float(np.mean(
            np.abs(rgb.astype(np.float32) - mapped.astype(np.float32)))), 3),
        'colors': entries,
        'pixels': ids.reshape(rgb.shape[:2]).tolist(),
    }
    base.with_suffix('.json').write_text(
        json.dumps(report, separators=(',', ':')), encoding='utf-8')
    print('SEVEN-COLOR palette (computed, not manually identified):')
    for entry in entries:
        print(f"  {entry['rgb']}: {entry['pixels']:,} pixels")
    print(f"  Mean absolute RGB error: {report['mean_absolute_rgb_error']}")
    print(f'  Preview: {base.with_suffix(".png")}')
    print(f'  Indexed pixel map: {base.with_suffix(".json")}')



def compare_palette_sizes(rgb, output, sizes=(4, 6, 8, 12, 16)):
    """Compare data-derived palettes; no preset 'correct' number of colors."""
    samples = rgb.reshape(-1, 3).astype(np.float32)
    reference = rgb.astype(np.float32)
    # Strong source edges receive their own error metric, so broad white
    # backgrounds do not hide loss of small dark features.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120) > 0
    results = []
    for count in sizes:
        cv2.setRNGSeed(42)
        _, labels, centers = cv2.kmeans(
            samples, count, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
            5, cv2.KMEANS_PP_CENTERS)
        centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
        ids = labels.reshape(rgb.shape[:2])
        rendered = centers[ids]
        error = np.mean(np.abs(reference - rendered.astype(np.float32)), axis=2)
        counts = np.bincount(ids.reshape(-1), minlength=count)
        order = np.argsort(-counts)
        colors = [
            {'rgb': '#%02X%02X%02X' % tuple(map(int, centers[i])),
             'pixels': int(counts[i])}
            for i in order
        ]
        name = output.with_name(output.stem + f'-palette{count}')
        cv2.imwrite(str(name.with_suffix('.png')),
                    cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        results.append({
            'count': count,
            'mean_absolute_rgb_error': round(float(error.mean()), 4),
            'edge_mean_absolute_rgb_error': round(
                float(error[edges].mean()) if np.any(edges) else float(error.mean()), 4),
            'colors': colors,
            'preview': str(name.with_suffix('.png')),
        })
        print(f'{count:2} colors: RGB MAE={results[-1]["mean_absolute_rgb_error"]:.3f}, '
              f'edge MAE={results[-1]["edge_mean_absolute_rgb_error"]:.3f}')
    # This is a comparison report, not an ungrounded automatic declaration
    # that a specific palette preserves all visual details.
    report = {
        'method': 'kmeans-rgb-on-resized-image',
        'dimensions': [int(rgb.shape[1]), int(rgb.shape[0])],
        'note': 'Lower errors are better. Inspect previews for detail preservation; '
                'colors are not necessarily available as Paint swatches.',
        'results': results,
    }
    path = output.with_name(output.stem + '-palette-comparison.json')
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Palette comparison: {path}')



def map_sixteen_to_paint(rgb, output):
    """Preserve the 16 learned image colors; never silently substitute swatches.

    The local Paint palette has not been calibrated. This report gives the
    exact colors that Paint would need for a faithful 16-color drawing.
    """
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(
        pixels, 16, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
        5, cv2.KMEANS_PP_CENTERS)
    learned = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    ids = labels.reshape(-1)
    mapped = learned[ids].reshape(rgb.shape)
    counts = np.bincount(ids, minlength=16)
    order = np.argsort(-counts)
    rows = [
        {'rgb': '#%02X%02X%02X' % tuple(map(int, learned[i])),
         'pixels': int(counts[i])}
        for i in order
    ]
    base = output.with_name(output.stem + '-palette16-paint-mapping')
    cv2.imwrite(str(base.with_suffix('.png')),
                cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR))
    report = {
        'palette': '16-colors-learned-from-image-no-paint-substitution',
        'source_color_count': 16,
        'mapped_unique_color_count': int(len(np.unique(learned, axis=0))),
        'mean_absolute_rgb_error': round(float(np.mean(
            np.abs(rgb.astype(np.float32) - mapped.astype(np.float32)))), 4),
        'colors': rows,
        'paint_palette_verified': False,
        'warning': 'This is a faithful 16-color image preview, NOT a '
                   'verified existing-Paint-palette preview. Exact colors '
                   'may require custom swatches; verify the actual Paint '
                   'palette before connecting the drawing engine.',
    }
    base.with_suffix('.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('16 learned image colors (no forced yellow or other substitutions):')
    for row in rows:
        print(f"  {row['rgb']}: {row['pixels']:,} pixels")
    print(f"Distinct colors: {report['mapped_unique_color_count']}; "
          f"RGB MAE: {report['mean_absolute_rgb_error']}")
    print('Paint palette: NOT verified; preview uses source-derived colors.')
    print(f'Preview: {base.with_suffix(".png")}')
    print(f'Report: {base.with_suffix(".json")}')




def export_paint_plan_16(rgb, output):
    """Plan horizontal Paint strokes, grouped by 16 image-derived colors.

    Coordinates are relative to the requested image dimensions; no screen clicks are made.
    White is treated as a prefilled canvas, so only non-background runs draw.
    """
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(
        pixels, 16, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2),
        5, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    ids = labels.reshape(rgb.shape[:2])
    counts = np.bincount(ids.reshape(-1), minlength=16)
    background = int(np.argmin(np.sum(
        (centers.astype(np.int32) - np.array([255, 255, 255])) ** 2,
        axis=1)))
    groups = []
    for index in np.argsort(-counts):
        index = int(index)
        color = centers[index]
        strokes = []
        if index != background:
            for y, row in enumerate(ids):
                x = 0
                while x < len(row):
                    if int(row[x]) != index:
                        x += 1
                        continue
                    x0 = x
                    while x + 1 < len(row) and int(row[x + 1]) == index:
                        x += 1
                    strokes.append([x0, y, x, y])
                    x += 1
        groups.append({
            'rgb': '#%02X%02X%02X' % tuple(map(int, color)),
            'pixels': int(counts[index]),
            'background': index == background,
            'strokes': strokes,
        })
    plan = {
        'format': 'drawbuddy-paint-stroke-plan-v1',
        'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
        'coordinate_system': 'image-local-zero-based',
        'stroke_type': 'horizontal-inclusive-1px',
        'background': 'white-prefilled',
        'palette_verified_in_paint': False,
        'note': 'Requires setting each exact RGB color in Paint before drawing. '
                'This plan does not execute clicks or change the existing engine.',
        'groups': groups,
    }
    path = output.with_name(output.stem + '-paint-plan16.json')
    path.write_text(json.dumps(plan, separators=(',', ':')), encoding='utf-8')
    print(f'Paint 16-color plan: {path}')
    print(f'Colors: {len(groups)}; drawable strokes: '
          f'{sum(len(g["strokes"]) for g in groups):,}')
    print('Paint palette NOT calibrated; drawing engine unchanged.')


def build_pixel_map(source, output, width=500, height=547, grid_scale=4, paint_palette_16=False, palette_7=False, compare_palettes=False, map_paint_16=False, paint_plan_16=False):
    if width < 1 or height < 1 or grid_scale < 1:
        raise ValueError('Dimensions and grid scale must be positive')
    source = Path(source)
    output = Path(output)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read source image: {source}')
    src_h, src_w = image.shape[:2]
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_counts = Counter(tuple(map(int, pixel)) for pixel in original_rgb.reshape(-1, 3))
    print(f'ORIGINAL source: {src_w}x{src_h}; unique RGB colors: {len(original_counts)}')
    for color, count in original_counts.most_common(24):
        print(f'  #{color[0]:02X}{color[1]:02X}{color[2]:02X}: {count:,} pixels')
    if len(original_counts) > 24:
        print(f'  ... {len(original_counts) - 24} additional colors')
    factor = min(width / src_w, height / src_h)
    dest_w = max(1, min(width, round(src_w * factor)))
    dest_h = max(1, min(height, round(src_h * factor)))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (dest_w, dest_h), interpolation=interpolation)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x0, y0 = (width - dest_w) // 2, (height - dest_h) // 2
    canvas[y0:y0 + dest_h, x0:x0 + dest_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    if paint_plan_16:
        export_paint_plan_16(rgb, output)
    if map_paint_16:
        map_sixteen_to_paint(rgb, output)
    if compare_palettes:
        compare_palette_sizes(rgb, output)
    if palette_7:
        export_seven_color_preview(rgb, output, grid_scale)
    if paint_palette_16:
        export_paint_palette_preview(rgb, output, grid_scale)
    # Each entry is an explicit 6-digit RGB value. White is #FFFFFF.
    rows = [['#%02X%02X%02X' % tuple(int(c) for c in pixel)
             for pixel in row] for row in rgb]
    payload = {
        'format': 'drawbuddy-rgb-pixel-map-v1',
        'width': width, 'height': height,
        'source': str(source).replace('\\', '/'),
        'source_width': src_w, 'source_height': src_h,
        'content_box': {'x': x0, 'y': y0, 'width': dest_w, 'height': dest_h},
        'pixels': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False,
                                 separators=(',', ':')), encoding='utf-8')
    preview = output.with_suffix('.png')
    cv2.imwrite(str(preview), canvas)
    if grid_scale > 1:
        # Preview only: nearest-neighbor enlargement keeps each cell intact.
        enlarged = cv2.resize(canvas, (width * grid_scale, height * grid_scale),
                              interpolation=cv2.INTER_NEAREST)
        grid_color = (195, 195, 195)
        for x in range(0, width * grid_scale, grid_scale):
            enlarged[:, x] = grid_color
        for y in range(0, height * grid_scale, grid_scale):
            enlarged[y, :] = grid_color
        cv2.imwrite(str(output.with_name(output.stem + '-grid.png')), enlarged)
    print(f'Pixel map: {width}x{height} = {width * height:,} RGB cells')
    print(f'Source: {src_w}x{src_h}; fitted: {dest_w}x{dest_h}; origin: ({x0},{y0})')
    print(f'JSON: {output}')
    print(f'Image: {preview}')



def analyze_continuous_paths(plan_path, output_dir='output/pixel-map'):
    """Offline 4-connected same-color paths; never interact with Paint.

    Every path follows only pixels belonging to its own palette color.
    Tree traversal may revisit pixels; that cost is measured, not hidden.
    """
    from collections import deque

    data = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    width, height = data.get('width'), data.get('height')
    groups = data.get('groups')
    if (data.get('format') != 'drawbuddy-paint-stroke-plan-v1'
            or not isinstance(width, int) or not isinstance(height, int)
            or width < 1 or height < 1 or not isinstance(groups, list)):
        raise ValueError('Expected a valid pixel Paint plan')
    preview = np.full((height, width, 3), 255, dtype=np.uint8)
    output_groups = []
    original_strokes = 0
    original_length = 0
    path_count = 0
    path_length = 0
    repeated_pixels = 0
    for group in groups:
        rgb_hex = group['rgb']
        rgb = tuple(bytes.fromhex(rgb_hex.lstrip('#')))
        runs = group['strokes']
        if group['background']:
            if runs:
                raise ValueError('Background must not contain drawing runs')
            continue
        nodes = []
        by_row = [[] for _ in range(height)]
        expected = np.zeros((height, width), dtype=np.uint8)
        for x0, y0, x1, y1 in runs:
            if not (0 <= x0 <= x1 < width and 0 <= y0 == y1 < height):
                raise ValueError('Invalid horizontal run')
            if np.any(expected[y0, x0:x1 + 1]):
                raise ValueError('Overlapping runs in one color')
            expected[y0, x0:x1 + 1] = 1
            idx = len(nodes)
            nodes.append((x0, x1, y0))
            by_row[y0].append(idx)
            original_length += x1 - x0
        original_strokes += len(nodes)
        neighbors = [[] for _ in nodes]
        # Only overlap on adjacent rows: exact 4-neighbor connectivity.
        for y in range(1, height):
            above = sorted(by_row[y - 1], key=lambda i: nodes[i][0])
            below = sorted(by_row[y], key=lambda i: nodes[i][0])
            j = 0
            for a in above:
                ax0, ax1, _ = nodes[a]
                while j < len(below) and nodes[below[j]][1] < ax0:
                    j += 1
                k = j
                while k < len(below) and nodes[below[k]][0] <= ax1:
                    b = below[k]
                    overlap = max(ax0, nodes[b][0])
                    if overlap <= min(ax1, nodes[b][1]):
                        neighbors[a].append((b, overlap))
                        neighbors[b].append((a, overlap))
                    k += 1
        visited = set()
        color_paths = []
        actual = np.zeros_like(expected)
        for root in range(len(nodes)):
            if root in visited:
                continue
            visited.add(root)
            children = [[] for _ in nodes]
            queue = deque([root])
            while queue:
                parent = queue.popleft()
                for child, x in neighbors[parent]:
                    if child not in visited:
                        visited.add(child)
                        children[parent].append((child, x))
                        queue.append(child)
            x0, _, y0 = nodes[root]
            points = []
            def add(x, y):
                if points and points[-1] == [x, y]:
                    return
                if points:
                    px, py = points[-1]
                    if px != x and py != y:
                        raise AssertionError('Diagonal path segment')
                    cv2.line(actual, (px, py), (x, y), 1, 1)
                else:
                    actual[y, x] = 1
                points.append([x, y])
            # Iterative traversal prevents recursion overflow on tall images.
            stack = [('visit', root, x0)]
            while stack:
                op, node, anchor = stack.pop()
                x0, x1, y = nodes[node]
                if op == 'visit':
                    add(anchor, y)
                    add(x0, y)
                    add(x1, y)
                    add(anchor, y)
                    for child, cx in reversed(children[node]):
                        stack.append(('return', node, anchor))
                        stack.append(('visit', child, cx))
                        stack.append(('descend', child, cx))
                elif op == 'descend':
                    add(anchor, points[-1][1])
                    add(anchor, y)
                elif op == 'return':
                    # The child ends at its entry x; return to the parent row.
                    child_y = points[-1][1]
                    add(points[-1][0], y)
                    add(anchor, y)
            color_paths.append(points)
            path_count += 1
            path_length += sum(abs(a[0] - b[0]) + abs(a[1] - b[1])
                               for a, b in zip(points, points[1:]))
        missing = int(np.count_nonzero((expected == 1) & (actual == 0)))
        extra = int(np.count_nonzero((expected == 0) & (actual == 1)))
        if missing or extra:
            raise AssertionError(f'{rgb_hex}: coverage mismatch missing={missing}, extra={extra}')
        color_length = sum(sum(abs(a[0] - b[0]) + abs(a[1] - b[1])
                               for a, b in zip(path, path[1:])) for path in color_paths)
        repeated_pixels += max(0, color_length + len(color_paths) - int(expected.sum()))
        preview[expected.astype(bool)] = rgb[::-1]
        output_groups.append({'rgb': rgb_hex, 'paths': color_paths,
                              'components': len(color_paths),
                              'covered_pixels': int(expected.sum()),
                              'missing_pixels': missing, 'extra_pixels': extra})
    report = {
        'format': 'drawbuddy-continuous-path-analysis-v1',
        'source_plan': str(plan_path), 'width': width, 'height': height,
        'baseline_horizontal_strokes': original_strokes,
        'baseline_horizontal_distance_pixels': original_length,
        'continuous_paths': path_count,
        'continuous_distance_pixels': path_length,
        'distance_ratio_to_baseline': round(path_length / max(1, original_length), 3),
        'estimated_revisited_pixel_steps': repeated_pixels,
        'coverage': 'exact-4-connected-no-extra-pixels',
        'note': 'Offline comparison only. Paths retrace pixels and are not yet executed in Paint.',
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'pixel-map-continuous-paths.json').write_text(
        json.dumps({'report': report, 'groups': output_groups}, separators=(',', ':')),
        encoding='utf-8')
    (output / 'pixel-map-continuous-report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    cv2.imwrite(str(output / 'pixel-map-continuous-preview.png'), preview)
    print('CONTINUOUS PATH ANALYSIS (offline; Paint unchanged)')
    for key, value in report.items():
        print(f'  {key}: {value}')
    print(f'  Report: {output / "pixel-map-continuous-report.json"}')
    print(f'  Preview: {output / "pixel-map-continuous-preview.png"}')

def main():
    parser = argparse.ArgumentParser(description='Build a full-color pixel map')
    parser.add_argument('source', nargs='?', default='assets/dog-reference.png')
    parser.add_argument('--output', default='output/pixel-map/pixel-map.json')
    parser.add_argument('--width', type=int, default=500)
    parser.add_argument('--height', type=int, default=547)
    parser.add_argument('--grid-scale', type=int, default=4)
    parser.add_argument('--paint-palette-16', action='store_true',
                        help='Export comparison using 16 fixed Paint-style colors')
    parser.add_argument('--palette-7', action='store_true',
                        help='Generate an image and indexed pixel map with 7 computed colors')
    parser.add_argument('--compare-palettes', action='store_true',
                        help='Compare 4, 6, 8, 12, 16 color palettes with error metrics')
    parser.add_argument('--map-paint-16', action='store_true',
                        help='Export faithful 16-color preview and Paint color requirements')
    parser.add_argument('--paint-plan-16', action='store_true',
                        help='Export grouped 1px horizontal strokes for 16 image colors')
    parser.add_argument('--analyze-continuous', action='store_true',
                        help='Analyze continuous same-color paths from an existing 16-color Paint plan (no Paint)')
    args = parser.parse_args()
    if args.analyze_continuous:
        analyze_continuous_paths('output/pixel-map/pixel-map-paint-plan16.json')
        return
    build_pixel_map(args.source, args.output, args.width, args.height,
                    args.grid_scale, args.paint_palette_16, args.palette_7,
                    args.compare_palettes, args.map_paint_16,
                    args.paint_plan_16)


if __name__ == '__main__':
    main()
