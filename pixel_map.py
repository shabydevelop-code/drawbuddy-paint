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


def build_pixel_map(source, output, width=300, height=328, grid_scale=4, paint_palette_16=False, palette_7=False, compare_palettes=False):
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


def main():
    parser = argparse.ArgumentParser(description='Build a full-color pixel map')
    parser.add_argument('source', nargs='?', default='assets/dog-reference.png')
    parser.add_argument('--output', default='output/pixel-map/pixel-map.json')
    parser.add_argument('--width', type=int, default=300)
    parser.add_argument('--height', type=int, default=328)
    parser.add_argument('--grid-scale', type=int, default=4)
    parser.add_argument('--paint-palette-16', action='store_true',
                        help='Export comparison using 16 fixed Paint-style colors')
    parser.add_argument('--palette-7', action='store_true',
                        help='Generate an image and indexed pixel map with 7 computed colors')
    parser.add_argument('--compare-palettes', action='store_true',
                        help='Compare 4, 6, 8, 12, 16 color palettes with error metrics')
    args = parser.parse_args()
    build_pixel_map(args.source, args.output, args.width, args.height,
                    args.grid_scale, args.paint_palette_16, args.palette_7,
                    args.compare_palettes)


if __name__ == '__main__':
    main()
