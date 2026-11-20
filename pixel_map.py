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


def build_pixel_map(source, output, width=300, height=328, grid_scale=4):
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
    args = parser.parse_args()
    build_pixel_map(args.source, args.output, args.width, args.height,
                    args.grid_scale)


if __name__ == '__main__':
    main()
