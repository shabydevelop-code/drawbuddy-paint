"""Scene Engine v1: deterministic vector scene, offline render and fidelity report.

No mouse control. Preserve source resolution and original color samples.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def extract_scene(source, colors=24):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read source: {source}')
    height, width = image.shape[:2]
    # Quantize colors for closed region topology. Never resize source.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    samples = rgb.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    _, labels, palette = cv2.kmeans(
        samples, colors, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.25),
        3, cv2.KMEANS_PP_CENTERS)
    label_image = labels.reshape(height, width)
    palette = np.clip(np.rint(palette), 0, 255).astype(np.uint8)
    shapes = []
    shape_masks = []
    for color_index in range(colors):
        mask = np.uint8(label_image == color_index) * 255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        # Each connected outer component becomes one shape with holes.
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] != -1:
                continue
            if cv2.contourArea(contour) < 1:
                continue
            rings = [contour[:, 0, :].tolist()]
            child = int(hierarchy[0, i, 2])
            while child != -1:
                rings.append(contours[child][:, 0, :].tolist())
                child = int(hierarchy[0, child, 0])
            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, contours, i, 255, -1)
            child_mask = int(hierarchy[0, i, 2])
            while child_mask != -1:
                cv2.drawContours(component_mask, contours, child_mask, 0, -1)
                child_mask = int(hierarchy[0, child_mask, 0])
            shape_masks.append(component_mask)
            shapes.append({'palette_index': color_index,
                           'color': palette[color_index].tolist(),
                           'rings': rings,
                           'area': float(cv2.contourArea(contour))})
    # Keep a lossless pixel ownership layer for precise validation and
    # future shared-edge extraction; curves alone cannot ensure exactness.
    return {'width': width, 'height': height, 'palette': palette.tolist(),
            'labels': label_image.tolist(), 'shapes': shapes,
            'source': str(source)}


def render_scene(scene):
    h, w = scene['height'], scene['width']
    palette = np.asarray(scene['palette'], np.uint8)
    labels = np.asarray(scene['labels'], np.int32)
    if labels.shape != (h, w) or labels.min() < 0 or labels.max() >= len(palette):
        raise ValueError('Scene label map is invalid')
    return cv2.cvtColor(palette[labels], cv2.COLOR_RGB2BGR)


def validate_vector_geometry(scene):
    """Rasterize actual shape rings and compare to exact color ownership.

    Reports vector coverage separately from palette fidelity. No inference
    from a pixel-label render is counted as vector success.
    """
    labels = np.asarray(scene['labels'], dtype=np.int32)
    h, w = labels.shape
    vector_masks = [np.zeros((h, w), np.uint8)
                    for _ in scene['palette']]
    for shape in scene['shapes']:
        rings = shape['rings']
        if not rings:
            continue
        mask = np.zeros((h, w), np.uint8)
        outer = np.asarray(rings[0], np.int32)
        if len(outer) < 3:
            continue
        cv2.fillPoly(mask, [outer], 255)
        for hole in rings[1:]:
            hole = np.asarray(hole, np.int32)
            if len(hole) >= 3:
                cv2.fillPoly(mask, [hole], 0)
        vector_masks[shape['palette_index']] |= mask
    # Count ambiguous overlaps as failures rather than hiding them via
    # an arbitrary draw order.
    votes = np.stack([(m != 0) for m in vector_masks], axis=0)
    coverage = votes.sum(axis=0)
    predicted = votes.argmax(axis=0)
    correct = (coverage == 1) & (predicted == labels)
    accuracy = float(correct.mean())
    missing = int(np.count_nonzero(coverage == 0))
    overlaps = int(np.count_nonzero(coverage > 1))
    per_color = []
    for index in range(len(vector_masks)):
        actual = labels == index
        proposed = votes[index]
        union = np.count_nonzero(actual | proposed)
        intersection = np.count_nonzero(actual & proposed)
        per_color.append({'index': index,
                          'iou': round(intersection / union, 5) if union else 1.0})
    diagnostic = np.full((h, w, 3), 255, np.uint8)
    diagnostic[(coverage == 1) & ~correct] = (0, 165, 255)
    diagnostic[coverage == 0] = (0, 0, 255)
    diagnostic[coverage > 1] = (255, 0, 255)
    return {'exact_pixel_coverage': round(accuracy, 6),
            'uncovered_pixels': missing, 'overlap_pixels': overlaps,
            'per_color': per_color}, diagnostic


def svg_scene(scene, output):
    # SVG is an inspectable vector proposal; the label map is authoritative.
    from xml.sax.saxutils import escape
    w, h = scene['width'], scene['height']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for shape in sorted(scene['shapes'], key=lambda s: -s['area']):
        r, g, b = shape['color']
        rings = []
        for ring in shape['rings']:
            if len(ring) < 3:
                continue
            rings.append('M ' + ' L '.join(f'{x} {y}' for x, y in ring) + ' Z')
        if rings:
            parts.append(f'<path d="{" ".join(rings)}" fill="#{r:02x}{g:02x}{b:02x}" fill-rule="evenodd"/>')
    parts.append('</svg>')
    output.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Build and validate a drawing scene; does not open Paint')
    parser.add_argument('source', type=Path)
    parser.add_argument('--colors', type=int, default=24)
    args = parser.parse_args()
    if not 2 <= args.colors <= 64:
        parser.error('--colors must be between 2 and 64')
    source = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    scene = extract_scene(args.source, args.colors)
    output = Path('output/scene-engine')
    output.mkdir(parents=True, exist_ok=True)
    render = render_scene(scene)
    cv2.imwrite(str(output / 'render.png'), render)
    delta = cv2.absdiff(source, render)
    cv2.imwrite(str(output / 'difference.png'), delta)
    svg_scene(scene, output / 'scene.svg')
    vector_metrics, vector_difference = validate_vector_geometry(scene)
    cv2.imwrite(str(output / 'vector-difference.png'), vector_difference)
    # Avoid enormous JSON arrays: save the exact pixel labels separately.
    labels = np.asarray(scene.pop('labels'), np.uint8)
    np.savez_compressed(output / 'ownership.npz', labels=labels,
                        palette=np.asarray(scene['palette'], np.uint8))
    (output / 'scene.json').write_text(json.dumps(scene, indent=2), encoding='utf-8')
    report = {'source': str(args.source), 'size': [scene['width'], scene['height']],
              'colors': args.colors, 'shapes': len(scene['shapes']),
              'mean_absolute_error': round(float(delta.mean()), 3),
              'max_channel_error': int(delta.max()),
              'vector_geometry': vector_metrics,
              'paint_ready': False,
              'note': 'Raster color fidelity and contour geometry are measured independently; Paint execution is not yet validated.'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Files:', output)


if __name__ == '__main__':
    main()
