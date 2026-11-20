"""Scene Engine v1: deterministic vector scene, offline render and fidelity report.

No mouse control. Preserve source resolution and original color samples.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scene_paths import export_paths


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


def pixel_cell_geometry(scene):
    """Exact nonoverlapping pixel-cell rectangles, merged into horizontal runs.

    Coordinates describe cell edges, not OpenCV contour pixel centers.
    Each pixel belongs to one and only one palette color.
    """
    labels = np.asarray(scene['labels'], dtype=np.int32)
    h, w = labels.shape
    runs = []
    for y in range(h):
        row = labels[y]
        starts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
        ends = np.r_[starts[1:], w]
        for x0, x1 in zip(starts, ends):
            runs.append((int(x0), int(y), int(x1 - x0), int(row[x0])))
    return runs


def validate_vector_geometry(scene, runs):
    """Validate the actual SVG rectangle geometry, not a label-map render."""
    labels = np.asarray(scene['labels'], np.int32)
    h, w = labels.shape
    reconstructed = np.full((h, w), -1, np.int32)
    coverage = np.zeros((h, w), np.uint16)
    for x, y, length, color in runs:
        if not (0 <= x < w and 0 <= y < h and
                0 < length <= w - x and 0 <= color < len(scene['palette'])):
            raise ValueError('Invalid vector cell geometry')
        coverage[y, x:x + length] += 1
        reconstructed[y, x:x + length] = color
    correct = (coverage == 1) & (reconstructed == labels)
    missing = int(np.count_nonzero(coverage == 0))
    overlaps = int(np.count_nonzero(coverage > 1))
    mismatch = int(np.count_nonzero(~correct))
    diagnostic = np.full((h, w, 3), 255, np.uint8)
    diagnostic[(coverage == 1) & ~correct] = (0, 165, 255)
    diagnostic[coverage == 0] = (0, 0, 255)
    diagnostic[coverage > 1] = (255, 0, 255)
    return {'exact_pixel_coverage': round(float(correct.mean()), 6),
            'uncovered_pixels': missing, 'overlap_pixels': overlaps,
            'mismatched_pixels': mismatch,
            'geometry_type': 'exact_pixel_cells',
            'paint_ready': False}, diagnostic


def svg_scene(scene, runs, output):
    """Emit gap-free SVG from exact pixel-cell edges.

    SVG is intentionally unsmoothed. Smooth curves will be a separate,
    explicitly measured approximation rather than silently losing fidelity.
    """
    w, h = scene['width'], scene['height']
    palette = scene['palette']
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" shape-rendering="crispEdges">']
    # Adjacent runs tile the canvas exactly; no black borders are invented.
    for x, y, length, color in runs:
        r, g, b = palette[color]
        parts.append(f'<rect x="{x}" y="{y}" width="{length}" height="1" fill="#{r:02x}{g:02x}{b:02x}"/>')
    parts.append('</svg>')
    output.write_text('\\n'.join(parts), encoding='utf-8')


def shared_boundary_report(scene):
    """Count each inter-color pixel edge once, including the canvas perimeter."""
    labels = np.asarray(scene['labels'], np.int32)
    h, w = labels.shape
    horizontal = int(np.count_nonzero(labels[1:, :] != labels[:-1, :]))
    vertical = int(np.count_nonzero(labels[:, 1:] != labels[:, :-1]))
    return {'internal_unique_edges': horizontal + vertical,
            'canvas_perimeter_edges': 2 * (w + h),
            'boundary_coordinate_system': 'integer_pixel_cell_edges'}



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
    runs = pixel_cell_geometry(scene)
    svg_scene(scene, runs, output / 'scene.svg')
    vector_metrics, vector_difference = validate_vector_geometry(scene, runs)
    path_metrics = export_paths(scene, output)
    cv2.imwrite(str(output / 'vector-difference.png'), vector_difference)
    # Avoid enormous JSON arrays: save the exact pixel labels separately.
    labels = np.asarray(scene['labels'], np.uint8)
    np.savez_compressed(output / 'ownership.npz', labels=labels,
                        palette=np.asarray(scene['palette'], np.uint8))
    scene_metadata = {key: value for key, value in scene.items() if key != 'labels'}
    (output / 'scene.json').write_text(json.dumps(scene_metadata, indent=2), encoding='utf-8')
    report = {'source': str(args.source), 'size': [scene['width'], scene['height']],
              'colors': args.colors, 'shapes': len(scene['shapes']),
              'mean_absolute_error': round(float(delta.mean()), 3),
              'max_channel_error': int(delta.max()),
              'vector_geometry': vector_metrics,
              'path_network': path_metrics,
              'shared_boundary_network': shared_boundary_report(scene),
              'vector_runs': len(runs),
              'paint_ready': False,
              'note': 'SVG is exact pixel-cell geometry, not yet smoothed into drawable curves. Paint execution is disabled.'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Files:', output)


if __name__ == '__main__':
    main()
