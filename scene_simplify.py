"""Measured, non-destructive structural cleanup of quantized scene labels.

Never modifies the exact ownership baseline; rejects candidates exceeding budget.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scene_paths import boundary_edges, trace_network, remove_collinear


def simplify_scene(scene, output, max_changed_fraction=0.015):
    original = np.asarray(scene['labels'], np.uint8)
    palette = np.asarray(scene['palette'], np.uint8)
    h, w = original.shape
    # Categorical majority vote: never median-filter arbitrary palette indices.
    counts = np.stack([
        cv2.boxFilter((original == k).astype(np.float32), -1, (3, 3),
                      normalize=False, borderType=cv2.BORDER_REPLICATE)
        for k in range(len(palette))
    ], axis=0)
    winner = np.argmax(counts, axis=0).astype(np.uint8)
    strength = np.max(counts, axis=0)
    # Change only decisive local islands; avoid eroding thin intentional lines.
    candidate = np.where((strength >= 6) & (counts[original,
                         np.arange(h)[:, None], np.arange(w)[None, :]] <= 2),
                         winner, original).astype(np.uint8)
    changed = candidate != original
    fraction = float(np.mean(changed))
    accepted = fraction <= max_changed_fraction
    selected = candidate if accepted else original

    # A preview with reduced polyline vertices is NOT substituted for
    # the exact shared-boundary geometry until raster validation exists.
    paths = trace_network(boundary_edges(selected.astype(np.int32)))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>']
    preview = np.full((h + 1, w + 1, 3), 255, np.uint8)
    raw_vertices = simplified_vertices = 0
    for path in paths:
        raw = np.asarray(remove_collinear(path['points']), np.int32)
        if len(raw) < 2:
            continue
        raw_vertices += len(raw)
        closed = len(raw) >= 4 and np.array_equal(raw[0], raw[-1])
        approx = cv2.approxPolyDP(raw.reshape(-1, 1, 2).astype(np.float32),
                                 epsilon=1.0, closed=closed).reshape(-1, 2)
        if len(approx) < 2:
            approx = raw
        if closed and not np.array_equal(approx[0], approx[-1]):
            approx = np.vstack([approx, approx[0]])
        simplified_vertices += len(approx)
        coords = ' L '.join(f'{x:g} {y:g}' for x, y in approx)
        svg.append(f'<path d="M {coords}" fill="none" stroke="#df6b1d" stroke-width="0.7"/>')
        cv2.polylines(preview, [np.rint(approx).astype(np.int32)], False,
                      (25, 105, 220), 1, cv2.LINE_8)
    svg.append('</svg>')
    out = Path(output)
    (out / 'simplified-paths.svg').write_text('\n'.join(svg), encoding='utf-8')
    cv2.imwrite(str(out / 'simplified-paths-preview.png'), preview)
    cv2.imwrite(str(out / 'simplified-render.png'),
                cv2.cvtColor(palette[selected], cv2.COLOR_RGB2BGR))
    diff = np.full((h, w, 3), 255, np.uint8)
    diff[changed] = (0, 0, 255)
    cv2.imwrite(str(out / 'simplification-difference.png'), diff)
    report = {'candidate_changed_pixels': int(changed.sum()),
              'candidate_changed_fraction': round(fraction, 6),
              'max_changed_fraction': max_changed_fraction,
              'candidate_accepted': accepted,
              'raw_vertices': raw_vertices,
              'preview_vertices': simplified_vertices,
              'preview_only': True,
              'bezier_validated': False,
              'paint_ready': False}
    (out / 'simplification-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
