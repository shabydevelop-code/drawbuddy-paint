"""Draw shared region boundaries in Paint; no automatic bucket filling."""
import json
from pathlib import Path
import cv2
import numpy as np
from .region_vectorizer import analyze_regions


def build_region_outline_drawing(source):
    report = analyze_regions(source)
    data = json.loads(Path(report['map']).read_text(encoding='utf-8'))
    w, h = data['width'], data['height']
    left = 310 + (240 - w) // 2
    top = 155 + (180 - h) // 2
    region_by_id = {r['id']: r for r in data['regions']}
    # An interface belongs to exactly one region pair; draw it once.
    outside, inside = [], []
    seen_edges = set()
    for interface in data['shared_boundaries']:
        a, b = interface['regions']
        target = outside if 0 in (a, b) else inside
        for path in interface['paths']:
            run = []
            for p, q in zip(path, path[1:]):
                u, v = tuple(p), tuple(q)
                edge = (u, v) if u < v else (v, u)
                if edge in seen_edges:
                    if len(run) >= 2:
                        target.append(run)
                    run = []
                    continue
                seen_edges.add(edge)
                if not run:
                    run = [list(u)]
                elif run[-1] != list(u):
                    if len(run) >= 2:
                        target.append(run)
                    run = [list(u)]
                run.append(list(v))
            if len(run) >= 2:
                target.append(run)

    # Preserve black facial and interior details as a separate final pass,
    # but avoid their thick double borders: use the centerline tracer.
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA if image.shape[1] > w else cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    from .centerline_painter import _thin, _trace_all
    black = (gray < 105).astype(np.uint8)
    # Shared-region boundaries own their drawn pixels. Remove ONLY ink
    # centerline segments already covered by those planned strokes, rather
    # than redrawing the same outline in the facial-detail pass.
    owned = np.zeros((h, w), dtype=np.uint8)
    for path in outside + inside:
        points = np.array(path, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(owned, [points], False, 255, 1,
                          lineType=cv2.LINE_8)
    skeleton = _thin(black)
    # A one-pixel tolerance accounts for cell-edge coordinates vs ink
    # centerlines. It is deliberately local: other nearby facial strokes
    # remain available for drawing.
    covered = cv2.dilate(owned, np.ones((3, 3), np.uint8))
    remaining = skeleton.copy()
    remaining[covered != 0] = 0
    ink = _trace_all(remaining, left, top) if np.any(remaining) else []
    removed_ink_pixels = int(np.count_nonzero(skeleton) -
                             np.count_nonzero(remaining))
    steps = []
    for name, paths in [('Exterior silhouette', outside),
                        ('Shared interior region boundaries', inside)]:
        if paths:
            steps.append({'name': name, 'tool': 'brush_paths',
                          'color': 'black', 'width': 1,
                          'paths': [[[left + x, top + y] for x, y in path]
                                   for path in paths]})
    if ink:
        steps.append({'name': 'Black source details', 'tool': 'brush_paths',
                      'color': 'black', 'width': 1, 'paths': ink})
    if not steps:
        raise ValueError('No drawable region boundaries')
    # Diagnostic only: do not infer safe fills from a preview.
    return {'name': f'Region boundaries {Path(source).stem}',
            'region_drawing': {'outer_paths': len(outside),
                               'inner_paths': len(inside),
                               'ink_paths': len(ink),
                               'removed_duplicate_ink_pixels': removed_ink_pixels,
                               'shared_edges': len(seen_edges),
                               'bucket_fill_enabled': False},
            'steps': steps}
