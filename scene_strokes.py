"""Trace diagnostic ink centerlines into strokes and measure variable-width reconstruction.

This is an offline experiment, not a Paint command generator.
"""
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(mask):
    """Consume each undirected skeleton edge once, including loops and branches."""
    coords = set(map(tuple, np.argwhere(mask > 0)))
    adj = {}
    for p in coords:
        y, x = p
        adj[p] = {q for dy, dx in NEIGHBORS
                  if (q := (y + dy, x + dx)) in coords}
    degrees = {p: len(qs) for p, qs in adj.items()}
    remaining = {tuple(sorted((p, q))) for p, qs in adj.items() for q in qs}

    def edge(a, b):
        return tuple(sorted((a, b)))

    paths = []

    def follow(start, nxt):
        path = [start]
        prev, current = start, nxt
        while True:
            e = edge(prev, current)
            if e not in remaining:
                break
            remaining.remove(e)
            path.append(current)
            if current == start or degrees[current] != 2:
                break
            choices = [p for p in adj[current] if edge(current, p) in remaining]
            if not choices:
                break
            prev, current = current, choices[0]
        paths.append(path)

    for p in sorted(coords):
        if degrees[p] != 2:
            for q in sorted(adj[p]):
                if edge(p, q) in remaining:
                    follow(p, q)
    while remaining:
        a, b = next(iter(remaining))
        follow(a, b)
    # Isolated pixels have no graph edges.
    for p in coords:
        if degrees[p] == 0:
            paths.append([p])
    return paths


def join_touching_paths(paths):
    """Join paths at identical endpoints only; never invent a bridging segment.

    A branch may be traversed more than once, but no new geometric edges
    are introduced. Deterministic pairing keeps diagnostic runs comparable.
    """
    paths = [list(p) for p in paths]
    while True:
        endpoints = defaultdict(list)
        for i, p in enumerate(paths):
            if len(p) > 1:
                endpoints[p[0]].append((i, True))
                endpoints[p[-1]].append((i, False))
        pair = None
        for point in sorted(endpoints):
            entries = sorted(set(endpoints[point]))
            distinct = sorted({i for i, _ in entries})
            if len(distinct) >= 2:
                i, j = distinct[:2]
                pair = (i, j, point)
                break
        if pair is None:
            break
        i, j, point = pair
        a, b = paths[i], paths[j]
        if a[0] == point:
            a = a[::-1]
        if b[-1] == point:
            b = b[::-1]
        assert a[-1] == b[0] == point
        paths[i] = a + b[1:]
        paths.pop(j)
    return paths


def reconstruct(paths, distance, ink, conservative=False):
    """Rasterize variable-width paths, optionally avoiding out-of-mask disks."""
    rebuilt = np.zeros_like(ink)
    records = []
    h, w = ink.shape
    for path in paths:
        pts = []
        for y, x in path:
            radius = max(1, int(round(float(distance[y, x]))))
            if conservative:
                # A disk must fit inside ink, including its image bounds.
                while radius > 1:
                    x0, x1 = max(0, x-radius), min(w, x+radius+1)
                    y0, y1 = max(0, y-radius), min(h, y+radius+1)
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    disk = (xx-x)**2 + (yy-y)**2 <= radius**2
                    if np.all(ink[y0:y1, x0:x1][disk]):
                        break
                    radius -= 1
            pts.append([int(x), int(y), radius])
            cv2.circle(rebuilt, (x, y), radius, 1, -1, cv2.LINE_8)
        records.append({'points': pts})
    return rebuilt, records


def fidelity(ink, rebuilt):
    union = (rebuilt != 0) | (ink != 0)
    intersection = (rebuilt != 0) & (ink != 0)
    return (float(intersection.sum() / max(1, union.sum())),
            int(np.count_nonzero((ink != 0) & (rebuilt == 0))),
            int(np.count_nonzero((ink == 0) & (rebuilt != 0))))


def export_strokes(ink, centerline, output):
    out = Path(output)
    ink = np.asarray(ink > 0, np.uint8)
    centerline = np.asarray(centerline > 0, np.uint8)
    original_paths = trace_skeleton(centerline)
    paths = join_touching_paths(original_paths)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    baseline, _ = reconstruct(original_paths, distance, ink)
    baseline_iou, baseline_missing, baseline_excess = fidelity(ink, baseline)
    candidate, candidate_records = reconstruct(paths, distance, ink, conservative=True)
    candidate_iou, _, _ = fidelity(ink, candidate)
    # Never regress the existing reconstruction score.
    if candidate_iou >= baseline_iou:
        rebuilt, records = candidate, candidate_records
        width_mode = 'ink-contained-radius'
    else:
        rebuilt, records = reconstruct(paths, distance, ink)
        width_mode = 'per-point-distance-transform-radius'
    iou, missing, excess = fidelity(ink, rebuilt)
    preview = np.full((*ink.shape, 3), 255, np.uint8)
    preview[rebuilt != 0] = (20, 20, 20)
    cv2.imwrite(str(out / 'stroke-reconstruction.png'),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    diff = np.full((*ink.shape, 3), 255, np.uint8)
    diff[(ink != 0) & (rebuilt == 0)] = (255, 0, 0)
    diff[(ink == 0) & (rebuilt != 0)] = (0, 0, 255)
    cv2.imwrite(str(out / 'stroke-difference.png'),
                cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    (out / 'ink-strokes.json').write_text(
        json.dumps({'width': int(ink.shape[1]), 'height': int(ink.shape[0]),
                    'paths': records}, separators=(',', ':')), encoding='utf-8')
    report = {'original_strokes': len(original_paths),
              'joined_strokes': len(paths),
              'baseline_iou': round(baseline_iou, 6),
              'baseline_missing_ink_pixels': baseline_missing,
              'baseline_excess_ink_pixels': baseline_excess,
              'strokes': len(records),
              'stroke_points': sum(len(p['points']) for p in records),
              'reconstruction_iou': round(iou, 6),
              'missing_ink_pixels': missing,
              'excess_ink_pixels': excess,
              'stroke_width_mode': width_mode,
              'strokes_are_diagnostic_only': True,
              'paint_ready': False}
    (out / 'stroke-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
