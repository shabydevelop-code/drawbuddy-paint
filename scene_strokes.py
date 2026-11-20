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


def export_strokes(ink, centerline, output):
    out = Path(output)
    ink = np.asarray(ink > 0, np.uint8)
    centerline = np.asarray(centerline > 0, np.uint8)
    paths = trace_skeleton(centerline)
    # Radius estimated from original ink mask at every centerline point.
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    rebuilt = np.zeros_like(ink)
    records = []
    for path in paths:
        pts = []
        for y, x in path:
            radius = max(1, int(round(float(distance[y, x]))))
            pts.append([int(x), int(y), radius])
            cv2.circle(rebuilt, (x, y), radius, 1, -1, cv2.LINE_8)
        records.append({'points': pts})
    union = (rebuilt != 0) | (ink != 0)
    intersection = (rebuilt != 0) & (ink != 0)
    iou = float(intersection.sum() / max(1, union.sum()))
    missing = int(np.count_nonzero((ink != 0) & (rebuilt == 0)))
    excess = int(np.count_nonzero((ink == 0) & (rebuilt != 0)))
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
    report = {'strokes': len(records),
              'stroke_points': sum(len(p['points']) for p in records),
              'reconstruction_iou': round(iou, 6),
              'missing_ink_pixels': missing,
              'excess_ink_pixels': excess,
              'stroke_width_mode': 'per-point-distance-transform-radius',
              'strokes_are_diagnostic_only': True,
              'paint_ready': False}
    (out / 'stroke-validation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    return report
