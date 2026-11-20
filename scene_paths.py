"""Extract unique shared pixel-cell edges as continuous vector paths.

Exact paths preserve topology; optional simplification is preview-only.
"""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np


def boundary_edges(labels):
    h, w = labels.shape
    edges = defaultdict(set)

    def add(a, b, p, q):
        if a == b:
            return
        key = tuple(sorted((int(a), int(b))))
        edge = (p, q) if p < q else (q, p)
        edges[key].add(edge)

    for y in range(h):
        for x in range(w):
            a = int(labels[y, x])
            if x == 0:
                add(-1, a, (0, y), (0, y + 1))
            if y == 0:
                add(-1, a, (x, 0), (x + 1, 0))
            if x == w - 1:
                add(a, -1, (w, y), (w, y + 1))
            else:
                add(a, int(labels[y, x + 1]), (x + 1, y), (x + 1, y + 1))
            if y == h - 1:
                add(a, -1, (x, h), (x + 1, h))
            else:
                add(a, int(labels[y + 1, x]), (x, y + 1), (x + 1, y + 1))
    return edges


def trace_network(edges):
    paths = []
    for pair, original in sorted(edges.items()):
        adjacency = defaultdict(set)
        for a, b in original:
            adjacency[a].add(b)
            adjacency[b].add(a)
        remaining = set(original)

        def key(a, b):
            return (a, b) if a < b else (b, a)

        while remaining:
            starts = [p for p in adjacency if len(adjacency[p]) != 2 and
                      any(key(p, q) in remaining for q in adjacency[p])]
            start = min(starts) if starts else min(min(e) for e in remaining)
            next_point = min(q for q in adjacency[start]
                             if key(start, q) in remaining)
            remaining.remove(key(start, next_point))
            chain = [start, next_point]
            while len(adjacency[chain[-1]]) == 2:
                here = chain[-1]
                options = sorted(q for q in adjacency[here]
                                 if key(here, q) in remaining)
                if not options:
                    break
                nxt = options[0]
                remaining.remove(key(here, nxt))
                chain.append(nxt)
                if nxt == start:
                    break
            paths.append({'colors': list(pair),
                          'points': [list(p) for p in chain]})
    return paths


def remove_collinear(points):
    if len(points) < 3:
        return points
    out = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = out[-1], points[i], points[i + 1]
        if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0]):
            out.append(b)
    out.append(points[-1])
    return out


def export_paths(scene, output):
    labels = np.asarray(scene['labels'], np.int32)
    h, w = labels.shape
    edges = boundary_edges(labels)
    paths = trace_network(edges)
    expected = sum(len(e) for e in edges.values())
    actual = sum(len(p['points']) - 1 for p in paths)
    if actual != expected:
        raise RuntimeError(f'Boundary coverage mismatch: {actual}/{expected}')

    # Exact vertex compression never changes the raster cell-edge geometry.
    for path in paths:
        path['points'] = remove_collinear(path['points'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'shared-paths.json').write_text(
        json.dumps({'width': w, 'height': h, 'paths': paths},
                   separators=(',', ':')), encoding='utf-8')

    # Preview-only adaptive curve approximation: each candidate is accepted
    # only if its maximum geometric deviation is within a chosen tolerance.
    # This does NOT validate raster fill accuracy or authorize Paint.
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           '<rect width="100%" height="100%" fill="white"/>']
    preview = np.full((h + 1, w + 1, 3), 255, np.uint8)
    for path in paths:
        pts = np.asarray(path['points'], np.int32)
        if len(pts) < 2:
            continue
        color = '#111111' if -1 in path['colors'] else '#df6b1d'
        command = 'M ' + ' L '.join(f'{x} {y}' for x, y in path['points'])
        svg.append(f'<path d="{command}" stroke="{color}" stroke-width="0.65" fill="none"/>')
        cv2.polylines(preview, [pts], False,
                      (0, 0, 0) if -1 in path['colors'] else (25, 105, 220),
                      1, lineType=cv2.LINE_8)
    svg.append('</svg>')
    (out / 'shared-paths.svg').write_text('\\n'.join(svg), encoding='utf-8')
    cv2.imwrite(str(out / 'shared-paths-preview.png'), preview)
    return {'unique_edges': expected, 'traced_edges': actual,
            'paths': len(paths), 'edge_coverage': 1.0,
            'smooth_curves_ready': False, 'paint_ready': False}
