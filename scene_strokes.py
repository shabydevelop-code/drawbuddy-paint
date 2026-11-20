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
    """Pair shared endpoints in near-linear time without inventing edges.

    Every input path is used exactly once, including loops and isolated
    points. Branch junctions are paired deterministically; unmatched
    endpoints remain separate. Only original vertices are retained.
    """
    paths = [list(p) for p in paths]
    endpoints = defaultdict(list)
    for i, path in enumerate(paths):
        if len(path) > 1 and path[0] != path[-1]:
            endpoints[path[0]].append((i, 0))
            endpoints[path[-1]].append((i, 1))

    mates = {}
    for point in sorted(endpoints):
        entries = sorted(endpoints[point])
        for k in range(0, len(entries) - 1, 2):
            left, right = entries[k], entries[k + 1]
            mates[left] = right
            mates[right] = left

    visited = set()
    joined = []

    def collect(start_id, start_side):
        result = []
        current_id, entry_side = start_id, start_side
        while current_id not in visited:
            visited.add(current_id)
            path = paths[current_id]
            oriented = path if entry_side == 0 else path[::-1]
            if result:
                assert result[-1] == oriented[0]
                result.extend(oriented[1:])
            else:
                result.extend(oriented)
            exit_side = 1 - entry_side
            nxt = mates.get((current_id, exit_side))
            if nxt is None or nxt[0] in visited:
                break
            current_id, entry_side = nxt
        return result

    # Start open chains at unpaired endpoints, then consume closed chains.
    for i, path in enumerate(paths):
        if i in visited or len(path) <= 1 or path[0] == path[-1]:
            continue
        if (i, 0) not in mates:
            joined.append(collect(i, 0))
        elif (i, 1) not in mates:
            joined.append(collect(i, 1))
    for i, path in enumerate(paths):
        if i not in visited:
            if len(path) <= 1 or path[0] == path[-1]:
                visited.add(i)
                joined.append(path)
            else:
                joined.append(collect(i, 0))
    assert len(visited) == len(paths)
    assert sum(max(0, len(p) - 1) for p in joined) == (
        sum(max(0, len(p) - 1) for p in paths))
    return joined


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



def recover_missing_ink(ink, rebuilt, records, excess_budget=300):
    """Expand existing stroke points only when added coverage outweighs spill.

    No new paths are created. Changes are raster-validated locally and the
    global excess budget is enforced. One bounded pass avoids quadratic work.
    """
    rebuilt = rebuilt.copy()
    missing = int(np.count_nonzero((ink != 0) & (rebuilt == 0)))
    excess = int(np.count_nonzero((ink == 0) & (rebuilt != 0)))
    h, w = ink.shape
    if excess >= excess_budget:
        return rebuilt, records, 0
    recovered = 0
    for record in records:
        for point in record['points']:
            x, y, radius = point
            # Try a single radius increment; this keeps runtime bounded.
            new_radius = radius + 1
            x0, x1 = max(0, x-new_radius), min(w, x+new_radius+1)
            y0, y1 = max(0, y-new_radius), min(h, y+new_radius+1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (xx-x)**2 + (yy-y)**2 <= new_radius**2
            fresh = disk & (rebuilt[y0:y1, x0:x1] == 0)
            gained = int(np.count_nonzero(fresh & (ink[y0:y1, x0:x1] != 0)))
            spill = int(np.count_nonzero(fresh & (ink[y0:y1, x0:x1] == 0)))
            if gained == 0 or gained <= spill or excess + spill > excess_budget:
                continue
            rebuilt[y0:y1, x0:x1][disk] = 1
            point[2] = new_radius
            missing -= gained
            excess += spill
            recovered += gained
            if missing == 0 or excess >= excess_budget:
                return rebuilt, records, recovered
    return rebuilt, records, recovered



def recover_missing_segments(ink, rebuilt, records, limit=120,
                             excess_budget=300):
    """Prioritize zero-spill missing-ink dots, then budgeted positive-gain dots.

    Candidate centers come from missing pixels, not just component centroids.
    A 3x3 convolution ranks potential radius-one coverage. Each accepted dot
    is checked against the current raster, avoiding duplicate coverage.
    """
    missing = np.uint8((ink != 0) & (rebuilt == 0))
    if not np.any(missing):
        return rebuilt, records, 0
    disk_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                           dtype=np.uint8)
    gain_map = cv2.filter2D(missing, cv2.CV_16S, disk_kernel,
                            borderType=cv2.BORDER_CONSTANT)
    background = np.uint8(ink == 0)
    spill_map = cv2.filter2D(background, cv2.CV_16S, disk_kernel,
                             borderType=cv2.BORDER_CONSTANT)
    # Only try centers on missing ink; prefer zero-spill and high gain.
    ys, xs = np.nonzero(missing)
    candidates = sorted(zip(xs.tolist(), ys.tolist()),
                        key=lambda p: (int(spill_map[p[1], p[0]]) != 0,
                                       -int(gain_map[p[1], p[0]]),
                                       int(spill_map[p[1], p[0]]),
                                       p[1], p[0]))
    excess = int(np.count_nonzero((ink == 0) & (rebuilt != 0)))
    added = 0
    for x, y in candidates:
        if added >= limit:
            break
        x0, x1 = max(0, x-1), min(ink.shape[1], x+2)
        y0, y1 = max(0, y-1), min(ink.shape[0], y+2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx-x)**2 + (yy-y)**2 <= 1
        fresh = disk & (rebuilt[y0:y1, x0:x1] == 0)
        gain = int(np.count_nonzero(fresh & (ink[y0:y1, x0:x1] != 0)))
        spill = int(np.count_nonzero(fresh & (ink[y0:y1, x0:x1] == 0)))
        if gain <= spill or gain == 0 or excess + spill > excess_budget:
            continue
        cv2.circle(rebuilt, (x, y), 1, 1, -1, cv2.LINE_8)
        records.append({'points': [[x, y, 1]]})
        excess += spill
        added += 1
    return rebuilt, records, added


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
    before_recovery_iou, before_recovery_missing, before_recovery_excess = fidelity(ink, rebuilt)
    recovered_pixels = 0
    if width_mode == 'ink-contained-radius':
        rebuilt, records, recovered_pixels = recover_missing_ink(
            ink, rebuilt, records, excess_budget=max(300, before_recovery_excess))
    before_segments_iou, _, _ = fidelity(ink, rebuilt)
    original_rebuilt = rebuilt.copy()
    original_count = len(records)
    rebuilt, records, supplemental_strokes = recover_missing_segments(
        ink, rebuilt, records, excess_budget=max(300, before_recovery_excess))
    iou, missing, excess = fidelity(ink, rebuilt)
    if iou < before_segments_iou:
        rebuilt = original_rebuilt
        del records[original_count:]
        supplemental_strokes = 0
        iou, missing, excess = fidelity(ink, rebuilt)
    assert iou >= before_recovery_iou - 1e-12
    preview = np.full((*ink.shape, 3), 255, np.uint8)
    preview[rebuilt != 0] = (20, 20, 20)
    cv2.imwrite(str(out / 'stroke-reconstruction.png'),
                cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    diff = np.full((*ink.shape, 3), 255, np.uint8)
    diff[(ink != 0) & (rebuilt == 0)] = (255, 0, 0)
    diff[(ink == 0) & (rebuilt != 0)] = (0, 0, 255)
    cv2.imwrite(str(out / 'stroke-difference.png'),
                cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    # Lossless source-ink horizontal runs for Paint fill correction.
    # Centerlines alone cannot reproduce solid spots or heavy mouth strokes.
    ink_runs = []
    for y, row in enumerate(ink):
        starts = np.flatnonzero((row != 0) & (np.r_[0, row[:-1]] == 0))
        ends = np.flatnonzero((row != 0) & (np.r_[row[1:], 0] == 0))
        ink_runs.extend([int(y), int(a), int(b)] for a, b in zip(starts, ends))
    (out / 'ink-strokes.json').write_text(
        json.dumps({'width': int(ink.shape[1]), 'height': int(ink.shape[0]),
                    'paths': records, 'ink_runs': ink_runs},
                   separators=(',', ':')), encoding='utf-8')
    report = {'original_strokes': len(original_paths),
              'joined_strokes': len(paths),
              'baseline_iou': round(baseline_iou, 6),
              'baseline_missing_ink_pixels': baseline_missing,
              'baseline_excess_ink_pixels': baseline_excess,
              'pre_recovery_iou': round(before_recovery_iou, 6),
              'pre_recovery_missing_ink_pixels': before_recovery_missing,
              'pre_recovery_excess_ink_pixels': before_recovery_excess,
              'recovered_ink_pixels': recovered_pixels,
              'pre_supplemental_iou': round(before_segments_iou, 6),
              'supplemental_strokes': supplemental_strokes,
              'source_ink_runs': len(ink_runs),
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
