"""Region-first vectorization preview. No Paint operations or bucket fills."""
from pathlib import Path
import json
import cv2
import numpy as np


def analyze_regions(source, max_width=240, max_height=180):
    source = Path(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read source: {source}')
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    w, h = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    # Preserve explicit dark ink separately from chromatic fur and light fur.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black = gray < 105
    colorful = (hsv[:, :, 1] >= 65) & (hsv[:, :, 2] >= 90) & ~black
    orange = colorful & (hsv[:, :, 2] >= 155)
    brown = colorful & ~orange
    cream = (~black & ~colorful & (gray < 248))
    classes = np.zeros((h, w), np.uint8)
    classes[cream] = 1
    classes[orange] = 2
    classes[brown] = 3
    classes[black] = 4

    # Segment connected regions independently, retaining every boundary.
    regions = []
    label_map = np.zeros((h, w), np.int32)
    next_id = 1
    for color_id, color_name in ((1, 'cream'), (2, 'orange'),
                                 (3, 'brown'), (4, 'black')):
        mask = (classes == color_id).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 3:
                continue
            region_mask = (labels == index).astype(np.uint8)
            contours, hierarchy = cv2.findContours(region_mask, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            boundaries = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                points = [[int(p[0][0]), int(p[0][1])] for p in contour]
                boundaries.append(points)
            if not boundaries:
                continue
            label_map[region_mask != 0] = next_id
            regions.append({'id': next_id, 'color': color_name,
                            'area': area, 'center': [float(v) for v in centers[index]],
                            'boundaries': boundaries})
            next_id += 1

    # Build a single shared edge network on the pixel-cell lattice.
    # Each interface between two cells is recorded once, regardless of
    # which region owns it. The outside boundary is against region 0.
    edges = {}
    def add_edge(a, b, first, second):
        if a == b:
            return
        pair = tuple(sorted((int(a), int(b))))
        edges.setdefault(pair, set()).add((first, second))

    for y in range(h):
        for x in range(w):
            region = int(label_map[y, x])
            if x == 0:
                add_edge(region, 0, (x, y), (x, y + 1))
            if y == 0:
                add_edge(region, 0, (x, y), (x + 1, y))
            right = int(label_map[y, x + 1]) if x + 1 < w else 0
            below = int(label_map[y + 1, x]) if y + 1 < h else 0
            add_edge(region, right, (x + 1, y), (x + 1, y + 1))
            add_edge(region, below, (x, y + 1), (x + 1, y + 1))

    # Turn shared edges into continuous polylines. Every edge is consumed
    # exactly once; junctions end a path instead of inventing connections.
    def trace_edges(edge_set):
        adjacency = {}
        for a, b in edge_set:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        unused = set(edge_set)
        paths = []
        def consume(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in unused:
                return False
            unused.remove(key)
            return True
        while unused:
            endpoints = sorted(v for v, n in adjacency.items()
                               if len(n) != 2 and any(
                                   ((v, q) if v < q else (q, v)) in unused
                                   for q in n))
            start = endpoints[0] if endpoints else min(min(e) for e in unused)
            options = sorted(q for q in adjacency[start]
                             if ((start, q) if start < q else (q, start)) in unused)
            if not options:
                raise RuntimeError('Shared boundary graph contains an orphan edge')
            nxt = options[0]
            consume(start, nxt)
            path = [start, nxt]
            while True:
                here = path[-1]
                if len(adjacency[here]) != 2:
                    break
                options = sorted(q for q in adjacency[here]
                                 if ((here, q) if here < q else (q, here)) in unused)
                if not options:
                    break
                q = options[0]
                consume(here, q)
                path.append(q)
                if q == start:
                    break
            paths.append([list(point) for point in path])
        return paths

    shared = []
    for pair, edge_set in sorted(edges.items()):
        paths = trace_edges(edge_set)
        shared.append({'regions': list(pair), 'edge_count': len(edge_set),
                       'paths': paths})
    if sum(len(p['paths']) for p in shared) == 0:
        raise ValueError('No shared boundary paths detected')

    # Render each shared interface once. Exterior edges first, interior
    # edges second, source black ink last. This preview is NOT a fill plan.
    preview = np.full((h + 1, w + 1, 3), 255, np.uint8)
    for item in sorted(shared, key=lambda e: (0 not in e['regions'], e['regions'])):
        for path in item['paths']:
            if len(path) >= 2:
                cv2.polylines(preview, [np.asarray(path, np.int32)], False,
                              (0, 110, 210) if 0 in item['regions'] else
                              (110, 110, 110), 1, lineType=cv2.LINE_8)
    preview[:h, :w][black] = (0, 0, 0)
    out = Path('output')
    out.mkdir(parents=True, exist_ok=True)
    preview_file = out / 'region-boundaries-preview.png'
    data_file = out / 'region-map.json'
    network_file = out / 'region-shared-boundaries-preview.png'
    if not cv2.imwrite(str(network_file), preview):
        raise OSError(f'Failed to write {network_file}')
    if not cv2.imwrite(str(preview_file), preview):
        raise OSError(f'Failed to write {preview_file}')
    data = {'source': str(source), 'width': w, 'height': h,
            'regions': regions, 'shared_boundaries': shared,
            'paint_ready': False, 'bucket_fill_enabled': False}
    data_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    return {'regions': len(regions), 'shared_boundaries': len(shared),
            'preview': str(preview_file), 'network_preview': str(network_file),
            'map': str(data_file),
            'paint_ready': False}
