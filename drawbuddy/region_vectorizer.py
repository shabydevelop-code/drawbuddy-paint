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

    # Shared boundaries are pixel adjacencies; each interface is recorded
    # once, irrespective of which two regions own it.
    adjacency = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = label_map[:h-dy, :w-dx]
        b = label_map[dy:, dx:]
        ys, xs = np.where((a != b) & (a > 0) & (b > 0))
        for y, x in zip(ys, xs):
            pair = tuple(sorted((int(a[y, x]), int(b[y, x]))))
            adjacency[pair] = adjacency.get(pair, 0) + 1
    shared = [{'regions': list(pair), 'contact_pixels': n}
              for pair, n in sorted(adjacency.items())]

    # Preview: region boundaries and source ink, without painting over regions.
    preview = np.full((h, w, 3), 255, np.uint8)
    palette = {'cream': (170, 200, 220), 'orange': (0, 130, 255),
               'brown': (70, 100, 160), 'black': (0, 0, 0)}
    for region in sorted(regions, key=lambda r: -r['area']):
        for boundary in region['boundaries']:
            contour = np.asarray(boundary, np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [contour], True, palette[region['color']], 1)
    preview[black] = (0, 0, 0)
    out = Path('output')
    out.mkdir(parents=True, exist_ok=True)
    preview_file = out / 'region-boundaries-preview.png'
    data_file = out / 'region-map.json'
    if not cv2.imwrite(str(preview_file), preview):
        raise OSError(f'Failed to write {preview_file}')
    data = {'source': str(source), 'width': w, 'height': h,
            'regions': regions, 'shared_boundaries': shared,
            'paint_ready': False, 'bucket_fill_enabled': False}
    data_file.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding='utf-8')
    return {'regions': len(regions), 'shared_boundaries': len(shared),
            'preview': str(preview_file), 'map': str(data_file),
            'paint_ready': False}
