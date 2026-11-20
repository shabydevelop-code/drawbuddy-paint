"""Order traced dog strokes by anatomical zones and nearby endpoints.

This is a deterministic, geometry-based heuristic, not image understanding.
Paths are never split or modified; only their execution order/direction changes.
"""
from copy import deepcopy
from math import dist


class DrawingSequencePlanner:
    """Group head-to-tail, then minimize pen-up travel inside each group."""

    ZONES = ("head", "ears", "face", "collar", "body", "legs", "tail")

    @staticmethod
    def zone(point, bounds):
        x0, y0, x1, y1 = bounds
        w, h = max(1, x1-x0), max(1, y1-y0)
        x, y = (point[0]-x0)/w, (point[1]-y0)/h
        # Coordinates are relative to the dog's traced bounding box.
        if x < 0.27 and y > 0.48:
            return "tail"
        if y >= 0.84:
            return "legs"
        if 0.38 <= y < 0.54 and x >= 0.33:
            return "collar"
        if y < 0.44:
            if x < 0.34 or x > 0.83:
                return "ears"
            if 0.32 < x < 0.82 and 0.16 < y < 0.43:
                return "face"
            return "head"
        return "body"

    @staticmethod
    def sample_points(points, count=21):
        if len(points) <= count:
            return points
        return [points[round(i*(len(points)-1)/(count-1))]
                for i in range(count)]

    def plan(self, drawing):
        result = deepcopy(drawing)
        steps = result["steps"]
        strokes = [s for s in steps if s.get("tool") == "stroke"
                   and len(s.get("points", [])) >= 2]
        if len(strokes) != len(steps):
            raise ValueError("Planner requires stroke-only drawing; preserving original order")
        xs = [p[0] for step in strokes for p in step["points"]]
        ys = [p[1] for step in strokes for p in step["points"]]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        buckets = {name: [] for name in self.ZONES}
        for index, step in enumerate(strokes):
            samples = self.sample_points(step["points"])
            counts = {zone: 0 for zone in self.ZONES}
            for point in samples:
                counts[self.zone(point, bounds)] += 1
            chosen = max(self.ZONES, key=lambda z: (counts[z], -self.ZONES.index(z)))
            buckets[chosen].append((index, step))
        ordered = []
        cursor = None
        for zone in self.ZONES:
            remaining = buckets[zone][:]
            while remaining:
                def score(item):
                    index, step = item
                    a, b = step["points"][0], step["points"][-1]
                    return (min(dist(cursor, a), dist(cursor, b)) if cursor else 0,
                            index)
                chosen = min(remaining, key=score)
                remaining.remove(chosen)
                _, step = chosen
                points = step["points"]
                if cursor and dist(cursor, points[-1]) < dist(cursor, points[0]):
                    step["points"] = list(reversed(points))
                ordered.append(step)
                cursor = step["points"][-1]
        result["steps"] = ordered
        return result, {zone: len(buckets[zone]) for zone in self.ZONES}
