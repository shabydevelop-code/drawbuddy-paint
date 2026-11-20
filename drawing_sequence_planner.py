"""Generic geometry-based stroke ordering for any traced drawing.

No object labels, fixed anatomical zones, or reference-image semantics.
Reorders and optionally reverses open strokes; never changes their geometry.
"""
from copy import deepcopy
from math import dist


class DrawingSequencePlanner:
    """Cluster nearby connected strokes and reduce pen-up travel."""

    def __init__(self, join_tolerance=8):
        self.join_tolerance = join_tolerance

    @staticmethod
    def distance_to_path(point, points):
        """Distance to polyline segments (not just its endpoints)."""
        px, py = point
        best = float("inf")
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            dx, dy = bx - ax, by - ay
            length2 = dx*dx + dy*dy
            t = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/length2)) if length2 else 0
            best = min(best, dist(point, (ax+t*dx, ay+t*dy)))
        return best

    def connected(self, first, second):
        """Connect strokes when an endpoint touches another stroke."""
        a, b = first["points"], second["points"]
        threshold = self.join_tolerance
        return (min(self.distance_to_path(p, b) for p in (a[0], a[-1])) <= threshold
                or min(self.distance_to_path(p, a) for p in (b[0], b[-1])) <= threshold)

    @staticmethod
    def choose(remaining, cursor):
        """Nearest reachable endpoint, stable by original index on ties."""
        def rank(item):
            index, step = item
            a, b = step["points"][0], step["points"][-1]
            distance = min(dist(cursor, a), dist(cursor, b)) if cursor else 0
            return distance, index
        return min(remaining, key=rank)

    def plan(self, drawing):
        result = deepcopy(drawing)
        steps = result.get("steps", [])
        if not steps:
            raise ValueError("Drawing has no steps")
        # Never reorder filled shapes or other operations: their layering
        # and fill semantics can depend on the original execution order.
        if any(s.get("tool") != "stroke" or len(s.get("points", [])) < 2
               for s in steps):
            raise ValueError("Generic planner requires stroke-only drawing")
        count = len(steps)
        parent = list(range(count))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        # Spatial bounding-box prefilter avoids expensive path comparisons.
        boxes = []
        for step in steps:
            xs = [p[0] for p in step["points"]]
            ys = [p[1] for p in step["points"]]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        for i in range(count):
            a = boxes[i]
            for j in range(i+1, count):
                b = boxes[j]
                t = self.join_tolerance
                if (a[2]+t < b[0] or b[2]+t < a[0] or
                        a[3]+t < b[1] or b[3]+t < a[1]):
                    continue
                if self.connected(steps[i], steps[j]):
                    union(i, j)

        groups = {}
        for i, step in enumerate(steps):
            groups.setdefault(find(i), []).append((i, step))
        pending = list(groups.values())
        ordered = []
        cursor = None
        while pending:
            # Choose the nearest group from current pen position; initial
            # group is anchored to the first original stroke.
            def group_rank(group):
                if cursor is None:
                    return min(i for i, _ in group), 0
                return (min(min(dist(cursor, s["points"][0]),
                                dist(cursor, s["points"][-1]))
                            for _, s in group),
                        min(i for i, _ in group))
            group = min(pending, key=group_rank)
            pending.remove(group)
            remaining = group[:]
            while remaining:
                item = self.choose(remaining, cursor)
                remaining.remove(item)
                _, step = item
                points = step["points"]
                if cursor is not None and dist(cursor, points[-1]) < dist(cursor, points[0]):
                    # Closed loops are not reversed: preserving their start
                    # and orientation avoids changing drawing semantics.
                    if points[0] != points[-1]:
                        step["points"] = list(reversed(points))
                ordered.append(step)
                cursor = step["points"][-1]
        result["steps"] = ordered
        return result, {"groups": len(groups), "strokes": count}
