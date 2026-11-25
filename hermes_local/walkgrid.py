"""Analytic walk grid — Phase 10 of the massive world upgrade.

The city is generated from data the framework already owns — the
heightmap (terrain.py), the building/site footprints
(buildings.py), the designer's layout objects and the road bands —
so "where can a citizen walk?" never needs a navmesh baked from
meshes. This module derives that answer analytically, on a 2m grid
aligned with the heightmap's own resolution:

  * one byte per cell in a flat bytearray: 0 = blocked,
    1 = sidewalk / plaza / road shoulder / door apron (cheapest),
    2 = grass / open terrain, 5 = road;
  * a cell is blocked when its ground is too steep (more than
    ~0.5m of rise across the 2m cell — the same threshold the
    client paints as rock), when it sits below the waterline, or
    when it falls inside a building/site/layout footprint
    inflated by 0.4m (the corner clearance that keeps citizens
    from clipping walls);
  * resource nodes never block: trees and goats are things
    citizens walk up to, not walls.

On top of the grid: octile A* (heapq) with corner-cut prevention
(a diagonal step needs both adjacent cardinal cells walkable),
line-of-sight smoothing over the A* cell path (a grid DDA, so a
route across open ground collapses to its endpoints), and a
spiral/BFS spawn snapper that pulls any point — a spawn, a
rendezvous, a task target — onto the nearest walkable cell.

The grid is DERIVED STATE, end to end: it is rebuilt from the
live worlds whenever they change (the bridge owns that wiring),
carries a monotonically increasing version so clients can tell
when to refetch, serializes as base64 of the flat bytearray for
/api/world — and is NEVER written into the world snapshot. A
restored world simply rebuilds its grid from its restored parts.
"""
from __future__ import annotations

import base64
import heapq
import math

from .buildings import BUILDINGS, _LAYOUT_OBJECT_SIZE
from .terrain import WATER_Y

CELL = 2.0               # metres per cell — the heightmap's step
INFLATE = 0.4            # footprint clearance, metres
SLOPE_MAX = 0.5          # metres of rise across one cell: steeper
                         # than this is rock, not a path (mirrors
                         # the client's biome rock threshold)

BLOCKED = 0
WALK = 1                 # sidewalk / plaza / shoulder / apron
GRASS = 2
ROAD = 5

# Founding road centrelines (mirrors buildings.py ROADS / the web
# client): a 12m carriageway (cost ROAD) inside a +4m sidewalk
# surround (cost WALK), drawn along both axes at each coordinate.
ROADS = (-96.0, -48.0, 0.0, 48.0, 96.0)
ROAD_HALF = 6.0
WALK_HALF = ROAD_HALF + 4.0

_SQRT2 = math.sqrt(2.0)


def _ring_road_coords(bound: float) -> list[float]:
    """Ring-road centrelines for a grown world, derived from the
    bound exactly as terrain.py / the client derive them: ring r
    (1-based) puts a road square at +/-(120 + 60r - 24)."""
    out: list[float] = []
    ring = 1
    while 120.0 + 60.0 * ring <= bound + 0.01:
        c = 120.0 + 60.0 * ring - 24.0
        out.extend([c, -c])
        ring += 1
    return out


class WalkGrid:
    """The walkability grid over one set of live world parts.

    Holds references, not copies: `terrain` (TerrainWorld),
    `construction` (ConstructionWorld or None) and
    `layout_objects_fn` (callable returning the layout document's
    object list, or None). rebuild() re-derives every cell from
    whatever those parts hold right now."""

    def __init__(self, terrain, construction=None,
                 layout_objects_fn=None):
        self.terrain = terrain
        self.construction = construction
        self.layout_objects_fn = layout_objects_fn
        self.bound = float(terrain.bound)
        self.w = self.h = 0
        self.cells = bytearray()
        self.version = 0
        self.rebuild()

    # -- geometry ---------------------------------------------------------
    @property
    def size(self) -> int:
        """Cells per axis at the live bound (120 founding, 360 cap)."""
        return int(round(2.0 * float(self.terrain.bound) / CELL))

    def _centre(self, ix: int, iz: int) -> tuple[float, float]:
        b = self.bound
        return (-b + (ix + 0.5) * CELL, -b + (iz + 0.5) * CELL)

    def _cell_of(self, x: float, z: float) -> tuple[int, int] | None:
        b = self.bound
        ix = int((x + b) / CELL)
        iz = int((z + b) / CELL)
        if 0 <= ix < self.w and 0 <= iz < self.h:
            return ix, iz
        return None

    def cost_at(self, x: float, z: float) -> int:
        """The traversal cost of the cell holding (x, z); 0 for
        blocked and for anything outside the grid."""
        cell = self._cell_of(x, z)
        if cell is None:
            return BLOCKED
        return self.cells[cell[1] * self.w + cell[0]]

    def walkable(self, x: float, z: float) -> bool:
        return self.cost_at(x, z) != BLOCKED

    # -- derivation ---------------------------------------------------------
    def _road_coords(self) -> tuple:
        """Founding + ring road centrelines at the live bound
        (cached per bound — _base_cost runs per cell)."""
        cached = getattr(self, "_roads_cache", None)
        if cached is None or cached[0] != self.bound:
            coords = ROADS + tuple(_ring_road_coords(self.bound))
            self._roads_cache = (self.bound, coords)
            return coords
        return cached[1]

    def _footprints(self) -> list[tuple[float, float, float, float]]:
        """Every blocking footprint as (cx, cz, half_w, half_d),
        already inflated: finished buildings and open sites (their
        catalog footprints) plus designer layout objects (their
        approximate sizes). Rotation is visual only throughout
        the framework — boxes are axis-aligned, as in placement."""
        out: list[tuple[float, float, float, float]] = []
        con = self.construction
        if con is not None:
            for b in con.buildings:
                spec = BUILDINGS.get(b.get("kind") or "", {})
                if b.get("archetype"):
                    spec = BUILDINGS.get(b["archetype"], spec)
                fw, fd = spec.get("footprint", (8.0, 8.0))
                out.append((float(b["position"][0]),
                            float(b["position"][1]),
                            fw / 2.0 + INFLATE, fd / 2.0 + INFLATE))
            for s in con.sites.values():
                if s.complete:
                    continue
                fw, fd = BUILDINGS[s.kind]["footprint"]
                out.append((s.x, s.z,
                            fw / 2.0 + INFLATE, fd / 2.0 + INFLATE))
        if self.layout_objects_fn is not None:
            for obj in self.layout_objects_fn() or ():
                size = _LAYOUT_OBJECT_SIZE.get(obj.get("kind"))
                if size is None:
                    continue
                pos = obj.get("position") or [0, 0, 0]
                out.append((float(pos[0]), float(pos[2]),
                            size[0] / 2.0 + INFLATE,
                            size[1] / 2.0 + INFLATE))
        return out

    def _aprons(self) -> list[tuple[float, float, float, float]]:
        """Door aprons of finished anchored buildings (houses,
        stalls): the strip of ground just off the front (+Z) edge
        — where a citizen stands to be "at" the place. Returned as
        (cx, cz, half_w, half_d) boxes, uninflated."""
        out: list[tuple[float, float, float, float]] = []
        con = self.construction
        if con is None:
            return out
        for b in con.buildings:
            if not b.get("anchor"):
                continue
            spec = BUILDINGS.get(b.get("kind") or "", {})
            if b.get("archetype"):
                spec = BUILDINGS.get(b["archetype"], spec)
            fw, fd = spec.get("footprint", (8.0, 8.0))
            bx, bz = float(b["position"][0]), float(b["position"][1])
            out.append((bx, bz + fd / 2.0 + CELL / 2.0,
                        fw / 2.0, CELL / 2.0))
        return out

    def _base_cost(self, cx: float, cz: float) -> int:
        """Terrain + road cost of one cell centre, before
        footprints and aprons: water and steep ground block;
        road bands and their sidewalk surrounds price the rest."""
        t = self.terrain
        h = t.sample(cx, cz)
        if h < WATER_Y:
            return BLOCKED
        slope = max(
            abs(t.sample(cx + CELL / 2.0, cz)
                - t.sample(cx - CELL / 2.0, cz)),
            abs(t.sample(cx, cz + CELL / 2.0)
                - t.sample(cx, cz - CELL / 2.0)))
        if slope > SLOPE_MAX:
            return BLOCKED
        cost = GRASS
        for c in self._road_coords():
            if abs(cx - c) <= WALK_HALF or abs(cz - c) <= WALK_HALF:
                cost = WALK
            if abs(cx - c) <= ROAD_HALF or abs(cz - c) <= ROAD_HALF:
                cost = ROAD
        return cost

    def _mark(self, cells: bytearray, box, value: int,
              only_if_not_blocked: bool = False) -> None:
        """Write `value` into every cell whose centre falls in the
        (cx, cz, half_w, half_d) box."""
        cx, cz, hw, hd = box
        b = self.bound
        ix0 = max(0, int((cx - hw + b) / CELL))
        ix1 = min(self.w - 1, int((cx + hw + b) / CELL))
        iz0 = max(0, int((cz - hd + b) / CELL))
        iz1 = min(self.h - 1, int((cz + hd + b) / CELL))
        for iz in range(iz0, iz1 + 1):
            gz = -b + (iz + 0.5) * CELL
            if abs(gz - cz) > hd:
                continue
            for ix in range(ix0, ix1 + 1):
                gx = -b + (ix + 0.5) * CELL
                if abs(gx - cx) > hw:
                    continue
                idx = iz * self.w + ix
                if only_if_not_blocked and cells[idx] == BLOCKED:
                    continue
                cells[idx] = value

    def _compute_all(self) -> bytearray:
        cells = bytearray(self.w * self.h)
        for iz in range(self.h):
            cz = -self.bound + (iz + 0.5) * CELL
            row = iz * self.w
            for ix in range(self.w):
                cx = -self.bound + (ix + 0.5) * CELL
                cells[row + ix] = self._base_cost(cx, cz)
        for box in self._footprints():
            self._mark(cells, box, BLOCKED)
        for box in self._aprons():
            self._mark(cells, box, WALK, only_if_not_blocked=True)
        return cells

    def _cell_cost_full(self, ix: int, iz: int) -> int:
        """One cell's cost with every rule applied — the regional
        recompute path (rebuild paints footprints by range; this
        checks them per cell, same result)."""
        cx, cz = self._centre(ix, iz)
        cost = self._base_cost(cx, cz)
        if cost == BLOCKED:
            return BLOCKED
        for fx, fz, hw, hd in self._footprints():
            if abs(cx - fx) <= hw and abs(cz - fz) <= hd:
                return BLOCKED
        for ax, az, hw, hd in self._aprons():
            if abs(cx - ax) <= hw and abs(cz - az) <= hd:
                return WALK
        return cost

    # -- rebuild / refresh ----------------------------------------------------
    def rebuild(self) -> bool:
        """Re-derive the whole grid from the live world parts.
        Bumps the version (and returns True) only when something
        actually changed — dimensions included."""
        self.bound = float(self.terrain.bound)
        self.w = self.h = self.size
        fresh = self._compute_all()
        if fresh != self.cells:
            self.cells = fresh
            self.version += 1
            return True
        return False

    def recompute_region(self, x: float, z: float,
                         radius: float) -> bool:
        """Re-derive the cells around (x, z) — the cheap refresh
        after a local change (a terraform patch settling). Bumps
        the version (and returns True) on any cell change."""
        if self.bound != float(self.terrain.bound) \
                or self.w != self.size:
            return self.rebuild()
        cell = self._cell_of(x, z)
        if cell is None:
            return False
        r = max(1, int(math.ceil(radius / CELL)))
        changed = False
        for iz in range(max(0, cell[1] - r),
                        min(self.h, cell[1] + r + 1)):
            for ix in range(max(0, cell[0] - r),
                            min(self.w, cell[0] + r + 1)):
                idx = iz * self.w + ix
                v = self._cell_cost_full(ix, iz)
                if v != self.cells[idx]:
                    self.cells[idx] = v
                    changed = True
        if changed:
            self.version += 1
        return changed

    # -- queries ----------------------------------------------------------------
    def snap_to_walkable(self, x: float, z: float):
        """The nearest walkable spot to (x, z): the point itself
        when its cell is walkable, else the centre of the closest
        non-blocked cell found by expanding rings (BFS) around the
        target cell. None only when the whole grid is blocked."""
        cell = self._cell_of(x, z)
        if cell is None:
            # Outside the land: clamp onto the grid, then snap.
            b = self.bound
            x = max(-b + 0.1, min(b - 0.1, x))
            z = max(-b + 0.1, min(b - 0.1, z))
            cell = self._cell_of(x, z)
            if cell is None:
                return None
        ix, iz = cell
        if self.cells[iz * self.w + ix] != BLOCKED:
            return (float(x), float(z))
        max_r = max(self.w, self.h)
        for r in range(1, max_r + 1):
            best, best_d = None, None
            for jz in range(max(0, iz - r), min(self.h, iz + r + 1)):
                for jx in range(max(0, ix - r), min(self.w, ix + r + 1)):
                    if max(abs(jx - ix), abs(jz - iz)) != r:
                        continue
                    if self.cells[jz * self.w + jx] == BLOCKED:
                        continue
                    cx, cz = self._centre(jx, jz)
                    d = (cx - x) ** 2 + (cz - z) ** 2
                    if best_d is None or d < best_d:
                        best, best_d = (cx, cz), d
            if best is not None:
                return best
        return None

    def line_clear(self, x0: float, z0: float,
                   x1: float, z1: float) -> bool:
        """Grid DDA: True when the straight segment (x0, z0) ->
        (x1, z1) stays on walkable cells (any non-zero cost may be
        crossed — roads are walkable, just pricey)."""
        dist = math.hypot(x1 - x0, z1 - z0)
        steps = max(1, int(dist / (CELL * 0.5)))
        for i in range(steps + 1):
            t = i / steps
            if self.cost_at(x0 + (x1 - x0) * t,
                            z0 + (z1 - z0) * t) == BLOCKED:
                return False
        return True

    # -- A* ---------------------------------------------------------------------
    def _astar(self, start: tuple[int, int],
               goal: tuple[int, int]) -> list[tuple[int, int]] | None:
        """Octile A* over cells. Diagonal steps require BOTH
        adjacent cardinal cells walkable (corner-cut prevention);
        step cost is step length x the entered cell's cost."""
        w, h, cells = self.w, self.h, self.cells
        gx, gz = goal

        def heuristic(ix: int, iz: int) -> float:
            dx, dz = abs(ix - gx), abs(iz - gz)
            return (dx + dz) + (_SQRT2 - 2.0) * min(dx, dz)

        start_idx = start[1] * w + start[0]
        g_score = {start_idx: 0.0}
        came: dict[int, int] = {}
        heap = [(heuristic(*start), start_idx)]
        closed: set[int] = set()
        while heap:
            _f, idx = heapq.heappop(heap)
            if idx in closed:
                continue
            closed.add(idx)
            ix, iz = idx % w, idx // w
            if (ix, iz) == goal:
                path = [idx]
                while path[-1] != start_idx:
                    path.append(came[path[-1]])
                path.reverse()
                return [(i % w, i // w) for i in path]
            for dx, dz, diag in ((1, 0, False), (-1, 0, False),
                                 (0, 1, False), (0, -1, False),
                                 (1, 1, True), (1, -1, True),
                                 (-1, 1, True), (-1, -1, True)):
                nx, nz = ix + dx, iz + dz
                if not (0 <= nx < w and 0 <= nz < h):
                    continue
                n_idx = nz * w + nx
                cost = cells[n_idx]
                if cost == BLOCKED or n_idx in closed:
                    continue
                if diag:
                    if cells[iz * w + nx] == BLOCKED \
                            or cells[nz * w + ix] == BLOCKED:
                        continue
                step = (_SQRT2 if diag else 1.0) * cost
                ng = g_score[idx] + step
                if ng < g_score.get(n_idx, float("inf")):
                    g_score[n_idx] = ng
                    came[n_idx] = idx
                    heapq.heappush(
                        heap, (ng + heuristic(nx, nz), n_idx))
        return None

    def find_route(self, x0: float, z0: float,
                   x1: float, z1: float):
        """A walkable route from (x0, z0) to (x1, z1) as world-
        space waypoints [[x, z], ...]: both ends snapped onto the
        grid, A* over cells, then line-of-sight smoothing drops
        every waypoint the straight line makes unnecessary. The
        final waypoint is the goal itself when its cell is
        walkable. None when no route exists."""
        start = self.snap_to_walkable(x0, z0)
        goal = self.snap_to_walkable(x1, z1)
        if start is None or goal is None:
            return None
        sc, gc = self._cell_of(*start), self._cell_of(*goal)
        if sc is None or gc is None:
            return None
        if sc == gc:
            return [[goal[0], goal[1]]]
        path = self._astar(sc, gc)
        if path is None:
            return None
        centres = [self._centre(ix, iz) for ix, iz in path]
        centres[-1] = goal
        # Line-of-sight smoothing: from each kept waypoint, jump
        # to the farthest later point still in clear sight.
        out = [centres[0]]
        i = 0
        while i < len(centres) - 1:
            j = len(centres) - 1
            while j > i + 1 and not self.line_clear(
                    centres[i][0], centres[i][1],
                    centres[j][0], centres[j][1]):
                j -= 1
            out.append(centres[j])
            i = j
        return [[round(x, 1), round(z, 1)] for x, z in out]

    # -- wire format --------------------------------------------------------------
    def public(self) -> dict:
        """The /api/world walkgrid payload: geometry + version +
        the flat cost grid as base64 (a Uint8Array on the client;
        gzip takes the founding grid to a few KB on the wire)."""
        return {"bound": self.bound, "cell": CELL, "w": self.w,
                "h": self.h, "version": self.version,
                "data_b64": base64.b64encode(
                    bytes(self.cells)).decode("ascii")}
