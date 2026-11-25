"""Terraforming — Phase 4 of the massive world upgrade, grown by
Phase 9 (city growth, design Addendum 3: "Even the city can grow").

Design doc: files/world-upgrade-design.md, system 5 + phase P4,
and Addendum 3 for the growth rules.

The city's ground is a **heightmap**: a grid of samples over the
+/-bound world at 2m resolution, heights in metres stored rounded
to 0.1. The bound is world STATE, not a constant: the founding
city spans +/-120m (121x121 samples), and each expansion grows it
by one +60m ring — 180, 240, 300, up to the engine cap of +/-360m.
The cap is a payload/storage limit (a 361x361 grid is ~130k
samples, ~1MB of JSON), not a design refusal: inside it, the land
grows whenever the Keeper wills it or the citizens vote it
(civics' "expand" action). Two ways to reshape the land itself:

  * **God mode (instant)** — the player applies an op directly:
    raise, lower, flatten (toward the height at the clicked point)
    or smooth, each with a radial cosine falloff so edits feather
    out at the rim instead of cliffing.
  * **Citizen crews (gradual)** — the same op as a *job*: a
    "terraform" task in the P2/P3 queue (see work.py /
    ActivityPlanner.assign_terraform). The assigned citizen travels
    to the patch and works; the heights move from their start values
    toward the job's goal values over ~30 sim-seconds of attended
    work. Jobs are created by the player (crew mode) and, from P5,
    by adopted civic proposals — the foreman never invents them, it
    only staffs ones that exist.

Movement and logic stay 2D throughout the framework: positions are
(x, z), and heights are world state + visuals (the client samples
the heightmap to seat its ground mesh, roads, buildings and
citizens). Building placement ignores slope entirely (P3's rules
are unchanged); sites and buildings simply render at the sampled
height.

Terrain is world SITUATION, not soul: the heightmap, the bound and
any active crew jobs ride the city layout document
({"terrain": {...}}) and GET /api/world, and a souls load never
touches them. Serialize / restore follows buildings.py's pattern —
validate the whole slice first, adopt it only if it is entirely
clean (all-or-nothing). A document written before growth existed
carries no bound and restores as the founding +/-120m world.
"""
from __future__ import annotations

import math

# -- the grid -------------------------------------------------------------
BOUND = 120.0            # the FOUNDING bound: samples span
                         # -BOUND..+BOUND on both axes
STEP = 2.0               # metres between samples
SIZE = 121               # founding samples per axis: (2*BOUND / STEP) + 1
SAMPLES = SIZE * SIZE    # 14641, the founding grid
WATER_Y = -0.35          # the client's pond plane: ground below this
                         # height reads as underwater

# -- growth (P9) --------------------------------------------------------------
RING = 60.0              # metres of new land per expansion
MAX_BOUND = 360.0        # the engine cap (payload/storage, see above)
BOUNDS = (120.0, 180.0, 240.0, 300.0, 360.0)   # every legal bound
_RELIEF_EASE_M = 24.0    # new-ring relief eases in over this distance
                         # past the old edge, so growth has no cliff seam

# -- op rules ---------------------------------------------------------------
OPS = ("raise", "lower", "flatten", "smooth")
MIN_RADIUS, MAX_RADIUS = 2.0, 20.0     # metres
MIN_AMOUNT, MAX_AMOUNT = 0.2, 2.0      # metres per application
CREW_SECONDS = 30.0      # attended sim-seconds for a crew job to
                         # apply its full delta
_HEIGHT_MIN, _HEIGHT_MAX = -20.0, 50.0   # restore-time sanity clamp


def _r1(v: float) -> float:
    """Heights live at 0.1m precision — every mutation rounds, so a
    serialize -> restore round-trip is exact."""
    return round(float(v), 1)


def _size_for(bound: float) -> int:
    """Samples per axis for a grid spanning +/-bound at STEP."""
    return int(round(2.0 * bound / STEP)) + 1


def _relief(x: float, z: float) -> float:
    """The new land's natural relief: gentle deterministic seeded
    rolling ground, 0 to ~1.5m. A fixed sum of broad sines — the
    same formula on every run and in the web client's JS twin, so
    a ring grows the same hills whoever grows it."""
    raw = 0.5 + 0.28 * math.sin(x * 0.021) * math.cos(z * 0.019) \
        + 0.22 * math.sin((x + 2.0 * z) * 0.013 + 1.3)
    return 1.5 * max(0.0, min(1.0, raw))


def _ease(d: float) -> float:
    """Smoothstep 0 -> 1 over _RELIEF_EASE_M metres past the old
    edge: relief is zero AT the seam and full a field's width in."""
    if d <= 0.0:
        return 0.0
    t = min(1.0, d / _RELIEF_EASE_M)
    return t * t * (3.0 - 2.0 * t)


def _clamp_num(v, lo: float, hi: float, default: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _falloff(d: float, radius: float) -> float:
    """Cosine falloff: 1 at the centre, 0 at (and beyond) the rim."""
    if d >= radius:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * d / radius))


def _sample_grid(heights: list[float], bound: float, size: int,
                 x: float, z: float) -> float:
    """Bilinear height at (x, z) over an explicit grid, clamped to
    it — the one sampling rule (the client mirrors it)."""
    gx = min(max((x + bound) / STEP, 0.0), size - 1.0)
    gz = min(max((z + bound) / STEP, 0.0), size - 1.0)
    ix, iz = int(gx), int(gz)
    if ix >= size - 1:
        ix, tx = size - 2, 1.0
    else:
        tx = gx - ix
    if iz >= size - 1:
        iz, tz = size - 2, 1.0
    else:
        tz = gz - iz
    a = heights[iz * size + ix]
    b = heights[iz * size + ix + 1]
    c = heights[(iz + 1) * size + ix]
    d = heights[(iz + 1) * size + ix + 1]
    return (a * (1 - tx) + b * tx) * (1 - tz) \
        + (c * (1 - tx) + d * tx) * tz


class TerrainWorld:
    """The heightmap, the terraforming ops and the crew job book."""

    def __init__(self, bound: float = BOUND):
        if bound not in BOUNDS:
            raise ValueError(f"terrain bound must be one of {BOUNDS}")
        self.bound = float(bound)
        self.heights: list[float] = [0.0] * (self.size * self.size)
        self.jobs: dict[str, dict] = {}
        self._seq = 0
        # Set once the land has been shaped (or restored, or grown):
        # the bridge only merges terrain into a served layout
        # document once there is something worth carrying.
        self.modified = False

    # -- grid basics ------------------------------------------------------
    @property
    def size(self) -> int:
        """Samples per axis at the live bound."""
        return _size_for(self.bound)

    @property
    def samples(self) -> int:
        return self.size * self.size

    @property
    def expansions(self) -> int:
        """How many growth rings the city has taken (0 = founding)."""
        return int(round((self.bound - BOUND) / RING))

    def index(self, ix: int, iz: int) -> int:
        return iz * self.size + ix

    def coords(self, idx: int) -> tuple[float, float]:
        iz, ix = divmod(idx, self.size)
        return (-self.bound + ix * STEP, -self.bound + iz * STEP)

    def sample(self, x: float, z: float) -> float:
        """Bilinear height at (x, z), clamped to the grid — the one
        sampling rule the client mirrors for seating its meshes."""
        return _sample_grid(self.heights, self.bound, self.size, x, z)

    def _goal_map(self, heights: list[float], op: str, x: float,
                  z: float, radius: float, amount: float,
                  target: float | None) -> dict[int, float]:
        """The full-application result of one op, as {index: goal
        height} over the samples inside the radius — computed
        against `heights` without mutating it. God mode adopts the
        map at once; a crew job lerps from the start heights to
        exactly this map."""
        size, bound = self.size, self.bound
        goal: dict[int, float] = {}
        ix0 = max(0, int((x - radius + bound) / STEP))
        ix1 = min(size - 1, int((x + radius + bound) / STEP) + 1)
        iz0 = max(0, int((z - radius + bound) / STEP))
        iz1 = min(size - 1, int((z + radius + bound) / STEP) + 1)
        for iz in range(iz0, iz1 + 1):
            gz = -bound + iz * STEP
            for ix in range(ix0, ix1 + 1):
                gx = -bound + ix * STEP
                w = _falloff(math.hypot(gx - x, gz - z), radius)
                if w <= 0.0:
                    continue
                idx = self.index(ix, iz)
                h = heights[idx]
                if op == "raise":
                    g = h + amount * w
                elif op == "lower":
                    g = h - amount * w
                elif op == "flatten":
                    g = h + (target - h) * w
                else:   # smooth: toward the 3x3 neighbourhood mean
                    total, count = 0.0, 0
                    for jz in range(max(0, iz - 1), min(size, iz + 2)):
                        for jx in range(max(0, ix - 1),
                                        min(size, ix + 2)):
                            total += heights[self.index(jx, jz)]
                            count += 1
                    avg = total / count
                    g = h + (avg - h) * w * min(1.0, amount)
                goal[idx] = _r1(g)
        return goal

    # -- validation ---------------------------------------------------------
    def validate_op(self, op: str, x: float, z: float, radius: float,
                    amount: float) -> str | None:
        """Why this op application is refused, or None when it may
        proceed: known op, centre inside the LIVE world bounds,
        radius 2-20m, amount 0.2-2m per application."""
        if op not in OPS:
            return (f"Unknown terraform op {op!r} — want one of: "
                    + ", ".join(OPS) + ".")
        if not (abs(x) <= self.bound and abs(z) <= self.bound):
            return "That spot is outside the city bounds."
        if not (MIN_RADIUS <= radius <= MAX_RADIUS):
            return (f"Terraform radius must be between "
                    f"{MIN_RADIUS:.0f} and {MAX_RADIUS:.0f} m.")
        if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
            return (f"Terraform amount must be between {MIN_AMOUNT} "
                    f"and {MAX_AMOUNT} m per application.")
        return None

    # -- god mode (instant) ---------------------------------------------------
    def apply_op(self, op: str, x: float, z: float, radius: float,
                 amount: float, target: float | None = None) -> dict:
        """Apply one op at full strength, immediately. Flatten's
        target defaults to the height at the clicked point. Callers
        validate first (validate_op). Returns {"changed": n,
        "target": t}."""
        if target is None and op == "flatten":
            target = self.sample(x, z)
        goal = self._goal_map(self.heights, op, x, z, radius, amount,
                              target)
        for idx, g in goal.items():
            self.heights[idx] = g
        self.modified = True
        return {"changed": len(goal), "target": target}

    @staticmethod
    def god_event(op: str, x: float, z: float, radius: float,
                  amount: float, target: float | None = None) -> dict:
        ev = {"type": "terraform", "mode": "god", "op": op,
              "position": [float(x), float(z)], "radius": radius,
              "amount": amount, "by": "Player"}
        if target is not None:
            ev["target"] = _r1(target)
        return ev

    # -- growth (P9) ---------------------------------------------------------
    def expand(self) -> dict | None:
        """Grow the world by one ring (+60m of bound). The existing
        land is preserved EXACTLY — every old sample keeps its
        world coordinates and its height (the grid simply gains a
        margin of samples on each side). The new ring is seeded
        with gentle deterministic relief (_relief) eased in from
        the old edge height, so the seam is a slope, never a cliff.
        Active crew jobs survive: their start/goal index maps are
        pure index shifts under the re-grid. Returns
        {"bound", "ring", "previous"} — or None at the engine cap,
        where nothing changes."""
        if self.bound >= MAX_BOUND:
            return None
        old_bound, old_size, old = self.bound, self.size, self.heights
        new_bound = old_bound + RING
        new_size = _size_for(new_bound)
        shift = int(round(RING / STEP))
        new = [0.0] * (new_size * new_size)
        for iz in range(new_size):
            z = -new_bound + iz * STEP
            for ix in range(new_size):
                x = -new_bound + ix * STEP
                idx = iz * new_size + ix
                if abs(x) <= old_bound and abs(z) <= old_bound:
                    new[idx] = old[(iz - shift) * old_size
                                   + (ix - shift)]
                else:
                    edge = _sample_grid(
                        old, old_bound, old_size,
                        max(-old_bound, min(old_bound, x)),
                        max(-old_bound, min(old_bound, z)))
                    d = max(abs(x), abs(z)) - old_bound
                    new[idx] = _r1(edge + _relief(x, z) * _ease(d))
        for job in self.jobs.values():
            for field in ("start", "goal"):
                job[field] = {
                    (idx // old_size + shift) * new_size
                    + (idx % old_size + shift): h
                    for idx, h in job[field].items()}
        self.bound = new_bound
        self.heights = new
        self.modified = True
        return {"bound": new_bound,
                "ring": int(round((new_bound - BOUND) / RING)),
                "previous": old_bound}

    @staticmethod
    def expansion_event(info: dict, districts: list[str],
                        by: str = "The Keeper") -> dict:
        """The {"type": "expansion"} protocol event: the new bound,
        the ring number, and the names of the districts that
        joined the world."""
        return {"type": "expansion", "bound": info["bound"],
                "ring": info["ring"], "district": list(districts),
                "districts": list(districts), "by": by}

    # -- citizen crews (gradual) ----------------------------------------------
    def create_job(self, op: str, x: float, z: float, radius: float,
                   amount: float, target: float | None = None) -> dict:
        """Open a crew job: snapshot the start heights over the
        patch and compute the goal map the crew will lerp toward.
        Callers validate first (validate_op)."""
        if target is None and op == "flatten":
            target = self.sample(x, z)
        goal = self._goal_map(self.heights, op, x, z, radius, amount,
                              target)
        self._seq += 1
        job = {"id": f"tf-{self._seq}", "op": op,
               "x": float(x), "z": float(z), "radius": radius,
               "amount": amount,
               "target": _r1(target) if target is not None else None,
               "progress": 0.0, "state": "active",
               "start": {idx: self.heights[idx] for idx in goal},
               "goal": goal}
        self.jobs[job["id"]] = job
        self.modified = True
        return job

    def work_at(self, job_id: str, dt: float, by: str = "") -> list[dict]:
        """One attending citizen works a crew job for `dt`
        sim-seconds: progress advances and the patch heights lerp
        from start to goal. At full progress the goal heights land
        exactly, the job closes, and the completion event (mode
        "crew") is returned."""
        job = self.jobs.get(job_id)
        if job is None or job["state"] != "active" or dt <= 0:
            return []
        job["progress"] = min(1.0, job["progress"] + dt / CREW_SECONDS)
        p = job["progress"]
        for idx, goal_h in job["goal"].items():
            start_h = job["start"][idx]
            self.heights[idx] = _r1(start_h + (goal_h - start_h) * p)
        self.modified = True
        if p < 1.0:
            return []
        for idx, goal_h in job["goal"].items():
            self.heights[idx] = goal_h
        job["state"] = "complete"
        del self.jobs[job_id]
        return [self.job_event(job, "complete", by=by)]

    @staticmethod
    def job_event(job: dict, state: str, **extra) -> dict:
        ev = {"type": "terraform", "mode": "crew", "state": state,
              "job": job["id"], "op": job["op"],
              "position": [job["x"], job["z"]],
              "radius": job["radius"], "amount": job["amount"]}
        if job.get("target") is not None:
            ev["target"] = job["target"]
        ev.update(extra)
        return ev

    # -- read models ------------------------------------------------------------
    @staticmethod
    def _public_job(job: dict) -> dict:
        """The full public job record — /api/world, layout documents
        and serialize() all use exactly this shape (start/goal maps
        included, so a layout save keeps a crew job's trajectory
        intact), and restore() reads it back."""
        return {"id": job["id"], "op": job["op"],
                "position": [job["x"], job["z"]],
                "radius": job["radius"], "amount": job["amount"],
                "target": job["target"],
                "progress": round(job["progress"], 4),
                "state": job["state"],
                "start": {str(k): v for k, v in job["start"].items()},
                "goal": {str(k): v for k, v in job["goal"].items()}}

    def public(self) -> dict:
        """The /api/world terrain payload: grid geometry (at the
        LIVE bound) + the heightmap itself + active crew jobs."""
        return {"bound": self.bound, "step": STEP, "size": self.size,
                "water_y": WATER_Y, "heights": list(self.heights),
                "jobs": [self._public_job(j) for j in self.jobs.values()]}

    def summary(self) -> dict:
        """The terrain picture for digests (the interpreter): the
        highest/lowest sampled points, whether any ground dips
        below the water line (a pond), the open crew jobs — and
        the city's span (bound + how many rings it has grown)."""
        total = self.samples
        max_i = max(range(total), key=lambda i: self.heights[i])
        min_i = min(range(total), key=lambda i: self.heights[i])
        pond = self.heights[min_i] < WATER_Y
        return {"max_h": self.heights[max_i],
                "max_at": list(self.coords(max_i)),
                "min_h": self.heights[min_i],
                "min_at": list(self.coords(min_i)),
                "pond": pond,
                "pond_at": list(self.coords(min_i)) if pond else None,
                "jobs": len(self.jobs),
                "bound": self.bound,
                "expansions": self.expansions}

    # -- save / load --------------------------------------------------------------
    def serialize(self) -> dict:
        """Full-fidelity world slice for layout documents: the
        bound rides with the grid, so a grown world restores
        grown."""
        return {"terrain": {"bound": self.bound,
                            "heights": list(self.heights),
                            "jobs": [self._public_job(j)
                                     for j in self.jobs.values()]}}

    def restore(self, doc) -> dict:
        """Replace all terrain state from a serialized slice
        (validated first — a malformed document changes nothing).
        The document's bound is adopted with its grid: restoring
        a grown world grows this one, restoring a founding-era
        document (no bound) returns it to +/-120m. Returns
        {"jobs": n_active, "bound": b}."""
        clean = validate_terrain(doc)["terrain"]
        # Adopt the bound BEFORE any goal map is recomputed below:
        # index math runs against the restored grid.
        self.bound = clean["bound"]
        heights = clean["heights"]
        jobs: dict[str, dict] = {}
        seq = 0
        for rec in clean["jobs"]:
            start = {int(k): v for k, v in rec["start"].items()}
            goal = {int(k): v for k, v in rec["goal"].items()}
            job = {"id": rec["id"], "op": rec["op"],
                   "x": rec["position"][0], "z": rec["position"][1],
                   "radius": rec["radius"], "amount": rec["amount"],
                   "target": rec["target"], "progress": rec["progress"],
                   "state": "active", "start": start, "goal": goal}
            if not goal:
                # A record without its trajectory (hand-written or
                # trimmed): recompute the goal against the restored
                # heights and start from them.
                goal = self._goal_map(heights, job["op"], job["x"],
                                      job["z"], job["radius"],
                                      job["amount"], job["target"])
                job["goal"] = goal
            if not start:
                job["start"] = {idx: heights[idx] for idx in goal}
            jobs[job["id"]] = job
            try:
                seq = max(seq, int(job["id"].rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                pass
        self.heights = heights
        self.jobs = jobs
        self._seq = max(self._seq, seq)
        self.modified = True
        return {"jobs": len(jobs), "bound": self.bound}


def _clean_height_map(raw, i: int, field_name: str,
                      size: int) -> dict:
    """One job's start/goal map: {sample index: height}. Keys must
    parse as in-range indices (of the document's own grid);
    values clamp into the sane band."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"terrain.jobs[{i}].{field_name} must be an object")
    out = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"terrain.jobs[{i}].{field_name} key {key!r} "
                f"is not a sample index")
        if not 0 <= idx < size * size:
            raise ValueError(
                f"terrain.jobs[{i}].{field_name} key {key!r} "
                f"is outside the heightmap")
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"terrain.jobs[{i}].{field_name}[{key!r}] "
                f"must be a number")
        out[str(idx)] = _r1(max(_HEIGHT_MIN,
                                min(_HEIGHT_MAX, float(value))))
    return out


def _clean_job_record(rec, i: int, bound: float, size: int) -> dict:
    """Validate one serialized crew job (layout documents,
    restore): op/position are structural (against the document's
    own bound); tuning fields clamp into range rather than
    reject."""
    if not isinstance(rec, dict):
        raise ValueError(f"terrain.jobs[{i}] must be an object")
    op = rec.get("op")
    if op not in OPS:
        raise ValueError(f"terrain.jobs[{i}].op {op!r} is not a terraform op")
    pos = rec.get("position")
    if not (isinstance(pos, list) and len(pos) == 2
            and all(isinstance(v, (int, float)) for v in pos)):
        raise ValueError(f"terrain.jobs[{i}].position must be [x, z]")
    x, z = float(pos[0]), float(pos[1])
    if abs(x) > bound or abs(z) > bound:
        raise ValueError(f"terrain.jobs[{i}].position is outside the world")
    target = rec.get("target")
    return {"id": str(rec.get("id") or f"tf-{i + 1}"), "op": op,
            "position": [x, z],
            "radius": _clamp_num(rec.get("radius"),
                                 MIN_RADIUS, MAX_RADIUS, 8.0),
            "amount": _clamp_num(rec.get("amount"),
                                 MIN_AMOUNT, MAX_AMOUNT, 1.0),
            "target": _r1(target) if isinstance(target, (int, float))
            else None,
            "progress": _clamp_num(rec.get("progress"), 0.0, 1.0, 0.0),
            "start": _clean_height_map(rec.get("start"), i, "start",
                                       size),
            "goal": _clean_height_map(rec.get("goal"), i, "goal",
                                      size)}


def validate_terrain(doc) -> dict:
    """Clean the {"terrain": {...}} slice of a layout/world
    document. The slice carries its bound (a founding-era document
    without one means +/-120m); the heights list is structural —
    exactly one float per sample of THAT bound's grid — anything
    else rejects the whole document."""
    if not isinstance(doc, dict):
        raise ValueError("terrain document must be an object")
    ter = doc.get("terrain")
    if not isinstance(ter, dict):
        raise ValueError("'terrain' must be an object")
    raw_bound = ter.get("bound", BOUND)
    try:
        bound = float(raw_bound)
    except (TypeError, ValueError):
        raise ValueError(f"terrain.bound {raw_bound!r} is not a number")
    if bound not in BOUNDS:
        raise ValueError(
            f"terrain.bound {bound} is not a grown size of the "
            f"world (one of {', '.join(str(int(b)) for b in BOUNDS)})")
    size = _size_for(bound)
    heights_in = ter.get("heights")
    if not isinstance(heights_in, list) or len(heights_in) != size * size:
        raise ValueError(
            f"terrain.heights must be a list of {size * size} samples")
    heights = []
    for v in heights_in:
        if not isinstance(v, (int, float)):
            raise ValueError("terrain.heights entries must be numbers")
        heights.append(_r1(max(_HEIGHT_MIN,
                               min(_HEIGHT_MAX, float(v)))))
    jobs_in = ter.get("jobs") or []
    if not isinstance(jobs_in, list):
        raise ValueError("terrain.jobs must be a list")
    return {"terrain": {"bound": bound, "heights": heights,
                        "jobs": [_clean_job_record(r, i, bound, size)
                                 for i, r in enumerate(jobs_in)]}}
