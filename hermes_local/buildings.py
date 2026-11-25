"""Building system — Phase 3 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, system 4 + phase P3.

The player (god mode) places a structure; placement creates a
**construction site** in the world. A site rises in two stages —
"site" (0-50% of the work) and "frame" (50-100%) — and finishes as
a permanent **building**. Citizens do the work through the P2 task
queue (a "build" task, see work.py / ActivityPlanner.assign_build);
this module owns the sites, the buildings and the build rules, the
same way resources.py owns nodes and harvesting.

**The delivery rule** (deliberately simple and deterministic):
materials move from the city inventory to a site in two stage
tranches. The first stage needs the first half (rounded up) of
every required material; the second stage needs the rest. While a
builder works at the site, the site tries to draw its current
stage's tranche from the inventory: if the stores hold it, the
tranche is deducted (delivered) and work accrues; if they don't,
no work accrues, nothing is consumed, and the site sits in
"awaiting_materials" until the stores suffice. A site therefore
never half-eats a tranche it cannot finish, and progress is a pure
function of (inventory, builder presence, sim time).

Completion effects: the building joins the permanent building list;
a house adds a "front" (home) anchor and a market stall a "browse"
(market) anchor to the planner's anchor set — the planner applies
that from the completion event this module returns, so this module
never imports npc.py. Lamps and fences are decorative.

Sites and buildings are world SITUATION, not soul: they ride the
city layout document ({"sites": [...], "buildings": [...]}) and
/api/world, and a souls load never touches them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# City geometry, mirrored from npc.py / the web client: roads run
# along both axes at these coordinates with a 12m carriageway, and
# the founding city is the +/-120 square. Since P9 the playable
# bound is live world state (terrain.py: +60m per expansion, cap
# +/-360) — placement reads it through ConstructionWorld.bound_fn,
# which the bridge wires to the terrain; unwired worlds (unit
# tests, headless sims) keep the founding bound below.
ROADS = (-96.0, -48.0, 0.0, 48.0, 96.0)
ROAD_HALF = 6.0
WORLD_BOUND = 120.0

# -- the buildable catalog -------------------------------------------------
# costs: city-inventory materials. work: sim-seconds of builder time
# at the site. footprint: (width, depth) metres, for placement
# overlap checks and the client's renderers. anchor: what a finished
# building adds to the planner's anchor set — (kind, name prefix) —
# or None for decorative structures.
BUILDINGS = {
    "stall": {"name": "Market stall", "costs": {"planks": 4},
              "work": 30.0, "footprint": (6.0, 5.0),
              "anchor": ("browse", "Market stall")},
    "lamp": {"name": "Lamp", "costs": {"planks": 2},
             "work": 15.0, "footprint": (1.5, 1.5), "anchor": None},
    "fence": {"name": "Fence segment", "costs": {"planks": 1},
              "work": 10.0, "footprint": (4.0, 0.6), "anchor": None},
    "house": {"name": "House", "costs": {"planks": 10, "wood": 4},
              "work": 90.0, "footprint": (10.0, 10.0),
              "anchor": ("front", "House")},
}

# Approximate footprints of designer layout objects (mirrors the web
# client's KIND_SIZE), for overlap checks against a saved layout.
# Roads are excluded (the road-band check covers them) and spawn
# markers are points, not obstacles.
_LAYOUT_OBJECT_SIZE = {"building": (16.0, 16.0), "house": (10.0, 10.0),
                       "industrial": (16.0, 16.0),
                       "vehicle": (4.4, 2.0), "prop": (1.5, 1.5)}
_OVERLAP_MARGIN = 0.5


def stage_tranches(costs: dict) -> tuple[dict, dict]:
    """Split a cost dict into the two stage tranches: stage 1 (site)
    takes the first half of each material rounded up, stage 2
    (frame) takes the remainder."""
    first, second = {}, {}
    for key, amount in costs.items():
        first[key] = (amount + 1) // 2
        second[key] = amount - first[key]
    return first, second


@dataclass
class ConstructionSite:
    id: str
    kind: str                      # stall | lamp | fence | house
    name: str                      # display name ("Market stall")
    x: float
    z: float
    rotation: float = 0.0          # degrees, around Y
    delivered: dict = field(default_factory=dict)   # resource -> count
    work_done: float = 0.0
    anchor_name: str | None = None  # set at completion, if it has one

    @property
    def spec(self) -> dict:
        return BUILDINGS[self.kind]

    @property
    def costs(self) -> dict:
        return dict(self.spec["costs"])

    @property
    def work_required(self) -> float:
        return float(self.spec["work"])

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.z)

    @property
    def complete(self) -> bool:
        return self.work_done >= self.work_required

    @property
    def stage(self) -> str:
        """site -> frame (at 50% of the work) -> complete."""
        if self.complete:
            return "complete"
        return "frame" if self.work_done >= self.work_required / 2.0 \
            else "site"

    @property
    def stage_index(self) -> int:
        """0 while in the site stage, 1 in the frame stage."""
        return 0 if self.work_done < self.work_required / 2.0 else 1

    def tranche(self, stage_index: int | None = None) -> dict:
        """The materials the given (default: current) stage needs."""
        first, second = stage_tranches(self.costs)
        idx = self.stage_index if stage_index is None else stage_index
        return first if idx == 0 else second

    def delivered_in_stage(self, stage_index: int | None = None) -> dict:
        """How much of the given stage's tranche is already on site."""
        idx = self.stage_index if stage_index is None else stage_index
        first, _second = stage_tranches(self.costs)
        out = {}
        for key, need in self.tranche(idx).items():
            prior = first.get(key, 0) if idx == 1 else 0
            out[key] = max(0, min(need, self.delivered.get(key, 0) - prior))
        return out

    def stage_supplied(self) -> bool:
        """True when the current stage's tranche is fully delivered."""
        have = self.delivered_in_stage()
        return all(have.get(k, 0) >= v for k, v in self.tranche().items())

    @property
    def state(self) -> str:
        """awaiting_materials | building | complete."""
        if self.complete:
            return "complete"
        return "building" if self.stage_supplied() else "awaiting_materials"

    @property
    def progress(self) -> float:
        return min(1.0, self.work_done / self.work_required)

    def to_dict(self) -> dict:
        """The full public record — /api/world, layout documents and
        serialize() all use exactly this shape, and restore() reads
        it back. Dynamic kinds (P6 innovation buildings) also carry
        their rendering "archetype" — a seed kind whose shape the
        client draws them with."""
        out = {"id": self.id, "kind": self.kind, "name": self.name,
               "position": [self.x, self.z], "rotation": self.rotation,
               "stage": self.stage, "state": self.state,
               "progress": round(self.progress, 4),
               "materials_required": self.costs,
               "materials_delivered": dict(self.delivered),
               "work_done": round(self.work_done, 3),
               "work_required": self.work_required,
               "anchor": self.anchor_name}
        archetype = self.spec.get("archetype")
        if archetype:
            out["archetype"] = archetype
        return out


def _clean_site_record(rec, i: int) -> dict:
    """Validate one serialized site record (layout documents,
    restore): kind/position are structural; progress fields are
    clamped into range rather than rejected."""
    if not isinstance(rec, dict):
        raise ValueError(f"sites[{i}] must be an object")
    kind = rec.get("kind")
    if kind not in BUILDINGS:
        raise ValueError(f"sites[{i}].kind {kind!r} is not buildable")
    pos = rec.get("position")
    if not (isinstance(pos, list) and len(pos) == 2
            and all(isinstance(v, (int, float)) for v in pos)):
        raise ValueError(f"sites[{i}].position must be [x, z]")
    delivered = rec.get("materials_delivered") or {}
    if not isinstance(delivered, dict):
        raise ValueError(f"sites[{i}].materials_delivered must be an object")
    costs = BUILDINGS[kind]["costs"]
    clean_delivered = {}
    for key, amount in delivered.items():
        if key in costs and isinstance(amount, (int, float)):
            clean_delivered[key] = max(0, min(int(amount), costs[key]))
    work = BUILDINGS[kind]["work"]
    try:
        work_done = float(rec.get("work_done", 0.0))
    except (TypeError, ValueError):
        work_done = 0.0
    rotation = rec.get("rotation", 0)
    if not isinstance(rotation, (int, float)):
        rotation = 0
    anchor = rec.get("anchor")
    return {"id": str(rec.get("id") or f"site-{i + 1}"), "kind": kind,
            "name": BUILDINGS[kind]["name"],
            "position": [float(pos[0]), float(pos[1])],
            "rotation": float(rotation),
            "materials_delivered": clean_delivered,
            "work_done": max(0.0, min(work_done, work)),
            "anchor": anchor if isinstance(anchor, str) else None}


def _clean_building_record(rec, i: int) -> dict:
    if not isinstance(rec, dict):
        raise ValueError(f"buildings[{i}] must be an object")
    kind = rec.get("kind")
    if kind not in BUILDINGS:
        raise ValueError(f"buildings[{i}].kind {kind!r} is not buildable")
    pos = rec.get("position")
    if not (isinstance(pos, list) and len(pos) == 2
            and all(isinstance(v, (int, float)) for v in pos)):
        raise ValueError(f"buildings[{i}].position must be [x, z]")
    rotation = rec.get("rotation", 0)
    if not isinstance(rotation, (int, float)):
        rotation = 0
    anchor = rec.get("anchor")
    out = {"id": str(rec.get("id") or f"building-{i + 1}"), "kind": kind,
           "name": BUILDINGS[kind]["name"],
           "position": [float(pos[0]), float(pos[1])],
           "rotation": float(rotation),
           "anchor": anchor if isinstance(anchor, str) else None}
    archetype = rec.get("archetype")
    if isinstance(archetype, str) and archetype in BUILDINGS:
        out["archetype"] = archetype
    return out


def validate_construction(doc) -> dict:
    """Clean the {"sites": [...], "buildings": [...]} slice of a
    layout/world document. Missing keys clean to empty lists."""
    if not isinstance(doc, dict):
        raise ValueError("construction document must be an object")
    sites_in = doc.get("sites") or []
    buildings_in = doc.get("buildings") or []
    if not isinstance(sites_in, list) or not isinstance(buildings_in, list):
        raise ValueError("'sites' and 'buildings' must be lists")
    return {"sites": [_clean_site_record(r, i)
                       for i, r in enumerate(sites_in)],
            "buildings": [_clean_building_record(r, i)
                          for i, r in enumerate(buildings_in)]}


class ConstructionWorld:
    """Construction sites + finished buildings + the build rules."""

    def __init__(self):
        self.sites: dict[str, ConstructionSite] = {}
        self.buildings: list[dict] = []
        self._seq = 0
        self._anchor_counts: dict[str, int] = {}
        # The live world bound (P9): a callable returning the
        # current playable +/- bound, wired by WorldBridge to the
        # terrain world. None = the founding bound, always.
        self.bound_fn = None

    def _bound(self) -> float:
        return float(self.bound_fn()) if self.bound_fn is not None \
            else WORLD_BOUND

    # -- placement ------------------------------------------------------
    def validate_placement(self, kind: str, x: float, z: float,
                           layout_objects=()) -> str | None:
        """Why this spot refuses a building of `kind`, or None when
        it is clear: inside the world bounds, off the road bands,
        and not overlapping a layout object (per the bridge's saved
        layout), another site, or a finished building. Footprints
        are axis-aligned boxes — rotation is visual only."""
        if kind not in BUILDINGS:
            return (f"Unknown building {kind!r} — want one of: "
                    + ", ".join(BUILDINGS) + ".")
        fw, fd = BUILDINGS[kind]["footprint"]
        hw, hd = fw / 2.0, fd / 2.0
        bound = self._bound()
        if abs(x) > bound - hw or abs(z) > bound - hd:
            return "That spot is outside the city bounds."
        for c in ROADS:
            if abs(x - c) < ROAD_HALF + hw or abs(z - c) < ROAD_HALF + hd:
                return "That spot is on a road — buildings need a clear plot."

        def overlaps(ox: float, oz: float, ow: float, od: float) -> bool:
            return (abs(x - ox) < hw + ow / 2.0 + _OVERLAP_MARGIN
                    and abs(z - oz) < hd + od / 2.0 + _OVERLAP_MARGIN)

        for obj in layout_objects or ():
            size = _LAYOUT_OBJECT_SIZE.get(obj.get("kind"))
            if size is None:
                continue
            pos = obj.get("position") or [0, 0, 0]
            if overlaps(float(pos[0]), float(pos[2]), size[0], size[1]):
                return "That spot overlaps an existing structure."
        for site in self.sites.values():
            sw, sd = BUILDINGS[site.kind]["footprint"]
            if overlaps(site.x, site.z, sw, sd):
                return "That spot overlaps another construction site."
        for b in self.buildings:
            bw, bd = BUILDINGS[b["kind"]]["footprint"]
            if overlaps(b["position"][0], b["position"][1], bw, bd):
                return "That spot overlaps a finished building."
        return None

    def place(self, kind: str, x: float, z: float, rotation: float = 0.0,
              layout_objects=()):
        """Create a construction site. Returns (site, None) or
        (None, human_reason)."""
        reason = self.validate_placement(kind, x, z, layout_objects)
        if reason is not None:
            return None, reason
        self._seq += 1
        site = ConstructionSite(
            id=f"site-{self._seq}", kind=kind,
            name=BUILDINGS[kind]["name"],
            x=float(x), z=float(z), rotation=float(rotation))
        self.sites[site.id] = site
        return site, None

    # -- the build itself --------------------------------------------------
    def _try_deliver(self, site: ConstructionSite, inventory: dict) -> bool:
        """Draw the site's current-stage tranche from the inventory
        if the stores hold all of it (all-or-nothing per tranche —
        see the module docstring). Returns True when the current
        stage stands fully supplied."""
        if site.stage_supplied():
            return True
        tranche = site.tranche()
        if any(inventory.get(k, 0) < v for k, v in tranche.items()):
            return False
        for k, v in tranche.items():
            if v:
                inventory[k] = inventory.get(k, 0) - v
                site.delivered[k] = site.delivered.get(k, 0) + v
        return True

    def work_at(self, site_id: str, dt: float, inventory: dict,
                by: str = "") -> list[dict]:
        """One builder works at a site for `dt` sim-seconds: deliver
        the current stage's tranche if the stores allow, then accrue
        work — but only while the stage stands supplied (the module
        docstring's rule). Returns the build events this tick
        produced (stage reached / completion). Completion also
        registers the finished building and reports the planner
        anchor to add, if the structure has one."""
        site = self.sites.get(site_id)
        if site is None or site.complete or dt <= 0:
            return []
        events: list[dict] = []
        stage_before = site.stage
        if self._try_deliver(site, inventory):
            site.work_done = min(site.work_required,
                                 site.work_done + dt)
        if site.stage != stage_before and site.stage == "frame":
            events.append(self._event(site, "frame", by=by))
        if site.complete:
            anchor = None
            spec_anchor = site.spec.get("anchor")
            if spec_anchor is not None:
                kind_a, prefix = spec_anchor
                n = self._anchor_counts.get(prefix, 0) + 1
                self._anchor_counts[prefix] = n
                site.anchor_name = f"{prefix} {n}"
                anchor = {"name": site.anchor_name, "kind": kind_a,
                          "x": site.x, "z": site.z}
            building = {"id": site.id, "kind": site.kind,
                        "name": site.name,
                        "position": [site.x, site.z],
                        "rotation": site.rotation,
                        "anchor": site.anchor_name}
            if site.spec.get("archetype"):
                building["archetype"] = site.spec["archetype"]
            self.buildings.append(building)
            events.append(self._event(site, "complete", by=by,
                                      building=building, anchor=anchor))
        return events

    @staticmethod
    def _event(site: ConstructionSite, state: str, **extra) -> dict:
        ev = {"type": "build", "site": site.id, "kind": site.kind,
              "name": site.name, "state": state,
              "position": [site.x, site.z],
              "stage": site.stage, "progress": round(site.progress, 4)}
        ev.update(extra)
        return ev

    def placed_event(self, site: ConstructionSite) -> dict:
        return self._event(site, "placed")

    # -- read models ---------------------------------------------------------
    def public_sites(self) -> list[dict]:
        return [s.to_dict() for s in self.sites.values()
                if not s.complete]

    def public_buildings(self) -> list[dict]:
        return [dict(b) for b in self.buildings]

    def summary(self) -> dict:
        """The construction picture for digests (the interpreter)."""
        return {"sites": [s.to_dict() for s in self.sites.values()
                          if not s.complete],
                "buildings": self.public_buildings()}

    # -- save / load -----------------------------------------------------------
    def serialize(self) -> dict:
        """Full-fidelity world slice for layout documents: every
        site (finished ones included — restore sorts them into the
        building list) plus the anchor counters."""
        return {"sites": [s.to_dict() for s in self.sites.values()],
                "buildings": self.public_buildings(),
                "anchor_counts": dict(self._anchor_counts)}

    def restore(self, doc) -> dict:
        """Replace all construction state from a serialized slice
        (validated first — a malformed document changes nothing).
        Returns {"sites": n_open, "buildings": n}."""
        clean = validate_construction(doc)
        sites: dict[str, ConstructionSite] = {}
        buildings: list[dict] = []
        seq = 0
        for rec in clean["sites"]:
            site = ConstructionSite(
                id=rec["id"], kind=rec["kind"], name=rec["name"],
                x=rec["position"][0], z=rec["position"][1],
                rotation=rec["rotation"],
                delivered=dict(rec["materials_delivered"]),
                work_done=rec["work_done"],
                anchor_name=rec.get("anchor"))
            sites[site.id] = site
            try:
                seq = max(seq, int(site.id.rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                pass
            if site.complete:
                buildings.append({"id": site.id, "kind": site.kind,
                                  "name": site.name,
                                  "position": [site.x, site.z],
                                  "rotation": site.rotation,
                                  "anchor": site.anchor_name})
        seen = {b["id"] for b in buildings}
        for rec in clean["buildings"]:
            if rec["id"] not in seen:
                buildings.append(rec)
                seen.add(rec["id"])
        self.sites = sites
        self.buildings = buildings
        self._seq = max(self._seq, seq)
        counts = doc.get("anchor_counts") if isinstance(doc, dict) \
            else None
        if isinstance(counts, dict):
            self._anchor_counts = {str(k): int(v) for k, v in counts.items()
                                   if isinstance(v, (int, float))}
        return {"sites": sum(1 for s in sites.values() if not s.complete),
                "buildings": len(buildings)}
