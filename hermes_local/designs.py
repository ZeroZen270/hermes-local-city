"""Brain-designed world content — Phase 8 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, Addendum 4 ("Use the
brain as the world's designer").

The strong free brain (LongCat on the tablet build) is a good
*designer* and a bad *authority* — so, exactly as innovations.py
handles adopted discoveries, this engine lets the model DESIGN
and keeps the engine in charge of what exists:

  1. **Design.** An injectable design brain writes a STRICT JSON
     design spec (schema below) — a building style, a building
     interior, or an outdoor asset composition — from a hint (a
     citizen's voted wish, the Keeper's god-mode request).
  2. **Validation.** The engine checks the spec deterministically:
     palettes are real hex in a sane range, roofs and props come
     from fixed vocabularies, props stand inside the footprint,
     counts clamp. A failing spec is fed back ONCE with its
     errors for revision; a second failure **shelves** the design
     (recorded with its reason, nothing rendered) — never faked.
  3. **Registration.** A valid design goes live in the running
     world: styles re-skin their archetype's buildings, interiors
     furnish their building kind (or one specific building), and
     assets are placed in the world as layout props (a
     "design-asset:<id>" marker object in the layout document)
     for the client to render from the composition.

Three hand-authored seed designs (a house interior, a stall
interior, one plaza fountain) are registered at construction so
interiors exist before any brain call — authored baseline
content, labeled origin "seeded", not a fake "generation".
With no design brain wired, an adopted asset falls back to the
closest seed template by hint keyword, labeled origin
"template"; styles and interiors honestly shelve ("no design
brain wired") rather than inventing a design.

Designs are durable world truth: they ride the world snapshot's
"world" section (persistence.py) and the layout document's
"designs" slice, both additively.
"""
from __future__ import annotations

import queue
import re
import threading

from .buildings import BUILDINGS, ROADS, ROAD_HALF, WORLD_BOUND
from .innovations import slugify

# -- the design vocabularies -------------------------------------------------
SPEC_KINDS = ("building_style", "interior", "asset")
ROOF_STYLES = ("flat", "gable", "hip")
# Furniture an interior may hold.
INTERIOR_PROPS = ("table", "chair", "bed", "shelf", "counter",
                  "stool", "plant", "lamp", "rug", "bookshelf",
                  "crate")
# An outdoor asset composes from the furniture set plus pieces
# that only make sense under the sky.
ASSET_PROPS = INTERIOR_PROPS + ("fountain", "bench", "planter",
                                "statue", "arch")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_CHANNEL_MIN, _CHANNEL_MAX = 0x22, 0xEE   # a sane range: no void, no glare

# -- clamps (a spec outside these is pulled into range, not rejected) -------
MAX_ROOMS = 5
MAX_PROPS_PER_ROOM = 12
MAX_PROPS_TOTAL = 40
FOOTPRINT_RANGE = (2.0, 14.0)

# Spot-search ring around an anchor for an adopted asset — the
# same ring civics.py searches for its builds and plantings.
_SPOT_OFFSETS = [(0, 0), (7, 0), (-7, 0), (0, 7), (0, -7),
                 (11, 5), (-11, 5), (5, 11), (5, -11), (-5, -11),
                 (-11, -5), (14, 0), (0, 14), (-14, 0), (0, -14),
                 (18, 8), (-18, -8), (8, -18), (-8, 18),
                 (22, 0), (0, 22), (-22, 0), (0, -22)]
# Approximate footprints of designer layout objects (mirrors
# buildings.py's table) for asset overlap checks.
_LAYOUT_OBJECT_SIZE = {"building": (16.0, 16.0), "house": (10.0, 10.0),
                       "industrial": (16.0, 16.0),
                       "vehicle": (4.4, 2.0), "prop": (1.5, 1.5)}

# A worked example of a valid spec (the design brain's prompt
# shows it; the validator is tuned against its shape).
EXAMPLE_DESIGN = {
    "kind": "interior", "name": "Cosy cottage rooms",
    "building_kind": "house",
    "rooms": [{"name": "Main room", "props": [
        {"kind": "table", "dx": 0.0, "dz": 0.5, "rot": 0},
        {"kind": "chair", "dx": -1.2, "dz": 0.5, "rot": 90},
        {"kind": "rug", "dx": 0.0, "dz": 0.8, "rot": 0}]}]}

# -- the hand-authored seeds ---------------------------------------------------
SEED_HOUSE_INTERIOR = {
    "kind": "interior", "name": "Cosy cottage rooms",
    "building_kind": "house",
    "rooms": [
        {"name": "Main room", "props": [
            {"kind": "table", "dx": 0.0, "dz": 0.5, "rot": 0},
            {"kind": "chair", "dx": -1.2, "dz": 0.5, "rot": 90},
            {"kind": "chair", "dx": 1.2, "dz": 0.5, "rot": -90},
            {"kind": "rug", "dx": 0.0, "dz": 0.9, "rot": 0},
            {"kind": "bookshelf", "dx": -4.2, "dz": -1.0, "rot": 90},
            {"kind": "lamp", "dx": 4.2, "dz": -4.0, "rot": 0},
            {"kind": "plant", "dx": -4.2, "dz": 3.8, "rot": 0}]},
        {"name": "Sleeping corner", "props": [
            {"kind": "bed", "dx": 3.2, "dz": 3.4, "rot": 0},
            {"kind": "crate", "dx": 4.4, "dz": 1.8, "rot": 20},
            {"kind": "stool", "dx": 1.8, "dz": 4.2, "rot": 0}]}]}
SEED_STALL_INTERIOR = {
    "kind": "interior", "name": "Trader's stall dressing",
    "building_kind": "stall",
    "rooms": [{"name": "Stall floor", "props": [
        {"kind": "counter", "dx": 0.0, "dz": 1.4, "rot": 0},
        {"kind": "stool", "dx": -1.6, "dz": 0.2, "rot": 0},
        {"kind": "crate", "dx": 2.2, "dz": -1.4, "rot": 15},
        {"kind": "crate", "dx": 2.4, "dz": -0.4, "rot": -10},
        {"kind": "shelf", "dx": -2.4, "dz": -1.6, "rot": 90},
        {"kind": "plant", "dx": 2.6, "dz": 1.8, "rot": 0}]}]}
SEED_PLAZA_FOUNTAIN = {
    "kind": "asset", "name": "Founders' fountain",
    "footprint": [5.0, 5.0], "position": [12.0, 12.0],
    "props": [
        {"kind": "fountain", "dx": 0.0, "dz": 0.0, "rot": 0},
        {"kind": "bench", "dx": -2.0, "dz": 1.6, "rot": 45},
        {"kind": "bench", "dx": 2.0, "dz": -1.6, "rot": -135},
        {"kind": "planter", "dx": 2.1, "dz": 2.1, "rot": 0},
        {"kind": "planter", "dx": -2.1, "dz": -2.1, "rot": 0},
        {"kind": "lamp", "dx": 0.0, "dz": -2.2, "rot": 0}]}

# Template fallbacks for adopted assets when no design brain is
# wired: the seed compositions, honestly labeled "template". The
# keyword picks the closest one; anything else gets the garden.
ASSET_TEMPLATES = (
    (("fountain", "spring", "well"), SEED_PLAZA_FOUNTAIN),
    (("arch", "gate", "gateway"), {
        "kind": "asset", "name": "Market arch",
        "footprint": [6.0, 3.0],
        "props": [
            {"kind": "arch", "dx": 0.0, "dz": 0.0, "rot": 0},
            {"kind": "planter", "dx": -2.6, "dz": 0.8, "rot": 0},
            {"kind": "planter", "dx": 2.6, "dz": 0.8, "rot": 0},
            {"kind": "lamp", "dx": -2.6, "dz": -0.9, "rot": 0},
            {"kind": "lamp", "dx": 2.6, "dz": -0.9, "rot": 0}]}),
    (("bench", "garden", "park", "seat", "green"), {
        "kind": "asset", "name": "Garden benches",
        "footprint": [6.0, 6.0],
        "props": [
            {"kind": "bench", "dx": -1.8, "dz": 0.0, "rot": 90},
            {"kind": "bench", "dx": 1.8, "dz": 0.0, "rot": -90},
            {"kind": "plant", "dx": 0.0, "dz": -1.8, "rot": 0},
            {"kind": "plant", "dx": 0.0, "dz": 1.8, "rot": 0},
            {"kind": "planter", "dx": -2.4, "dz": -2.4, "rot": 0},
            {"kind": "planter", "dx": 2.4, "dz": 2.4, "rot": 0},
            {"kind": "lamp", "dx": 0.0, "dz": 0.0, "rot": 0}]}),
)


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clean_hex(value, errors: list, what: str):
    """A palette color: strict #rrggbb, every channel inside the
    sane range (no void black, no glare white)."""
    text = str(value or "").strip()
    if not _HEX.match(text):
        errors.append(f"{what} must be a hex color like '#a8804f', "
                      f"got {value!r}")
        return None
    channels = [int(text[i:i + 2], 16) for i in (1, 3, 5)]
    if any(c < _CHANNEL_MIN or c > _CHANNEL_MAX for c in channels):
        errors.append(f"{what} {text} is outside the sane range — "
                      f"each channel must sit between "
                      f"#{_CHANNEL_MIN:02x} and #{_CHANNEL_MAX:02x}")
        return None
    return text.lower()


def _clean_props(raw, vocab, hw: float, hd: float, errors: list,
                 what: str) -> list:
    """A prop composition: known kinds only, every prop standing
    inside the footprint (half extents hw/hd), rotation wrapped
    to degrees. Counts clamp at the caller."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"{what} must be a list of props")
        return []
    out = []
    for i, prop in enumerate(raw):
        if not isinstance(prop, dict):
            errors.append(f"{what}[{i}] must be an object")
            continue
        kind = str(prop.get("kind") or "")
        if kind not in vocab:
            errors.append(f"{what}[{i}] names unknown prop "
                          f"{kind!r} — props must be one of: "
                          f"{', '.join(vocab)}")
            continue
        dx = _clamp(prop.get("dx"), -1e6, 1e6, None)
        dz = _clamp(prop.get("dz"), -1e6, 1e6, None)
        if dx is None or dz is None:
            errors.append(f"{what}[{i}] ({kind}) needs numeric "
                          f"dx/dz offsets")
            continue
        if abs(dx) > hw or abs(dz) > hd:
            errors.append(f"{what}[{i}] ({kind}) stands at "
                          f"({dx}, {dz}), outside the footprint "
                          f"(+/-{hw} by +/-{hd})")
            continue
        rot = _clamp(prop.get("rot"), -360.0, 720.0, 0.0) % 360.0
        out.append({"kind": kind, "dx": round(dx, 2),
                    "dz": round(dz, 2), "rot": round(rot, 1)})
    return out


def validate_design(raw, ctx: dict):
    """Check a design spec against the world. `ctx` carries the
    world's truth: {"building_kinds", "building_ids",
    "design_ids", "position" (the engine's chosen spot for an
    asset, or None), "spot_check" (callable(x, z, w, d) ->
    reason | None)}. Returns (clean_spec, []) or
    (None, [human error, ...]) — every rule that fired, so the
    revision pass can fix them all at once. Counts and sizes
    CLAMP; structural problems — unknown kinds/props/archetypes,
    off-palette colors, props outside the footprint, an id the
    world already holds — are errors."""
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["the design is not a JSON object"]
    kind = str(raw.get("kind") or "")
    if kind not in SPEC_KINDS:
        errors.append("kind must be one of: " + ", ".join(SPEC_KINDS))
        return None, errors
    name = str(raw.get("name") or "").strip()[:48]
    if not name:
        errors.append("the design needs a name")
    design_id = slugify(raw.get("id") or name)
    if not design_id:
        errors.append("the design needs an id")
    elif design_id in (ctx.get("design_ids") or ()):
        errors.append(f"the world already holds a design "
                      f"called {design_id!r}")
    if errors:
        return None, errors

    if kind == "building_style":
        archetype = str(raw.get("archetype") or "")
        if archetype not in BUILDINGS:
            errors.append("archetype must be one of: "
                          + ", ".join(BUILDINGS))
        palette_in = raw.get("palette")
        if not isinstance(palette_in, dict):
            errors.append("palette must be an object with wall, "
                          "roof and trim hex colors")
            palette = {}
        else:
            palette = {}
            for key in ("wall", "roof", "trim"):
                color = _clean_hex(palette_in.get(key), errors,
                                   f"palette.{key}")
                if color is not None:
                    palette[key] = color
        roof_style = str(raw.get("roof_style") or "")
        if roof_style not in ROOF_STYLES:
            errors.append("roof_style must be one of: "
                          + ", ".join(ROOF_STYLES))
        if errors:
            return None, errors
        return {"kind": kind, "id": design_id, "name": name,
                "archetype": archetype, "palette": palette,
                "roof_style": roof_style,
                "floors": int(_clamp(raw.get("floors"), 1, 3, 1))}, []

    if kind == "interior":
        building_kind = raw.get("building_kind")
        building_id = raw.get("building_id")
        if building_kind is not None:
            building_kind = str(building_kind)
            if building_kind not in (ctx.get("building_kinds")
                                     or set(BUILDINGS)):
                errors.append(f"building_kind {building_kind!r} is "
                              f"not a building of this world")
                building_kind = None
        if building_id is not None:
            building_id = str(building_id)
            if building_id not in (ctx.get("building_ids") or ()):
                errors.append(f"no finished building with id "
                              f"{building_id!r} stands in this world")
                building_id = None
        if building_kind is None and building_id is None \
                and not errors:
            errors.append("an interior needs a building_kind (the "
                          "buildings it furnishes) or a building_id "
                          "(one specific building)")
        # The footprint props must fit: the specific building's
        # kind when an id names one, else the named kind.
        foot_kind = building_kind
        if foot_kind is None and building_id is not None:
            foot_kind = (ctx.get("building_kinds_by_id") or {}
                         ).get(building_id)
        rooms_in = raw.get("rooms")
        if not isinstance(rooms_in, list) or not rooms_in:
            errors.append("an interior needs at least one room")
            rooms_in = []
        rooms = []
        total = 0
        if foot_kind in BUILDINGS:
            fw, fd = BUILDINGS[foot_kind]["footprint"]
            hw, hd = fw / 2.0, fd / 2.0
            for i, room in enumerate(rooms_in[:MAX_ROOMS]):
                if not isinstance(room, dict):
                    errors.append(f"rooms[{i}] must be an object")
                    continue
                props = _clean_props(
                    room.get("props"), INTERIOR_PROPS, hw, hd,
                    errors, f"rooms[{i}].props")
                total += len(props)
                rooms.append({
                    "name": str(room.get("name") or f"Room {i + 1}")[:32],
                    "props": props[:MAX_PROPS_PER_ROOM]})
        if total > MAX_PROPS_TOTAL:
            errors.append(f"an interior holds at most "
                          f"{MAX_PROPS_TOTAL} props, not {total}")
        if errors:
            return None, errors
        return {"kind": kind, "id": design_id, "name": name,
                "building_kind": building_kind,
                "building_id": building_id, "rooms": rooms}, []

    # kind == "asset"
    fp_in = raw.get("footprint") or [4.0, 4.0]
    if not (isinstance(fp_in, list) and len(fp_in) == 2):
        errors.append("footprint must be [width, depth] in metres")
        fp_in = [4.0, 4.0]
    fw = _clamp(fp_in[0], *FOOTPRINT_RANGE, 4.0)
    fd = _clamp(fp_in[1], *FOOTPRINT_RANGE, 4.0)
    props = _clean_props(raw.get("props"), ASSET_PROPS, fw / 2.0,
                         fd / 2.0, errors, "props")
    if not props and not errors:
        errors.append("an asset needs at least one prop")
    # Placement: a spec may suggest a spot, but the engine owns
    # the ground — an unusable suggestion falls back to the spot
    # the engine chose (ctx), and only having NO valid spot at
    # all is an error.
    spot_check = ctx.get("spot_check")
    position = None
    candidates = []
    if isinstance(raw.get("position"), list) \
            and len(raw["position"]) == 2:
        candidates.append(raw["position"])
    if ctx.get("position"):
        candidates.append(ctx["position"])
    for cand in candidates:
        try:
            x, z = float(cand[0]), float(cand[1])
        except (TypeError, ValueError):
            continue
        if spot_check is None or spot_check(x, z, fw, fd) is None:
            position = [round(x, 1), round(z, 1)]
            break
    if position is None and not errors:
        errors.append("no clear ground could be found for the "
                      "asset — every candidate spot is taken, on "
                      "a road, or out of bounds")
    if errors:
        return None, errors
    return {"kind": kind, "id": design_id, "name": name,
            "footprint": [fw, fd], "position": position,
            "props": props}, []


def validate_designs(doc) -> dict:
    """Clean the {"styles", "interiors", "assets", "shelved"}
    slice of a world save / layout document. Records were
    validated at registration; restore checks structure only and
    is all-or-nothing, like innovations.validate_innovations."""
    if not isinstance(doc, dict):
        raise ValueError("designs document must be an object")
    out = {"styles": [], "interiors": [], "assets": [],
           "shelved": []}
    for key in ("styles", "interiors", "assets"):
        records = doc.get(key) or []
        if not isinstance(records, list):
            raise ValueError(f"designs.{key} must be a list")
        want = {"styles": "building_style", "interiors": "interior",
                "assets": "asset"}[key]
        for i, rec in enumerate(records):
            if not isinstance(rec, dict) or not rec.get("id") \
                    or not rec.get("name") \
                    or rec.get("kind") != want:
                raise ValueError(
                    f"designs.{key}[{i}] needs an id, a name and "
                    f"kind {want!r}")
            clean = {"kind": want, "id": str(rec["id"]),
                     "name": str(rec["name"]),
                     "origin": str(rec.get("origin") or "brain"),
                     "proposer": str(rec.get("proposer") or ""),
                     "hint": str(rec.get("hint") or "")}
            if want == "building_style":
                palette = rec.get("palette") or {}
                clean.update({
                    "archetype": str(rec.get("archetype") or ""),
                    "palette": {k: str(palette.get(k) or "")
                                for k in ("wall", "roof", "trim")},
                    "roof_style": str(rec.get("roof_style")
                                      or "gable"),
                    "floors": int(rec.get("floors") or 1)})
            elif want == "interior":
                rooms = []
                for room in rec.get("rooms") or []:
                    if not isinstance(room, dict):
                        continue
                    props = [{"kind": str(p.get("kind")),
                              "dx": float(p.get("dx") or 0),
                              "dz": float(p.get("dz") or 0),
                              "rot": float(p.get("rot") or 0)}
                             for p in room.get("props") or []
                             if isinstance(p, dict)
                             and p.get("kind") in INTERIOR_PROPS]
                    rooms.append({"name": str(
                        room.get("name") or "Room"), "props": props})
                clean.update({
                    "building_kind": (str(rec["building_kind"])
                                      if rec.get("building_kind")
                                      else None),
                    "building_id": (str(rec["building_id"])
                                    if rec.get("building_id")
                                    else None),
                    "rooms": rooms})
            else:
                fp = rec.get("footprint") or [4.0, 4.0]
                pos = rec.get("position") or [0.0, 0.0]
                clean.update({
                    "footprint": [float(fp[0]), float(fp[1])],
                    "position": [float(pos[0]), float(pos[1])],
                    "props": [{"kind": str(p.get("kind")),
                               "dx": float(p.get("dx") or 0),
                               "dz": float(p.get("dz") or 0),
                               "rot": float(p.get("rot") or 0)}
                              for p in rec.get("props") or []
                              if isinstance(p, dict)
                              and p.get("kind") in ASSET_PROPS]})
            out[key].append(clean)
    shelved = doc.get("shelved") or []
    if not isinstance(shelved, list):
        raise ValueError("designs.shelved must be a list")
    for rec in shelved:
        if isinstance(rec, dict) and rec.get("hint") is not None:
            out["shelved"].append({
                "kind": str(rec.get("kind") or "asset"),
                "hint": str(rec.get("hint") or "")[:120],
                "reason": str(rec.get("reason") or "")[:240],
                "proposer": str(rec.get("proposer") or "")})
    return out


class DesignEngine:
    """The design registry + the brain-design pipeline.

    Wiring: WorldBridge constructs the engine with itself and
    exposes it as planner.designs (the civics executor reaches
    adopt_asset through the planner). `design_brain` (callable(ctx)
    -> raw spec dict | None) is injected: build_world wires the
    live brain; tests wire fakes. With no brain, adopted assets
    fall back to labeled seed templates and other kinds shelve —
    the engine registers nothing it cannot stand behind."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.styles: dict[str, dict] = {}      # archetype -> record
        self.interiors: list[dict] = []
        self.assets: list[dict] = []
        self.shelved: list[dict] = []
        self.design_brain = None
        self._results: queue.Queue = queue.Queue()
        self._seed()

    # -- the seeds ---------------------------------------------------------
    def _seed(self) -> None:
        """Register the hand-authored baseline designs (see the
        module docstring): two interiors + the plaza fountain,
        all labeled origin "seeded". They pass the same
        validation as any brain design."""
        for raw in (SEED_HOUSE_INTERIOR, SEED_STALL_INTERIOR,
                    SEED_PLAZA_FOUNTAIN):
            clean, errors = validate_design(raw, self._ctx())
            if clean is not None:
                self._register(clean, origin="seeded", proposer="",
                               hint="")

    # -- world access ------------------------------------------------------
    @property
    def _planner(self):
        manager = getattr(self.bridge, "manager", None)
        return manager.planner if manager is not None else None

    def _design_ids(self) -> set:
        ids = {r["id"] for r in self.styles.values()}
        ids |= {r["id"] for r in self.interiors}
        ids |= {r["id"] for r in self.assets}
        return ids

    def _ctx(self, position=None) -> dict:
        buildings = self.bridge.construction.buildings
        return {"building_kinds": set(BUILDINGS),
                "building_ids": {b["id"] for b in buildings},
                "building_kinds_by_id": {
                    b["id"]: (b.get("archetype") or b["kind"])
                    for b in buildings},
                "design_ids": self._design_ids(),
                "position": position,
                "spot_check": self.spot_check}

    def spot_check(self, x: float, z: float, w: float,
                   d: float) -> str | None:
        """Why this ground refuses an asset of footprint (w, d),
        or None when it is clear — the building placement rules
        (bounds, road bands, overlap) applied to a prop."""
        hw, hd = w / 2.0, d / 2.0
        # The live bound (P9): the bridge's terrain carries it; a
        # bare engine keeps the founding bound.
        bound = WORLD_BOUND
        ter = getattr(self.bridge, "terrain", None)
        if ter is not None:
            bound = ter.bound
        if abs(x) > bound - hw or abs(z) > bound - hd:
            return "outside the city bounds"
        for c in ROADS:
            if abs(x - c) < ROAD_HALF + hw or abs(z - c) < ROAD_HALF + hd:
                return "on a road"

        def overlaps(ox, oz, ow, od):
            return (abs(x - ox) < hw + ow / 2.0 + 0.5
                    and abs(z - oz) < hd + od / 2.0 + 0.5)

        layout = getattr(self.bridge, "layout", None) or {}
        for obj in layout.get("objects", []):
            if str(obj.get("asset", "")).startswith("design-asset:"):
                continue        # asset markers check via self.assets
            size = _LAYOUT_OBJECT_SIZE.get(obj.get("kind"))
            if size is None:
                continue
            pos = obj.get("position") or [0, 0, 0]
            if overlaps(float(pos[0]), float(pos[2]), size[0], size[1]):
                return "overlaps an existing structure"
        con = self.bridge.construction
        for site in con.sites.values():
            if site.complete:
                continue
            sw, sd = BUILDINGS[site.kind]["footprint"]
            if overlaps(site.x, site.z, sw, sd):
                return "overlaps a construction site"
        for b in con.buildings:
            bw, bd = BUILDINGS[b["kind"]]["footprint"]
            if overlaps(b["position"][0], b["position"][1], bw, bd):
                return "overlaps a finished building"
        for a in self.assets:
            aw, ad = a["footprint"]
            if overlaps(a["position"][0], a["position"][1], aw, ad):
                return "overlaps another designed asset"
        return None

    def _find_spot(self, anchor_name: str = "Central Plaza"):
        """The first clear ground in the spot ring around an
        anchor (a 4x4 asset's footprint decides road clearance),
        or None."""
        planner = self._planner
        anchor = None
        if planner is not None:
            for a in planner.anchors:
                if a.name == anchor_name:
                    anchor = a
                    break
        cx, cz = (anchor.x, anchor.z) if anchor else (8.0, 8.0)
        for ox, oz in _SPOT_OFFSETS:
            x, z = cx + ox, cz + oz
            if self.spot_check(x, z, 4.0, 4.0) is None:
                return [x, z]
        return None

    # -- read models ---------------------------------------------------------
    def style_for(self, archetype: str) -> dict | None:
        return self.styles.get(archetype)

    def interior_for(self, building: dict) -> dict | None:
        """The interior furnishing a finished building record: a
        design naming the building itself first, then its kind,
        then its archetype — latest registration wins."""
        bid = building.get("id")
        kind = building.get("kind")
        arche = building.get("archetype")
        for rec in reversed(self.interiors):
            if bid and rec.get("building_id") == bid:
                return rec
        for rec in reversed(self.interiors):
            if kind and rec.get("building_kind") == kind:
                return rec
        for rec in reversed(self.interiors):
            if arche and rec.get("building_kind") == arche:
                return rec
        return None

    def interior_for_kind(self, kind: str) -> dict | None:
        for rec in reversed(self.interiors):
            if rec.get("building_kind") == kind:
                return rec
        return None

    def public_state(self) -> dict:
        return {"styles": [dict(r) for r in self.styles.values()],
                "interiors": [dict(r) for r in self.interiors],
                "assets": [dict(r) for r in self.assets],
                "shelved": [dict(s) for s in self.shelved],
                "counts": {"styles": len(self.styles),
                           "interiors": len(self.interiors),
                           "assets": len(self.assets)}}

    # -- registration ---------------------------------------------------------
    def register_spec(self, raw, origin: str = "brain",
                      proposer: str = "", hint: str = ""):
        """Validate + register one design spec synchronously.
        Returns (record, []) or (None, [errors]). The async
        pipeline (request_design / adopt_asset) is this plus the
        one-revision loop; god tooling and tests use this direct
        path."""
        clean, errors = validate_design(raw, self._ctx())
        if clean is None:
            return None, errors
        return self._register(clean, origin, proposer, hint), []

    def _register(self, clean: dict, origin: str, proposer: str,
                  hint: str) -> dict:
        rec = {**clean, "origin": origin, "proposer": proposer,
               "hint": hint}
        if clean["kind"] == "building_style":
            self.styles[clean["archetype"]] = rec
        elif clean["kind"] == "interior":
            self.interiors.append(rec)
        else:
            self.assets.append(rec)
            self.sync_markers()
        return rec

    def sync_markers(self) -> None:
        """Keep the layout document's objects carrying one
        "design-asset:<id>" marker per placed asset, so the
        asset rides the layout like any other prop (the client
        renders the composition from the designs payload)."""
        layout = getattr(self.bridge, "layout", None)
        if not isinstance(layout, dict):
            return
        objects = layout.setdefault("objects", [])
        have = {str(o.get("asset")) for o in objects
                if isinstance(o, dict)}
        for a in self.assets:
            marker = f"design-asset:{a['id']}"
            if marker in have:
                continue
            objects.append({
                "asset": marker,
                "position": [a["position"][0], 0, a["position"][1]],
                "rotation": [0, 0, 0], "scale": [1, 1, 1],
                "kind": "prop"})
            have.add(marker)

    # -- the async design pipeline -------------------------------------------
    def adopt_asset(self, prop: dict, now: float):
        """Civics executor entry: the city voted for a new asset.
        The design work runs on a background thread; the outcome
        lands through tick() as a design event. Returns
        (events, executed_text) for the civic result."""
        hint = prop["action"].get("asset_hint", "")
        spot = self._find_spot()
        if spot is None:
            return [], ("no clear ground near the plaza could be "
                        "found for the new work — the design waits")
        job = {"kind": "asset", "hint": hint,
               "proposer": prop["proposer"], "position": spot,
               "revised": False}
        self._start_design(job, errors=None)
        return [], (f"the city's designers took up '{hint}' — "
                    f"if the design holds up, it will stand near "
                    f"the plaza")

    def request_design(self, kind: str, hint: str, proposer: str,
                       now: float):
        """God-mode design request ({"type": "design"}): start
        the pipeline for one design kind. Returns (True, text) or
        (False, human_reason)."""
        if kind not in SPEC_KINDS:
            return False, ("A design kind must be one of: "
                           + ", ".join(SPEC_KINDS) + ".")
        job = {"kind": kind, "hint": hint or "", "proposer": proposer,
               "position": None, "revised": False}
        if kind == "asset":
            job["position"] = self._find_spot()
            if job["position"] is None:
                return False, ("No clear ground near the plaza "
                               "for a new asset right now.")
        self._start_design(job, errors=None)
        return True, (f"The city's designers took up the {kind} "
                      f"commission — the result lands when the "
                      f"drawing is done.")

    def _brain_ctx(self, job: dict, errors) -> dict:
        return {"kind": job["kind"], "hint": job["hint"],
                "proposer": job["proposer"],
                "position": job.get("position"),
                "building_kinds": sorted(BUILDINGS),
                "building_ids": sorted(
                    b["id"] for b in
                    self.bridge.construction.buildings),
                "existing": sorted(self._design_ids()),
                "interior_props": list(INTERIOR_PROPS),
                "asset_props": list(ASSET_PROPS),
                "roof_styles": list(ROOF_STYLES),
                "example": EXAMPLE_DESIGN, "errors": errors}

    def _template_for(self, hint: str) -> dict:
        low = (hint or "").lower()
        for keywords, template in ASSET_TEMPLATES:
            if any(k in low for k in keywords):
                return template
        return ASSET_TEMPLATES[-1][1]

    def _unique_template(self, hint: str) -> dict:
        """A deep copy of the closest template, renamed when the
        world already holds a design under its name (the seeded
        plaza fountain vs. a voted second fountain) — a template
        placement is a new physical asset, not a re-registration."""
        import copy
        raw = copy.deepcopy(self._template_for(hint))
        base = raw.get("name") or "Designed asset"
        name, n = base, 1
        while slugify(name) in self._design_ids():
            n += 1
            name = f"{base} {n}"
        raw["name"] = name
        return raw

    def _start_design(self, job: dict, errors) -> None:
        if self.design_brain is None:
            if job["kind"] == "asset":
                # No brain: the closest seed template by keyword,
                # honestly labeled — a real composition the world
                # already knows how to hold.
                raw = self._unique_template(job["hint"])
                self._results.put({"job": job, "raw": raw,
                                   "origin": "template"})
            else:
                self._results.put({
                    "job": job, "raw": None,
                    "fatal": "no design brain is wired to draw "
                             "this design"})
            return

        def work():
            ctx = self._brain_ctx(job, errors)
            try:
                raw = self.design_brain(ctx)
            except Exception:
                raw = None
            self._results.put({"job": job, "raw": raw,
                               "origin": "brain"})

        threading.Thread(target=work, daemon=True,
                         name="hermes-design").start()

    def _drain_results(self, now: float) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            job = res["job"]
            if res.get("fatal"):
                events.append(self._shelve(job, res["fatal"]))
                continue
            clean, errors = validate_design(
                res.get("raw"), self._ctx(job.get("position")))
            if clean is not None:
                rec = self._register(clean, res.get("origin",
                                                   "brain"),
                                     job["proposer"], job["hint"])
                events.append({"type": "design",
                               "state": "registered",
                               "kind": clean["kind"],
                               "id": clean["id"],
                               "name": clean["name"],
                               "origin": rec["origin"],
                               "proposer": job["proposer"],
                               "design": rec})
            elif not job["revised"]:
                # ONE revision, errors fed back verbatim.
                job["revised"] = True
                self._start_design(job, errors=errors)
            else:
                events.append(self._shelve(job, "; ".join(errors)))
        return events

    def _shelve(self, job: dict, reason: str) -> dict:
        entry = {"kind": job["kind"], "hint": job["hint"],
                 "reason": reason, "proposer": job["proposer"]}
        self.shelved.append(entry)
        return {"type": "design", "state": "shelved",
                "kind": job["kind"], "hint": job["hint"],
                "reason": reason, "proposer": job["proposer"]}

    def tick(self, now: float) -> list[dict]:
        """Advance designs one world tick: land finished design
        work as design events (registered / shelved)."""
        return self._drain_results(now)

    # -- persistence --------------------------------------------------------------
    def serialize(self) -> dict:
        keys = ("kind", "id", "name", "archetype", "palette",
                "roof_style", "floors", "building_kind",
                "building_id", "rooms", "footprint", "position",
                "props", "origin", "proposer", "hint")
        def pub(rec):
            return {k: rec[k] for k in keys if k in rec}
        return {"styles": [pub(r) for r in self.styles.values()],
                "interiors": [pub(r) for r in self.interiors],
                "assets": [pub(r) for r in self.assets],
                "shelved": [dict(s) for s in self.shelved]}

    def restore(self, doc) -> dict:
        """Replace the registry from a saved slice (validated
        first — a malformed document changes nothing) and
        re-place the asset markers in the layout document."""
        clean = validate_designs(doc)
        self.styles = {r["archetype"]: r for r in clean["styles"]}
        self.interiors = clean["interiors"]
        self.assets = clean["assets"]
        self.shelved = clean["shelved"]
        self.sync_markers()
        return {"styles": len(self.styles),
                "interiors": len(self.interiors),
                "assets": len(self.assets)}
