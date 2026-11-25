"""Dynamic innovation — Phase 6 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, Addendum 2 + the
innovation loop + "Dynamic innovation: ANY idea is possible".

A discovery a citizen brings back from the research terminal can be
put to the city as an **adopt_innovation** proposal (civics.py).
When the vote adopts it, this engine makes the idea REAL:

  1. **Deep research.** The idea is searched again, properly — how
     it is made, from what, with what tools, how it is used and
     sold (the same injectable web search as research.py).
  2. **Synthesis.** The world-planner brain writes the findings up
     as a STRICT JSON content spec (schema below) — a new item, how
     the world gets it (a gather source and/or a recipe at a
     station), the workshop that must be built for it, the building
     variant it unlocks, its uses and its market price.
  3. **Validation.** The engine checks the spec deterministically
     against what the world can physically express — the same
     discipline as every other synthesis in this framework: the
     model proposes, the engine disposes. A failing spec is fed
     back ONCE with its errors for revision; a second failure files
     the idea as **under study** (city knowledge + the proposer's
     memory) and registers nothing. Nothing is ever faked.
  4. **Registration.** A valid spec goes live in the running world:
     a new inventory key; gather nodes for its raw material (dug
     where the research says — low wet ground for clay-like
     things); a craft recipe in the RecipeBook (work.py) worked at
     a seed station or at the innovation's own station building; the
     station joins the build catalog (buildings.py); a building
     variant (a brick house) joins it too. Citizens then gather,
     craft, haul and build with the new thing through the ordinary
     P2/P3 machinery — and the market (below) sells it.

**The market.** Built stalls hold stock of sellable goods: when a
craft of a sellable item completes, half the output (rounded up)
routes to market stock, the rest to the city stores. On a slow
tick a citizen buys one stocked good — a {"type": "sale"} event,
stock down, the city's abstract **trade** counter up by the price,
buyer and seller remember the trade (kind="trade"). Currency stays
abstract on purpose: the counter is the city's commerce, not a
wallet. Food-kind innovations are eaten straight from the stores by
citizens on the same slow tick. Tool-kind innovations, while the
city holds any, make gather/craft work a little faster (durations
x TOOL_FACTOR in the planner — the documented simplification).

**Persistence.** Registered innovations, the under-study file and
the market ride the world save (the layout document's
"innovations" slice) and restore on load — re-applying their
registrations, since node kinds, recipes and catalog entries are
runtime state. (Inventory counts and node stock are session state
everywhere in this framework and are not persisted either.)

The seeded flagship is masonry — clay pits -> kiln -> bricks ->
brick houses — carried below as MASONRY_EXAMPLE: the worked
example the synthesizer prompt shows the brain, and the shape the
validator is tuned against. It is NOT pre-registered content; it
enters the world only by being researched, voted and adopted like
anything else.
"""
from __future__ import annotations

import queue
import random
import re
import threading

from . import work as work_mod
from .buildings import BUILDINGS

# -- the spec vocabulary ---------------------------------------------------
KINDS = ("material", "food", "good", "tool")
USES = ("building_material", "recipe_input", "food", "sellable", "tool")
ARCHETYPES = ("stall", "lamp", "fence", "house")

# -- clamps (a spec outside these is pulled into range, not rejected) -----
PRICE_RANGE = (1, 50)
WORK_S_RANGE = (5.0, 60.0)
OUT_QTY_RANGE = (1, 6)
INPUT_QTY_RANGE = (1, 10)
COST_QTY_RANGE = (1, 20)
COST_TOTAL_MAX = 40
STATION_WORK_S = 45.0        # builder time for a dynamic station

SOURCE_NODE_COUNT = 5        # gather nodes a registered source opens
SOURCE_MAX_STOCK = 3
SOURCE_REGROW_S = 45.0
SALE_INTERVAL_S = 45.0       # the market's slow tick
EAT_INTERVAL_S = 60.0        # citizens eat new foods on this tick
TOOL_FACTOR = 0.8            # gather/craft duration while a tool is held

# Spot-search ring around a source centre — mirrors civics.py's
# planting ring (kept local so this module imports nothing civic).
_SPOT_OFFSETS = [(0, 0), (7, 0), (-7, 0), (0, 7), (0, -7),
                 (11, 5), (-11, 5), (5, 11), (5, -11), (-5, -11),
                 (-11, -5), (14, 0), (0, 14), (-14, 0), (0, -14),
                 (18, 8), (-18, -8), (8, -18), (-8, 18),
                 (22, 0), (0, 22), (-22, 0), (0, -22)]
_ROADS = (-96.0, -48.0, 0.0, 48.0, 96.0)

# The flagship, as a spec: the synthesizer's worked example and the
# validator's reference shape. NOT registered anywhere by default.
MASONRY_EXAMPLE = {
    "id": "bricks", "name": "Bricks", "kind": "material",
    "source": {"node_label": "Clay pit", "yields": "clay",
               "area_hint": "low wet ground near a pond"},
    "recipe": {"inputs": {"clay": 2}, "out_qty": 2,
               "station_id": "kiln", "work_s": 12},
    "station_building": {"name": "Kiln", "cost": {"planks": 4},
                         "archetype": "stall"},
    "variant_building": {"name": "Brick house",
                         "costs": {"bricks": 6, "planks": 2},
                         "archetype": "house"},
    "uses": ["building_material", "sellable"], "price": 6,
}


def slugify(text) -> str:
    """An inventory/catalog-safe id: lowercase words joined by
    underscores, starting with a letter."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")
    if slug and not slug[0].isalpha():
        slug = "item_" + slug
    return slug[:32]


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clean_costs(raw, allowed: set, errors: list, what: str) -> dict:
    """A cost/input dict: keys must be known materials, quantities
    clamp into range, the total clamps to COST_TOTAL_MAX by
    trimming the largest entries first (deterministic)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append(f"{what} must be an object of material: amount")
        return {}
    out = {}
    for key, qty in raw.items():
        key = str(key)
        if key not in allowed:
            errors.append(f"{what} names unknown material {key!r}")
            continue
        out[key] = int(_clamp(qty, *COST_QTY_RANGE, 1))
    total = sum(out.values())
    while total > COST_TOTAL_MAX and out:
        key = max(out, key=lambda k: (out[k], k))
        out[key] -= 1
        total -= 1
        if out[key] <= 0:
            del out[key]
    return out


def validate_spec(raw, ctx: dict):
    """Check a synthesized content spec against the world.

    `ctx` carries the world's current truth: {"inventory_keys",
    "station_ids", "building_kinds", "item_ids"}. Returns
    (clean_spec, []) or (None, [human error, ...]) — every rule
    that fired is listed, so the revision pass can fix them all at
    once. Quantities/prices/durations CLAMP (see the ranges above);
    structural problems — unknown materials, an unknown station or
    archetype, an id that collides with something the world already
    has, an item with no way to come into being — are errors.
    """
    errors: list[str] = []
    if not isinstance(raw, dict):
        return None, ["the spec is not a JSON object"]
    inventory = set(ctx.get("inventory_keys") or ())
    stations = set(ctx.get("station_ids") or ())
    buildings = set(ctx.get("building_kinds") or ())
    items = set(ctx.get("item_ids") or ())

    name = str(raw.get("name") or "").strip()[:48]
    if not name:
        errors.append("the spec needs a name")
    item_id = slugify(raw.get("id") or name)
    if not item_id:
        errors.append("the spec needs an id")
    elif item_id in inventory or item_id in items:
        errors.append(f"the world already has a material "
                      f"called {item_id!r}")
    elif item_id in buildings or item_id in stations:
        errors.append(f"{item_id!r} already names a building or "
                      f"station in this world")

    kind = str(raw.get("kind") or "")
    if kind not in KINDS:
        errors.append(f"kind must be one of: {', '.join(KINDS)}")

    uses_in = raw.get("uses")
    uses = [u for u in (uses_in if isinstance(uses_in, list) else [])
            if u in USES]
    if not uses:
        errors.append("uses must name at least one of: "
                      + ", ".join(USES))
    if kind == "food" and "food" not in uses:
        uses.append("food")
    if kind == "tool" and "tool" not in uses:
        uses.append("tool")

    # The source: where the raw stuff comes from. `yields` names
    # the gathered material (default: the item itself).
    source = None
    raw_source = raw.get("source")
    yields = item_id
    if raw_source is not None:
        if not isinstance(raw_source, dict):
            errors.append("source must be an object or null")
        else:
            label = str(raw_source.get("node_label") or "").strip()[:40]
            if not label:
                errors.append("a source needs a node_label "
                              "(e.g. 'Clay pit')")
            yields = slugify(raw_source.get("yields") or item_id)
            if not yields:
                errors.append("the source's yields needs a name")
            elif yields != item_id and yields in inventory:
                errors.append(f"the source yields {yields!r}, which "
                              f"the world already gathers another way")
            elif yields in buildings or yields in stations:
                errors.append(f"the source yields {yields!r}, which "
                              f"already names a building or station")
            source = {"node_label": label or "Resource site",
                      "yields": yields,
                      "area_hint": str(
                          raw_source.get("area_hint") or "")[:80]}

    # The station building this innovation needs, if any. Its id
    # derives from its name, and a recipe may then name it.
    station_building = None
    raw_sb = raw.get("station_building")
    station_id_new = None
    if raw_sb is not None:
        if not isinstance(raw_sb, dict):
            errors.append("station_building must be an object or null")
        else:
            sb_name = str(raw_sb.get("name") or "").strip()[:40]
            if not sb_name:
                errors.append("station_building needs a name")
            station_id_new = slugify(sb_name)
            archetype = str(raw_sb.get("archetype") or "")
            if archetype not in ARCHETYPES:
                errors.append("station_building.archetype must be "
                              "one of: " + ", ".join(ARCHETYPES))
            if station_id_new and (station_id_new in stations
                                   or station_id_new in buildings):
                errors.append(f"a station or building called "
                              f"{station_id_new!r} already exists")
            allowed = inventory | {item_id, yields}
            costs = _clean_costs(raw_sb.get("cost"), allowed, errors,
                                 "station_building.cost")
            if not costs and not errors:
                errors.append("station_building.cost cannot be empty")
            station_building = {"id": station_id_new,
                                "name": sb_name or "Workshop",
                                "costs": costs,
                                "archetype": archetype}

    # The recipe: how the item is made, and where.
    recipe = None
    raw_recipe = raw.get("recipe")
    if raw_recipe is not None:
        if not isinstance(raw_recipe, dict):
            errors.append("recipe must be an object or null")
        else:
            allowed = inventory | {item_id, yields}
            inputs = _clean_costs(raw_recipe.get("inputs"), allowed,
                                  errors, "recipe.inputs")
            for key in list(inputs):
                if inputs[key] > INPUT_QTY_RANGE[1]:
                    inputs[key] = INPUT_QTY_RANGE[1]
            station_id = str(raw_recipe.get("station_id") or "")
            if station_id not in stations \
                    and station_id != station_id_new:
                errors.append(f"recipe.station_id {station_id!r} is "
                              f"not a station of this world")
            if not inputs:
                errors.append("recipe.inputs cannot be empty")
            recipe = {"inputs": inputs,
                      "out_qty": int(_clamp(
                          raw_recipe.get("out_qty"), *OUT_QTY_RANGE, 1)),
                      "station_id": station_id,
                      "work_s": _clamp(raw_recipe.get("work_s"),
                                       *WORK_S_RANGE, 12.0)}
    if source is None and recipe is None:
        errors.append("the item needs a way to come into being: "
                      "a gather source, a recipe, or both")

    # The building variant a new material unlocks.
    variant = None
    raw_var = raw.get("variant_building")
    if raw_var is None and "building_material" in uses and not errors:
        raw_var = {"name": f"{name} house",
                   "costs": {item_id: 8}, "archetype": "house"}
    if raw_var is not None:
        if not isinstance(raw_var, dict):
            errors.append("variant_building must be an object or null")
        else:
            var_name = str(raw_var.get("name") or "").strip()[:40]
            if not var_name:
                errors.append("variant_building needs a name")
            var_id = slugify(var_name)
            archetype = str(raw_var.get("archetype") or "")
            if archetype not in ARCHETYPES:
                errors.append("variant_building.archetype must be "
                              "one of: " + ", ".join(ARCHETYPES))
            if var_id and (var_id in buildings or var_id in stations):
                errors.append(f"a building called {var_id!r} "
                              f"already exists")
            allowed = inventory | {item_id, yields}
            costs = _clean_costs(raw_var.get("costs"), allowed,
                                 errors, "variant_building.costs")
            if not costs and not errors:
                errors.append("variant_building.costs cannot be empty")
            variant = {"id": var_id, "name": var_name or "New house",
                       "costs": costs, "archetype": archetype}

    if errors:
        return None, errors
    return {"id": item_id, "name": name, "kind": kind, "uses": uses,
            "source": source, "recipe": recipe,
            "station_building": station_building,
            "variant_building": variant,
            "price": int(_clamp(raw.get("price"), *PRICE_RANGE, 5))}, []


def validate_innovations(doc) -> dict:
    """Clean the {"records", "under_study", "market"} slice of a
    world save. Specs were validated at adoption; restore checks
    structure only (ids, names, lists, non-negative numbers) and is
    all-or-nothing, like buildings.validate_construction."""
    if not isinstance(doc, dict):
        raise ValueError("innovations document must be an object")
    records_in = doc.get("records") or []
    study_in = doc.get("under_study") or []
    if not isinstance(records_in, list) or not isinstance(study_in, list):
        raise ValueError("'records' and 'under_study' must be lists")
    records = []
    for i, rec in enumerate(records_in):
        if not isinstance(rec, dict) or not rec.get("id") \
                or not rec.get("name"):
            raise ValueError(f"records[{i}] needs an id and a name")
        uses = [u for u in (rec.get("uses") or []) if u in USES]
        records.append({
            "id": str(rec["id"]), "name": str(rec["name"]),
            "kind": str(rec.get("kind") or "good"),
            "uses": uses,
            "source": rec.get("source")
            if isinstance(rec.get("source"), dict) else None,
            "recipe": rec.get("recipe")
            if isinstance(rec.get("recipe"), dict) else None,
            "station_building": rec.get("station_building")
            if isinstance(rec.get("station_building"), dict) else None,
            "variant_building": rec.get("variant_building")
            if isinstance(rec.get("variant_building"), dict) else None,
            "price": max(1, int(rec.get("price") or 5)),
            "state": "registered",
            "proposer": str(rec.get("proposer") or ""),
            "hint": str(rec.get("hint") or "")})
    under_study = []
    for rec in study_in:
        if isinstance(rec, dict) and rec.get("hint"):
            under_study.append({
                "hint": str(rec["hint"])[:120],
                "reason": str(rec.get("reason") or "")[:240],
                "proposer": str(rec.get("proposer") or "")})
    market_in = doc.get("market") or {}
    stock = {}
    if isinstance(market_in, dict):
        for key, qty in (market_in.get("stock") or {}).items():
            if isinstance(qty, (int, float)) and qty > 0:
                stock[str(key)] = int(qty)
    trade = market_in.get("trade", 0) if isinstance(market_in, dict) else 0
    return {"records": records, "under_study": under_study,
            "market": {"stock": stock,
                       "trade": max(0, int(trade))
                       if isinstance(trade, (int, float)) else 0}}


class InnovationEngine:
    """The registry, the market, and the adoption pipeline.

    Wiring: WorldBridge constructs the engine with itself; the
    planner reaches it as `planner.innovations` (craft routing,
    tool speed, foreman craft needs) and civics' executor calls
    adopt(). `spec_brain` (callable(ctx) -> raw spec dict | None)
    and `searcher` (callable(query) -> [result, ...]) are injected:
    build_world wires the live brain + real web search; tests wire
    fakes. With no spec_brain the engine can register nothing —
    adoptions go honestly to the under-study file."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.records: dict[str, dict] = {}     # item id -> record
        self.under_study: list[dict] = []
        self.market_stock: dict[str, int] = {}
        self.trade: int = 0
        self.spec_brain = None
        self.searcher = None
        self.tool_factor = TOOL_FACTOR
        self._results: queue.Queue = queue.Queue()
        self._rng = random.Random(41)
        self._sale_next: float | None = None
        self._eat_next: float | None = None

    # -- world access ------------------------------------------------------
    @property
    def _planner(self):
        manager = getattr(self.bridge, "manager", None)
        return manager.planner if manager is not None else None

    @property
    def _book(self):
        return getattr(self.bridge, "recipes", None) \
            or work_mod.default_book()

    def _ctx(self) -> dict:
        """The world's current truth, for spec validation."""
        book = self._book
        stations = set(work_mod.STATIONS)
        stations |= {r["station_building"]["id"]
                     for r in self.records.values()
                     if r.get("station_building")}
        return {"inventory_keys": set(
                    self.bridge.resources.inventory),
                "station_ids": stations,
                "building_kinds": set(BUILDINGS),
                "item_ids": set(self.records)}

    # -- read models ---------------------------------------------------------
    def get(self, item_id: str) -> dict | None:
        return self.records.get(item_id)

    def is_sellable(self, item_id: str) -> bool:
        rec = self.records.get(item_id)
        return rec is not None and rec.get("state") == "registered" \
            and "sellable" in rec.get("uses", [])

    def tool_active(self) -> bool:
        """A registered tool the city actually holds (stores or
        market stock) — the planner speeds work while this holds."""
        inv = self.bridge.resources.inventory
        for rec in self.records.values():
            if "tool" not in rec.get("uses", []):
                continue
            if inv.get(rec["id"], 0) > 0 \
                    or self.market_stock.get(rec["id"], 0) > 0:
                return True
        return False

    def food_stock(self) -> int:
        """How much innovation food the stores hold (the foreman's
        food picture counts it alongside garden food)."""
        inv = self.bridge.resources.inventory
        return sum(inv.get(r["id"], 0) for r in self.records.values()
                   if r.get("kind") == "food")

    def public_record(self, rec: dict) -> dict:
        return {"id": rec["id"], "name": rec["name"],
                "kind": rec["kind"], "uses": list(rec["uses"]),
                "price": rec["price"], "state": rec["state"],
                "proposer": rec.get("proposer", ""),
                "source": (rec["source"] or {}).get("node_label"),
                "recipe": ({"inputs": dict(rec["recipe"]["inputs"]),
                            "out_qty": rec["recipe"]["out_qty"],
                            "station": rec["recipe"]["station_id"]}
                           if rec.get("recipe") else None),
                "station": (rec["station_building"] or {}).get("name"),
                "variant": (rec["variant_building"] or {}).get("name")}

    def public_state(self) -> dict:
        return {"innovations": [self.public_record(r)
                                for r in self.records.values()],
                "under_study": [dict(u) for u in self.under_study],
                "market": {"stock": dict(self.market_stock),
                           "trade": self.trade}}

    # -- the market ------------------------------------------------------------
    def route_craft_outputs(self, outputs: dict) -> dict:
        """A finished craft's outputs, split for the stores: for a
        sellable good, half (rounded up) routes to market stock —
        the stalls sell from there — and the rest lands in the city
        inventory the planner is about to credit. Returns the dict
        that should actually reach the inventory."""
        out = {}
        for key, qty in outputs.items():
            qty = int(qty)
            if qty > 0 and self.is_sellable(key):
                share = (qty + 1) // 2
                self.market_stock[key] = \
                    self.market_stock.get(key, 0) + share
                qty -= share
            if qty > 0:
                out[key] = qty
        return out

    def craft_needs(self) -> list[str]:
        """Craft task types the foreman should queue: a sellable
        innovation whose stalls stand empty while its inputs sit in
        the stores and its station is standing."""
        inv = self.bridge.resources.inventory
        book = self._book
        out = []
        for rec in self.records.values():
            recipe = rec.get("recipe")
            if recipe is None or "sellable" not in rec["uses"]:
                continue
            if self.market_stock.get(rec["id"], 0) > 0:
                continue
            if any(inv.get(k, 0) < v
                   for k, v in recipe["inputs"].items()):
                continue
            if book.station_position(
                    recipe["station_id"],
                    self.bridge.construction) is None:
                continue        # the workshop isn't built yet
            out.append(f"craft_{rec['id']}")
        return out

    # -- adoption ----------------------------------------------------------------
    def adopt(self, prop: dict, now: float):
        """Civics executor entry: the city voted to adopt the
        discovery in prop["action"]["innovation_hint"]. Synthesis
        runs on a background thread (deep research + the brain);
        the outcome lands through tick() as an innovation event.
        Returns (events, executed_text) for the civic result."""
        job = {"hint": prop["action"].get("innovation_hint", ""),
               "proposer": prop["proposer"],
               "title": prop["title"], "revised": False}
        self._start_synthesis(job, errors=None, findings=None)
        return [], (f"the city's researchers took up the discovery "
                    f"'{job['hint']}' — if it can be made, the "
                    f"terminal will work out how")

    def _spec_ctx(self, job: dict, findings, errors) -> dict:
        return {"hint": job["hint"], "proposer": job["proposer"],
                "findings": findings or [],
                "inventory_keys": sorted(
                    self.bridge.resources.inventory),
                "station_ids": sorted(self._ctx()["station_ids"]),
                "building_kinds": sorted(BUILDINGS),
                "example": MASONRY_EXAMPLE, "errors": errors}

    def _start_synthesis(self, job: dict, errors, findings) -> None:
        if self.spec_brain is None:
            self._results.put({
                "job": job, "raw": None, "findings": findings or [],
                "fatal": "no world-planner brain is wired to "
                         "develop the idea"})
            return

        def work():
            found = findings
            if found is None:
                found = []
                if self.searcher is not None:
                    try:
                        found = self.searcher(
                            f"how to make {job['hint']}: materials, "
                            f"process, tools, uses and how it is "
                            f"sold") or []
                    except Exception:
                        found = []
            ctx = self._spec_ctx(job, found, errors)
            try:
                raw = self.spec_brain(ctx)
            except Exception:
                raw = None
            self._results.put({"job": job, "raw": raw,
                               "findings": found})

        threading.Thread(target=work, daemon=True,
                         name="hermes-innovation").start()

    def _drain_results(self, now: float) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            job = res["job"]
            if res.get("fatal"):
                events.append(self._under_study(job, res["fatal"], now))
                continue
            clean, errors = validate_spec(res.get("raw"), self._ctx())
            if clean is not None:
                events.append(self._register(clean, job, now))
            elif not job["revised"]:
                # ONE revision, with the validator's errors fed
                # back verbatim — then the idea stands or falls.
                job["revised"] = True
                self._start_synthesis(job, errors=errors,
                                      findings=res.get("findings"))
            else:
                events.append(self._under_study(
                    job, "; ".join(errors), now))
        return events

    def _register(self, clean: dict, job: dict, now: float) -> dict:
        rec = {**clean, "state": "registered",
               "proposer": job["proposer"], "hint": job["hint"],
               "adopted_at": now}
        self.records[rec["id"]] = rec
        self._apply(clean)
        proposer = self.bridge.npcs.get(job["proposer"])
        if proposer is not None:
            proposer.witness(
                f"The city adopted my discovery and made it real: "
                f"{clean['name']} is now part of the world — "
                f"citizens can make it, use it and trade it.",
                kind="research")
        self.bridge.city_knowledge.append({
            "kind": "innovation", "citizen": job["proposer"],
            "text": (f"The city adopted '{job['hint']}' and now "
                     f"makes {clean['name']}."),
            "innovation": clean["id"], "ts": now})
        return {"type": "innovation", "state": "registered",
                "id": clean["id"], "name": clean["name"],
                "proposer": job["proposer"],
                "innovation": self.public_record(rec)}

    def _under_study(self, job: dict, reason: str, now: float) -> dict:
        entry = {"hint": job["hint"], "reason": reason,
                 "proposer": job["proposer"]}
        self.under_study.append(entry)
        proposer = self.bridge.npcs.get(job["proposer"])
        if proposer is not None:
            proposer.witness(
                f"The city's researchers filed my discovery "
                f"'{job['hint']}' as under study for now — {reason}",
                kind="research")
        self.bridge.city_knowledge.append({
            "kind": "under_study", "citizen": job["proposer"],
            "text": (f"'{job['hint']}' is under study — {reason}"),
            "innovation": None, "ts": now})
        return {"type": "innovation", "state": "under_study",
                "hint": job["hint"], "reason": reason,
                "proposer": job["proposer"]}

    # -- registration side effects -------------------------------------------
    def _apply(self, spec: dict) -> None:
        """Make a validated spec physically true in the world:
        inventory keys, gather nodes + their task, the recipe, the
        station building and the variant building. Idempotent —
        restore() re-applies saved specs into a fresh world."""
        resources = self.bridge.resources
        book = self._book
        item = spec["id"]
        resources.inventory.setdefault(item, 0)
        source = spec.get("source")
        if source:
            yields = source.get("yields") or item
            resources.inventory.setdefault(yields, 0)
            kind_id = f"src_{yields}"
            if kind_id not in resources.kind_specs:
                resources.register_kind(
                    kind_id, resource=yields,
                    max_stock=SOURCE_MAX_STOCK,
                    regrow_seconds=SOURCE_REGROW_S,
                    per_harvest=1)
            book.register_gather(f"gather_{yields}", kind_id, yields,
                                 label=f"gather {yields}")
            existing = {n.id for n in resources.nodes.values()}
            for i, (x, z) in enumerate(
                    self._source_positions(spec), 1):
                node_id = f"{yields}-src-{i}"
                if node_id not in existing:
                    resources.spawn_node(node_id, kind_id,
                                         source["node_label"], x, z)
        recipe = spec.get("recipe")
        if recipe:
            book.register_recipe(f"craft_{item}", recipe["inputs"],
                                 {item: recipe["out_qty"]},
                                 recipe["station_id"],
                                 recipe["work_s"])
        station = spec.get("station_building")
        if station:
            arche = BUILDINGS[station["archetype"]]
            BUILDINGS[station["id"]] = {
                "name": station["name"], "costs": dict(station["costs"]),
                "work": STATION_WORK_S,
                "footprint": arche["footprint"], "anchor": None,
                "archetype": station["archetype"], "dynamic": True}
            book.register_station(station["id"], station["name"])
        variant = spec.get("variant_building")
        if variant:
            arche = BUILDINGS[variant["archetype"]]
            BUILDINGS[variant["id"]] = {
                "name": variant["name"],
                "costs": dict(variant["costs"]),
                "work": arche["work"],
                "footprint": arche["footprint"],
                "anchor": arche["anchor"],
                "archetype": variant["archetype"], "dynamic": True}

    def _source_positions(self, spec: dict) -> list[tuple]:
        """Where a source's gather nodes open: low wet ground for
        clay-like things (the terrain's pond / lowest point),
        otherwise near the anchor the research named, else the
        plaza. Deterministic — a restored world digs the same pits.
        Roads and existing nodes are avoided, as in civics'
        planting rule."""
        resources = self.bridge.resources
        terrain = getattr(self.bridge, "terrain", None)
        planner = self._planner
        hint = (spec["source"].get("area_hint") or "").lower()
        center = None
        if terrain is not None and any(
                w in hint for w in ("clay", "pond", "river", "wet",
                                    "low", "water", "shore", "bank")):
            summary = terrain.summary()
            center = summary.get("pond_at") or summary.get("min_at")
        if center is None and planner is not None:
            for anchor in planner.anchors:
                low = anchor.name.lower()
                if low and (low in hint or hint and hint in low):
                    center = (anchor.x, anchor.z)
                    break
        if center is None:
            center = (8.0, 8.0)
        # keep the cluster inside the playable area even when the
        # hinted low ground sits at the world edge (flat ground's
        # lowest sample is a corner) — against the LIVE bound (P9):
        # a grown world lets a source cluster sit in the new land.
        bound = terrain.bound if terrain is not None else 120.0
        cx = max(-(bound - 20.0), min(bound - 20.0, float(center[0])))
        cz = max(-(bound - 20.0), min(bound - 20.0, float(center[1])))
        center = (cx, cz)
        taken = [(n.x, n.z) for n in resources.nodes.values()]
        out = []
        for ox, oz in _SPOT_OFFSETS:
            if len(out) >= SOURCE_NODE_COUNT:
                break
            x, z = center[0] + ox, center[1] + oz
            if abs(x) > bound - 8.0 or abs(z) > bound - 8.0:
                continue
            if any(abs(x - c) < 7.5 or abs(z - c) < 7.5
                   for c in _ROADS):
                continue
            if any((x - tx) ** 2 + (z - tz) ** 2 < 6.25
                   for tx, tz in taken):
                continue
            out.append((x, z))
            taken.append((x, z))
        return out

    # -- the slow market / table tick -------------------------------------------
    def tick(self, now: float) -> list[dict]:
        """Advance innovation one world tick: land finished
        syntheses, then the market's slow commerce — a sale when
        the stalls hold stock, a citizen eating a new food now and
        then."""
        events = self._drain_results(now)
        if self._sale_next is None:
            self._sale_next = now + SALE_INTERVAL_S
        if self._eat_next is None:
            self._eat_next = now + EAT_INTERVAL_S
        if now >= self._sale_next:
            self._sale_next = now + SALE_INTERVAL_S
            sale = self._market_sale()
            if sale is not None:
                events.append(sale)
        if now >= self._eat_next:
            self._eat_next = now + EAT_INTERVAL_S
            self._eat_food()
        return events

    def _market_sale(self) -> dict | None:
        stocked = [(item, qty) for item, qty in self.market_stock.items()
                   if qty > 0 and item in self.records]
        if not stocked or not self.bridge.npcs:
            return None
        total = sum(qty for _item, qty in stocked)
        pick = self._rng.uniform(0, total)
        item = stocked[-1][0]
        for cand, qty in stocked:
            pick -= qty
            if pick <= 0:
                item = cand
                break
        rec = self.records[item]
        names = sorted(self.bridge.npcs)
        buyer = self._rng.choice(names)
        sellers = [n for n in names
                   if self.bridge.npcs[n].card.role
                   in ("vendor", "shopkeeper")]
        seller = self._rng.choice(sellers) if sellers else "the market"
        self.market_stock[item] -= 1
        self.trade += rec["price"]
        self.bridge.npcs[buyer].witness(
            f"Bought {rec['name']} at the market for {rec['price']}.",
            kind="trade")
        if seller in self.bridge.npcs:
            self.bridge.npcs[seller].witness(
                f"Sold {rec['name']} to {buyer} at the market for "
                f"{rec['price']}.", kind="trade")
        return {"type": "sale", "item": item, "name": rec["name"],
                "buyer": buyer, "seller": seller,
                "price": rec["price"],
                "stock": self.market_stock[item], "trade": self.trade}

    def _eat_food(self) -> None:
        inv = self.bridge.resources.inventory
        foods = [r for r in self.records.values()
                 if r.get("kind") == "food" and inv.get(r["id"], 0) > 2]
        if not foods or not self.bridge.npcs:
            return
        rec = self._rng.choice(foods)
        eater = self._rng.choice(sorted(self.bridge.npcs))
        inv[rec["id"]] -= 1
        self.bridge.npcs[eater].witness(
            f"Ate {rec['name']} from the city stores — a taste the "
            f"city only learned recently.", kind="episode")

    # -- persistence --------------------------------------------------------------
    def serialize(self) -> dict:
        return {"records": [{k: rec.get(k) for k in
                             ("id", "name", "kind", "uses", "source",
                              "recipe", "station_building",
                              "variant_building", "price", "proposer",
                              "hint")}
                            | {"state": "registered"}
                            for rec in self.records.values()],
                "under_study": [dict(u) for u in self.under_study],
                "market": {"stock": dict(self.market_stock),
                           "trade": self.trade}}

    def restore(self, doc) -> dict:
        """Replace the registry from a saved slice (validated
        first — a malformed document changes nothing) and
        re-apply every registration to the live worlds."""
        clean = validate_innovations(doc)
        self.records = {}
        for rec in clean["records"]:
            self.records[rec["id"]] = rec
            self._apply(rec)
        self.under_study = clean["under_study"]
        self.market_stock = clean["market"]["stock"]
        self.trade = clean["market"]["trade"]
        return {"innovations": len(self.records),
                "under_study": len(self.under_study)}
