"""RTS labor + crafting — Phase 2 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, systems 2 + 3, phase P2.

The city runs a small **task queue**. A task sends one citizen to a
resource node (gather) or a crafting station (craft), has them work
for a few sim-seconds, and lands the result in the shared city
inventory (resources.py). Two sources create tasks:

  * the **foreman** (in ActivityPlanner) keeps baseline stocks —
    wood >= TARGET_WOOD, food >= TARGET_FOOD — by enqueueing a
    gather task whenever a stock runs low and no equivalent task is
    already underway, assigned to the best idle citizen;
  * the **player**, in god mode, via the {"type": "order"} protocol
    message — naming a citizen or letting the same best-idle scoring
    pick one.

Execution lives in the ActivityPlanner (npc.py): a task is a strong
intent that drives its assignee exactly the way a conversation-born
plan does — travel to the target, dwell there working — and, like a
plan, it yields to the conversation lock: an engaged citizen does
not start (or pauses) until the player lets them go. Gathering goes
through ResourceWorld.try_harvest, the single harvest path, with
the NPC as the actor; crafting validates its inputs at completion
time and fails gracefully (with an event) if the stores no longer
hold them.

This module holds only the shared definitions — stations, recipes,
targets, affinities — so the planner, the bridge, the interpreter
and the web client mirror all read the same numbers.
"""
from __future__ import annotations

# -- foreman tuning -----------------------------------------------------
TARGET_WOOD = 10         # baseline city stocks the foreman maintains
TARGET_FOOD = 8
MAX_ACTIVE_TASKS = 4     # concurrent tasks, city-wide (queue stays small
                         # so the city never empties into the fields)

# -- work durations (sim seconds at the target) ---------------------------
GATHER_SECONDS = 6.0     # one harvest action (a tree felling, a picking)
CRAFT_SECONDS = 10.0     # one recipe at a station

# -- task types ------------------------------------------------------------
GATHER_WOOD = "gather_wood"
GATHER_FOOD = "gather_food"
CRAFT_PLANKS = "craft_planks"
CRAFT_MEAL = "craft_meal"
# Building (P3, see buildings.py): a build task's target is a
# construction site, not a node or station, so it is created by
# ActivityPlanner.assign_build — it is deliberately NOT in
# TASK_TYPES / task_target(), which only describe the generic
# node/station flow.
BUILD = "build"
# Terraforming (P4, see terrain.py): a terraform task's target is a
# land patch (a crew job in the TerrainWorld), created by
# ActivityPlanner.assign_terraform — likewise NOT in TASK_TYPES.
# The foreman staffs open jobs but never creates them: crew jobs
# come from the player (crew mode) and, from P5, adopted proposals.
TERRAFORM = "terraform"
TASK_TYPES = (GATHER_WOOD, GATHER_FOOD, CRAFT_PLANKS, CRAFT_MEAL)

# Which resource-node kinds each gather task harvests.
GATHER_NODE_KINDS = {GATHER_WOOD: ("tree",),
                     GATHER_FOOD: ("plant", "animal")}

# -- crafting stations -------------------------------------------------------
# Fixed world objects. The sawbench stands by the West Depot anchor;
# the kitchen stove by Market Row West, a few steps from the garden
# plots. Positions keep clear of roads and resource nodes; the web
# client mirrors these coordinates for its standalone mode.
STATIONS = {
    "sawbench": {"id": "sawbench", "name": "Sawbench",
                 "x": -100.0, "z": 12.0, "anchor": "West Depot"},
    "stove": {"id": "stove", "name": "Market stove",
              "x": -31.0, "z": 5.0, "anchor": "Market Row West"},
}

# -- recipes -------------------------------------------------------------------
# inputs are consumed from (and outputs added to) the city inventory
# at completion time — never at assignment — so a stock-out mid-task
# is a clean failure, not a negative inventory.
RECIPES = {
    CRAFT_PLANKS: {"station": "sawbench",
                   "inputs": {"wood": 1}, "outputs": {"planks": 1}},
    CRAFT_MEAL: {"station": "stove",
                 "inputs": {"food": 2}, "outputs": {"meals": 1}},
}

# -- assignment scoring -----------------------------------------------------
# Role affinity bonuses when the foreman / an unnamed order picks the
# best idle citizen (the same idea as the planner's activity scoring:
# a nudge, never a hard rule). Gathering is anyone's work; crafting
# leans to the hands-on trades and the food handlers.
WORK_AFFINITY = {
    GATHER_WOOD: {},
    GATHER_FOOD: {"vendor": 0.6, "shopkeeper": 0.3, "courier": 0.3},
    CRAFT_PLANKS: {"mechanic": 1.5, "engineer": 1.0, "vendor": 0.4},
    CRAFT_MEAL: {"vendor": 1.5, "shopkeeper": 0.8, "medic": 0.3},
    BUILD: {"mechanic": 1.5, "engineer": 1.0, "vendor": 0.2},
    TERRAFORM: {"engineer": 1.5, "mechanic": 1.0, "vendor": 0.2},
}

TASK_LABELS = {GATHER_WOOD: "gather wood", GATHER_FOOD: "gather food",
               CRAFT_PLANKS: "saw planks", CRAFT_MEAL: "cook a meal",
               BUILD: "build", TERRAFORM: "terraform"}


def station_list() -> list[dict]:
    """Public station dicts for /api/world (clients render from this)."""
    return [{"id": s["id"], "name": s["name"],
             "position": [s["x"], s["z"]], "anchor": s["anchor"]}
            for s in STATIONS.values()]


def task_target(task_type: str):
    """Where a task of this type is worked: ("station", station_id)
    for crafting, ("node", node_kinds) for gathering."""
    recipe = RECIPES.get(task_type)
    if recipe is not None:
        return ("station", recipe["station"])
    return ("node", GATHER_NODE_KINDS[task_type])


# ---------------------------------------------------------------------------
# The recipe registry (generalized in P6 for dynamic innovation).
#
# RECIPES / STATIONS above are the SEEDS: every RecipeBook starts
# holding exactly them, behaving exactly as the hardcoded tables did.
# An adopted innovation (see innovations.py) registers MORE at
# runtime: a craft recipe for its item (worked at a seed station or
# at a station building the city must first build), a gather task
# for a raw material its source nodes yield, and the station
# building itself. The planner works from a RecipeBook — the bridge
# owns one per world and wires it in; a planner without one falls
# back to a shared seed-only book, so headless sims and unit tests
# see precisely the P2 behavior.
# ---------------------------------------------------------------------------
class RecipeBook:
    """Seed recipes/stations + whatever innovation registers live."""

    def __init__(self):
        self._recipes: dict[str, dict] = {}
        for task_type, r in RECIPES.items():
            self._recipes[task_type] = {
                "station": r["station"], "inputs": dict(r["inputs"]),
                "outputs": dict(r["outputs"]), "work_s": CRAFT_SECONDS,
                "dynamic": False}
        self._stations: dict[str, dict] = {
            sid: {"id": s["id"], "name": s["name"], "dynamic": False}
            for sid, s in STATIONS.items()}
        # Dynamic gather tasks: task_type -> {"node_kinds": (kind,),
        # "resource": inventory key the nodes yield}.
        self._gathers: dict[str, dict] = {}

    # -- lookups ------------------------------------------------------
    def get(self, task_type: str) -> dict | None:
        """The recipe for a craft task type, or None."""
        return self._recipes.get(task_type)

    def is_craft(self, task_type: str) -> bool:
        return task_type in self._recipes

    def craft_types(self) -> list[str]:
        return list(self._recipes)

    def gather_kinds(self, task_type: str):
        """Node kinds a gather task harvests: the seed table, or a
        dynamic gather registered by an innovation."""
        if task_type in GATHER_NODE_KINDS:
            return GATHER_NODE_KINDS[task_type]
        entry = self._gathers.get(task_type)
        return entry["node_kinds"] if entry else None

    def is_gather(self, task_type: str) -> bool:
        return self.gather_kinds(task_type) is not None

    def known_task_types(self) -> list[str]:
        out = list(TASK_TYPES)
        for t in list(self._recipes) + list(self._gathers):
            if t not in out:
                out.append(t)
        return out

    def station_name(self, station_id: str,
                     construction=None) -> str | None:
        stn = self._stations.get(station_id)
        if stn is not None:
            return stn["name"]
        if construction is not None:
            for b in construction.buildings:
                if b.get("kind") == station_id:
                    return b.get("name", station_id)
        return None

    def station_position(self, station_id: str, construction=None):
        """Where a station stands: a seed station's fixed spot, or
        the position of a finished station BUILDING (a dynamic
        station exists only once the city has built it). None when
        the station is not standing yet."""
        seed = STATIONS.get(station_id)
        if seed is not None:
            return (seed["x"], seed["z"])
        if construction is not None:
            for b in construction.buildings:
                if b.get("kind") == station_id:
                    pos = b.get("position") or [0, 0]
                    return (float(pos[0]), float(pos[1]))
        return None

    def public_stations(self, construction=None) -> list[dict]:
        """Seed stations (always standing) + dynamic stations that
        have actually been built, for /api/world."""
        out = station_list()
        if construction is not None:
            built = {b.get("kind"): b for b in construction.buildings}
            for sid, stn in self._stations.items():
                if not stn.get("dynamic"):
                    continue
                b = built.get(sid)
                if b is None:
                    continue        # registered, but not built yet
                out.append({"id": sid, "name": stn["name"],
                            "position": [float(b["position"][0]),
                                         float(b["position"][1])],
                            "anchor": "Innovation"})
        return out

    # -- registration (innovations.py drives this) ----------------------
    def register_recipe(self, task_type: str, inputs: dict,
                        outputs: dict, station: str,
                        work_s: float) -> None:
        self._recipes[task_type] = {
            "station": station, "inputs": dict(inputs),
            "outputs": dict(outputs), "work_s": float(work_s),
            "dynamic": True}
        TASK_LABELS.setdefault(task_type, "craft "
                               + task_type.removeprefix("craft_")
                               .replace("_", " "))

    def register_gather(self, task_type: str, node_kind: str,
                        resource: str, label: str) -> None:
        self._gathers[task_type] = {"node_kinds": (node_kind,),
                                    "resource": resource}
        TASK_LABELS.setdefault(task_type, label)

    def register_station(self, station_id: str, name: str) -> None:
        self._stations[station_id] = {"id": station_id, "name": name,
                                      "dynamic": True}

    def dynamic_recipes(self) -> list[dict]:
        """The dynamic craft entries (for digests / the client)."""
        return [{"task_type": t, "station": r["station"],
                 "inputs": dict(r["inputs"]),
                 "outputs": dict(r["outputs"]), "work_s": r["work_s"]}
                for t, r in self._recipes.items() if r.get("dynamic")]


_DEFAULT_BOOK: RecipeBook | None = None


def default_book() -> RecipeBook:
    """A shared seed-only book for planners nobody wired one into
    (unit tests, headless sims) — exactly the P2 tables."""
    global _DEFAULT_BOOK
    if _DEFAULT_BOOK is None:
        _DEFAULT_BOOK = RecipeBook()
    return _DEFAULT_BOOK
