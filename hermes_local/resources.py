"""Resource world — Phase 1 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, system 1 + phase P1.

The city holds harvestable **resource nodes** and one shared **city
inventory**:

  * trees  -> wood. A mature tree is felled whole by one harvest
    (all its stock at once), leaving a stump that regrows through a
    sapling stage back to maturity.
  * garden plants -> food. One harvest takes one unit; the plot
    regrows unit by unit and is bare while empty.
  * animals -> food. Tending/milking: one harvest takes one unit
    and the animal stays right where it is (it is never removed);
    it then needs its regrow window before it can be tended again.

Regrowth is **tick-driven**: nothing here reads the wall clock.
`ResourceWorld.tick(now)` advances every node's regrow clock to the
world's sim/monotonic time, and the bridge ticks it from the same
message flow that drives the NPC planner (proximity heartbeats,
interactions, /api/world reads) — exactly the world loop the client
already runs.

`ResourceWorld.try_harvest()` is THE harvest path: the bridge's
player "interact" message calls it today, and the ActivityPlanner's
RTS labor (P2) will call the same function with an NPC's position
and name — range/stock rules live here, once, not in callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INTERACT_RANGE = 3.0     # metres: how close an actor must stand

# Per-kind tuning. regrow_seconds is per stock unit.
_KIND_SPECS = {
    "tree":   {"resource": "wood", "max_stock": 3, "regrow_seconds": 60.0,
               "per_harvest": None},        # None = take the whole stock
    "plant":  {"resource": "food", "max_stock": 2, "regrow_seconds": 45.0,
               "per_harvest": 1},
    "animal": {"resource": "food", "max_stock": 2, "regrow_seconds": 40.0,
               "per_harvest": 1},
}


@dataclass
class ResourceNode:
    id: str
    kind: str                      # tree | plant | animal
    name: str                      # display label ("Oak tree", ...)
    x: float
    z: float
    stock: int
    max_stock: int
    regrow_seconds: float
    resource: str                  # wood | food
    per_harvest: int | None = None     # None = whole stock (trees)
    regrow_anchor: float | None = field(default=None, repr=False)

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.z)

    @property
    def state(self) -> str:
        """Visual/life state, derived from stock — clients render
        exactly this: trees stump -> sapling -> mature; plants bare
        -> ready; animals resting (not yet tendable) -> ready.
        Dynamic kinds (an innovation's dig sites, P6) read the same
        way as tended things: ready while they hold stock, resting
        while they regrow."""
        if self.kind == "tree":
            if self.stock <= 0:
                return "stump"
            return "mature" if self.stock >= self.max_stock else "sapling"
        if self.kind == "plant":
            return "bare" if self.stock <= 0 else "ready"
        return "resting" if self.stock <= 0 else "ready"

    @property
    def harvestable(self) -> bool:
        # A tree is only worth felling once it is fully grown again;
        # plants and animals give a unit whenever they have one.
        if self.kind == "tree":
            return self.stock >= self.max_stock
        return self.stock > 0

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                "position": [self.x, self.z],
                "stock": self.stock, "max_stock": self.max_stock,
                "resource": self.resource, "state": self.state,
                "regrow_seconds": self.regrow_seconds}


def _node(node_id: str, kind: str, name: str, x: float, z: float) -> ResourceNode:
    spec = _KIND_SPECS[kind]
    return ResourceNode(
        id=node_id, kind=kind, name=name, x=float(x), z=float(z),
        stock=spec["max_stock"], max_stock=spec["max_stock"],
        regrow_seconds=spec["regrow_seconds"], resource=spec["resource"],
        per_harvest=spec["per_harvest"])


def default_resource_nodes() -> list[ResourceNode]:
    """The starter set, placed on the city grid (roads at
    -96/-48/0/48/96, blocks centred at -72/-24/24/72): a tree grove
    in each park, garden plots by the market rows, an animal pen
    near North Green plus a pair by the West Depot. Positions keep
    clear of the road bands. The web client mirrors this list
    exactly for standalone mode."""
    nodes: list[ResourceNode] = []
    trees = [  # Old Park grove (-72,-72)
        (-82, -78), (-62, -80), (-78, -62), (-60, -62), (-72, -86),
        (-86, -70),
        # North Green grove (72,72)
        (62, 78), (82, 66), (78, 84), (60, 62), (86, 76),
        # scattered near the plaza blocks
        (20, 20), (-20, 24), (24, -18)]
    for i, (x, z) in enumerate(trees, 1):
        nodes.append(_node(f"tree-{i:02d}", "tree", "Oak tree", x, z))
    plants = [  # Market Row West (-36,8)
        (-36, 12), (-33, 12), (-36, 15), (-33, 15),
        # Market Row South (8,-60)
        (10, -60), (10, -57), (16, -60), (16, -57)]
    for i, (x, z) in enumerate(plants, 1):
        nodes.append(_node(f"plant-{i:02d}", "plant", "Garden plot", x, z))
    animals = [  # pen near North Green
        (58, 58), (64, 58), (58, 64), (64, 64),
        # a pair by the West Depot
        (-88, 16), (-88, 24)]
    for i, (x, z) in enumerate(animals, 1):
        nodes.append(_node(f"animal-{i:02d}", "animal", "Goat", x, z))
    return nodes


def make_node(node_id: str, kind: str, name: str, x: float, z: float,
              spec: dict) -> ResourceNode:
    """A node of a caller-defined kind (P6 dynamic sources): the
    spec carries resource / max_stock / regrow_seconds /
    per_harvest exactly as _KIND_SPECS does for the seed kinds."""
    return ResourceNode(
        id=node_id, kind=kind, name=name, x=float(x), z=float(z),
        stock=int(spec["max_stock"]), max_stock=int(spec["max_stock"]),
        regrow_seconds=float(spec["regrow_seconds"]),
        resource=str(spec["resource"]),
        per_harvest=spec.get("per_harvest"))


class ResourceWorld:
    """Nodes + city inventory + the harvest/regrow rules."""

    def __init__(self, nodes: list[ResourceNode] | None = None):
        self.nodes: dict[str, ResourceNode] = {
            n.id: n for n in (nodes if nodes is not None
                              else default_resource_nodes())}
        # wood/food are gathered (P1); planks/meals are crafted from
        # them at stations (P2, see work.py). All four keys always
        # exist so clients and digests can read them unconditionally.
        self.inventory: dict[str, int] = {"wood": 0, "food": 0,
                                          "planks": 0, "meals": 0}
        # Node-kind tuning: the seed table, plus kinds an adopted
        # innovation registers at runtime (P6 — a clay pit is a
        # kind like any other once it exists).
        self.kind_specs: dict[str, dict] = {
            k: dict(v) for k, v in _KIND_SPECS.items()}

    # -- dynamic kinds + nodes (P6) --------------------------------------
    def register_kind(self, kind: str, resource: str,
                      max_stock: int = 3, regrow_seconds: float = 60.0,
                      per_harvest: int | None = 1) -> None:
        """Teach the world a new node kind: what its nodes yield,
        how much they hold and how fast they come back."""
        self.kind_specs[kind] = {
            "resource": resource, "max_stock": int(max_stock),
            "regrow_seconds": float(regrow_seconds),
            "per_harvest": per_harvest}
        self.inventory.setdefault(resource, 0)

    def spawn_node(self, node_id: str, kind: str, name: str,
                   x: float, z: float) -> ResourceNode:
        """Add a node of a registered kind, full of stock."""
        node = make_node(node_id, kind, name, x, z,
                         self.kind_specs[kind])
        self.nodes[node_id] = node
        return node

    # -- regrowth ------------------------------------------------------
    def tick(self, now: float) -> None:
        """Advance regrowth to sim time `now`: one stock unit per
        regrow_seconds spent below max. Called by the bridge from
        the world loop; safe to call as often as messages arrive."""
        for node in self.nodes.values():
            if node.stock >= node.max_stock:
                node.regrow_anchor = None
                continue
            if node.regrow_anchor is None:
                node.regrow_anchor = now
                continue
            units = int((now - node.regrow_anchor) // node.regrow_seconds)
            if units > 0:
                node.stock = min(node.max_stock, node.stock + units)
                node.regrow_anchor += units * node.regrow_seconds
                if node.stock >= node.max_stock:
                    node.regrow_anchor = None

    # -- harvesting ------------------------------------------------------
    def try_harvest(self, node_id: str, actor_pos, now: float,
                    by: str = "Player"):
        """Attempt one harvest. Returns (event, notice): on success
        the harvest event dict and None; on failure None and a short
        human reason. This is the single harvest path — the bridge's
        player interact calls it now; P2's planner labor will call
        it with NPC positions."""
        self.tick(now)
        node = self.nodes.get(node_id)
        if node is None:
            return None, f"No such resource: {node_id}."
        dist = ((node.x - float(actor_pos[0])) ** 2
                + (node.z - float(actor_pos[1])) ** 2) ** 0.5
        if dist > INTERACT_RANGE:
            return None, (f"Too far from the {node.name.lower()} "
                          f"({dist:.0f}m — get within "
                          f"{INTERACT_RANGE:.0f}m).")
        if not node.harvestable:
            if node.kind == "tree" and node.stock > 0:
                return None, "That oak is still a sapling — let it grow."
            return None, (f"The {node.name.lower()} has nothing to give "
                          f"right now — it needs time to regrow.")
        if node.per_harvest is None:
            amount = node.stock               # felling takes the tree
        else:
            amount = min(node.per_harvest, node.stock)
        node.stock -= amount
        if node.stock < node.max_stock and node.regrow_anchor is None:
            node.regrow_anchor = now
        self.inventory[node.resource] = \
            self.inventory.get(node.resource, 0) + amount
        return {"type": "harvest", "node": node.id, "kind": node.kind,
                "resource": node.resource, "amount": amount, "by": by,
                "stock": node.stock,
                "inventory": dict(self.inventory)}, None

    # -- read models ------------------------------------------------------
    def snapshot(self) -> dict:
        """GET /api/world payload: inventory + every node's public
        state (the client renders nodes straight from this)."""
        return {"inventory": dict(self.inventory),
                "nodes": [n.to_dict() for n in self.nodes.values()]}

    def summary(self) -> dict:
        """Compact resource picture for digests (the interpreter):
        inventory totals, node counts, how many stand depleted."""
        depleted = sum(1 for n in self.nodes.values() if n.stock <= 0)
        return {"inventory": dict(self.inventory),
                "nodes": len(self.nodes), "depleted": depleted}
