"""Headless flicker measurement: player standing still for 120 simulated
seconds at the web client's start position (6, 14). Counts how often the
ACTIVE set (== the chat-target list) changes, before vs after the
movement-system rework.

BEFORE is a faithful model of the old behavior, from the pre-rework code:
tiers by plain distance bands (ACTIVE <= 15m, nearest 4; AMBIENT <= 50m)
and the old client sim — AMBIENT NPCs random-wander between sidewalk
walk-nodes within 70m at 1.25 m/s, ACTIVE/DORMANT stand still.

AFTER uses the real NPCManager + ActivityPlanner with arrive steering
mirroring the web client's arriveMove, at the client's 0.7s proximity
heartbeat cadence.

Run:  python tests/measure_flicker.py
"""
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from hermes_local.npc import NPCManager, generate_roster  # noqa: E402
from hermes_local.world_bridge import city_spawn_positions  # noqa: E402

PLAYER = (6.0, 14.0)
DURATION = 120.0
SETTLE = 10.0
TICK = 0.7          # the client's proximity heartbeat cadence
SUB = 0.1           # movement integration step


class FakeNPC:
    """Planner/manager only read .card and call activate/deactivate."""

    def __init__(self, card):
        self.card = card
        self.active = False

    def activate(self):
        self.active = True
        return []

    def deactivate(self):
        self.active = False
        return ""

    def witness(self, event, kind="episode"):
        pass


def steer(pos, tx, tz, speed, dt, stop=0.35):
    dx, dz = tx - pos[0], tz - pos[1]
    d = (dx * dx + dz * dz) ** 0.5
    if d <= stop:
        return pos, True
    slow = min(1.0, max(0.22, (d - stop) / 3.0))
    step = min(d - stop, speed * slow * dt)
    return (pos[0] + dx / d * step, pos[1] + dz / d * step), False


def walk_nodes():
    nodes = []
    for c in (-96, -48, 0, 48, 96):
        for t in range(-108, 109, 24):
            nodes += [(c + 8, t), (t, c + 8), (c - 8, t), (t, c - 8)]
    return nodes


def run_legacy(spawns):
    rng = random.Random(42)
    nodes = walk_nodes()
    pos = dict(spawns)
    wander = {}
    tiers = {n: "dormant" for n in spawns}
    prev_active, changes, changes_post = frozenset(), 0, 0
    moving_time, total_time = 0.0, 0.0
    t = 0.0
    while t <= DURATION:
        steps = int(TICK / SUB)
        for _ in range(steps):
            for name in spawns:
                total_time += SUB
                if tiers[name] != "ambient":
                    continue
                tgt = wander.get(name)
                if tgt is None or (pos[name][0] - tgt[0]) ** 2 \
                        + (pos[name][1] - tgt[1]) ** 2 < 1.0:
                    near = [w for w in nodes
                            if (w[0] - pos[name][0]) ** 2
                            + (w[1] - pos[name][1]) ** 2 < 70 ** 2]
                    tgt = wander[name] = rng.choice(near) if near else None
                if tgt:
                    pos[name], done = steer(pos[name], tgt[0], tgt[1],
                                            1.25, SUB, stop=0.2)
                    if not done:
                        moving_time += SUB
        dists = {n: ((p[0] - PLAYER[0]) ** 2 + (p[1] - PLAYER[1]) ** 2) ** 0.5
                 for n, p in pos.items()}
        in15 = sorted((n for n, d in dists.items() if d <= 15),
                      key=lambda n: dists[n])
        active = frozenset(in15[:4])
        tiers = {n: ("active" if n in active
                     else "ambient" if dists[n] <= 50 else "dormant")
                 for n in spawns}
        if active != prev_active:
            changes += 1
            if t >= SETTLE:
                changes_post += 1
            prev_active = active
        t += TICK
    return changes, changes_post, moving_time / total_time


def run_new(spawns):
    cards = generate_roster(len(spawns))
    npcs = {c.name: FakeNPC(c) for c in cards}
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    planner = manager.planner
    pos = dict(spawns)
    prev_active, changes, changes_post = frozenset(), 0, 0
    social_events = 0
    t = 0.0
    while t <= DURATION:
        events = manager.update(PLAYER, pos, now=t)
        social_events += sum(1 for e in events if e.get("type") == "social")
        active = frozenset(manager.stats()["active_names"])
        if active != prev_active:
            changes += 1
            if t >= SETTLE:
                changes_post += 1
            prev_active = active
        steps = int(TICK / SUB)
        for _ in range(steps):
            for name, st in planner.state.items():
                if st["state"] == "travel":
                    pos[name], _ = steer(pos[name], st["tx"], st["tz"],
                                         st["speed"], SUB)
        t += TICK
    stats = planner.stats()
    return changes, changes_post, stats, social_events


def main():
    cards = generate_roster(54)
    spawns = city_spawn_positions([c.name for c in cards])
    lc, lcp, lmoving = run_legacy(spawns)
    nc, ncp, nstats, nsocial = run_new(spawns)
    print(f"Scenario: player stationary at {PLAYER}, 54 NPCs, "
          f"{DURATION:.0f}s, updates every {TICK}s, settle {SETTLE:.0f}s")
    print(f"BEFORE (wander + plain bands): ACTIVE-set changes "
          f"total={lc}, after settle={lcp}; NPC time moving={lmoving:.0%}")
    print(f"AFTER  (planner + hysteresis): ACTIVE-set changes "
          f"total={nc}, after settle={ncp}; dwell share="
          f"{nstats['dwell_share']:.0%} (dwell {nstats['dwell_s']}s vs "
          f"travel {nstats['travel_s']}s per NPC avg); "
          f"NPC-NPC social pairings={nsocial}")


if __name__ == "__main__":
    main()
