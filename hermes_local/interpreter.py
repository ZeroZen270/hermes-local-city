"""World Interpreter: an oracle that knows the whole city.

Aaron's ask: "an AI world interpreter that knows everything in this
world that I can ask things of." The interpreter is NOT a citizen —
it is the keeper of the world's knowledge, answering questions from
live world state rather than from a character's point of view:

  * a **digest** of the world right now (every citizen's role, tier,
    position, current activity and social partner; pending
    conversation-born plans; tier counts),
  * a **memory search** across ALL citizens' episodic memory
    namespaces for the question's terms (keyword pass over every
    namespace, merged with one global semantic recall, attributed
    back to the citizen who remembers it),
  * the bridge's **recent-events** ring buffer (speech, social
    exchanges, plans, engagements, tier changes).

With a live brain (an LLMClient) those three are handed to the model
with strict instructions to answer only from them. Without one
(mock/demo), a small rule-based reader answers the basic shapes from
the digest — "Where is Mara?", "Who is Mara?", "How many citizens
are here?", "Who is at Central Plaza?" — and says honestly that the
full interpreter needs the live brain for anything else.

The interpreter only ever READS the world: it never speaks to a
citizen, never engages anyone, and writes nothing into any NPC's
memory. It behaves identically in ghost mode — a ghost cannot talk
to citizens, but may still consult the interpreter.
"""
from __future__ import annotations

import re
import time

from .npc import district_anchor_specs
from .work import TASK_LABELS

_MAX_MEMORIES = 12
_MAX_EVENTS = 25

# Weather states as plain phrases (P8) — shared by the digest's
# sky line and the rule answers for "what's the weather?".
_WEATHER_PHRASE = {"clear": "the sky is clear",
                   "cloudy": "the sky is overcast",
                   "rain": "it is raining",
                   "storm": "a storm is rolling over the city",
                   "fog": "a thick fog lies over the city"}

_STOPWORDS = frozenset({
    "what", "where", "when", "which", "would", "could", "should",
    "about", "there", "their", "anyone", "someone", "everyone",
    "anything", "something", "happened", "happen", "recently",
    "lately", "today", "tell", "know", "does", "doing", "have",
    "with", "this", "that", "them", "then", "than", "from", "were",
    "been", "much", "many", "most", "some", "city", "citizen",
    "citizens", "people", "person", "anybody", "everybody",
})

_SYSTEM_PROMPT = (
    "You are the World Interpreter of Hermes City, a living "
    "simulation whose citizens have their own memories and plans. "
    "You are not a citizen; you keep the world's knowledge, and the "
    "player is asking you about the world.\n"
    "Answer ONLY from the WORLD KNOWLEDGE provided below: the "
    "citizen digest, the citizen memories matching the question, "
    "and the recent events. Be specific — use citizens' names, "
    "their roles, the places they are at and what they are doing. "
    "If the answer is not in the knowledge, say so plainly (for "
    "example: 'I don't know that — it isn't in the city's records.') "
    "instead of inventing anything. Never invent citizens, places, "
    "conversations or events. Keep the answer concise: one short "
    "paragraph at most.")


def _norm(text: str) -> str:
    """Lowercase, punctuation -> spaces, collapsed: the shared
    normal form for question/citizen/place matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _building_counts(buildings) -> dict:
    """Finished buildings (a list of building records) as a
    kind -> count map, for digest lines."""
    counts: dict[str, int] = {}
    for b in buildings or []:
        kind = b.get("kind") if isinstance(b, dict) else None
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _question_terms(question: str, names) -> set:
    """Significant terms of a question: every citizen name it
    mentions, plus its content words (stopwords removed)."""
    low = question.lower()
    terms = {name.lower() for name in names if name.lower() in low}
    for word in re.findall(r"[a-z][a-z'-]{3,}", low):
        if word not in _STOPWORDS:
            terms.add(word)
    return terms


def _activity(c: dict) -> str:
    """One-phrase description of what a snapshot citizen is doing."""
    state, kind = c.get("state"), c.get("kind")
    anchor, partner = c.get("anchor"), c.get("partner")
    if c.get("engaged") or state == "engaged":
        return "in conversation with the player"
    task = c.get("task")
    if task:
        label = TASK_LABELS.get(task["type"], task["type"])
        place = task.get("place") or task.get("target")
        if task.get("phase") == "work":
            return f"working: {label} at {place}"
        return f"traveling to {place} to {label}"
    if state == "social":
        text = f"chatting with {partner}" if partner else "chatting"
        return text + (f" at {anchor}" if anchor else "")
    if state == "travel":
        text = f"traveling ({kind})" if kind else "traveling"
        return text + (f" toward {anchor}" if anchor else "")
    if state == "dwell":
        return (f"at {anchor} ({kind})" if anchor
                else f"resting ({kind or 'idle'})")
    return f"near {anchor}" if anchor else "about the city"


class WorldInterpreter:
    """Reads a WorldBridge's world and answers questions about it."""

    def __init__(self, bridge, client=None, memory=None):
        self.bridge = bridge
        self.client = client        # LLMClient-shaped, or None (rules)
        self._memory = memory       # shared Memory store, or None

    # -- plumbing ------------------------------------------------------
    @property
    def memory(self):
        """The shared memory store: the one given at construction,
        else the first NPC's (build_world shares one store across
        the whole roster)."""
        if self._memory is not None:
            return self._memory
        for npc in self.bridge.npcs.values():
            store = getattr(npc, "memory", None)
            if store is not None:
                self._memory = store
                return store
        return None

    # -- knowledge assembly ---------------------------------------------
    def snapshot(self) -> dict:
        """Structured world state: per-citizen situation + identity,
        pending plans, tier counts, anchor names."""
        bridge, manager = self.bridge, self.bridge.manager
        tiers = manager.tiers if manager else {}
        positions = manager.positions if manager else {}
        planner = manager.planner if manager else None
        citizens = []
        for name, npc in bridge.npcs.items():
            card = npc.card
            st = planner.state.get(name) if planner is not None else None
            anchor_obj = st.get("anchor") if st else None
            pos = positions.get(name)
            citizens.append({
                "name": name, "role": card.role,
                "persona": card.persona, "backstory": card.backstory,
                "traits": list(card.traits or []),
                "goals": list(card.goals or []),
                "tier": tiers.get(name, "dormant"),
                "position": ([float(pos[0]), float(pos[1])]
                             if pos else None),
                "state": st.get("state") if st else None,
                "kind": st.get("kind") if st else None,
                "anchor": anchor_obj.name if anchor_obj is not None else None,
                "partner": st.get("partner") if st else None,
                "engaged": bool(manager is not None
                                and manager.is_engaged(name)),
            })
        plans = []
        if planner is not None:
            for plan in planner.pending_plans:
                plans.append({"a": plan["a"], "b": plan["b"],
                              "where": plan["anchor"].name,
                              "active": bool(plan.get("active"))})
        counts = {"active": 0, "ambient": 0, "dormant": 0}
        for c in citizens:
            counts[c["tier"]] = counts.get(c["tier"], 0) + 1
        counts["total"] = len(citizens)
        resources = None
        res = getattr(bridge, "resources", None)
        if res is not None:
            try:
                resources = res.summary()
            except Exception:
                resources = None
        work = None
        if planner is not None:
            try:
                work = planner.work_summary()
            except Exception:
                work = None
        if work:
            by_npc = {t["npc"]: t for t in work["tasks"]}
            for c in citizens:
                c["task"] = by_npc.get(c["name"])
        construction = None
        con = getattr(bridge, "construction", None)
        if con is not None:
            try:
                construction = con.summary()
                if planner is not None:
                    for site in construction["sites"]:
                        site["where"] = self._nearest_anchor(
                            planner.anchors, site["position"][0],
                            site["position"][1])
            except Exception:
                construction = None
        civics = None
        cv = getattr(bridge, "civics", None)
        if cv is not None:
            try:
                civics = cv.public_state()
            except Exception:
                civics = None
        terrain = None
        ter = getattr(bridge, "terrain", None)
        if ter is not None:
            try:
                terrain = ter.summary()
                if planner is not None:
                    terrain["high_where"] = self._nearest_anchor(
                        planner.anchors, terrain["max_at"][0],
                        terrain["max_at"][1])
                    terrain["low_where"] = self._nearest_anchor(
                        planner.anchors, terrain["min_at"][0],
                        terrain["min_at"][1])
                    if terrain.get("pond_at"):
                        terrain["pond_where"] = self._nearest_anchor(
                            planner.anchors, terrain["pond_at"][0],
                            terrain["pond_at"][1])
                    # The districts (P9): the planner anchors the
                    # city's growth rings opened, by name.
                    known = {spec[0] for spec in district_anchor_specs(
                        terrain.get("expansions", 0))}
                    terrain["districts"] = [
                        a.name for a in planner.anchors
                        if a.name in known]
            except Exception:
                terrain = None
        persistence = None
        psvc = getattr(bridge, "persistence", None)
        if psvc is not None:
            try:
                persistence = psvc.public_status()
            except Exception:
                persistence = None
        weather = None
        weng = getattr(bridge, "weather", None)
        if weng is not None:
            try:
                weather = weng.public()
            except Exception:
                weather = None
        designs = None
        deng = getattr(bridge, "designs", None)
        if deng is not None:
            try:
                pub = deng.public_state()
                designs = {
                    "counts": pub["counts"],
                    "styles": [r["name"] for r in pub["styles"]],
                    "assets": [r["name"] for r in pub["assets"]],
                    "interiors": [{
                        "name": r["name"],
                        "building_kind": r.get("building_kind"),
                        "building_id": r.get("building_id"),
                        "rooms": [{"name": rm["name"],
                                   "props": [p["kind"]
                                             for p in rm["props"]]}
                                  for rm in r["rooms"]]}
                        for r in pub["interiors"]]}
            except Exception:
                designs = None
        research = None
        rs_eng = getattr(bridge, "research", None)
        inno = getattr(bridge, "innovations", None)
        if rs_eng is not None or inno is not None:
            try:
                research = {
                    "knowledge": list(
                        getattr(bridge, "city_knowledge", []))[-5:],
                    "innovations": (inno.public_state()
                                    if inno is not None else
                                    {"innovations": [],
                                     "under_study": [],
                                     "market": {"stock": {},
                                                "trade": 0}}),
                    "recipes": (bridge.recipes.dynamic_recipes()
                                if getattr(bridge, "recipes", None)
                                is not None else [])}
            except Exception:
                research = None
        return {"citizens": citizens, "plans": plans, "counts": counts,
                "resources": resources, "work": work,
                "construction": construction, "terrain": terrain,
                "civics": civics, "research": research,
                "persistence": persistence, "weather": weather,
                "designs": designs,
                "anchors": ([a.name for a in planner.anchors]
                            if planner is not None else []),
                "brain": (bridge.brain_info or {}).get("brain", ""),
                "live": self.client is not None}

    @staticmethod
    def _nearest_anchor(anchors, x: float, z: float) -> str:
        """The name of the closest named place to a point — how the
        digest says where a construction site is ("at Market Row")."""
        best, best_d = None, float("inf")
        for a in anchors or []:
            d = (a.x - x) ** 2 + (a.z - z) ** 2
            if d < best_d:
                best, best_d = a, d
        return best.name if best is not None else "the city outskirts"

    def digest_text(self, snap: dict) -> str:
        counts = snap["counts"]
        brain = snap["brain"] or ("live model" if snap["live"]
                                   else "demo brain (rule-based)")
        lines = [f"Hermes City: {counts['total']} citizens. "
                 f"Brain: {brain}.",
                 f"Tiers now: {counts['active']} ACTIVE, "
                 f"{counts['ambient']} AMBIENT, "
                 f"{counts['dormant']} DORMANT."]
        sky = snap.get("weather")
        if sky:
            # The sky line (P8): time of day + weather, always —
            # the oracle should know whether it is raining tonight
            # without being asked.
            lines.append(
                f"Sky: it is {sky['daypart']} ({sky['clock']}) and "
                f"{_WEATHER_PHRASE.get(sky['state'], sky['state'])}.")
        lines.append("Citizens:")
        for c in snap["citizens"]:
            pos = c["position"]
            pos_text = (f" Position ({pos[0]:.0f}, {pos[1]:.0f})."
                        if pos else "")
            lines.append(f"- {c['name']} ({c['role']}, "
                         f"{c['tier'].upper()}): {_activity(c)}.{pos_text}")
        if snap["plans"]:
            lines.append("Pending plans:")
            for p in snap["plans"]:
                lines.append(
                    f"- {p['a']} and {p['b']} plan to meet at {p['where']}"
                    + (" (underway now)." if p["active"] else "."))
        res = snap.get("resources")
        if res:
            inv = res["inventory"]
            lines.append(
                f"Resources: the city stores hold "
                f"{inv.get('wood', 0)} wood and {inv.get('food', 0)} "
                f"food, plus {inv.get('planks', 0)} planks and "
                f"{inv.get('meals', 0)} meals crafted at stations, "
                f"gathered from {res['nodes']} resource nodes "
                f"(trees, garden plots, animals); {res['depleted']} "
                f"of them are depleted and regrowing.")
        work = snap.get("work")
        if work:
            tg = work["targets"]
            if work["tasks"]:
                doing = "; ".join(
                    f"{t['npc']} — "
                    f"{TASK_LABELS.get(t['type'], t['type'])} at "
                    f"{t.get('place') or t['target']} ({t['phase']})"
                    for t in work["tasks"])
                lines.append(
                    f"Work: the foreman keeps baseline stocks of "
                    f"{tg['wood']} wood and {tg['food']} food. "
                    f"Tasks underway: {doing}.")
            else:
                lines.append(
                    f"Work: the foreman keeps baseline stocks of "
                    f"{tg['wood']} wood and {tg['food']} food. "
                    f"No tasks underway.")
        construction = snap.get("construction")
        if construction:
            for site in construction.get("sites", []):
                needs = []
                for key in ("wood", "food", "planks", "meals"):
                    short = site["materials_required"].get(key, 0) \
                        - site["materials_delivered"].get(key, 0)
                    if short > 0:
                        needs.append(f"{short} more {key}")
                lines.append(
                    f"Construction: a {site['kind']} is going up at "
                    f"{site.get('where', 'the city')} — "
                    f"{site['stage']}, "
                    f"{int(round(site['progress'] * 100))}% built, "
                    f"{site['state'].replace('_', ' ')}"
                    + (f", needs {', '.join(needs)}" if needs else "")
                    + ".")
            done = _building_counts(construction.get("buildings"))
            if done:
                lines.append(
                    "Finished buildings: "
                    + ", ".join(f"{n} {k}" for k, n in done.items())
                    + ".")
        terrain = snap.get("terrain")
        if terrain:
            # The land line (P9): the city's span is always worth
            # reporting — how far the ground runs, how many rings
            # it has grown, and the districts those rings opened.
            bound = terrain.get("bound", 120.0)
            grown = terrain.get("expansions", 0)
            land_line = (f"Land: the city spans "
                         f"{int(bound * 2)}m across "
                         f"(±{int(bound)}m)")
            if grown:
                land_line += (f", grown by {grown} expansion"
                              f"{'s' if grown != 1 else ''} beyond "
                              f"the founding core")
            else:
                land_line += " — the founding core, not yet grown"
            districts = terrain.get("districts") or []
            if districts:
                land_line += (f". Its districts: "
                              f"{', '.join(districts)}")
            lines.append(land_line + ".")
        if terrain and (terrain["max_h"] - terrain["min_h"] > 0.05
                        or terrain.get("pond") or terrain.get("jobs")):
            # A still-flat world says nothing (the digest stays the
            # P3 shape); once the land is shaped, describe it.
            line = (f"Terrain: the land runs from "
                    f"{terrain['min_h']:.1f} m near "
                    f"{terrain.get('low_where', 'the city outskirts')} "
                    f"up to {terrain['max_h']:.1f} m near "
                    f"{terrain.get('high_where', 'the city outskirts')}.")
            if terrain.get("pond"):
                pond_where = terrain.get("pond_where",
                                         "the city outskirts")
                line += f" A pond fills the lowest ground near {pond_where}."
            else:
                line += (" No ground dips below the water line, so "
                         "there are no ponds.")
            if terrain.get("jobs"):
                line += (f" Citizen land crews are working on "
                         f"{terrain['jobs']} patch"
                         f"{'es' if terrain['jobs'] != 1 else ''}.")
            lines.append(line)
        civics = snap.get("civics")
        if civics:
            open_p = civics.get("open")
            if open_p:
                t = open_p["tally"]
                lines.append(
                    f"Civics: the city is deciding on "
                    f"'{open_p['title']}' (proposed by "
                    f"{open_p['proposer']} — {open_p['rationale']}) "
                    f"Ballots so far: {t['for']} for, "
                    f"{t['against']} against, {t['abstain']} "
                    f"abstained — {t['cast']} of {t['total']} cast; "
                    f"the proposal is {open_p['state']}.")
            elif civics.get("history"):
                last = civics["history"][-1]
                lt = last["tally"]
                lines.append(
                    f"Civics: the city's last decision — "
                    f"'{last['title']}' (proposed by "
                    f"{last['proposer']}) was "
                    f"{last['outcome'].upper()} ({lt['for']} for, "
                    f"{lt['against']} against, {lt['abstain']} "
                    f"abstained)."
                    + (f" {last['executed']}" if last.get("executed")
                       else ""))
        research = snap.get("research")
        if research:
            bits = []
            insights = [e for e in research["knowledge"]
                        if e.get("kind") == "insight"]
            if insights:
                last = insights[-1]
                bits.append(
                    f"the latest insight at the research terminal "
                    f"({last.get('citizen', 'a citizen')} asked "
                    f"\"{last.get('question', '')}\"): "
                    f"{last.get('summary', '')}")
            regs = research["innovations"]["innovations"]
            if regs:
                bits.append(
                    "the city now makes "
                    + ", ".join(f"{r['name']} ({r['kind']})"
                                for r in regs))
            study = research["innovations"]["under_study"]
            if study:
                bits.append(
                    "under study: "
                    + ", ".join(f"'{u['hint']}'" for u in study))
            market = research["innovations"]["market"]
            stock = ", ".join(f"{v} {k}" for k, v
                              in market["stock"].items() if v)
            if market["trade"] or stock:
                bits.append(
                    f"the market has traded {market['trade']} in "
                    f"goods so far"
                    + (f"; the stalls hold {stock}" if stock else ""))
            if bits:
                lines.append("Research & innovation: "
                             + "; ".join(bits) + ".")
        designs = snap.get("designs")
        if designs and (designs["counts"]["styles"]
                        or designs["counts"]["interiors"]
                        or designs["counts"]["assets"]):
            dc = designs["counts"]
            names = (designs["styles"]
                     + [i["name"] for i in designs["interiors"]]
                     + designs["assets"])
            lines.append(
                f"Designs: the city holds {sum(dc.values())} "
                f"registered designs ({dc['styles']} building "
                f"styles, {dc['interiors']} interiors, "
                f"{dc['assets']} placed assets): "
                + ", ".join(names) + ".")
        persistence = snap.get("persistence")
        if persistence and persistence.get("repo") \
                and persistence.get("state") in ("synced", "offline"):
            # Persistence line only when repo sync is enabled
            # (P7): the durable copy's repo + freshness, or the
            # honest offline note.
            synced_at = max(persistence.get("last_push") or 0.0,
                            persistence.get("last_pull") or 0.0)
            when = (time.strftime("%H:%M:%S",
                                  time.localtime(synced_at))
                    if synced_at else "not yet")
            if persistence["state"] == "offline":
                lines.append(
                    f"Persistence: world data syncs to the GitHub "
                    f"repo {persistence['repo']}, but the sync is "
                    f"currently OFFLINE (last success {when}) — the "
                    f"local snapshot remains authoritative and the "
                    f"repo copy may be stale.")
            else:
                line = (
                    f"Persistence: the world's live data is saved "
                    f"to the GitHub repo {persistence['repo']} "
                    f"(last synced {when}), so the city persists "
                    f"between loads.")
                if persistence.get("keep_versions"):
                    line += (
                        " Dated copies of earlier states are "
                        "kept in the repo's versions/ folder.")
                lines.append(line)
        return "\n".join(lines)

    def search_memories(self, question: str,
                        limit: int = _MAX_MEMORIES) -> list[dict]:
        """Memories across ALL citizens matching the question, each
        attributed to its citizen: [{citizen, text, kind, ts, score}].

        Two passes, merged: a keyword pass over every namespace (the
        question's terms literally in the memory text — deterministic
        even with the offline embedder) and one global semantic
        recall whose rows are attributed through the text->citizen
        map the keyword pass built."""
        memory = self.memory
        if memory is None:
            return []
        terms = _question_terms(question, list(self.bridge.npcs))
        hits: dict[tuple, dict] = {}
        owner: dict[str, str] = {}
        for name, npc in self.bridge.npcs.items():
            try:
                entries = memory.list_namespace(npc.card.namespace)
            except Exception:
                entries = []
            for entry in entries:
                text = str(entry.get("text", ""))
                if not text:
                    continue
                owner.setdefault(text, name)
                matched = [t for t in terms if t in text.lower()]
                if matched:
                    hits[(name, text)] = {
                        "citizen": name, "text": text,
                        "kind": entry.get("kind", "note"),
                        "ts": entry.get("ts"),
                        "score": float(len(matched))}
        try:
            rows = memory.recall(question, k=limit)
        except Exception:
            rows = []
        for row in rows:
            score = float(row.get("score") or 0.0)
            if score <= 0.0:
                continue    # no signal (e.g. offline embedder): skip
            text = str(row.get("text", ""))
            citizen = owner.get(text)
            if citizen is None:
                continue
            key = (citizen, text)
            if key in hits:
                hits[key]["score"] += score
            else:
                hits[key] = {"citizen": citizen, "text": text,
                             "kind": row.get("kind", "note"),
                             "ts": row.get("ts"), "score": score}
        out = sorted(hits.values(), key=lambda h: -h["score"])
        return out[:limit]

    def events_text(self, limit: int = _MAX_EVENTS) -> str:
        """The bridge's recent events as short chronological lines."""
        lines = []
        for rec in self.bridge.recent_events()[-limit:]:
            stamp = time.strftime("%H:%M:%S",
                                  time.localtime(rec.get("ts", 0.0)))
            kind = rec.get("type")
            if kind == "dialogue":
                lines.append(f"[{stamp}] {rec.get('npc', '?')} said: "
                             f"\"{rec.get('text', '')}\"")
            elif kind == "social":
                who = rec.get("a") or rec.get("npc", "?")
                lines.append(f"[{stamp}] {who} → {rec.get('b', '?')}: "
                             f"\"{rec.get('text', '')}\"")
            elif kind == "plan":
                lines.append(f"[{stamp}] {rec.get('a', '?')} and "
                             f"{rec.get('b', '?')} agreed to meet at "
                             f"{rec.get('where', '?')}.")
            elif kind == "tier":
                lines.append(f"[{stamp}] {rec.get('npc', '?')} became "
                             f"{str(rec.get('tier', '')).upper()}"
                             f" ({rec.get('reason', '')}).")
            elif kind == "engage":
                verb = ("started talking with the player"
                        if rec.get("engaged")
                        else "stopped talking with the player")
                lines.append(f"[{stamp}] {rec.get('npc', '?')} {verb}"
                             f" ({rec.get('reason', '')}).")
            elif kind == "harvest":
                lines.append(f"[{stamp}] {rec.get('by', '?')} harvested "
                             f"{rec.get('amount', '?')} "
                             f"{rec.get('resource', '')} from a "
                             f"{rec.get('kind', 'node')} "
                             f"({rec.get('node', '?')}).")
            elif kind == "task":
                label = TASK_LABELS.get(rec.get("task_type"),
                                        rec.get("task_type"))
                lines.append(f"[{stamp}] {rec.get('npc', '?')} "
                             f"{rec.get('state', '')} the task "
                             f"'{label}' ({rec.get('target', '?')}).")
            elif kind == "build":
                state = rec.get("state", "")
                what = rec.get("name") or rec.get("kind", "building")
                if state == "placed":
                    lines.append(f"[{stamp}] A construction site for "
                                 f"a {what} was marked out.")
                elif state == "frame":
                    lines.append(f"[{stamp}] The frame of the {what} "
                                 f"went up.")
                elif state == "complete":
                    lines.append(f"[{stamp}] The {what} was finished "
                                 f"and now stands in the city.")
            elif kind == "terraform":
                op = rec.get("op", "shape")
                pos = rec.get("position") or [0, 0]
                where = f"({pos[0]:.0f}, {pos[1]:.0f})"
                if rec.get("mode") == "crew":
                    if rec.get("state") == "complete":
                        lines.append(
                            f"[{stamp}] Citizens finished the {op} "
                            f"land work at {where}.")
                    else:
                        lines.append(
                            f"[{stamp}] A citizen crew took on a "
                            f"{op} job at {where}.")
                else:
                    past = {"raise": "raised", "lower": "lowered",
                            "flatten": "flattened",
                            "smooth": "smoothed"}.get(op, "reshaped")
                    lines.append(f"[{stamp}] The land was {past} by "
                                 f"god mode at {where}.")
            elif kind == "proposal":
                lines.append(f"[{stamp}] {rec.get('proposer', '?')} "
                             f"put a proposal to the city: "
                             f"\"{rec.get('title', '')}\".")
            elif kind == "debate":
                lines.append(f"[{stamp}] In the civic debate, "
                             f"{rec.get('npc', '?')} said: "
                             f"\"{rec.get('text', '')}\"")
            elif kind == "ballot":
                lines.append(f"[{stamp}] {rec.get('name', '?')} "
                             f"voted {rec.get('choice', '?')} — "
                             f"{rec.get('reason', '')}")
            elif kind == "civic_result":
                lines.append(f"[{stamp}] The city decided: "
                             f"\"{rec.get('title', '')}\" was "
                             f"{str(rec.get('outcome', '')).upper()}.")
            elif kind == "research":
                state = rec.get("state", "")
                if state == "insight":
                    lines.append(
                        f"[{stamp}] {rec.get('by', '?')} researched "
                        f"\"{rec.get('question', '')}\" at the "
                        f"terminal and learned: "
                        f"{rec.get('summary', '')}")
                elif state == "no_connection":
                    lines.append(
                        f"[{stamp}] The research terminal could not "
                        f"reach the web for "
                        f"\"{rec.get('question', '')}\" — nothing "
                        f"was learned.")
                else:
                    lines.append(
                        f"[{stamp}] {rec.get('by', '?')} sat down "
                        f"at the research terminal with a question.")
            elif kind == "innovation":
                if rec.get("state") == "registered":
                    lines.append(
                        f"[{stamp}] A discovery became real: the "
                        f"city can now make {rec.get('name', '?')}.")
                else:
                    lines.append(
                        f"[{stamp}] The discovery "
                        f"'{rec.get('hint', '')}' was filed as "
                        f"under study — the city could not make it "
                        f"real yet.")
            elif kind == "sale":
                lines.append(
                    f"[{stamp}] {rec.get('buyer', '?')} bought "
                    f"{rec.get('name', 'a good')} at the market "
                    f"from {rec.get('seller', 'the stalls')} for "
                    f"{rec.get('price', '?')}.")
            elif kind == "weather":
                lines.append(
                    f"[{stamp}] The sky changed: "
                    f"{rec.get('state', '?')} — "
                    f"{rec.get('daypart', '')} "
                    f"(hour {rec.get('hour', '?')}).")
            elif kind == "design":
                if rec.get("state") == "registered":
                    lines.append(
                        f"[{stamp}] A design was registered: "
                        f"'{rec.get('name', '?')}' "
                        f"({rec.get('kind', '?')}, "
                        f"{rec.get('origin', '')}).")
                elif rec.get("state") == "shelved":
                    lines.append(
                        f"[{stamp}] The design "
                        f"'{rec.get('hint', '')}' was shelved: "
                        f"{rec.get('reason', '')}")
                else:
                    lines.append(
                        f"[{stamp}] The designers took up a "
                        f"{rec.get('kind', 'design')} commission: "
                        f"'{rec.get('hint', '')}'.")
        return "\n".join(lines)

    # -- answering --------------------------------------------------------
    def ask(self, question: str) -> str:
        q = str(question or "").strip()
        if not q:
            raise ValueError("missing question")
        if self.client is not None:
            try:
                return self._ask_live(q)
            except Exception:
                pass    # a dead model call degrades to the digest rules
        return self._ask_rules(q)

    def _ask_live(self, question: str) -> str:
        snap = self.snapshot()
        mems = self.search_memories(question)
        mem_text = "\n".join(
            f"- [{m['citizen']} · {m['kind']}] {m['text']}"
            for m in mems) or "(no citizen memory matched)"
        user = (f"WORLD KNOWLEDGE\n{self.digest_text(snap)}\n\n"
                f"CITIZEN MEMORIES MATCHING THE QUESTION\n{mem_text}\n\n"
                f"RECENT EVENTS\n{self.events_text() or '(nothing yet)'}\n\n"
                f"QUESTION: {question}")
        resp = self.client.chat([{"role": "system",
                                  "content": _SYSTEM_PROMPT},
                                 {"role": "user", "content": user}])
        answer = (resp.get("content") or "").strip()
        if not answer:
            raise ValueError("the interpreter's brain returned no answer")
        return answer

    # -- rule-based reader (mock/demo mode) --------------------------------
    @staticmethod
    def _match_citizen(norm_q: str, snap: dict):
        padded = f" {_norm(norm_q)} "
        best = None
        for c in snap["citizens"]:
            if f" {_norm(c['name'])} " in padded:
                if best is None or len(c["name"]) > len(best["name"]):
                    best = c
        return best

    @staticmethod
    def _match_place(norm_q: str, snap: dict):
        padded = f" {_norm(norm_q)} "
        anchors = snap["anchors"]
        for name in sorted(anchors, key=len, reverse=True):
            if f" {_norm(name)} " in padded:
                return name
        # One distinctive word ("plaza") is enough when it names
        # exactly one place; "market" names four and stays ambiguous.
        words = set(_norm(norm_q).split())
        hits = [name for name in anchors
                if any(len(w) >= 4 and w in words
                       for w in _norm(name).split())]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _citizen_line(c: dict) -> str:
        traits = ", ".join(c["traits"]) or "ordinary"
        return (f"{c['name']} is a {c['role']} ({traits}). "
                f"Right now: {_activity(c)}.")

    def _ask_rules(self, question: str) -> str:
        snap = self.snapshot()
        q = _norm(question)
        padded = f" {q} "
        counts = snap["counts"]
        citizen = self._match_citizen(q, snap)
        place = self._match_place(q, snap)

        sky = snap.get("weather")
        if sky and (" weather " in padded or " raining " in padded
                    or " rain " in padded or " storm " in padded
                    or " fog " in padded or " cloudy " in padded
                    or " sunny " in padded or " overcast " in padded):
            phrase = _WEATHER_PHRASE.get(sky["state"], sky["state"])
            return (f"{phrase[0].upper() + phrase[1:]} — it is "
                    f"{sky['daypart']}, {sky['clock']}.")
        if sky and (" time " in padded or " o clock " in padded):
            return (f"It is {sky['clock']} — {sky['daypart']}, and "
                    f"{_WEATHER_PHRASE.get(sky['state'], sky['state'])}.")

        work = snap.get("work")
        if work and (" working " in padded or " task " in padded
                     or " tasks " in padded or " foreman " in padded):
            tg = work["targets"]
            if not work["tasks"]:
                return (f"No tasks are underway. The foreman keeps "
                        f"baseline stocks of {tg['wood']} wood and "
                        f"{tg['food']} food and assigns citizens when "
                        f"the stores run low.")
            doing = "; ".join(
                f"{t['npc']} — {TASK_LABELS.get(t['type'], t['type'])} "
                f"at {t.get('place') or t['target']} ({t['phase']})"
                for t in work["tasks"])
            return f"Tasks underway: {doing}."

        construction = snap.get("construction")
        if construction and (" construction " in padded
                              or " building " in padded
                              or " built " in padded
                              or " buildings " in padded):
            bits = []
            for site in construction.get("sites", []):
                bits.append(
                    f"a {site['kind']} going up at "
                    f"{site.get('where', 'the city')} "
                    f"({int(round(site['progress'] * 100))}% built, "
                    f"{site['state'].replace('_', ' ')})")
            done = _building_counts(construction.get("buildings"))
            if done:
                bits.append("finished: " + ", ".join(
                    f"{n} {k}" for k, n in done.items()))
            if not bits:
                return ("Nothing is under construction and no "
                        "buildings have been finished yet.")
            return "Construction: " + "; ".join(bits) + "."

        designs = snap.get("designs")
        if designs and (" inside " in padded or " interior " in padded
                        or " interiors " in padded
                        or " furniture " in padded
                        or " rooms " in padded):
            kind = None
            if " stall " in padded:
                kind = "stall"
            elif " house " in padded or " home " in padded \
                    or " cottage " in padded:
                kind = "house"
            recs = [i for i in designs["interiors"]
                    if kind is None
                    or i.get("building_kind") == kind]
            if not recs:
                return ("No interior is registered for that kind "
                        "of building yet — the designers have not "
                        "dressed one.")
            bits = []
            for rec in recs:
                rooms = []
                for rm in rec["rooms"]:
                    tally: dict[str, int] = {}
                    for p in rm["props"]:
                        tally[p] = tally.get(p, 0) + 1
                    props = ", ".join(
                        f"{k} ×{n}" if n > 1 else k
                        for k, n in tally.items())
                    rooms.append(f"the {rm['name']} holds {props}")
                bits.append(
                    f"Inside a {rec.get('building_kind') or 'building'} "
                    f"('{rec['name']}'): " + "; ".join(rooms))
            return ". ".join(bits) + "."

        terrain = snap.get("terrain")
        if terrain and (" how big " in padded or " grow " in padded
                        or " grown " in padded or " expand " in padded
                        or " expansion " in padded
                        or " districts " in padded
                        or " how far " in padded):
            bound = terrain.get("bound", 120.0)
            grown = terrain.get("expansions", 0)
            districts = terrain.get("districts") or []
            answer = (f"The city spans {int(bound * 2)}m across "
                      f"(±{int(bound)}m)")
            if grown:
                answer += (f" — it has grown by {grown} expansion"
                           f"{'s' if grown != 1 else ''} beyond "
                           f"the founding core")
                if districts:
                    answer += (f", opening the districts of "
                               f"{', '.join(districts)}")
            else:
                answer += " — still the founding core"
            answer += (". And yes, the city can grow: the land "
                       "opens in +60m rings, by the Keeper's hand "
                       "or by a citizen vote, up to ±360m — the "
                       "widest ground this world can hold.")
            return answer
        if terrain and (" pond " in padded or " ponds " in padded):
            if terrain.get("pond"):
                pond_where = terrain.get("pond_where",
                                         "the city outskirts")
                return (f"Yes — a pond fills the lowest ground near "
                        f"{pond_where}, where the land dips to "
                        f"{terrain['min_h']:.1f} m, below the water "
                        f"line.")
            return ("No — no ground anywhere in the city dips below "
                    "the water line, so there are no ponds.")
        if terrain and (" highest " in padded or " high ground " in padded
                        or " hill " in padded or " terrain " in padded
                        or " land like " in padded
                        or " lowest " in padded):
            low_where = terrain.get("low_where", "the city outskirts")
            high_where = terrain.get("high_where", "the city outskirts")
            return (f"The land runs from {terrain['min_h']:.1f} m "
                    f"near {low_where} up to {terrain['max_h']:.1f} m "
                    f"near {high_where}.")

        civics = snap.get("civics")
        if civics and (" vot" in padded or " proposal " in padded
                       or " decid" in padded or " civic" in padded):
            open_p = civics.get("open")
            if open_p:
                t = open_p["tally"]
                return (f"The city is voting on "
                        f"'{open_p['title']}', proposed by "
                        f"{open_p['proposer']}: "
                        f"{open_p['rationale']} So far: {t['for']} "
                        f"for, {t['against']} against, "
                        f"{t['abstain']} abstained — {t['cast']} of "
                        f"{t['total']} ballots cast.")
            history = civics.get("history") or []
            if history:
                last = history[-1]
                lt = last["tally"]
                return (f"The city last decided on "
                        f"'{last['title']}' (proposed by "
                        f"{last['proposer']}): it was "
                        f"{last['outcome'].upper()} — {lt['for']} "
                        f"for, {lt['against']} against, "
                        f"{lt['abstain']} abstained."
                        + (f" {last['executed']}"
                           if last.get("executed") else ""))
            return ("The city has no open proposal, and nothing "
                    "has been put to a vote yet.")

        research = snap.get("research")
        if research and (" learned " in padded or " learnt " in padded
                         or " research " in padded
                         or " discover" in padded
                         or " terminal " in padded):
            insights = [e for e in research["knowledge"]
                        if e.get("kind") == "insight"]
            if not insights:
                return ("The city has not learned anything at the "
                        "research terminal yet — no citizen has "
                        "brought back an insight.")
            bits = [f"{e.get('citizen', 'Someone')} asked "
                    f"\"{e.get('question', '')}\" and learned: "
                    f"{e.get('summary', '')}"
                    for e in insights[-3:]]
            return "What the city has learned: " + " ".join(bits)
        if research and (" make " in padded or " craft " in padded
                         or " produce " in padded
                         or " manufactur" in padded):
            bits = ["planks at the sawbench (1 wood each)",
                    "meals at the market stove (2 food each)"]
            for r in research["recipes"]:
                inputs = ", ".join(f"{v} {k}" for k, v
                                   in r["inputs"].items())
                out = next(iter(r["outputs"]), "?")
                bits.append(f"{out} at the {r['station']} "
                            f"({inputs})")
            for rec in research["innovations"]["innovations"]:
                bits.append(f"{rec['name']} — a {rec['kind']} the "
                            f"city adopted and now produces")
            return "The city can make: " + "; ".join(bits) + "."

        res = snap.get("resources")
        if res and (" wood " in padded or " food " in padded
                    or " stores " in padded or " inventory " in padded):
            inv = res["inventory"]
            return (f"The city stores hold {inv.get('wood', 0)} wood "
                    f"and {inv.get('food', 0)} food, plus "
                    f"{inv.get('planks', 0)} planks and "
                    f"{inv.get('meals', 0)} meals, gathered from "
                    f"{res['nodes']} resource nodes; {res['depleted']} "
                    f"of them are depleted and regrowing.")

        if " how many " in padded:
            if " active " in padded:
                return (f"{counts['active']} of the city's "
                        f"{counts['total']} citizens are ACTIVE "
                        f"right now.")
            if " ambient " in padded:
                return (f"{counts['ambient']} of the city's "
                        f"{counts['total']} citizens are AMBIENT "
                        f"right now.")
            if " dormant " in padded or " asleep " in padded:
                return (f"{counts['dormant']} of the city's "
                        f"{counts['total']} citizens are DORMANT "
                        f"right now.")
            return (f"There are {counts['total']} citizens in the "
                    f"city: {counts['active']} ACTIVE, "
                    f"{counts['ambient']} AMBIENT, "
                    f"{counts['dormant']} DORMANT.")

        if place and (" who " in padded or " at " in padded):
            here = [c for c in snap["citizens"] if c["anchor"] == place]
            if not here:
                return f"Nobody is at {place} right now."
            return (f"At {place}: "
                    + "; ".join(f"{c['name']} ({_activity(c)})"
                                for c in here) + ".")

        if citizen and (" where " in padded or " doing " in padded
                        or " up to " in padded):
            pos = citizen["position"]
            pos_text = (f" Position ({pos[0]:.0f}, {pos[1]:.0f})."
                        if pos else "")
            return f"{citizen['name']} is {_activity(citizen)}.{pos_text}"

        if citizen and (" who " in padded or " what does " in padded
                        or " about " in padded):
            traits = ", ".join(citizen["traits"]) or "ordinary"
            goals = "; ".join(citizen["goals"])
            return (f"{citizen['name']} is the city's "
                    f"{citizen['role']}. {citizen['persona']} "
                    f"{citizen['backstory']} Traits: {traits}."
                    + (f" Goals: {goals}." if goals else "")
                    + f" Right now: {_activity(citizen)}.")

        # Fallback: honest about the limit, plus any digest fact the
        # question did touch (a citizen, a place, a matching memory).
        facts = []
        if citizen is not None:
            facts.append(self._citizen_line(citizen))
        if place is not None:
            names = [c["name"] for c in snap["citizens"]
                     if c["anchor"] == place]
            facts.append(f"At {place} right now: "
                         f"{', '.join(names) if names else 'nobody'}.")
        mems = self.search_memories(question, limit=1)
        if mems:
            facts.append(f"Closest record — [{mems[0]['citizen']}] "
                         f"{mems[0]['text']}")
        honest = ("In demo mode I can only read the city digest — "
                  "try 'Where is Mara?', 'Who is Mara?', 'How many "
                  "citizens are here?' or 'Who is at Central "
                  "Plaza?'. The full interpreter needs the live brain.")
        return honest + (" " + " ".join(facts) if facts else "")
