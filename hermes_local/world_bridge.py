"""World bridge: the engine-agnostic interface between 3D worlds and NPCs.

One JSON protocol, three transports (stdlib only):
  * **JSON-lines TCP socket** — for game engines and tools.
  * **HTTP POST /api/message** — same protocol, for browser clients (CORS on).
  * **Server-Sent Events GET /api/events** — framework -> world pushes
    (dialogue, actions, tier changes) for browser clients.
Plus GET /api/roster, GET /api/status, GET/POST /api/layout,
GET/POST /api/souls (NPC soul save/load: cards + memories + situations),
POST /api/ask (the World Interpreter: ask anything about the world —
see hermes_local/interpreter.py), GET /api/world (resource nodes +
city inventory — see hermes_local/resources.py), static serving of
the web world demo, and the launch portal at GET /portal.

Protocol (world -> framework):
  {"type": "register_world", "actions": {<action>: {"required": [...]}, ...}}
  {"type": "perception", "npc": "Mara", "event": "saw the gate open",
   "nearby": ["Player", "red toolbox"]}
  {"type": "speech", "to": "Mara", "from": "Player", "text": "Hello."}
  {"type": "proximity", "player": [x, z], "npcs": {"Mara": [x, z], ...},
   "ghost": true}   (optional: player is an unseen listener — tiers still
   run, but nobody engages with or answers the player)
  {"type": "npc_chat", "a": "Mara", "b": "Joss", "opener": "Nice day.", "turns": 4}
  {"type": "interact", "node": "tree-01"}   (harvest a resource node;
   validated against the last reported player position)
  {"type": "order", "task": "gather_wood", "npc": "Mara"}   (god-mode
   work order — task is gather_wood | gather_food | craft_planks |
   craft_meal, see hermes_local/work.py; "npc" is optional: omitted,
   the best idle citizen is assigned. Answers with a task event, or a
   notice explaining why no assignment was possible)
  {"type": "place", "kind": "stall", "position": [x, z], "rotation": 0}
   (god-mode building placement, P3 — creates a construction site,
   see hermes_local/buildings.py; citizens raise it via "build"
   tasks: {"type": "order", "task": "build", "site": "site-1",
   "npc": "Mara"}, "site" optional — the oldest waiting site is used.
   Answers with a build event, or a notice explaining the refusal)
  {"type": "terraform", "op": "raise|lower|flatten|smooth",
   "position": [x, z], "radius": m, "amount": m, "mode": "god"|"crew"}
   (god-mode terraforming, P4 — see hermes_local/terrain.py. God
   mode (the default) applies the op instantly and answers with a
   terraform event; "crew" opens a terraform JOB instead — a
   "terraform" task whose citizen applies the delta over ~30
   attended sim-seconds. Invalid ops answer with a notice)
  {"type": "propose", "title": "...", "rationale": "...",
   "action": {"type": "build", "kind": "house", "anchor": "..."}}
   (civics, P5 — see hermes_local/civics.py: the player puts a
   proposal to the city. The citizens debate it and EVERY citizen
   votes; the player gets no ballot and cannot override the tally.
   Answers with a proposal event, or a notice explaining the
   refusal — one open proposal at a time)
  {"type": "close_vote"}   (civics, P5: end the voting window
   early; the tally stands as cast)
  {"type": "research_ask", "question": "...", "npc": "Mara"}
   (research, P6 — see hermes_local/research.py: the world planner
   searches the real web from the research terminal and synthesizes
   an insight into city knowledge + the named citizen's memory.
   Answers with a "searching" research event; the insight — or an
   honest "no_connection" — lands on a later tick)
  {"type": "sync_now"}   (persistence, P7 — see
   hermes_local/persistence.py: force a world snapshot + repo push
   on the next tick. Answers with a persistence status event)
  {"type": "expand"}   (city growth, P9 — see hermes_local/
   terrain.py: grow the world by one +60m ring, instantly, with
   its four new district anchors. Answers with an expansion
   event; at the engine cap (+/-360m) a notice answers instead
   and nothing changes. Citizens can also vote the same act:
   {"type": "propose", "action": {"type": "expand"}})
  {"type": "set_weather", "state": "rain"}   (sky, P8 — see
   hermes_local/weather.py: force the weather state; answers with
   the weather event itself)
  {"type": "set_time", "hour": 21.5}   (sky, P8: set the world
   clock; answers with a weather event carrying the new hour)
  {"type": "design", "kind": "asset", "hint": "a fountain"}
   (designs, P8 — see hermes_local/designs.py: commission one
   design — a building_style, an interior, or an asset. The
   validated result lands as a design event on a later tick, or
   the design is shelved. Asset proposals are also a civics
   action: {"type": "propose", "action": {"type": "asset",
   "asset_hint": "..."}})
  {"type": "world_state", "state": {...}}
  {"type": "save_layout", "layout": {<city layout document>}}
  {"type": "load_layout"}

Protocol (framework -> world):
  {"type": "dialogue", "npc": "Mara", "text": "..."}
  {"type": "action", "npc": "Mara", "action": "move_to", "args": {"target": "..."}}
  {"type": "tier", "npc": "Mara", "tier": "active"|"ambient"|"dormant", ...}
  {"type": "plan", "a": "Mara", "b": "Joss", "where": "Central Plaza"}
  {"type": "harvest", "node": "tree-01", "kind": "tree", "amount": 3, "by": "Player"}
  {"type": "task", "task": "task-1", "task_type": "gather_wood",
   "npc": "Mara", "target": "tree-01", "state": "assigned"|"started"|
   "completed"|"failed"}   (RTS labor lifecycle, P2)
  {"type": "build", "site": "site-1", "kind": "stall",
   "state": "placed"|"frame"|"complete", ...}   (construction
   lifecycle, P3; completion carries "building" and maybe "anchor")
  {"type": "terraform", "mode": "god"|"crew", "op": "raise", ...}
   (terraforming, P4: a god application, a crew job's creation, or
   a crew job's completion with "state": "complete")
  {"type": "proposal", "id": "prop-1", "proposer": "Mara",
   "title": "...", "proposal": {...}}   (civics, P5: a proposal was
   put to the city — by a citizen, or by the Keeper (the player))
  {"type": "debate", "proposal": "prop-1", "npc": "Mara",
   "text": "...", "stance": "for"|"against"}   (civics, P5)
  {"type": "ballot", "proposal": "prop-1", "name": "Mara",
   "choice": "FOR"|"AGAINST"|"ABSTAIN", "reason": "..."}   (civics,
   P5: one citizen's vote, cast live as the window fills)
  {"type": "civic_result", "proposal": "prop-1", "outcome":
   "adopted"|"rejected", "tally": {...}, "executed": "..."|null}
   (civics, P5: the decision, and what the city did about it)
  {"type": "research", "state": "searching"|"insight"|
   "no_connection", "question": "...", "by": "Mara", ...}
   (research, P6: a terminal session's progress and findings)
  {"type": "innovation", "state": "registered"|"under_study",
   "id": "bricks", "name": "Bricks", ...}   (innovation, P6: an
   adopted discovery became real content — or was filed under
   study when its spec could not be made valid)
  {"type": "sale", "item": "bricks", "buyer": "Mara",
   "seller": "Joss", "price": 6, "stock": 3, "trade": 42}
   (the market, P6: a citizen bought a stocked innovation good)
  {"type": "persistence", "action": "push"|"pull"|"save_local"|...,
   "state": "synced"|"offline"|"local-only"|"disabled", ...}
   (persistence, P7: quiet status notes — snapshot saves and repo
   sync — for the god panel; the same status rides /api/status)
  {"type": "expansion", "bound": 180, "ring": 1,
   "district": ["East Fields", ...], "by": "The Keeper"|"civic vote"}
   (city growth, P9: the world grew by one ring; the named
   districts joined the city and their anchors are live)
  {"type": "walkgrid", "version": 3}   (walk grid, P10: the
   analytic walkability grid changed — refetch /api/world, whose
   "walkgrid" field carries {bound, cell, w, h, version,
   data_b64}; travel intents may also carry a "route" of
   waypoints around blocked ground)
  {"type": "status", ...} / {"type": "ack", ...} / {"type": "error", ...}
  {"type": "layout", "layout": {...}}
"""
from __future__ import annotations

import collections
import json
import queue
import re
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .buildings import BUILDINGS, ConstructionWorld, validate_construction
from .civics import CivicsEngine
from .designs import DesignEngine, validate_designs
from .innovations import InnovationEngine, validate_innovations
from .interpreter import WorldInterpreter
from .npc import (DEFAULT_ACTION_SCHEMA, ActivityAnchor,
                  ring_district_names)
from .research import ResearchEngine, validate_knowledge, web_search
from .resources import ResourceWorld
from .terrain import MAX_BOUND as TERRAIN_MAX_BOUND
from .terrain import TerrainWorld, validate_terrain
from .walkgrid import WalkGrid
from .weather import WeatherEngine
from .work import RecipeBook, station_list

# --------------------------------------------------------------------------
# City layout documents: {version, objects: [{asset, position, rotation,
# scale, kind}]} — the god-mode designer's format, shared by web client,
# bridge, and any external tool that wants to generate a city.
# --------------------------------------------------------------------------
def validate_layout(doc) -> dict:
    if not isinstance(doc, dict):
        raise ValueError("layout must be a JSON object")
    objects = doc.get("objects")
    if not isinstance(objects, list):
        raise ValueError("layout.objects must be a list")
    cleaned = []
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict) or not obj.get("asset"):
            raise ValueError(f"objects[{i}] needs an 'asset' name")
        pos = obj.get("position", [0, 0, 0])
        if not (isinstance(pos, list) and len(pos) == 3
                and all(isinstance(v, (int, float)) for v in pos)):
            raise ValueError(f"objects[{i}].position must be [x, y, z]")
        rot = obj.get("rotation", 0)
        if isinstance(rot, (int, float)):
            rot = [0, float(rot), 0]
        scale = obj.get("scale", 1)
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * 3
        cleaned.append({
            "asset": str(obj["asset"]),
            "position": [float(v) for v in pos],
            "rotation": [float(v) for v in rot],
            "scale": [float(v) for v in scale],
            "kind": str(obj.get("kind", "prop")),
        })
    out = {"version": int(doc.get("version", 1)), "objects": cleaned}
    # Construction (P3) rides the layout document: a layout saved
    # with sites/buildings keeps them (validated + cleaned); a
    # layout without them stays exactly the P2-era shape.
    if "sites" in doc or "buildings" in doc:
        out.update(validate_construction(doc))
        counts = doc.get("anchor_counts")
        if isinstance(counts, dict):
            out["anchor_counts"] = {
                str(k): int(v) for k, v in counts.items()
                if isinstance(v, (int, float))}
    # Terraforming (P4) rides the layout document the same way: a
    # layout saved with a shaped heightmap keeps it (validated +
    # cleaned); a layout without one stays exactly the P3-era shape.
    if "terrain" in doc:
        out.update(validate_terrain(doc))
    # Research + innovation (P6) ride it too: the innovation
    # registry/market and the shared city knowledge. (The bridge
    # adopts these slices BEFORE this validation runs — dynamic
    # building kinds must exist in the catalog before construction
    # records naming them can validate. See _adopt_world_slices.)
    if "innovations" in doc:
        out["innovations"] = validate_innovations(doc["innovations"])
    if "knowledge" in doc:
        out["knowledge"] = validate_knowledge(doc["knowledge"])
    # Designs (P8) ride it the same way: brain-designed styles,
    # interiors and placed assets (validated + cleaned); a layout
    # without them stays exactly the P7-era shape.
    if "designs" in doc:
        out["designs"] = validate_designs(doc["designs"])
    return out


# --------------------------------------------------------------------------
# Demo brain: a rule-based stand-in LLM so the web world works out of the
# box with no model server. Implements the same chat() interface; NOT a
# real model — live mode swaps in llm.LLMClient with the identical shape.
# --------------------------------------------------------------------------
class DemoBrain:
    name = "demo-brain (rule-based, no LLM)"

    def chat(self, messages, tools=None):
        system = messages[0].get("content", "") if messages else ""
        if "Summarize" in system or "summarize" in system[:60].lower():
            last = next((m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"), "")
            return {"content": "Encounter (demo summary): " + last[:140],
                    "tool_calls": []}
        m = re.search(r"You are ([^,]+), a ([^.]+)\.", system)
        name, role = (m.group(1), m.group(2)) if m else ("Someone", "townsperson")
        last_user = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        text = re.sub(r"^[^:]{1,30}:\s*", "", last_user).strip()
        low = text.lower()
        acted = any(msg.get("role") == "tool" for msg in messages) or \
            "Action results:" in last_user or \
            any("Action results:" in (msg.get("content") or "") for msg in messages)
        if acted:
            return {"content": f"Done. What else can a {role} do for you?",
                    "tool_calls": []}

        def emit(action, args):
            if tools:
                return {"content": "", "tool_calls": [
                    {"id": "demo_1", "name": action, "arguments": args}]}
            return {"content": f"```tool\n{json.dumps({'name': action, 'arguments': args})}\n```",
                    "tool_calls": []}

        follow = re.search(r"\bfollow me\b", low)
        goto = re.search(r"(?:go to|walk to|head to|move to)\s+(.+)", low)
        grab = re.search(r"(?:pick up|grab|take|fetch)\s+(.+)", low)
        if follow:
            return emit("follow", {"entity": "Player"})
        if goto:
            return emit("move_to", {"target": goto.group(1).strip(" .!")})
        if grab:
            return emit("interact_with", {"object": grab.group(1).strip(" .!")})
        if "wave" in low:
            return emit("play_animation", {"name": "wave"})
        if "where" in low and "Things you remember:" in system:
            lines = [ln[2:] for ln in system.splitlines() if ln.startswith("- ")]
            words = {w for w in re.findall(r"[a-z]{4,}", low)}
            for line in lines:
                if words & set(re.findall(r"[a-z]{4,}", line.lower())):
                    return {"content": f"Aye, I remember: {line}", "tool_calls": []}
        if any(g in low for g in ("hello", "hi", "hey")):
            # Location gating (P11): the demo brain reads the
            # situation the manager put in the system prompt and
            # never hosts from somewhere it is not. A vendor AT a
            # market anchor may pitch the cart; the same vendor
            # met mid-road is just passing through, like you. The
            # checks read ONLY the situation segment (the line
            # after "Where you are right now:") — the grounding
            # instruction below it talks about roads and
            # pavements too and must not trip the gate.
            sit = ""
            if "Where you are right now:" in system:
                sit = system.split("Where you are right now:", 1)[1] \
                            .split("\n", 1)[0]
            if "middle of the road" in sit \
                    or "on the pavement by the road" in sit:
                return {"content": f"Hello there. I'm {name}, the {role} "
                                   f"— just passing through myself. "
                                   f"Mind the traffic.", "tool_calls": []}
            if role == "vendor" and "by the market stalls" in sit:
                return {"content": f"Hello there! You've come to the "
                                   f"right cart — I'm {name}. Today's "
                                   f"special is the best in the district.",
                        "tool_calls": []}
            if role == "shopkeeper" and "your workplace" in sit:
                return {"content": f"Hello there. I'm {name}, the "
                                   f"{role}. Everything in my shop is "
                                   f"fairly priced — looking for "
                                   f"anything in particular?",
                        "tool_calls": []}
            return {"content": f"Hello there. I'm {name}, the {role}. "
                               f"What brings you by?", "tool_calls": []}
        return {"content": f"({name} the {role} nods) Tell me more — or ask me "
                           f"to go somewhere, fetch something, or follow you.",
                "tool_calls": []}


# --------------------------------------------------------------------------
# Ambient conversation generator: LLM-written NPC<->NPC exchanges for
# social pairings (see ActivityPlanner). Live worlds only — mock/demo
# worlds keep the canned line set and never build one of these.
# --------------------------------------------------------------------------
def _weather_line(weather_fn) -> str:
    """The sky as one line of prompt context (P8): "It is raining
    this evening." — from the bridge's WeatherEngine via the
    injected callable. Any failure or absence means no line: the
    weather colors small talk, it never blocks it."""
    if weather_fn is None:
        return ""
    try:
        line = weather_fn()
    except Exception:
        return ""
    return f"\nSky: {line}" if line else ""


def make_conversation_generator(client, anchor_names=None, memory=None,
                                weather_fn=None):
    """Build the planner's conversation-generator hook over a live
    LLMClient. Deliberately uses a dedicated client with a 25s timeout
    and a single attempt (NOT the brain's 120s / 3-retry settings):
    ambient chatter must never hang or hammer the model server. Returns
    (lines, plan_anchor | None) — lines is [(speaker, line), ...] — or
    None on any failure (bad HTTP, unparseable output, wrong speakers);
    in live mode the planner treats None as silence for that pairing
    (there is no canned fallback once a generator is wired).

    Soul-fed prompts: when `memory` (the shared Memory store) is given,
    each participant's prompt block carries a few of that NPC's actual
    memories — a plan they share with this partner first, then other
    memories mentioning the partner, then the most recent ones — so
    the model writes for the character's soul, not just their card.

    Continuations: the planner hands back the transcript so far as
    `history` ([(speaker, line), ...]); the prompt then asks for the
    next beat of the SAME conversation instead of a fresh opener.

    Location context (P11): the hook's fifth argument, `situations`,
    carries each speaker's factual situation string (clock + where
    they stand, from the planner's world data). When present it is
    the prompt's ground truth for place, under the grounding rule:
    nobody claims a shop, stall or home they are not standing at.

    Conversation-born plans: on a first exchange (no history) the
    prompt invites the pair to *sometimes* end by agreeing to meet
    later at a named place from `anchor_names`. That agreement is
    signalled by one final machine-readable line, `PLAN: <place
    name>`, which is parsed out here — it is never spoken or shown as
    dialogue — and validated against the anchor set (an unknown place
    means no plan). The planner turns a valid plan into a rendezvous
    a few sim-minutes later (see ActivityPlanner)."""
    from .llm import LLMClient

    ambient = LLMClient(base_url=client.base_url, model=client.model,
                        api=client.api, timeout=25, retries=1,
                        temperature=client.temperature,
                        api_key=client.api_key)
    anchors = list(anchor_names or [])

    def _who(card) -> str:
        traits = ", ".join(card.traits) or "ordinary"
        return (f"{card.name} — a {card.role}; traits: {traits}. "
                f"{card.persona} Voice: {card.voice}")

    def _soul(card, partner_name: str) -> str:
        """Up to ~3 of this NPC's own memories, as prompt lines: their
        shared plan with this partner first (kind="plan"), then other
        memories that mention the partner, then the most recent ones.
        Compact and truncated — flavour for the writer, not a dossier."""
        if memory is None:
            return ""
        try:
            entries = memory.list_namespace(card.namespace)
        except Exception:
            return ""
        if not entries:
            return ""
        plans = [e for e in entries
                 if e.get("kind") == "plan"
                 and partner_name in str(e.get("text", ""))]
        mentions = [e for e in entries
                    if partner_name in str(e.get("text", ""))
                    and e not in plans]
        recent = list(reversed(entries))     # list is oldest-first
        picked: list[str] = []
        for pool in (plans, mentions, recent):
            for entry in pool:
                text = " ".join(str(entry.get("text", "")).split())
                if text and text not in picked:
                    picked.append(text)
                if len(picked) >= 3:
                    break
            if len(picked) >= 3:
                break
        if not picked:
            return ""
        return (f"\n{card.name} remembers:\n"
                + "\n".join(f"- {text[:120]}" for text in picked))

    def generate(card_a, card_b, anchor_name, history=None,
                 situations=None):
        history = list(history or [])
        if history:
            transcript = "\n".join(f"{s}: {t}" for s, t in history[-8:])
            ask = (f"They are mid-conversation at {anchor_name}. "
                   f"It has gone so far:\n{transcript}\n"
                   f"Write 2-4 MORE short lines continuing this exact "
                   f"conversation naturally — a new beat, no repeats, "
                   f"alternating speakers. Each line under 20 words, "
                   f"in character.")
            plan_prompt = ""
        else:
            ask = (f"Write 2-4 short lines of conversation between "
                   f"{card_a.name} and {card_b.name}, alternating "
                   f"speakers, as they chat at {anchor_name}. Each "
                   f"line under 20 words, in character.")
            plan_prompt = ""
            if anchors:
                plan_prompt = (
                    "\nSometimes — not always — the two may end the "
                    "chat by agreeing to meet later at one specific "
                    "place from this list: " + ", ".join(anchors) +
                    ". If they do agree, append ONE final line by "
                    "itself, formatted exactly as 'PLAN: <place name>' "
                    "using a name from the list verbatim. That line is "
                    "not dialogue. If they don't agree to meet, add "
                    "nothing after the dialogue.")
        # Location context (P11): when the planner hands over
        # per-speaker situations, they replace the bare anchor
        # name as the prompt's ground truth — each speaker talks
        # from where they factually stand, under the grounding
        # rule (no claiming a shop/stall/home that isn't theirs
        # at this spot).
        place = f"Place: {anchor_name}."
        if situations:
            from .npc import LOCATION_GROUNDING

            grounded = []
            for card in (card_a, card_b):
                sit = situations.get(card.name)
                if sit:
                    grounded.append(f"{card.name} — {sit}")
            if grounded:
                place = ("Where each of them is:\n"
                         + "\n".join(grounded)
                         + f"\n{LOCATION_GROUNDING}")
        messages = [
            {"role": "system", "content":
             "You write tiny, natural spoken exchanges for a living city "
             "simulation. Output ONLY dialogue lines, one per line, each "
             "formatted exactly as 'Name: line'. No narration, no stage "
             "directions, no quotation marks, no preamble."},
            {"role": "user", "content":
             "Setting: a lively city district where the neighbors know "
             "each other and cross paths all day.\n"
             f"{_who(card_a)}{_soul(card_a, card_b.name)}\n"
             f"{_who(card_b)}{_soul(card_b, card_a.name)}\n"
             f"{place}\n" + ask + plan_prompt
             + _weather_line(weather_fn)}]
        try:
            resp = ambient.chat(messages)
        except Exception:
            return None
        lines, plan = [], None
        for raw in (resp.get("content") or "").splitlines():
            pm = re.match(r"^PLAN:\s*(.+)$", raw.strip())
            if pm:
                candidate = pm.group(1).strip()
                plan = candidate if candidate in anchors else None
                continue
            m = re.match(r"^([A-Za-z][\w .'-]*):\s*(.+)$", raw.strip())
            if not m:
                continue
            speaker = m.group(1).strip()
            text = m.group(2).strip().strip('"')
            if speaker not in (card_a.name, card_b.name) or not text:
                return None
            lines.append((speaker, text))
        if not 2 <= len(lines) <= 4:
            return None
        if {s for s, _ in lines} != {card_a.name, card_b.name}:
            return None
        return lines, plan

    return generate


def make_civic_generators(client, memory=None):
    """Build the CivicsEngine's live generators (P5) over a live
    LLMClient — the drafter, the debater and the ballot caster.
    Like the conversation generator, they deliberately use a
    dedicated client with a 25s timeout and a single attempt: civic
    calls must never hang the world or hammer the model server.

    Every prompt is soul-fed (the engine assembles each citizen's
    soul text — card memories, past civic votes first — and hands
    it over), and every reply is parsed STRICTLY:

      * drafter -> (title, rationale) from a "TITLE: ..." first
        line plus the sentences after it; anything else -> None
        (the engine keeps its template draft).
      * debater -> exactly one line per debater, "Name: line",
        speakers exactly the cast; anything else -> None (live
        mode then simply has no debate — never canned lines).
      * ballot_caster -> (choice, reason) from "VOTE: FOR|AGAINST|
        ABSTAIN" + "REASON: ..."; garbage -> None (the engine
        retries once, then records ABSTAIN "no clear answer").
        Transport failures RAISE out of the client and the engine
        records ABSTAIN "couldn't be reached" — in live mode no
        rule-based ballot is ever substituted (Aaron's directive).
    """
    from .llm import LLMClient

    civic = LLMClient(base_url=client.base_url, model=client.model,
                      api=client.api, timeout=25, retries=1,
                      temperature=client.temperature,
                      api_key=client.api_key)

    def _who(card) -> str:
        traits = ", ".join(card.traits) or "ordinary"
        return (f"{card.name} — a {card.role}; traits: {traits}. "
                f"{card.persona} Voice: {card.voice}")

    def _soul_block(card, soul_text: str) -> str:
        return f"{_who(card)}" + (
            f"\n{card.name} remembers:\n{soul_text}" if soul_text
            else "")

    def drafter(card, soul_text, seed):
        from .civics import action_summary
        messages = [
            {"role": "system", "content":
             "You write for a living city simulation. A citizen is "
             "putting a proposal to a city vote. Write the proposal "
             "in their voice. Output exactly: a first line "
             "'TITLE: <short title>', then 1-2 sentences of "
             "rationale. Nothing else."},
            {"role": "user", "content":
             f"{_soul_block(card, soul_text)}\n"
             f"What the city faces: {seed.get('fact', '')}\n"
             f"The proposal: {action_summary(seed.get('action', {}))} "
             f"(action: {seed.get('action', {})}).\n"
             "Write the proposal as this citizen would put it."}]
        resp = civic.chat(messages)
        title, rationale_lines = None, []
        for raw in (resp.get("content") or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^TITLE:\s*(.+)$", line, re.IGNORECASE)
            if m and title is None:
                title = m.group(1).strip().strip('"')
            else:
                rationale_lines.append(line)
        rationale = " ".join(rationale_lines).strip()
        if not title or not rationale:
            return None
        return title, rationale

    def debater(cards, souls, proposal):
        cast = "\n".join(
            _soul_block(card, souls.get(card.name, ""))
            for card in cards)
        supporters = [c.name for c in cards[:-1]] if len(cards) > 2 \
            else [c.name for c in cards]
        skeptic = cards[-1].name if len(cards) > 2 else None
        stance = (f"{' and '.join(supporters)} argue FOR it"
                  + (f"; {skeptic} argues AGAINST it" if skeptic
                     else "") + ".")
        messages = [
            {"role": "system", "content":
             "You write a tiny public debate in a living city "
             "simulation. Output ONLY one line per speaker, each "
             "formatted exactly as 'Name: line'. No narration, no "
             "preamble."},
            {"role": "user", "content":
             f"The proposal before the city: \"{proposal['title']}\" "
             f"— {proposal['rationale']}\nThe speakers:\n{cast}\n"
             f"{stance} Each speaks ONE short line (under 22 words), "
             "in character, about this proposal."}]
        resp = civic.chat(messages)
        lines, seen = [], set()
        names = {c.name for c in cards}
        for raw in (resp.get("content") or "").splitlines():
            m = re.match(r"^([A-Za-z][\w .'-]*):\s*(.+)$", raw.strip())
            if not m:
                continue
            speaker = m.group(1).strip()
            text = m.group(2).strip().strip('"')
            if speaker not in names or speaker in seen or not text:
                return None
            seen.add(speaker)
            lines.append((speaker, text))
        if seen != names or not lines:
            return None
        order = {c.name: i for i, c in enumerate(cards)}
        lines.sort(key=lambda lt: order[lt[0]])
        return lines

    def ballot_caster(card, soul_text, proposal, debate_digest):
        from .civics import action_summary
        debate = (f"\nThe debate so far:\n{debate_digest}"
                  if debate_digest else
                  "\nThere was no public debate.")
        messages = [
            {"role": "system", "content":
             "You ARE one citizen of a simulated city, voting on a "
             "proposal. Decide as this citizen truly would — their "
             "role, traits, memories and past votes are who they "
             "are; there is no formula. Output exactly two lines: "
             "'VOTE: FOR' or 'VOTE: AGAINST' or 'VOTE: ABSTAIN', "
             "then 'REASON: <one short line, in their voice>'."},
            {"role": "user", "content":
             f"{_soul_block(card, soul_text)}\n"
             f"The proposal: \"{proposal['title']}\" — "
             f"{proposal['rationale']}\n"
             f"If adopted, the city will: "
             f"{action_summary(proposal['action'])}.{debate}\n"
             "Cast this citizen's ballot."}]
        resp = civic.chat(messages)      # raises on transport failure
        choice, reason = None, ""
        for raw in (resp.get("content") or "").splitlines():
            line = raw.strip()
            m = re.match(r"^VOTE:\s*(FOR|AGAINST|ABSTAIN)\b", line,
                         re.IGNORECASE)
            if m and choice is None:
                choice = m.group(1).upper()
                continue
            m = re.match(r"^REASON:\s*(.+)$", line, re.IGNORECASE)
            if m:
                reason = m.group(1).strip().strip('"')
        if choice is None:
            return None
        return choice, reason or "no reason given"

    return drafter, debater, ballot_caster


# --------------------------------------------------------------------------
# Research + innovation brains (P6): the world planner's two pens.
# Like the conversation/civic generators, both use a dedicated
# client with a ~45s timeout and a single attempt, and both parse
# STRICT JSON — anything else is a failure the engines handle
# honestly (a findings-only insight; a spec revision, then the
# under-study file). No output of either is ever trusted past the
# engines' own validation.
# --------------------------------------------------------------------------
def _extract_json(text: str):
    """The first complete JSON object in a model reply (fences and
    prose tolerated), or None."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    return None
    return None


def make_research_brain(client):
    """The ResearchEngine's insight synthesizer over a live
    LLMClient: (citizen, question, world_ctx, results) ->
    {"summary", "innovation_hint" | None}. Raises/propagates
    nothing special — the engine treats any failure as 'fall back
    to the findings themselves'."""
    from .llm import LLMClient

    planner_client = LLMClient(
        base_url=client.base_url, model=client.model, api=client.api,
        timeout=45, retries=1, temperature=client.temperature,
        api_key=client.api_key)

    def synthesize(citizen, question, ctx, results):
        findings = "\n".join(
            f"- {r['title']} ({r['url']}): {r['snippet']}"
            for r in results)
        shortages = ", ".join(ctx.get("shortages") or []) or "none"
        known = ", ".join(ctx.get("innovations") or []) or "nothing new yet"
        messages = [
            {"role": "system", "content":
             "You are the world planner of a living city simulation, "
             "answering a citizen's question from REAL web search "
             "results. Output ONLY a JSON object: {\"summary\": "
             "\"<2-3 sentences: what was learned, grounded in the "
             "results>\", \"innovation_hint\": \"<a short phrase "
             "naming ONE concrete thing the city could adopt and "
             "make — a material, food, good or tool the results "
             "support — or null>\"}. Never claim anything the "
             "results do not support."},
            {"role": "user", "content":
             f"{citizen} asks the research terminal: \"{question}\"\n"
             f"The city: stores {ctx.get('inventory', {})}; "
             f"shortages: {shortages}; finished houses: "
             f"{ctx.get('houses', 0)}; the city already makes: "
             f"{known}.\nSearch results:\n{findings}"}]
        resp = planner_client.chat(messages)
        data = _extract_json(resp.get("content") or "")
        if not isinstance(data, dict) \
                or not str(data.get("summary") or "").strip():
            return None
        hint = data.get("innovation_hint")
        return {"summary": str(data["summary"]).strip(),
                "innovation_hint": (str(hint).strip()
                                    if hint else None)}

    return synthesize


def make_spec_synthesizer(client):
    """The InnovationEngine's spec brain over a live LLMClient:
    (ctx) -> raw spec dict | None. The ctx carries the adopted
    hint, the deep-research findings, the world's capabilities and
    — on the one revision pass — the validator's errors verbatim.
    The engine validates whatever comes back; this function's only
    job is a well-formed attempt."""
    from .llm import LLMClient

    spec_client = LLMClient(
        base_url=client.base_url, model=client.model, api=client.api,
        timeout=45, retries=1, temperature=client.temperature,
        api_key=client.api_key)

    def synthesize(ctx):
        findings = "\n".join(
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in ctx.get("findings") or []) or "(web research unavailable)"
        errors = ""
        if ctx.get("errors"):
            errors = ("\nYour previous spec was REJECTED by the "
                      "world engine for these reasons — fix every "
                      "one:\n" + "\n".join(f"- {e}"
                                           for e in ctx["errors"]))
        messages = [
            {"role": "system", "content":
             "You are the world planner of a living city simulation. "
             "The city has VOTED to adopt a discovery; your job is "
             "to specify, as STRICT JSON only, how it becomes real "
             "in the world. Schema: {\"id\": slug, \"name\": str, "
             "\"kind\": \"material|food|good|tool\", \"source\": "
             "{\"node_label\": str, \"yields\": <raw material id>, "
             "\"area_hint\": str} | null, \"recipe\": {\"inputs\": "
             "{material: qty}, \"out_qty\": int, \"station_id\": "
             "str, \"work_s\": number} | null, \"station_building\": "
             "{\"name\": str, \"cost\": {material: qty}, "
             "\"archetype\": \"stall|lamp|fence|house\"} | null, "
             "\"variant_building\": {\"name\": str, \"costs\": "
             "{material: qty}, \"archetype\": str} | null, "
             "\"uses\": [\"building_material|recipe_input|food|"
             "sellable|tool\"], \"price\": int}. Rules: recipe "
             "inputs and building costs may only name materials the "
             "world already has, the new item, or its source's "
             "yields; a recipe's station_id must be an existing "
             "station or the station_building you specify; the item "
             "needs a source, a recipe, or both. Output ONLY the "
             "JSON object."},
            {"role": "user", "content":
             f"The adopted discovery: \"{ctx.get('hint', '')}\" "
             f"(proposed by {ctx.get('proposer', 'a citizen')}).\n"
             f"Deep research findings:\n{findings}\n"
             f"The world already has these materials: "
             f"{', '.join(ctx.get('inventory_keys') or [])}.\n"
             f"Existing stations: "
             f"{', '.join(ctx.get('station_ids') or [])}. "
             f"Existing buildings: "
             f"{', '.join(ctx.get('building_kinds') or [])}.\n"
             f"A worked example of a valid spec:\n"
             f"{json.dumps(ctx.get('example') or {})}"
             f"{errors}"}]
        resp = spec_client.chat(messages)
        data = _extract_json(resp.get("content") or "")
        return data if isinstance(data, dict) else None

    return synthesize


def make_design_brain(client):
    """The DesignEngine's design brain over a live LLMClient
    (P8): (ctx) -> raw design spec dict | None. Like the other
    world-planner pens it uses a dedicated ~45s client with a
    single attempt and parses STRICT JSON; the engine validates
    whatever comes back — a design that fails validation is fed
    back once with its errors (the ctx carries them), then
    shelved. This function's only job is a well-formed attempt in
    the requested kind: a building style, an interior, or an
    outdoor asset composition."""
    from .llm import LLMClient

    design_client = LLMClient(
        base_url=client.base_url, model=client.model, api=client.api,
        timeout=45, retries=1, temperature=client.temperature,
        api_key=client.api_key)

    def design(ctx):
        errors = ""
        if ctx.get("errors"):
            errors = ("\nYour previous design was REJECTED by the "
                      "world engine for these reasons — fix every "
                      "one:\n" + "\n".join(f"- {e}"
                                           for e in ctx["errors"]))
        messages = [
            {"role": "system", "content":
             "You are the designer of a living city simulation. "
             "Design ONE thing, as STRICT JSON only, in the "
             "requested kind. Schemas: building_style: {\"kind\": "
             "\"building_style\", \"name\": str, \"archetype\": "
             "<an existing building kind>, \"palette\": {\"wall\": "
             "\"#rrggbb\", \"roof\": \"#rrggbb\", \"trim\": "
             "\"#rrggbb\"}, \"roof_style\": \"flat|gable|hip\", "
             "\"floors\": 1-3}. interior: {\"kind\": \"interior\", "
             "\"name\": str, \"building_kind\": <an existing "
             "building kind>, \"rooms\": [{\"name\": str, \"props\": "
             "[{\"kind\": <a furniture prop>, \"dx\": number, "
             "\"dz\": number, \"rot\": degrees}]}]} — dx/dz are "
             "metres from the building's centre and must stay "
             "inside its footprint (a house is 10x10, a stall 6x5). "
             "asset: {\"kind\": \"asset\", \"name\": str, "
             "\"footprint\": [w, d], \"props\": [{\"kind\": <a "
             "prop>, \"dx\": number, \"dz\": number, \"rot\": "
             "degrees}]} — an outdoor composition (a fountain, "
             "benches, an arch...) whose props stay inside its "
             "footprint. Palette channels must each sit between "
             "#22 and #ee. Output ONLY the JSON object."},
            {"role": "user", "content":
             f"Design a {ctx.get('kind', 'asset')} for the city. "
             f"The commission: \"{ctx.get('hint', '')}\" "
             f"(from {ctx.get('proposer', 'the city')}).\n"
             f"Buildings of this world: "
             f"{', '.join(ctx.get('building_kinds') or [])}.\n"
             f"Interior props: "
             f"{', '.join(ctx.get('interior_props') or [])}.\n"
             f"Asset props: "
             f"{', '.join(ctx.get('asset_props') or [])}.\n"
             f"Roof styles: "
             f"{', '.join(ctx.get('roof_styles') or [])}.\n"
             f"Designs the world already holds: "
             f"{', '.join(ctx.get('existing') or []) or 'none'}.\n"
             f"A worked example of a valid design:\n"
             f"{json.dumps(ctx.get('example') or {})}"
             f"{errors}"}]
        resp = design_client.chat(messages)
        data = _extract_json(resp.get("content") or "")
        return data if isinstance(data, dict) else None

    return design


# --------------------------------------------------------------------------
# Bridge core (transport-independent)
# --------------------------------------------------------------------------
# Recent-events ring buffer: the World Interpreter reads the world's
# short-term past from here. Only the event kinds a client actually
# receives are recorded (speech arrives as "dialogue"), each as a
# compact dict with a wall-clock timestamp; per kind, only the listed
# fields are kept and text is truncated.
_EVENT_KEEP = {"dialogue": ("npc", "text"),
               "social": ("a", "b", "npc", "text"),
               "plan": ("a", "b", "where"),
               "tier": ("npc", "tier", "reason"),
               "engage": ("npc", "engaged", "reason"),
               "harvest": ("node", "kind", "resource", "amount", "by"),
               "task": ("task", "task_type", "npc", "target", "state"),
               "build": ("site", "kind", "name", "state"),
               "terraform": ("op", "mode", "state", "position", "job",
                             "by"),
               "expansion": ("bound", "ring", "district", "by"),
               "proposal": ("id", "proposer", "title", "state"),
               "debate": ("proposal", "npc", "text"),
               "ballot": ("proposal", "name", "choice", "reason"),
               "civic_result": ("proposal", "title", "outcome"),
               "research": ("state", "question", "by", "summary",
                            "innovation_hint"),
               "innovation": ("id", "name", "state", "hint"),
               "weather": ("state", "hour", "daypart"),
               "design": ("kind", "id", "name", "state", "origin",
                          "hint", "reason"),
               "sale": ("item", "name", "buyer", "seller", "price")}
_EVENT_BUFFER_MAX = 120


class WorldBridge:
    def __init__(self, npcs: dict, manager=None, director=None,
                 layout_path: str | None = None, resources=None,
                 construction=None, terrain=None):
        self.npcs = npcs
        self.manager = manager
        self.director = director
        # Resource world (P1): nodes + city inventory. Every bridge
        # gets one — the default starter set unless a world is handed
        # in — so /api/world and the interact protocol always work.
        self.resources: ResourceWorld = resources \
            if resources is not None else ResourceWorld()
        # Construction world (P3): sites + finished buildings. Every
        # bridge gets one, so /api/world and the place protocol
        # always work; the planner builds against it.
        self.construction: ConstructionWorld = construction \
            if construction is not None else ConstructionWorld()
        # Terrain world (P4): the heightmap + crew jobs. Every
        # bridge gets one, so /api/world and the terraform protocol
        # always work; the planner crews against it.
        self.terrain: TerrainWorld = terrain \
            if terrain is not None else TerrainWorld()
        # City growth (P9): building placement validates against
        # the LIVE bound — the terrain's, which grows by ring.
        self.construction.bound_fn = lambda: self.terrain.bound
        # RTS labor (P2): the planner works against this same
        # ResourceWorld — one inventory, one set of nodes, whether
        # the hands are the player's (interact) or a citizen's (task).
        if manager is not None and getattr(manager, "planner", None) \
                is not None:
            manager.planner.resources = self.resources
            manager.planner.construction = self.construction
            manager.planner.terrain = self.terrain
        # Research + innovation (P6): the RecipeBook (seed recipes +
        # whatever adopted innovations register), the shared city
        # knowledge, and the two engines. They exist before any
        # layout loads: restoring a world must re-register its
        # innovations before construction records naming dynamic
        # buildings can validate.
        self.recipes: RecipeBook = RecipeBook()
        self.city_knowledge: list[dict] = []
        self.research: ResearchEngine = ResearchEngine(self)
        self.innovations: InnovationEngine = InnovationEngine(self)
        if manager is not None and getattr(manager, "planner", None) \
                is not None:
            manager.planner.recipes = self.recipes
            manager.planner.research = self.research
            manager.planner.innovations = self.innovations
        # Living-world quality (P8): the sky (world clock + weather)
        # and the designers (brain-designed building styles,
        # interiors and placed assets). The design engine seeds its
        # default house/stall interiors and plaza asset at
        # construction; its placed-asset layout markers sync once
        # the layout is loaded (below, next to the other restores).
        self.weather: WeatherEngine = WeatherEngine()
        self.designs: DesignEngine = DesignEngine(self)
        if manager is not None and getattr(manager, "planner", None) \
                is not None:
            manager.planner.weather = self.weather
            manager.planner.designs = self.designs
        self.layout_path = Path(layout_path) if layout_path else None
        self.layout: dict | None = None
        if self.layout_path and self.layout_path.is_file():
            raw_layout = None
            try:
                raw_layout = json.loads(self.layout_path.read_text())
            except Exception:
                raw_layout = None
            if isinstance(raw_layout, dict):
                self._adopt_world_slices(raw_layout)
            try:
                self.layout = validate_layout(raw_layout) \
                    if raw_layout is not None else None
            except Exception:
                self.layout = None
        if self.layout and ("sites" in self.layout
                            or "buildings" in self.layout):
            # A layout document carrying construction state seeds
            # the construction world with it (P3).
            try:
                self.construction.restore(self.layout)
                self._sync_building_anchors()
            except ValueError:
                pass
        if self.layout and "terrain" in self.layout:
            # A layout document carrying a shaped heightmap seeds
            # the terrain world with it (P4) — bound included (P9):
            # a grown world's districts come back with its land.
            try:
                self.terrain.restore(self.layout)
                self._sync_district_anchors()
            except ValueError:
                pass
        # Placed design assets live in the layout as prop markers
        # (P8): every registered asset gets its marker back — a
        # layout adopted with its designs slice carries them
        # already; a bare layout regains them here.
        self.designs.sync_markers()
        # Walk grid (P10, hermes_local/walkgrid.py): the analytic
        # walkability grid over the terrain + construction +
        # layout, built AFTER every restore above so it derives
        # from the adopted world. Derived state only — rebuilt
        # whenever its sources change (see _refresh_walkgrid),
        # never persisted. The planner routes against the same
        # grid the client clamps against.
        self.walkgrid: WalkGrid = WalkGrid(
            self.terrain, self.construction,
            layout_objects_fn=lambda: (self.layout or {})
            .get("objects", []))
        if manager is not None and getattr(manager, "planner", None) \
                is not None:
            manager.planner.walkgrid = self.walkgrid
        self._wg_sig = self._walkgrid_signature()
        self.world_state: dict = {}
        self.brain_info: dict = {}   # set by build_world; surfaced in /api/status
        self._outbox: list[dict] = []
        self._lock = threading.Lock()
        self._subscribers: set[queue.Queue] = set()
        self._events: collections.deque = collections.deque(
            maxlen=_EVENT_BUFFER_MAX)
        self._events_lock = threading.Lock()
        # Civics (P5): the proposal/debate/voting engine. Every
        # bridge gets one, wired to the planner (which holds the
        # resource/construction/terrain worlds); it ticks from the
        # tail of the planner's tick. Mock by default — build_world
        # wires the live AI generators for live worlds.
        planner = manager.planner if manager is not None else None
        self.civics: CivicsEngine | None = None
        if planner is not None:
            self.civics = CivicsEngine(
                planner,
                layout_objects_fn=lambda: (self.layout or {})
                .get("objects", []),
                knowledge_fn=lambda: self.city_knowledge)
            planner.civics = self.civics
        # World-data persistence (P7, hermes_local/persistence.py):
        # the snapshot/repo-sync service. build_world constructs
        # and starts it (it needs the config + the startup pull);
        # a bare bridge simply has none, and /api/status reports
        # the persistence state as "disabled".
        self.persistence = None
        # The World Interpreter (hermes_local/interpreter.py): rule-
        # based over the digest by default; build_world swaps in a
        # live-brain one for live worlds.
        self.interpreter = WorldInterpreter(self)
        for npc in npcs.values():
            npc.ctx.world = self

    # -- world sink: NPC body-action tools call this ---------------------------
    def send_action(self, npc_name: str, action: str, args: dict) -> None:
        if self.manager is not None:
            # Brain-directed movement stands the itinerary planner down so
            # it doesn't immediately walk the NPC off on its next activity.
            if action == "follow":
                self.manager.note_directed(npc_name, None)
            elif action == "move_to":
                self.manager.note_directed(npc_name, 45.0)
            elif action == "interact_with":
                self.manager.note_directed(npc_name, 10.0)
        if action == "say":
            msg = {"type": "dialogue", "npc": npc_name,
                   "text": str(args.get("text", ""))}
        else:
            msg = {"type": "action", "npc": npc_name, "action": action,
                   "args": args}
        with self._lock:
            self._outbox.append(msg)

    def _drain_outbox(self) -> list[dict]:
        with self._lock:
            out, self._outbox = self._outbox, []
        return out

    # -- SSE fan-out -------------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        self._subscribers.discard(q)

    def broadcast(self, msg: dict) -> None:
        for q in list(self._subscribers):
            q.put(msg)

    # -- recent events (the interpreter's short-term memory of the world) --------
    def _record_event(self, resp: dict) -> None:
        keep = _EVENT_KEEP.get(resp.get("type"))
        if keep is None:
            return
        rec = {"ts": time.time(), "type": resp["type"]}
        for key in keep:
            if key in resp:
                rec[key] = resp[key]
        if "text" in rec:
            rec["text"] = str(rec["text"])[:160]
        with self._events_lock:
            self._events.append(rec)

    def recent_events(self) -> list[dict]:
        """The last ~120 world events (oldest first), compact dicts
        with wall-clock `ts`. Fed from the same responses clients
        receive, so the interpreter's past is exactly what the world
        showed."""
        with self._events_lock:
            return list(self._events)

    # -- world interpreter ---------------------------------------------------------
    def ask(self, question: str) -> str:
        """Ask the World Interpreter anything about this world.
        Raises ValueError on an empty question."""
        return self.interpreter.ask(question)

    # -- research + innovation slices of the world save (P6) ---------------------
    def _adopt_world_slices(self, doc: dict) -> None:
        """Adopt a world document's P6 slices: the innovation
        registry/market (re-registering every saved innovation
        into the live worlds) and the shared city knowledge. Each
        slice is independently validated and all-or-nothing; a bad
        slice is skipped, never half-applied. This runs BEFORE
        layout validation so dynamic building kinds exist in the
        catalog when construction records are checked."""
        if "innovations" in doc:
            try:
                self.innovations.restore(doc["innovations"])
            except ValueError:
                pass
        if "knowledge" in doc:
            try:
                self.city_knowledge = validate_knowledge(
                    doc["knowledge"])
            except ValueError:
                pass
        if "designs" in doc:
            try:
                self.designs.restore(doc["designs"])
            except ValueError:
                pass

    # -- construction (P3: sites + buildings) -----------------------------------------
    def _sync_building_anchors(self) -> None:
        """Re-register planner anchors for restored finished
        buildings (P3): a house/stall completed before a layout
        reload keeps its home/market anchor in the planner's set.
        Additive only — anchors whose names are already present are
        left alone, and city anchors are never touched."""
        planner = self.manager.planner if self.manager else None
        if planner is None:
            return
        have = {a.name for a in planner.anchors}
        for b in self.construction.buildings:
            name = b.get("anchor")
            spec = BUILDINGS.get(b.get("kind") or "", {}).get("anchor")
            if not name or spec is None or name in have:
                continue
            planner.anchors.append(ActivityAnchor(
                name, spec[0], float(b["position"][0]),
                float(b["position"][1])))
            have.add(name)

    def _sync_district_anchors(self) -> None:
        """Rebuild the planner's district anchors from the
        terrain's live bound (P9): the expansion count derives
        from the bound, and the anchors are a deterministic
        function of the count (npc.district_anchor_specs) — they
        are never persisted per anchor. Called after every
        terrain restore (layout seed, save_layout, snapshot
        apply); expansions themselves go through the planner,
        which keeps its own set in step."""
        planner = self.manager.planner if self.manager else None
        if planner is not None:
            planner.set_expansions(self.terrain.expansions)

    # -- walk grid (P10: analytic walkability) -----------------------------------
    def _walkgrid_signature(self) -> tuple:
        """A cheap fingerprint of the walk grid's sources: the
        terrain bound, the construction counts, and the layout
        document's identity + object count. Height changes ride
        in through terraform/build events (forced refreshes in
        handle_message), so the signature stays O(1)."""
        return (self.terrain.bound, len(self.construction.buildings),
                len(self.construction.sites), id(self.layout),
                len((self.layout or {}).get("objects", [])))

    def _refresh_walkgrid(self, force: bool = False):
        """Re-derive the walk grid when its sources changed (or
        unconditionally when `force`). Returns the
        {"type": "walkgrid", "version": N} event when the grid
        actually changed, else None — clients refetch /api/world
        on the event (and on any version change they notice)."""
        sig = self._walkgrid_signature()
        if not force and sig == self._wg_sig:
            return None
        self._wg_sig = sig
        if self.walkgrid.rebuild():
            return {"type": "walkgrid",
                    "version": self.walkgrid.version}
        return None

    # -- resources (P1: nodes + city inventory) --------------------------------------
    def world_payload(self) -> dict:
        """GET /api/world: the resource snapshot (regrowth first
        advanced to now — the read itself is a world-loop tick),
        plus the crafting stations and the underway task list (P2),
        plus construction sites and finished buildings (P3), so one
        read renders the whole working world."""
        self.resources.tick(time.monotonic())
        payload = self.resources.snapshot()
        payload["stations"] = self.recipes.public_stations(
            self.construction)
        planner = self.manager.planner if self.manager else None
        payload["tasks"] = planner.active_tasks() if planner else []
        payload["sites"] = self.construction.public_sites()
        payload["buildings"] = self.construction.public_buildings()
        payload["terrain"] = self.terrain.public()
        payload["civics"] = self.civics.public_state() \
            if self.civics is not None else {"open": None,
                                             "history": []}
        # Research + innovation (P6): the build catalog (dynamic
        # kinds included, so the client can offer/place them), the
        # terminal + city knowledge, and the innovation registry
        # with its market.
        payload["catalog"] = [
            {"kind": kind, "name": spec["name"],
             "costs": dict(spec["costs"]),
             "archetype": spec.get("archetype")}
            for kind, spec in BUILDINGS.items()]
        payload["research"] = self.research.public()
        payload["innovations"] = self.innovations.public_state()
        # Living-world quality (P8): the design registry (styles,
        # interiors, placed assets — the client renders from this)
        # and the sky (clock + weather).
        payload["designs"] = self.designs.public_state()
        payload["weather"] = self.weather.public()
        # Walk grid (P10): refreshed from its live sources, then
        # the whole grid rides along (base64 bytearray — the
        # client decodes it into a Uint8Array and clamps citizen
        # movement against it; gzip keeps it small on the wire).
        self._refresh_walkgrid()
        payload["walkgrid"] = self.walkgrid.public()
        return payload

    # -- protocol ------------------------------------------------------------------
    def handle_message(self, msg: dict) -> list[dict]:
        try:
            responses = self._handle(msg)
        except Exception as err:
            responses = [{"type": "error", "error": f"{type(err).__name__}: {err}"}]
        # Walk grid (P10): anything that reshapes the walkable
        # world — a placement, a terraform op or completion, a
        # finished building, an expansion, a layout swap — forces
        # a re-derivation; everything else rides the cheap
        # signature check. A changed grid appends its event so
        # clients know to refetch.
        force = msg.get("type") in ("place", "terraform", "expand",
                                    "save_layout") or any(
            r.get("type") in ("terraform", "expansion")
            or (r.get("type") == "build"
                and r.get("state") == "complete")
            for r in responses)
        ev = self._refresh_walkgrid(force=force)
        if ev is not None:
            responses.append(ev)
        for resp in responses:
            self._record_event(resp)
            self.broadcast(resp)
        return responses

    def _handle(self, msg: dict) -> list[dict]:
        kind = msg.get("type")
        if kind == "register_world":
            schema = msg.get("actions") or DEFAULT_ACTION_SCHEMA
            for npc in self.npcs.values():
                npc.ctx.action_schema = schema
            return [{"type": "ack", "ok": True,
                     "npcs": sorted(self.npcs)}]
        if kind == "perception":
            npc = self.npcs.get(msg.get("npc", ""))
            if npc is None:
                return [{"type": "error", "error": f"unknown npc {msg.get('npc')}"}]
            event = str(msg.get("event", ""))
            nearby = msg.get("nearby") or []
            if nearby:
                event += f" (nearby: {', '.join(map(str, nearby))})"
            npc.witness(event)
            return [{"type": "ack", "ok": True, "npc": npc.card.name}]
        if kind == "speech":
            npc = self.npcs.get(msg.get("to", ""))
            if npc is None:
                return [{"type": "error", "error": f"unknown npc {msg.get('to')}"}]
            if self.manager is not None and (self.manager.ghost
                                             or msg.get("ghost")):
                # Ghost mode: the citizens cannot hear the player. The
                # speech is dropped (no lock, no reply) and the client
                # gets a small system notice instead of dialogue.
                return [{"type": "notice", "npc": npc.card.name,
                         "text": "(You are unseen — the citizens cannot "
                                 "hear you in ghost mode.)"}]
            if self.manager is not None:
                if self.manager.tiers.get(npc.card.name) != "active" and \
                        not self.manager.is_engaged(npc.card.name):
                    return [{"type": "notice", "npc": npc.card.name,
                             "text": f"({npc.card.name} is too far away to hear you.)"}]
                # Conversation lock: the NPC stops, faces the player, and
                # stays pinned ACTIVE until the chat goes quiet or the
                # player walks off (see NPCManager).
                self.manager.note_speech(npc.card.name)
            reply = npc.hear(str(msg.get("text", "")),
                             speaker=str(msg.get("from", "Player")),
                             situation=(self.manager.situation_for(
                                 npc.card.name)
                                        if self.manager is not None
                                        else None))
            out = self._drain_outbox()
            if not any(m.get("type") == "dialogue" and m.get("npc") == npc.card.name
                       for m in out):
                out.append({"type": "dialogue", "npc": npc.card.name,
                            "text": reply})
            return out
        if kind == "proximity":
            if self.manager is None:
                return [{"type": "error", "error": "no NPC manager configured"}]
            positions = {k: (v[0], v[1]) for k, v in (msg.get("npcs") or {}).items()}
            # Optional "ghost": true — the player is an unseen listener;
            # tiers still run from their position, engagement never does.
            events = self.manager.update(msg.get("player", [0, 0]), positions,
                                         ghost=bool(msg.get("ghost", False)))
            return events + [{"type": "status", **self.manager.stats()}]
        if kind == "npc_chat":
            if self.director is None:
                return [{"type": "error", "error": "no director configured"}]
            transcript = self.director.converse(
                msg["a"], msg["b"], str(msg.get("opener", "Hello.")),
                turns=int(msg.get("turns", 4)))
            return [{"type": "dialogue", "npc": t["npc"], "text": t["text"]}
                    for t in transcript]
        if kind == "interact":
            # Player harvests a resource node. Range is validated
            # against the last reported player position (the manager
            # keeps it from proximity heartbeats); the harvest rules
            # themselves live in ResourceWorld.try_harvest — the same
            # function P2's NPC labor will call.
            if self.manager is None:
                return [{"type": "error",
                         "error": "no NPC manager configured"}]
            event, notice = self.resources.try_harvest(
                str(msg.get("node", "")), self.manager.player_pos,
                time.monotonic(), by=str(msg.get("from", "Player")))
            if event is not None:
                return [event]
            return [{"type": "notice", "text": notice}]
        if kind == "order":
            # God-mode work order (P2): assign a citizen to gather or
            # craft. The planner owns assignment + execution; the
            # answer is the "assigned" task event, or a notice saying
            # why not (unknown task/citizen, queue full, nothing ripe).
            if self.manager is None or self.manager.planner is None:
                return [{"type": "error",
                         "error": "no NPC manager configured"}]
            planner = self.manager.planner
            engaged = {n for n in self.npcs
                       if self.manager.is_engaged(n)}
            if str(msg.get("task", "")) == "build":
                # Build order (P3): assign a citizen to a construction
                # site — the named site, or the oldest site still
                # waiting for a builder.
                site_id = msg.get("site") or None
                if site_id is None:
                    staffed = {t["target"] for t in planner.tasks
                               if t["state"] == "active"
                               and t["target_kind"] == "site"}
                    site_id = next(
                        (s.id for s in self.construction.sites.values()
                         if not s.complete and s.id not in staffed), None)
                    if site_id is None:
                        return [{"type": "notice",
                                 "text": "No construction site is "
                                         "waiting for a builder."}]
                task, reason = planner.assign_build(
                    str(site_id), time.monotonic(),
                    self.manager.positions,
                    npc_name=msg.get("npc") or None, engaged=engaged)
                if task is None:
                    return [{"type": "notice", "text": reason}]
                return [planner.task_event(task, "assigned")]
            task, reason = planner.assign_task(
                str(msg.get("task", "")), time.monotonic(),
                self.manager.positions,
                npc_name=msg.get("npc") or None, engaged=engaged)
            if task is None:
                return [{"type": "notice", "text": reason}]
            return [planner.task_event(task, "assigned")]
        if kind == "place":
            # God-mode building placement (P3): create a construction
            # site. Spatial validation (bounds, road bands, footprint
            # overlap — against the bridge's saved layout objects
            # when it has one, plus all sites/buildings) lives in
            # ConstructionWorld.validate_placement.
            pos = msg.get("position")
            if not (isinstance(pos, list) and len(pos) == 2
                    and all(isinstance(v, (int, float)) for v in pos)):
                return [{"type": "notice",
                         "text": "Placement needs a position [x, z]."}]
            rotation = msg.get("rotation", 0)
            if not isinstance(rotation, (int, float)):
                rotation = 0
            objects = (self.layout or {}).get("objects", [])
            site, reason = self.construction.place(
                str(msg.get("kind", "")), float(pos[0]), float(pos[1]),
                rotation=float(rotation), layout_objects=objects)
            if site is None:
                return [{"type": "notice", "text": reason}]
            return [self.construction.placed_event(site)]
        if kind == "terraform":
            # God-mode terraforming (P4): reshape the heightmap.
            # Default mode is instant — the op applies now and a
            # terraform event answers. With "mode": "crew" the same
            # fields open a crew JOB instead: a "terraform" task in
            # the planner queue whose citizen applies the delta
            # gradually (~30 attended sim-seconds). Bounds, radius
            # (2-20m) and amount (0.2-2m) are validated by the
            # TerrainWorld; anything off is a notice and no change.
            # Flatten's target is the height at the clicked point.
            pos = msg.get("position")
            if not (isinstance(pos, list) and len(pos) == 2
                    and all(isinstance(v, (int, float)) for v in pos)):
                return [{"type": "notice",
                         "text": "Terraforming needs a position [x, z]."}]

            def _num(key, default):
                v = msg.get(key, default)
                return float(v) if isinstance(v, (int, float)) else None

            radius = _num("radius", 8.0)
            amount = _num("amount", 1.0)
            if radius is None or amount is None:
                return [{"type": "notice",
                         "text": "Terraform radius and amount must "
                                 "be numbers."}]
            op = str(msg.get("op", ""))
            x, z = float(pos[0]), float(pos[1])
            reason = self.terrain.validate_op(op, x, z, radius, amount)
            if reason is not None:
                return [{"type": "notice", "text": reason}]
            target = self.terrain.sample(x, z) if op == "flatten" \
                else None
            if str(msg.get("mode", "god")) == "crew":
                job = self.terrain.create_job(op, x, z, radius, amount,
                                              target)
                out = [self.terrain.job_event(job, "created")]
                planner = self.manager.planner if self.manager else None
                if planner is not None:
                    # Staff it right away when a citizen is free
                    # (the foreman also staffs open jobs on its
                    # interval, so a queued job is never stranded).
                    engaged = {n for n in self.npcs
                               if self.manager.is_engaged(n)}
                    task, _reason = planner.assign_terraform(
                        job["id"], time.monotonic(),
                        self.manager.positions,
                        npc_name=msg.get("npc") or None,
                        engaged=engaged)
                    if task is not None:
                        out.append(planner.task_event(task, "assigned"))
                return out
            self.terrain.apply_op(op, x, z, radius, amount, target)
            return [self.terrain.god_event(op, x, z, radius, amount,
                                           target)]
        if kind == "expand":
            # City growth (P9): the Keeper grows the world by one
            # +60m ring, instantly — the same act the citizens can
            # vote through (civics "expand"). The expansion event
            # answers, carrying the new bound and the district
            # names that joined the city; at the engine cap a
            # notice answers and nothing changes.
            planner = self.manager.planner if self.manager else None
            if planner is not None:
                info = planner.apply_expansion()
            else:
                info = self.terrain.expand()
            if info is None:
                return [{"type": "notice",
                         "text": f"The city already spans "
                                 f"±{int(TERRAIN_MAX_BOUND)}m — the "
                                 f"widest ground this world can "
                                 f"hold. It can grow no further."}]
            return [TerrainWorld.expansion_event(
                info, ring_district_names(info["ring"]),
                by="The Keeper")]
        if kind == "propose":
            # Civics (P5): the player puts a proposal to the city.
            # The citizens debate it and EVERY citizen votes (AI
            # from their souls in live mode; rule ballots in the
            # demo path). The player gets NO ballot and cannot
            # override the tally — the answer is the proposal event,
            # or a notice explaining the refusal.
            if self.civics is None:
                return [{"type": "error",
                         "error": "no NPC manager configured"}]
            action = msg.get("action")
            if not isinstance(action, dict):
                return [{"type": "notice",
                         "text": "A proposal needs an action — "
                                 "{\"type\": \"build\"|\"terraform\"|"
                                 "\"plant_trees\"|\"gather_drive\"|"
                                 "\"expand\", ...}."}]
            events, reason = self.civics.open_proposal(
                "The Keeper", str(msg.get("title", "") or ""),
                str(msg.get("rationale", "") or ""), action,
                time.monotonic())
            if events is None:
                return [{"type": "notice", "text": reason}]
            return events
        if kind == "close_vote":
            # Civics (P5): end the voting window early. The tally
            # stands as cast; uncast citizens are recorded as
            # abstaining when the window closes on the next tick.
            if self.civics is None:
                return [{"type": "error",
                         "error": "no NPC manager configured"}]
            _ok, text = self.civics.request_close_vote()
            return [{"type": "notice", "text": text}]
        if kind == "research_ask":
            # Research (P6): put a question to the research
            # terminal. The world planner searches the real web and
            # synthesizes an insight — stored in city knowledge and,
            # for a named citizen, in their memory. The answer is a
            # "searching" event now and the insight event when it
            # lands on a later tick; if the terminal cannot reach
            # the web, a "no_connection" research event answers
            # instead and nothing is stored. See research.py.
            return self.research.ask(
                str(msg.get("question", "") or ""),
                by=str(msg.get("from", "The Keeper")),
                npc_name=msg.get("npc") or None,
                now=time.monotonic())
        if kind == "set_weather":
            # Sky (P8): force the weather state. The answer is the
            # weather event itself, so every client repaints its
            # sky from the same message.
            ev = self.weather.set_weather(
                str(msg.get("state", "") or ""))
            if ev is None:
                return [{"type": "error",
                         "error": "unknown weather state: "
                                  f"{msg.get('state')!r}"}]
            return [ev]
        if kind == "set_time":
            # Sky (P8): set the world clock (hour 0-24). The
            # answer is a weather event carrying the new hour.
            ev = self.weather.set_time(msg.get("hour"))
            if ev is None:
                return [{"type": "error",
                         "error": "set_time needs a numeric hour"}]
            return [ev]
        if kind == "design":
            # Designs (P8): commission one design of a kind —
            # {"type": "design", "kind": "building_style"|"interior"
            # |"asset", "hint": "..."}. The design brain (live
            # worlds) or the seeded templates (assets, mock worlds)
            # draw it; the validated result lands as a design event
            # on a later tick, or the design is shelved, honestly.
            ok, text = self.designs.request_design(
                str(msg.get("kind", "") or ""),
                str(msg.get("hint", "") or ""),
                str(msg.get("from", "The Keeper")),
                time.monotonic())
            return [{"type": "notice" if ok else "error",
                     "text" if ok else "error": text}]
        if kind == "sync_now":
            # Persistence (P7): force a snapshot + repo push on the
            # next world tick. The answer is the current status;
            # the push itself lands as a persistence event when it
            # completes.
            if self.persistence is None:
                return [{"type": "notice",
                         "text": "World persistence is not wired "
                                 "in this world — snapshots and repo "
                                 "sync are off."}]
            return [self.persistence.request_sync()]
        if kind == "world_state":
            self.world_state = msg.get("state") or {}
            return [{"type": "ack", "ok": True}]
        if kind == "save_layout":
            incoming = msg.get("layout")
            if isinstance(incoming, dict):
                # P6 slices adopt first (independently validated):
                # a saved world's innovations must be registered
                # before its construction records can validate.
                self._adopt_world_slices(incoming)
            try:
                self.layout = validate_layout(incoming)
            except ValueError as err:
                return [{"type": "error", "error": f"invalid layout: {err}"}]
            if "sites" in self.layout or "buildings" in self.layout:
                # The document carries construction state: adopt it,
                # so loading a saved city restores its sites and
                # finished buildings, progress included (P3).
                self.construction.restore(self.layout)
                self._sync_building_anchors()
            if "terrain" in self.layout:
                # The document carries a shaped heightmap: adopt it
                # the same way (P4) — crew jobs included, and the
                # bound (P9): district anchors follow the land.
                self.terrain.restore(self.layout)
                self._sync_district_anchors()
            # Note: placed-asset markers are NOT force-synced into
            # a client-saved document (P8) — the layout document is
            # the client's own and round-trips exactly; markers are
            # synced when the world itself owns the document (at
            # construction from its layout file, and on snapshot
            # apply), and live registrations append to the live
            # layout as they land.
            if self.layout_path:
                self.layout_path.parent.mkdir(parents=True, exist_ok=True)
                self.layout_path.write_text(json.dumps(self.layout, indent=1))
            return [{"type": "ack", "ok": True,
                     "objects": len(self.layout["objects"])}]
        if kind == "load_layout":
            doc = self.layout or {"version": 1, "objects": []}
            if self.construction.sites or self.construction.buildings:
                # Serve the construction world's authoritative state
                # with the layout, so a fresh client renders the
                # sites and buildings the server actually holds.
                doc = {**doc, **self.construction.serialize()}
            if self.terrain.modified:
                # Same for the shaped heightmap (P4): once the land
                # has been terraformed, the served layout carries
                # the terrain world's authoritative grid + jobs.
                doc = {**doc, **self.terrain.serialize()}
            if self.innovations.records or self.innovations.under_study:
                # Same for the innovation registry + market (P6):
                # the served layout carries the authoritative slice
                # so a saved world keeps everything it has learned
                # to make.
                doc = {**doc,
                       "innovations": self.innovations.serialize()}
            if self.city_knowledge:
                doc = {**doc, "knowledge": list(self.city_knowledge)}
            if self.designs.styles or self.designs.interiors \
                    or self.designs.assets:
                # Same for the design registry (P8): the served
                # layout carries the authoritative slice so a saved
                # world keeps its styles, interiors and assets.
                doc = {**doc, "designs": self.designs.serialize()}
            return [{"type": "layout", "layout": doc}]
        return [{"type": "error", "error": f"unknown message type: {kind}"}]

    # -- souls: NPC identity + knowledge snapshots --------------------------------
    def export_souls(self) -> dict:
        if self.manager is None:
            raise ValueError("no NPC manager configured")
        return self.manager.export_souls()

    def import_souls(self, payload) -> dict:
        """POST /api/souls body: {"mode": "knowledge"|"full",
        "snapshot": {...}}. Returns the import summary; ValueError
        (bad mode, missing snapshot, malformed snapshot) -> HTTP 400."""
        if self.manager is None:
            raise ValueError("no NPC manager configured")
        if not isinstance(payload, dict):
            raise ValueError("souls load body must be a JSON object")
        snapshot = payload.get("snapshot")
        if snapshot is None:
            raise ValueError("souls load body is missing 'snapshot'")
        return self.manager.import_souls(
            snapshot, mode=str(payload.get("mode", "knowledge")))

    # -- payloads for the web client ----------------------------------------------
    def roster_payload(self, positions: dict) -> dict:
        return {"characters": [
            {**npc.card.to_dict(),
             "position": list(positions.get(name, (0, 0))),
             "namespace": npc.card.namespace}
            for name, npc in self.npcs.items()]}

    def status_payload(self, mode: str = "") -> dict:
        stats = self.manager.stats() if self.manager else {}
        self.resources.tick(time.monotonic())
        return {"mode": mode, "npc_count": len(self.npcs), **self.brain_info,
                **stats,
                "weather": self.weather.public(),
                "inventory": dict(self.resources.inventory),
                "persistence": (self.persistence.public_status()
                                 if self.persistence is not None else
                                 {"state": "disabled", "repo": None,
                                  "last_push": None, "last_pull": None,
                                  "last_error": None}),
                "tiers": dict(self.manager.tiers) if self.manager else {}}


# --------------------------------------------------------------------------
# City spawn layout: deterministic sidewalk-ish spread over a street grid.
# --------------------------------------------------------------------------
def city_spawn_positions(names: list[str]) -> dict[str, tuple[float, float]]:
    positions = {}
    for i, name in enumerate(names):
        col, row = i % 9, i // 9
        x = (col - 4) * 24 + (i % 3) * 4
        z = (row - 3) * 24 + ((i // 3) % 3) * 4
        positions[name] = (float(x), float(z))
    return positions


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------
class _SocketHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                msg = {}
            for resp in self.server.bridge.handle_message(msg):
                self.wfile.write((json.dumps(resp) + "\n").encode())
            self.wfile.flush()


def make_socket_server(bridge: WorldBridge, host: str, port: int):
    server = socketserver.ThreadingTCPServer((host, port), _SocketHandler)
    server.daemon_threads = True
    server.bridge = bridge
    return server


_CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                  ".json": "application/json",
                  ".glb": "model/gltf-binary",
                  ".md": "text/plain; charset=utf-8",
                  ".png": "image/png", ".jpg": "image/jpeg"}


def make_http_server(bridge: WorldBridge, host: str, port: int,
                     static_dir: str | Path, positions: dict,
                     mode: str = ""):
    static_root = Path(static_dir).resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the console clean
            pass

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                msg = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send_json({"messages": [
                    {"type": "error", "error": "invalid JSON body"}]}, 400)
            if self.path == "/api/message":
                return self._send_json(
                    {"messages": bridge.handle_message(msg)})
            if self.path == "/api/layout":
                return self._send_json({"messages": bridge.handle_message(
                    {"type": "save_layout", "layout": msg})})
            if self.path == "/api/souls":
                try:
                    return self._send_json(bridge.import_souls(msg))
                except ValueError as err:
                    return self._send_json({"error": str(err)}, 400)
            if self.path == "/api/ask":
                question = msg.get("question") if isinstance(msg, dict) \
                    else None
                if not isinstance(question, str) or not question.strip():
                    return self._send_json(
                        {"error": "missing 'question'"}, 400)
                try:
                    return self._send_json({"answer": bridge.ask(question)})
                except Exception as err:
                    return self._send_json(
                        {"error": f"{type(err).__name__}: {err}"}, 500)
            return self._send_json({"messages": [
                {"type": "error", "error": f"unknown path {self.path}"}]}, 404)

        def do_GET(self):
            if self.path == "/api/roster":
                # Prefer the manager's live positions (seeded with the
                # spawns at build time, then kept current by proximity
                # reports and soul loads) so a client refreshing after a
                # souls load sees restored positions, not stale spawns.
                pos = (bridge.manager.positions
                       if bridge.manager is not None else positions)
                return self._send_json(bridge.roster_payload(pos))
            if self.path == "/api/souls":
                try:
                    return self._send_json(bridge.export_souls())
                except ValueError as err:
                    return self._send_json({"error": str(err)}, 400)
            if self.path == "/api/status":
                return self._send_json(bridge.status_payload(mode))
            if self.path == "/api/world":
                return self._send_json(bridge.world_payload())
            if self.path == "/api/layout":
                return self._send_json(
                    bridge.layout or {"version": 1, "objects": []})
            if self.path == "/api/events":
                return self._sse()
            return self._static()

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            q = bridge.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        payload = f"data: {json.dumps(msg)}\n\n".encode()
                    except queue.Empty:
                        payload = b": heartbeat\n\n"
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bridge.unsubscribe(q)

        def _static(self):
            rel = self.path.split("?", 1)[0].lstrip("/") or "demo.html"
            if rel == "portal":
                rel = "portal.html"          # the launch portal lives at /portal
            if rel == "README.md":
                # Serve the project README (one level up from webworld/) so
                # the portal's documentation links work under the bridge.
                target = (static_root.parent / "README.md").resolve()
            else:
                target = (static_root / rel).resolve()
                if static_root not in target.parents and target != static_root:
                    return self._send_json({"error": "forbidden"}, 403)
            if not target.is_file():
                return self._send_json({"error": f"not found: {rel}"}, 404)
            body = target.read_bytes()
            self.send_response(200)
            # No-store on everything static: tablet browsers were
            # heuristically caching demo.html and model files across
            # updates, so a fixed build could keep rendering the old
            # cached page/models (the 2026-11-24 "buildings are
            # still boxes" report). This server is localhost — a
            # full re-fetch per load is cheap and always current.
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type",
                             _CONTENT_TYPES.get(target.suffix,
                                                "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True

    _orig_handle_error = ThreadingHTTPServer.handle_error

    def _handle_error(self, request, client_address):
        # A browser that reloads, navigates away or drops its fetch
        # mid-response surfaces as a broken pipe in the handler
        # thread. That is the client leaving, not a server fault —
        # swallow it quietly instead of printing a traceback.
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        return _orig_handle_error(self, request, client_address)

    server.handle_error = _handle_error.__get__(server,
                                                ThreadingHTTPServer)
    return server


# --------------------------------------------------------------------------
# Assembly: build a full world (brain + memory + roster + manager + bridge)
# --------------------------------------------------------------------------
def build_world(cfg: dict, mock: bool = False, characters_dir: str | None = None,
                roster_count: int = 54, layout_path: str | None = None):
    """Returns (bridge, manager, npcs, positions, mode_str)."""
    from .embeddings import EmbeddingClient
    from .llm import LLMClient, detect_resources, pick_brain_model
    from .memory import Memory
    from .npc import Director, NPC, NPCManager, generate_roster, load_cards

    ecfg = cfg.get("embeddings", {})
    embedder = EmbeddingClient(
        base_url=ecfg.get("base_url", "http://localhost:11434"),
        api=ecfg.get("api", "ollama"), models=ecfg.get("models"),
        timeout=int(ecfg.get("timeout", 10)),
        local_dim=int(ecfg.get("local_dim", 512)))
    mcfg = cfg.get("memory", {})
    memory = Memory(mcfg.get("db_path", "~/.hermes_local/memory.db"),
                    mcfg.get("notes_path", "~/.hermes_local/notes.md"),
                    embedder,
                    search_dim=int(ecfg.get("search_dim", 256)),
                    recall_k=int(mcfg.get("recall_k", 4)))

    lcfg = cfg.get("llm", {})
    resources = detect_resources()
    ladder_model, ladder_why = pick_brain_model(
        resources, lcfg.get("model", "auto"))
    if mock:
        client, mode = DemoBrain(), "mock (demo brain — no LLM server)"
        brain_info = {"brain": DemoBrain.name, "provider": "mock",
                      "endpoint": "",
                      "ladder_pick": ladder_model, "ladder_why": ladder_why}
    else:
        model = ladder_model
        client = LLMClient(base_url=lcfg.get("base_url", "http://localhost:11434"),
                           model=model, api=lcfg.get("api", "ollama"),
                           timeout=int(lcfg.get("timeout", 120)),
                           retries=int(lcfg.get("retries", 3)),
                           temperature=float(lcfg.get("temperature", 0.7)),
                           api_key=lcfg.get("api_key", ""))
        mode = f"live (brain: {model})"
        brain_info = {"brain": model,
                      "provider": lcfg.get("provider") or "local",
                      "endpoint": lcfg.get("base_url",
                                           "http://localhost:11434"),
                      "ladder_pick": ladder_model, "ladder_why": ladder_why}

    cards = []
    if characters_dir:
        cards.extend(load_cards(characters_dir))
    seen = {c.name for c in cards}
    for card in generate_roster(roster_count):
        if card.name not in seen:
            cards.append(card)
            seen.add(card.name)
    npcs = {card.name: NPC(card, client, memory=memory, embedder=embedder,
                           config=cfg) for card in cards}
    positions = city_spawn_positions(list(npcs))
    # Live worlds (a real LLMClient, not the DemoBrain) get LLM-written
    # ambient NPC<->NPC exchanges for EVERY pairing (no tier gate, no
    # canned fallback — a failed generation is silence); mock worlds
    # keep canned lines only. The generator knows the city's anchor
    # names so a pair's "let's meet later at ..." agreement can be
    # validated into a real rendezvous plan, and it reads the shared
    # memory store so prompts carry each citizen's own memories.
    from .npc import build_city_anchors
    # The sky colors small talk (P8): the conversation generator
    # reads the bridge's weather through this cell, filled in once
    # the bridge exists (the generator is built before it).
    weather_line: dict = {"fn": None}
    conversation_gen = (None if mock else make_conversation_generator(
        client, anchor_names=[a.name for a in build_city_anchors()],
        memory=memory,
        weather_fn=lambda: weather_line["fn"]()
        if weather_line["fn"] else ""))
    acfg = cfg.get("agent", {})
    manager = NPCManager(
        npcs,
        active_radius=float(acfg.get("active_radius", 15)),
        ambient_radius=float(acfg.get("ambient_radius", 50)),
        max_active=int(acfg.get("max_active", 4)),
        active_exit_radius=float(acfg.get("active_exit_radius", 20)),
        ambient_exit_radius=float(acfg.get("ambient_exit_radius", 60)),
        min_hold_s=float(acfg.get("tier_min_hold_s", 5)),
        engage_timeout_s=float(acfg.get("engage_timeout_s", 20)),
        engage_range_m=float(acfg.get("engage_range_m", 25)),
        spawns=positions, conversation_generator=conversation_gen)
    director = Director(npcs)
    bridge = WorldBridge(npcs, manager=manager, director=director,
                         layout_path=layout_path)
    bridge.brain_info = brain_info
    weather_line["fn"] = bridge.weather.context_line
    # Sky config (P8): the day length is world config, brain or no
    # brain — a full day/night cycle every N sim-seconds.
    wcfg = cfg.get("weather") or {}
    if wcfg.get("day_length_s"):
        try:
            bridge.weather.day_length_s = max(
                60.0, float(wcfg["day_length_s"]))
        except (TypeError, ValueError):
            pass
    if not mock:
        # Live worlds get a live World Interpreter: its own dedicated
        # short-timeout client over the same brain (the ambient
        # conversation generator does the same) so an oracle answer
        # never hangs on the brain's 120s / 3-retry settings. Mock
        # worlds keep the bridge's default rule-based interpreter.
        bridge.interpreter = WorldInterpreter(
            bridge,
            client=LLMClient(base_url=client.base_url, model=client.model,
                             api=client.api, timeout=30, retries=1,
                             temperature=client.temperature,
                             api_key=client.api_key),
            memory=memory)
        # Live worlds get live civics (P5): the drafter, debater
        # and — per Aaron's directive — the per-citizen AI ballot
        # caster, all soul-fed, over the same dedicated-client
        # discipline. ai_ballots=True is the switch that forbids
        # rule ballots outright in this world: a failed ballot call
        # is an abstention, never a formula vote.
        if bridge.civics is not None:
            (bridge.civics.drafter, bridge.civics.debater,
             bridge.civics.ballot_caster) = make_civic_generators(
                client, memory=memory)
            bridge.civics.ai_ballots = True
        # Live worlds get the world planner (P6): the research
        # terminal's insight synthesizer and the innovation
        # engine's spec synthesizer, both over dedicated ~45s
        # clients, plus the real keyless web search. Mock worlds
        # wire neither — their terminal stays honestly dark.
        bridge.research.brain = make_research_brain(client)
        bridge.innovations.spec_brain = make_spec_synthesizer(client)
        bridge.innovations.searcher = \
            lambda q: web_search(q, fetcher=bridge.research.fetcher)
        # Live worlds get a live design brain (P8): styles,
        # interiors and assets drawn by the brain and validated by
        # the design engine. Mock worlds keep design_brain=None —
        # their assets fall back to the seeded templates, honestly
        # labeled, and styles/interiors are seeded only.
        bridge.designs.design_brain = make_design_brain(client)
    # World-data persistence (P7): attach the service and run its
    # startup — the local snapshot loads, then a bounded-wait pull
    # applies a newer repo copy — so the world that comes up is the
    # newest state anywhere, and this process keeps it saved from
    # here on (interval snapshots on the planner tick + sync_now).
    from .persistence import PersistenceService
    service = PersistenceService(bridge, cfg.get("persistence") or {},
                                 layout_path=layout_path)
    bridge.persistence = service
    if manager.planner is not None:
        manager.planner.persistence = service
    service.startup()
    manager.update((0.0, 0.0), positions)
    return bridge, manager, npcs, positions, mode
