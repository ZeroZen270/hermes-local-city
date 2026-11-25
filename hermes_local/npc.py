"""NPC character engine: cards, episodic memory, dialogue, body actions,
a turn-taking director, and the proximity-activation manager.

Design (low-resource first):
  * ONE shared brain model serves every NPC. Each NPC is a lightweight
    object: a character card, a memory *namespace* in the shared
    EmbeddingGemma 2 vector store, and a short-term conversation window
    managed by the agent's ContextManager.
  * Body actions are ordinary framework tools (say / move_to / follow /
    interact_with / play_animation / emote), emitted through the same dual
    tool-calling modes as the main agent (native tools, or the ReAct
    fenced-JSON fallback) and validated against a per-world action schema
    before anything reaches the world.
  * Proximity activation keeps 50+ NPCs cheap: only ACTIVE NPCs (player
    within `active_radius`, at most `max_active` of them) ever touch the
    LLM. AMBIENT NPCs are scripted by the world client; DORMANT NPCs cost
    nothing. Activation loads persona + recalled memories; deactivation
    summarizes the encounter into episodic memory and unloads context.
  * Movement is an itinerary planner (ActivityPlanner), not a wander:
    NPCs pick activities by utility score, travel to anchors, and dwell
    there 30-180s. Tiers use hysteresis bands plus a minimum hold, and a
    conversation lock pins an NPC the player is talking to — so the
    ACTIVE set, and the chat-target list built from it, stays stable.
"""
from __future__ import annotations

import json
import queue
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import tools as tools_mod
from . import work as work_mod
from .agent import ContextManager, parse_tool_blocks
from .llm import ToolUnsupportedError

# --------------------------------------------------------------------------
# World action schema + tools (registered in the shared tool registry)
# --------------------------------------------------------------------------
DEFAULT_ACTION_SCHEMA: dict[str, dict] = {
    "say": {"required": ["text"], "description": "Speak a line of dialogue."},
    "move_to": {"required": ["target"], "description": "Walk to a named place or object."},
    "follow": {"required": ["entity"], "description": "Follow an entity (e.g. Player)."},
    "interact_with": {"required": ["object"], "description": "Use / pick up / examine an object."},
    "play_animation": {"required": ["name"], "description": "Play a body animation (wave, bow, work...)."},
    "emote": {"required": ["name"], "description": "Show an emotion (happy, angry, surprised...)."},
}
WORLD_TOOL_NAMES = list(DEFAULT_ACTION_SCHEMA)


def _dispatch(ctx, action: str, args: dict) -> str:
    """Validate an action against the world schema and hand it to the world."""
    if ctx.world is None:
        return "TOOL ERROR: no 3D world is connected, so body actions are unavailable."
    schema = ctx.action_schema or DEFAULT_ACTION_SCHEMA
    if action not in schema:
        return (f"TOOL ERROR: action '{action}' is not allowed in this world. "
                f"Allowed: {', '.join(sorted(schema))}")
    missing = [k for k in schema[action].get("required", []) if not args.get(k)]
    if missing:
        return f"TOOL ERROR: {action} is missing required argument(s): {', '.join(missing)}"
    try:
        ctx.world.send_action(ctx.npc_name, action, args)
    except Exception as err:  # the world sink must never crash the loop
        return f"TOOL ERROR: world rejected {action}: {err}"
    if action == "say":
        return f"(you said: {args.get('text', '')})"
    return f"({action} {args} sent to the world)"


def _make_action_tool(action: str, description: str, arg: str):
    def fn(ctx, **kwargs):
        return _dispatch(ctx, action, kwargs)
    fn.__name__ = action
    return fn


for _action, _spec in DEFAULT_ACTION_SCHEMA.items():
    _arg = _spec["required"][0]
    tools_mod.TOOLS[_action] = {
        "name": _action,
        "description": _spec["description"],
        "parameters": {"type": "object",
                       "properties": {_arg: {"type": "string"}},
                       "required": _spec["required"]},
        "fn": _make_action_tool(_action, _spec["description"], _arg),
    }


def world_schemas() -> list[dict]:
    """Native-mode tool schemas for just the world action tools."""
    return [s for s in tools_mod.schemas()
            if s["function"]["name"] in WORLD_TOOL_NAMES]


def world_catalog_text() -> str:
    lines = ["Body actions — call one by replying with a fenced block:",
             '```tool', '{"name": "<action>", "arguments": {...}}', '```']
    for name in WORLD_TOOL_NAMES:
        spec = DEFAULT_ACTION_SCHEMA[name]
        lines.append(f"- {name}({spec['required'][0]}): {spec['description']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Character cards
# --------------------------------------------------------------------------
@dataclass
class CharacterCard:
    name: str
    role: str = "townsperson"
    persona: str = ""
    backstory: str = ""
    traits: list = field(default_factory=list)
    voice: str = ""                       # style notes for how they talk
    goals: list = field(default_factory=list)
    relationships: dict = field(default_factory=dict)

    @property
    def namespace(self) -> str:
        return "npc:" + re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")

    def to_dict(self) -> dict:
        return asdict(self)


def load_card(path: str | Path) -> CharacterCard:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # PyYAML is a declared dependency

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    known = {f for f in CharacterCard.__dataclass_fields__}
    return CharacterCard(**{k: v for k, v in data.items() if k in known})


def load_cards(directory: str | Path) -> list[CharacterCard]:
    directory = Path(directory)
    cards = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.suffix in (".json", ".yaml", ".yml"):
                cards.append(load_card(path))
    return cards


# --------------------------------------------------------------------------
# Procedural roster: real character cards generated from persona templates,
# so a 50+ NPC city is 50+ real framework characters, each with its own
# memory namespace — not dummies.
# --------------------------------------------------------------------------
_FIRST_NAMES = [
    "Mara", "Joss", "Priya", "Tomas", "Ines", "Kofi", "Lena", "Rurik",
    "Sana", "Dario", "Mina", "Oskar", "Tessa", "Yusuf", "Alba", "Cato",
    "Nia", "Felix", "Rhea", "Bruno", "Ilsa", "Marco", "Odette", "Piotr",
    "Suki", "Anders", "Vera", "Hugo", "Lila", "Emeka", "Rosa", "Kian",
    "Marta", "Silas", "Anouk", "Dmitri", "Farah", "Goran", "Hana", "Ivor",
    "Jana", "Kaspar", "Lior", "Mette", "Nils", "Opal", "Ravi", "Selin",
    "Tariq", "Uma", "Viktor", "Wren", "Yara", "Zoran",
]
_ROLES = [
    ("shopkeeper", "runs a small shop and knows every price in the district",
     "Speak warmly, mention goods and prices, offer help finding things."),
    ("guard", "patrols the streets and keeps the peace",
     "Speak briefly and formally. Watch for trouble."),
    ("student", "studies at the city institute and is always in a hurry",
     "Speak fast and curious, ask questions back."),
    ("mechanic", "repairs vehicles and machines at the depot",
     "Speak practically, use workshop slang, wipe grease off everything."),
    ("medic", "staffs the clinic on the corner",
     "Speak calmly and precisely. Ask about symptoms like a professional."),
    ("vendor", "sells food from a street cart",
     "Speak loudly and cheerfully, always pitching today's special."),
    ("courier", "delivers packages across the city on foot",
     "Speak breathlessly, always mid-errand, knows every shortcut."),
    ("elder", "has lived in this district for sixty years",
     "Speak slowly, tell small stories about how the city used to be."),
    ("artist", "paints murals on the district walls",
     "Speak dreamily, notice colors and light in everything."),
    ("engineer", "maintains the city's power and water systems",
     "Speak in systems and numbers, dry humor."),
]
_TRAITS = ["cheerful", "gruff", "curious", "cautious", "witty", "sleepy",
           "proud", "kind", "suspicious", "chatty", "stoic", "nervous",
           "generous", "blunt", "dreamy", "meticulous"]


def generate_roster(count: int = 54, seed: int = 7) -> list[CharacterCard]:
    rng = random.Random(seed)
    cards: list[CharacterCard] = []
    for i in range(count):
        name = _FIRST_NAMES[i % len(_FIRST_NAMES)]
        if i >= len(_FIRST_NAMES):
            name = f"{name} {chr(65 + (i // len(_FIRST_NAMES)))}"
        role, persona, voice = _ROLES[i % len(_ROLES)]
        traits = rng.sample(_TRAITS, 2)
        cards.append(CharacterCard(
            name=name, role=role,
            persona=f"{name} is a {role} who {persona}.",
            backstory=f"{name} has lived in the city for {3 + (i * 7) % 30} years.",
            traits=traits, voice=voice,
            goals=[f"Do the work of a {role} well",
                   "Look out for neighbors in the district"],
            relationships={}))
    return cards


# --------------------------------------------------------------------------
# NPC
# --------------------------------------------------------------------------
class NPC:
    """One character: card + episodic memory namespace + conversation window."""

    def __init__(self, card: CharacterCard, client, memory=None, embedder=None,
                 world=None, action_schema: dict | None = None,
                 config: dict | None = None):
        cfg = (config or {}).get("agent", {})
        self.card = card
        self.client = client
        self.memory = memory
        self.mode = cfg.get("tool_mode", "auto")
        self.cm = ContextManager(
            max_tokens=int(cfg.get("context_tokens", 16000)),
            keep_recent=int(cfg.get("keep_recent_turns", 6)))
        self.ctx = tools_mod.ToolContext(
            memory=memory, embedder=embedder,
            world=world, action_schema=action_schema, npc_name=card.name)
        self.history: list[dict] = []       # short-term conversation window
        self.encounter_log: list[str] = []  # this activation's events, for summary
        self.active = False
        self.recalled_on_activation: list[dict] = []
        self.stats = {"activations": 0, "turns": 0, "summaries": 0}

    # -- memory ---------------------------------------------------------------
    @property
    def ns(self) -> str:
        return self.card.namespace

    def witness(self, event: str, kind: str = "episode") -> None:
        """Write something this NPC saw/heard into its episodic memory."""
        if self.memory is not None:
            self.memory.remember(event, kind=kind, namespace=self.ns)
        self.encounter_log.append(event)

    def recall(self, query: str, k: int = 4) -> list[dict]:
        if self.memory is None:
            return []
        return self.memory.recall(query, k=k, namespace=self.ns)

    # -- activation lifecycle (driven by NPCManager) --------------------------
    def activate(self) -> list[dict]:
        """Load persona context: recall relevant episodic memories."""
        self.recalled_on_activation = self.recall(
            f"{self.card.role} encounters, the player, important events", k=4)
        self.active = True
        self.stats["activations"] += 1
        self.encounter_log.append(
            f"{self.card.name} noticed the player nearby and became alert.")
        return self.recalled_on_activation

    def deactivate(self) -> str:
        """Summarize the encounter into episodic memory; unload context.

        Only a SUBSTANTIVE encounter is filed: activation itself
        logs a "<name> noticed the player nearby" line, so a
        citizen the player merely walked past used to file (and
        announce, and spend a brain call summarizing) a memory of
        nothing happening — repeatedly, on every pass. When the
        log holds nothing beyond that notice, deactivate quietly:
        no summary, no memory write, no event text."""
        summary = ""
        substantive = [e for e in self.encounter_log
                       if "noticed the player nearby" not in e]
        if substantive:
            text = "\n".join(self.encounter_log)[-6000:]
            try:
                resp = self.client.chat([
                    {"role": "system",
                     "content": "Summarize this encounter from the character's "
                                "point of view in 1-3 sentences. Keep names, "
                                "places, promises, and anything learned."},
                    {"role": "user", "content": text}])
                summary = (resp.get("content") or "").strip()
            except Exception:
                summary = ""
            if not summary:
                summary = "Encounter: " + "; ".join(self.encounter_log[-4:])
            if self.memory is not None:
                self.memory.remember(summary, kind="encounter", namespace=self.ns)
            self.stats["summaries"] += 1
        self.encounter_log = []
        self.history = []
        self.recalled_on_activation = []
        self.active = False
        return summary

    # -- conversation -----------------------------------------------------------
    def system_prompt(self, recalled: list[dict],
                      situation: str | None = None) -> str:
        c = self.card
        rel = "; ".join(f"{k}: {v}" for k, v in c.relationships.items()) or "none yet"
        prompt = (
            f"You are {c.name}, a {c.role}. {c.persona} {c.backstory}\n"
            f"Traits: {', '.join(c.traits)}. Voice: {c.voice}\n"
            f"Goals: {'; '.join(c.goals)}. Relationships: {rel}.\n"
            "Stay in character at all times. Keep replies short (1-3 "
            "sentences) unless asked for more. Speak by replying normally, "
            "or use the say tool. Use body action tools (move_to, follow, "
            "interact_with, play_animation, emote) when you physically act. "
            "Keep the conversation going: ask the other person questions "
            "back, react to what they actually said instead of only "
            "answering, and refer to things you remember about them when "
            "that is relevant.")
        if situation:
            # Location context (P11): where the citizen factually
            # is, computed by the manager/planner from the world's
            # own data — the character must speak from THERE.
            prompt += (f"\n\nWhere you are right now: {situation}\n"
                       f"{LOCATION_GROUNDING}")
        if self.mode == "react":
            prompt += "\n\n" + world_catalog_text()
        if recalled:
            mem = "\n".join(f"- {h['text']}" for h in recalled if h["score"] > 0.03)
            if mem:
                prompt += f"\n\nThings you remember:\n{mem}"
        return prompt

    def _chat(self, messages: list[dict]) -> dict:
        if self.mode in ("auto", "native"):
            try:
                return self.client.chat(messages, tools=world_schemas())
            except Exception as err:
                # Providers reject native tools with many error shapes,
                # not just ToolUnsupportedError (Nous LongCat free 400s
                # the tools array with a generic error). Degrade
                # permanently to ReAct and retry this same turn as a
                # plain chat call; only a failure of the plain call
                # itself may propagate.
                self.mode = "react"
                print(f"[hermes] {self.card.name}: native tools failed "
                      f"({type(err).__name__}: {str(err)[:200]}); "
                      "degrading to ReAct mode", flush=True)
                return self.client.chat(messages)
        return self.client.chat(messages)

    def hear(self, text: str, speaker: str = "Player",
             situation: str | None = None) -> str:
        """One dialogue turn. Returns the NPC's spoken reply.
        `situation` (P11) is the citizen's factual location
        context, computed by whoever owns their position (the
        manager) — an NPC never guesses where it is standing."""
        recalled = self.recall(text, k=3) or self.recalled_on_activation
        messages = ([{"role": "system",
                      "content": self.system_prompt(recalled, situation)}]
                    + list(self.history)
                    + [{"role": "user", "content": f"{speaker}: {text}"}])
        spoken: list[str] = []
        reply = ""
        for _ in range(4):  # small action loop: act, observe, then speak
            messages, _did = self.cm.compress(messages, self._summarize)
            try:
                resp = self._chat(messages)
            except Exception as err:
                # Last resort (the tools call AND its ReAct retry both
                # failed): keep the in-world line, but surface the real
                # error on the server log instead of swallowing it.
                print(f"[hermes] {self.card.name}: chat failed "
                      f"({type(err).__name__}: {str(err)[:200]}); "
                      "replying distracted", flush=True)
                reply = f"({self.card.name} seems distracted.)"
                break
            content = resp.get("content") or ""
            calls = resp.get("tool_calls") or []
            native = bool(calls)
            if not calls:
                calls = parse_tool_blocks(content)
            if not calls:
                reply = content.strip()
                break
            if native:
                messages.append({"role": "assistant", "content": content,
                                 "tool_calls": calls})
            else:
                messages.append({"role": "assistant", "content": content})
            results = []
            for call in calls:
                result = tools_mod.run_tool(self.ctx, call["name"],
                                            call.get("arguments"))
                results.append(f"- {call['name']}: {result}")
                if call["name"] == "say":
                    spoken.append(str((call.get("arguments") or {}).get("text", "")))
                if native:
                    messages.append({"role": "tool", "name": call["name"],
                                     "tool_call_id": call.get("id", ""),
                                     "content": result})
            if not native:
                messages.append({"role": "user",
                                 "content": "Action results:\n" + "\n".join(results) +
                                            "\nNow reply in character."})
        if spoken:
            reply = " ".join(s for s in spoken if s) or reply
        reply = reply or f"({self.card.name} nods.)"
        self.history += [{"role": "user", "content": f"{speaker}: {text}"},
                         {"role": "assistant", "content": reply}]
        self.encounter_log.append(f"{speaker} said: {text} | "
                                  f"{self.card.name} replied: {reply}")
        self.stats["turns"] += 1
        return reply

    def _summarize(self, text: str) -> str:
        resp = self.client.chat([
            {"role": "system", "content": "Summarize briefly, keeping facts and names."},
            {"role": "user", "content": text}])
        return resp.get("content") or "(summary unavailable)"


# --------------------------------------------------------------------------
# Director: schedules NPC<->NPC conversations so nobody talks over anyone.
# --------------------------------------------------------------------------
class Director:
    def __init__(self, npcs: dict[str, NPC]):
        self.npcs = npcs

    def converse(self, name_a: str, name_b: str, opener: str,
                 turns: int = 4) -> list[dict]:
        """Alternate turns between two NPCs; returns the transcript."""
        a, b = self.npcs[name_a], self.npcs[name_b]
        transcript = []
        speaker, listener, line = a, b, opener
        for _ in range(turns):
            reply = listener.hear(line, speaker=speaker.card.name)
            transcript.append({"npc": listener.card.name, "text": reply})
            speaker, listener, line = listener, speaker, reply
        return transcript


# --------------------------------------------------------------------------
# Activity anchors + itinerary planner — the modern movement system.
#
# inZOI / GTA-style: NPCs do not random-wander. Each NPC runs an itinerary
# loop — pick an activity by utility score, travel purposefully to an
# activity anchor, DWELL there performing it (30-180s), repeat. Most NPC
# time is spent dwelling, so the city reads as lived-in rather than
# restless, and NPCs stop drifting across the ACTIVE boundary every few
# seconds (which is what made the chat-target list flicker). Anchors
# derive from the same city grid the world layout is generated from
# (roads at -96/-48/0/48/96, blocks centered at -72/-24/24/72, building
# fronts at block corners +-11) plus each NPC's spawn point: an NPC's
# workplace is the role-appropriate anchor nearest its spawn. Vendors and
# workers weight their workplace heavily. The web client mirrors these
# exact constants and rules for its local sim / standalone engine.
# --------------------------------------------------------------------------
CITY_ROADS = (-96, -48, 0, 48, 96)
CITY_BLOCKS = (-72, -24, 24, 72)
FRONT_OFFSETS = ((-11, -11), (11, -11), (-11, 11), (11, 11))

# Every dwell lands inside 30-180s; most of an NPC's life is spent here.
DWELL_RANGES = {"work": (90.0, 180.0), "browse": (45.0, 120.0),
                "social": (60.0, 150.0), "idle": (30.0, 90.0),
                "errand": (30.0, 75.0), "research": (60.0, 150.0),
                "home": (240.0, 480.0)}


@dataclass
class ActivityAnchor:
    name: str
    kind: str            # front | social | idle | browse | errand
    x: float
    z: float


def build_city_anchors() -> list[ActivityAnchor]:
    """Deterministic anchor set derived from the city grid (see above)."""
    anchors: list[ActivityAnchor] = []
    for bx in CITY_BLOCKS:
        for bz in CITY_BLOCKS:
            for k, (ox, oz) in enumerate(FRONT_OFFSETS):
                anchors.append(ActivityAnchor(
                    f"Block {bx},{bz} front {k + 1}", "front",
                    float(bx + ox), float(bz + oz)))
    for bx in CITY_BLOCKS:
        for bz in CITY_BLOCKS:
            if (bx, bz) in ((-72, -72), (72, 72)):
                continue  # those two block centers are parks, below
            anchors.append(ActivityAnchor(
                f"Courtyard {bx},{bz}", "social", float(bx), float(bz)))
    anchors.append(ActivityAnchor("Old Park", "idle", -72.0, -72.0))
    anchors.append(ActivityAnchor("North Green", "idle", 72.0, 72.0))
    anchors.append(ActivityAnchor("Central Plaza", "social", 8.0, 8.0))
    anchors.append(ActivityAnchor("Market Row West", "browse", -36.0, 8.0))
    anchors.append(ActivityAnchor("Market Row East", "browse", 36.0, -8.0))
    anchors.append(ActivityAnchor("Market Row North", "browse", -8.0, 60.0))
    anchors.append(ActivityAnchor("Market Row South", "browse", 8.0, -60.0))
    anchors.append(ActivityAnchor("West Depot", "errand", -104.0, 8.0))
    anchors.append(ActivityAnchor("East Depot", "errand", 104.0, -8.0))
    anchors.append(ActivityAnchor("North Depot", "errand", -8.0, 104.0))
    anchors.append(ActivityAnchor("South Depot", "errand", 8.0, -104.0))
    # The research computer (P6, see research.py): a terminal a few
    # steps from Central Plaza where citizens sit down and ask the
    # wider world questions. Position mirrors research.TERMINAL_POS.
    anchors.append(ActivityAnchor("Research Terminal", "research",
                                  18.0, 10.0))
    return anchors


# --------------------------------------------------------------------------
# Districts (P9, city growth — design Addendum 3). Each expansion of
# the world (terrain bound +60m) opens one new ring of land, and the
# ring's band gets four planner anchors at its cardinal midpoints —
# deterministic names by ring number, so a district is a pure
# function of HOW MANY expansions the city has taken, never a
# hand-persisted record: the layout document carries the terrain
# bound (terrain.py), the expansion count derives from it, and the
# anchors are rebuilt from the count (ActivityPlanner.set_expansions,
# WorldBridge._sync_district_anchors). Ring r's band midpoint sits
# at +/-(120 + 60r - 30) on each axis — the middle of the new land.
# Kinds are ordinary planner kinds: fields and woods are places to
# idle, the plots are work ground (errand), the meadows social.
# --------------------------------------------------------------------------
_DISTRICT_SUFFIXES = {
    "East": ("Fields", "Reaches", "Marches", "Wilds"),
    "North": ("Woods", "Pines", "Fells", "Highlands"),
    "West": ("Plots", "Farms", "Commons", "Barrens"),
    "South": ("Meadows", "Downs", "Shores", "Flats"),
}
_DISTRICT_KINDS = {"East": "idle", "North": "idle",
                   "West": "errand", "South": "social"}
MAX_EXPANSIONS = 4        # rings to the terrain engine cap (+/-360m)


def district_anchor_specs(expansions: int) -> list[tuple]:
    """(name, kind, x, z) for every district anchor of the first
    `expansions` rings, ring by ring, East/North/West/South within
    each ring. Pure + deterministic — the one definition the
    planner, the civics executor and the interpreter all read."""
    specs: list[tuple] = []
    for ring in range(1, max(0, min(int(expansions),
                                   MAX_EXPANSIONS)) + 1):
        c = 120.0 + 60.0 * ring - 30.0     # the ring band's midpoint
        for direction, (dx, dz) in (("East", (1.0, 0.0)),
                                    ("North", (0.0, 1.0)),
                                    ("West", (-1.0, 0.0)),
                                    ("South", (0.0, -1.0))):
            specs.append((f"{direction} "
                          f"{_DISTRICT_SUFFIXES[direction][ring - 1]}",
                          _DISTRICT_KINDS[direction],
                          dx * c, dz * c))
    return specs


def district_anchors(expansions: int) -> list[ActivityAnchor]:
    """The district anchors for a city grown by `expansions` rings."""
    return [ActivityAnchor(name, kind, x, z)
            for name, kind, x, z in district_anchor_specs(expansions)]


def ring_district_names(ring: int) -> list[str]:
    """Just the four names one ring adds, in spec order."""
    return [name for name, _k, _x, _z
            in district_anchor_specs(ring)
            if name not in {n for n, _k2, _x2, _z2
                            in district_anchor_specs(ring - 1)}]


_ROLE_AFFINITY = {  # role -> {activity kind: utility bonus}
    "vendor": {"browse": 1.5}, "shopkeeper": {"browse": 0.6},
    "artist": {"idle": 1.0, "social": 0.6},
    "elder": {"idle": 1.2, "social": 0.4, "research": 0.3},
    "student": {"social": 0.8, "browse": 0.3, "research": 1.2},
    "guard": {"social": 0.4, "errand": 0.4}, "courier": {"errand": 0.8},
    "mechanic": {"errand": 0.3}, "medic": {"social": 0.3},
    "engineer": {"idle": 0.3, "research": 0.8},
}
_TRAIT_AFFINITY = {  # trait -> {activity kind: utility bonus}
    "chatty": {"social": 0.9}, "curious": {"browse": 0.8, "research": 0.9},
    "sleepy": {"idle": 0.9}, "stoic": {"idle": 0.5},
    "meticulous": {"work": 0.7}, "proud": {"work": 0.5},
    "kind": {"social": 0.6}, "generous": {"social": 0.5},
    "suspicious": {"idle": 0.4, "social": -0.5},
    "cautious": {"idle": 0.4, "social": -0.3},
    "dreamy": {"idle": 0.7}, "nervous": {"social": -0.3},
    "witty": {"social": 0.4}, "cheerful": {"social": 0.4},
    "gruff": {"idle": 0.3}, "blunt": {"work": 0.3},
}
_SOCIAL_LINES = [
    "Nice weather for the district, isn't it?",
    "Did you hear the market prices dropped this week?",
    "I was just saying — this city never sits still.",
    "You look well. The district treating you kindly?",
    "Mind the traffic by the depot, it's wild today.",
    "I keep meaning to visit the plaza more often.",
    "Work's steady, can't complain. You?",
    "Have you tried the food cart on Market Row? Superb.",
    "I remember when all this was half-built.",
    "Quiet moment like this — rare, and welcome.",
    "The engineers finally fixed the lights on our street.",
    "If you're heading north, take the shortcut by the park.",
]


def _anchor_dist(a: ActivityAnchor, pos) -> float:
    return ((a.x - pos[0]) ** 2 + (a.z - pos[1]) ** 2) ** 0.5


def workplace_anchor(card: CharacterCard, spawn, anchors) -> ActivityAnchor:
    """The role-appropriate anchor nearest the NPC's spawn point.
    Pools follow the P11 role classes (_ROLE_CLASS): trade roles
    work the market (browse), works roles the depots (errand),
    guards the plaza + depots; everyone else (medic included —
    there is no clinic anchor) takes a building front."""
    if card.role in ("vendor", "shopkeeper"):
        pool = [a for a in anchors if a.kind == "browse"]
    elif card.role in ("mechanic", "courier", "engineer"):
        pool = [a for a in anchors if a.kind == "errand"]
    elif card.role == "guard":
        pool = [a for a in anchors if a.name == "Central Plaza"]
        pool += [a for a in anchors if a.kind == "errand"]
    else:
        pool = [a for a in anchors if a.kind == "front"]
    best = min(range(len(pool)),
               key=lambda i: (_anchor_dist(pool[i], spawn), i))
    return pool[best]


def _activity_kind(card: CharacterCard, anchor: ActivityAnchor,
                   workplace: ActivityAnchor) -> str:
    if anchor is workplace:
        return "errand" if anchor.kind == "errand" else "work"
    return "browse" if anchor.kind == "front" else anchor.kind


def _hook_arity(gen) -> int:
    """How many positional arguments a conversation-generator
    hook accepts: 3 = the original (card_a, card_b, anchor)
    contract, 4 = history-aware, 5 = history + per-speaker
    situation strings (P11 location context). *args counts as 5.
    Inspected once, up front, so every hook is always called with
    exactly the shape it was written against."""
    if gen is None:
        return 0
    try:
        import inspect

        params = list(inspect.signature(gen).parameters.values())
    except (TypeError, ValueError):
        return 3
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return 5
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return min(5, len(positional))


def _hook_takes_history(gen) -> bool:
    """Does this conversation-generator hook accept the fourth
    `history` argument? (Kept for the original callers; new code
    reads _hook_arity.)"""
    return _hook_arity(gen) >= 4


# --------------------------------------------------------------------------
# Daily schedules + location context (P11). Citizens keep a rough
# shape to their day: the planner's utility choice is bent by the
# world clock (weather.daypart names — morning / afternoon /
# evening / night) toward the places a person of their role would
# be at that hour, and after dark toward home. The bend is a score
# adjustment inside _choose only — conversation locks, tasks,
# plans and brain-directed orders keep their existing priority,
# and a planner with no weather engine attached keeps the exact
# pre-P11 behaviour (neutral multipliers).
# --------------------------------------------------------------------------
_ROLE_CLASS = {
    "vendor": "trade", "shopkeeper": "trade",
    "guard": "watch",
    "mechanic": "works", "engineer": "works", "courier": "works",
    "medic": "care",
    "student": "student",
    "artist": "creative",
    "elder": "elder",
}
# Additive score bonus per (daypart, activity kind): positive
# pulls citizens toward that kind of place at that hour.
_SCHEDULE_KIND = {
    "morning":   {"work": 1.8, "browse": 0.9, "errand": 1.2,
                  "research": 0.6, "social": -0.5, "idle": -0.6},
    "afternoon": {"work": 1.5, "browse": 1.2, "errand": 0.9,
                  "research": 0.3, "social": 0.0, "idle": -0.3},
    "evening":   {"work": -0.6, "browse": 0.2, "errand": -0.5,
                  "research": -0.3, "social": 1.8, "idle": 0.9},
    "night":     {"work": -1.7, "browse": -1.1, "errand": -0.9,
                  "research": -0.8, "social": -0.8, "idle": -0.3},
}
# Per role-class extra bend, same shape, sparser: only the
# entries that differ from the city-wide table.
_SCHEDULE_CLASS = {
    "trade":    {"morning": {"work": 0.9, "browse": 0.6},
                 "afternoon": {"browse": 0.6}},
    "watch":    {"night": {"social": 3.0, "errand": 2.6}},
    "works":    {"morning": {"errand": 0.6, "work": 0.4}},
    "care":     {"morning": {"social": 0.7}, "afternoon": {"social": 0.5}},
    "student":  {"morning": {"research": 1.4, "social": 0.5},
                 "afternoon": {"browse": 0.5}},
    "creative": {"afternoon": {"idle": 0.8}, "evening": {"idle": 0.5}},
    "elder":    {"morning": {"idle": 0.7, "social": 0.5},
                 "evening": {"social": 0.4}},
}
# The pull of home itself, by daypart (guards on night watch are
# exempt — see _schedule_bonus). Night is deliberately far
# stronger than any single other term in _choose: after dark,
# home is where citizens should end up.
_SCHEDULE_HOME = {"morning": -1.5, "afternoon": -1.0,
                  "evening": 2.5, "night": 7.0}
_HOME_SYNTH_NAME = "home"          # synthetic per-NPC home anchor name

# The grounding rule every location-fed prompt carries (P11):
# a citizen speaks as someone standing where they actually are.
LOCATION_GROUNDING = (
    "Speak as someone standing where you are. Only call a shop, "
    "stall or home yours if your situation says you are at it; "
    "if you are on the road, on the pavement or out on open "
    "ground, you are passing through — talk like a traveller, "
    "not like a host receiving guests.")

_ANCHOR_WHERE = {"browse": "by the market stalls",
                 "social": "on the open plaza ground",
                 "idle": "on the green",
                 "errand": "by the depot ground",
                 "research": "at the research terminal",
                 "front": "outside the homes there"}
# Ownership claims in situation_text ("at X", "your workplace",
# "at home") are only honest within this distance of the anchor:
# a citizen 10m away on the grass is NEAR their shop, not in it.
# Between this radius and the ~12m naming range the situation
# says "near <name>" and lets the surface speak for itself.
_AT_CLAIM_RADIUS = 5.0


class ActivityPlanner:
    """Per-NPC itinerary state machines: travel -> dwell -> (social) ->
    travel ... Utility-scored activity choice; emits additive protocol
    events ({"type": "intent", ...} / {"type": "social", ...}) that the
    world client turns into locomotion. The planner never moves anyone
    itself — positions stay owned by the world — it reads reported
    positions to detect arrivals and dwell expiry."""

    ARRIVE_RADIUS = 1.6      # m: close enough to the target to start dwelling
    SOCIAL_RANGE = 4.0       # m: partner must stay within this while chatting
    # Ambient LLM chatter discipline (sim seconds). In live mode EVERY
    # social pairing asks the generator for its exchange — there is no
    # tier gate and no canned fallback (a failed generation is silence,
    # never a scripted line). The discipline that keeps a free-tier
    # brain affordable: a pair that just started a generated conversation
    # waits this long before a NEW pairing with the same partner may
    # generate again (continuations of a live conversation are exempt),
    # generations city-wide start at most this often, at most this many
    # run at once, and a run of consecutive failures trips a breaker.
    GEN_PAIR_COOLDOWN = 90.0
    GEN_GLOBAL_GAP = 3.0
    GEN_MAX_INFLIGHT = 2
    GEN_RETRY_DELAY = 5.0        # one retry, witnessed pairs only
    GEN_BREAKER_FAILURES = 3     # consecutive failures before the pause
    GEN_BREAKER_PAUSE = 60.0
    # Ongoing witnessed conversation: after an exchange lands, the pair
    # keeps talking — a continuation is requested every cadence tick,
    # and the pairing itself may stretch past its rolled duration up to
    # the cap so a good chat is not cut mid-thread.
    CONT_CADENCE = (15.0, 30.0)
    CONT_CAP_S = 90.0
    GEN_HISTORY_MAX = 12         # lines of transcript handed back as context
    # Conversation-born plans (see _maybe_form_plan): a pair that agrees
    # in a generated exchange to meet later waits far longer before it
    # can plan again; the rendezvous itself falls due a few sim-minutes
    # after the chat and stays completable for a generous window.
    PLAN_PAIR_COOLDOWN = 600.0
    PLAN_DELAY_RANGE = (120.0, 300.0)
    PLAN_EXPIRY_S = 600.0
    PLAN_MEET_RADIUS = 4.0   # m from the anchor center: "arrived"

    def __init__(self, npcs: dict[str, NPC], spawns: dict, seed: int = 11,
                 conversation_generator=None):
        """`conversation_generator`: optional callable producing an
        LLM-written exchange for a social pairing. Live-mode contract:
        EVERY pairing asks for a generated exchange (no tier gate);
        a hook that fails (None / exception / invalid lines) yields
        silence for that pairing — the canned line set is used only
        when no generator is wired at all (demo/mock worlds).

        Call shape: (card_a, card_b, anchor_name) -> exchange | None,
        or, for hooks that accept a fourth argument, (card_a, card_b,
        anchor_name, history) where history is the [(speaker, line)]
        transcript of this conversation so far (empty for the first
        exchange, non-empty for continuations); hooks accepting a
        fifth argument additionally receive `situations` — {name:
        location-context string} for both speakers (P11). The exchange is either
        a plain list of (speaker_name, line) pairs (the original
        contract), or a tuple (lines, plan_anchor_name | None) when
        the pair also agreed to meet later at a named anchor — see
        _maybe_form_plan. It is always invoked on a background thread."""
        self.npcs = npcs
        self.anchors = build_city_anchors()
        # City growth (P9): how many expansion rings the world has
        # taken, and the district anchor names that came with them.
        # The anchors themselves are rebuilt deterministically from
        # the count (district_anchor_specs) — never persisted.
        self.expansions = 0
        self._district_names: set[str] = set()
        self.rng = random.Random(seed)
        self.conversation_generator = conversation_generator
        self._gen_takes_history = _hook_takes_history(conversation_generator)
        self._gen_arity = _hook_arity(conversation_generator)
        self._gen_results: queue.Queue = queue.Queue()
        self._gen_pair_last: dict[tuple, float] = {}
        self._gen_last_at: float = -1e9
        # Live conversation records, keyed by the sorted name pair:
        # {key, a, b, anchor, history, started, witnessed, state,
        # next_at, attempt}. state walks pending -> awaiting -> live
        # (continuations scheduled) with retry/dead offshoots. A record
        # lives only while its pair stays social together; it is live
        # situation, never part of a souls snapshot.
        self._conversations: dict[tuple, dict] = {}
        self._gen_inflight = 0
        self._gen_lock = threading.Lock()
        self._gen_fail_streak = 0
        self._gen_paused_until: float = -1e9
        # Pending conversation-born plans: [{a, b, anchor, due, expires,
        # active}]. A plan is KNOWLEDGE, not situation: the memory
        # entries written at formation (kind="plan") are what a souls
        # snapshot carries, so this live queue is deliberately NOT part
        # of the snapshot — a souls load never resurrects it, and
        # reset_npc() drops queued plans for the restored NPC.
        self.pending_plans: list[dict] = []
        self._plan_pair_last: dict[tuple, float] = {}
        # RTS labor (P2, see work.py): the task queue. `resources` is
        # the bridge's ResourceWorld, wired in by WorldBridge — a
        # planner without one (unit tests, headless sims) simply runs
        # no economy: no foreman, no orders, no work. Tasks are live
        # situation like plans: never part of a souls snapshot.
        self.resources = None
        # Building (P3, see buildings.py): the ConstructionWorld,
        # wired in by WorldBridge next to `resources`. A planner
        # without one runs no construction: no build tasks, and the
        # foreman staffs no sites.
        self.construction = None
        # Terraforming (P4, see terrain.py): the TerrainWorld,
        # wired in by WorldBridge the same way. Crew jobs live
        # there; the planner only staffs and works them (the
        # foreman never creates one). Movement stays 2D — a crew
        # job's target is an (x, z) patch; heights are world state.
        self.terrain = None
        # Civics (P5, see civics.py): the CivicsEngine, wired in by
        # WorldBridge the same way. It reads the same worlds off
        # this planner and ticks at the tail of tick() below —
        # proposals, debate and the all-citizen vote ride the world
        # loop like everything else. A planner without one (unit
        # tests, headless sims) simply runs no democracy.
        self.civics = None
        # Research + innovation (P6, see research.py /
        # innovations.py): the RecipeBook the task machinery reads
        # (seed recipes plus whatever adopted innovations register;
        # _book() falls back to a seed-only book when unwired), the
        # ResearchEngine (terminal sessions land through its tick)
        # and the InnovationEngine (synthesis results, the market).
        # All wired by WorldBridge; a planner without them runs the
        # P2-P5 world unchanged.
        self.recipes = None
        self.research = None
        self.innovations = None
        # World-data persistence (P7, see persistence.py): the
        # PersistenceService, wired by build_world (and set on the
        # bridge). It ticks at the tail of tick() below — snapshot
        # saves and repo pushes ride the world loop like everything
        # else, with all network work on its own background thread.
        self.persistence = None
        # Living-world quality (P8, see weather.py / designs.py):
        # the WeatherEngine (clock + sky) and the DesignEngine
        # (brain-designed styles, interiors and assets), wired in
        # by WorldBridge next to the other worlds. They tick at the
        # tail of tick() below, before persistence — a design that
        # lands on this tick is in this tick's snapshot. A planner
        # without them runs the P7 world unchanged.
        self.weather = None
        self.designs = None
        # Walk grid (P10, see walkgrid.py): the analytic
        # walkability grid over the live world, wired in by
        # WorldBridge next to the other worlds. When a travel
        # target's straight line crosses blocked cells, the
        # planner attaches the grid's route to the travel intent
        # (see _route_for); a planner without one walks citizens
        # direct, exactly as before. The grid is derived state —
        # never part of a souls snapshot.
        self.walkgrid = None
        # Gathering drives (P5): temporary foreman target raises
        # from adopted civic proposals — {resource: {"extra": int,
        # "until": sim_now}}. Expired boosts simply stop counting.
        self.target_boosts: dict[str, dict] = {}
        self.tasks: list[dict] = []
        self._task_by_npc: dict[str, dict] = {}
        self._task_seq = 0
        self._foreman_next: float | None = None
        self._tick_dt = 0.0      # sim seconds covered by the last tick
        self.directed_until: dict[str, float] = {}
        self.state: dict[str, dict] = {}
        self.time_in = {"dwell": 0.0, "travel": 0.0}
        self._last_tick: float | None = None
        self.spawns = {name: (float(p[0]), float(p[1]))
                       for name, p in spawns.items()}
        for name in npcs:
            self.state[name] = self._fresh_state(name)

    def _book(self):
        """The RecipeBook this planner works from: the bridge's
        book (seed + registered innovations) when wired, else the
        shared seed-only book — the P2 tables, unchanged."""
        return self.recipes if self.recipes is not None \
            else work_mod.default_book()

    def _fresh_state(self, name: str) -> dict:
        """A brand-new itinerary for one NPC: dwelling at its spawn, no
        anchor, no partner, no cooldowns. Soul loads in 'knowledge' mode
        reset to exactly this — who the NPC is survives, where they were
        and what they were doing does not."""
        spawn = self.spawns.get(name, (0.0, 0.0))
        if self.walkgrid is not None:
            # Respawn snapping (P10): a spawn point that lands on
            # a blocked cell (a citizen built over it, the land
            # sank) respawns at the nearest walkable cell instead
            # of inside a wall.
            snapped = self.walkgrid.snap_to_walkable(spawn[0], spawn[1])
            if snapped is not None:
                spawn = snapped
        return {
            "state": "dwell", "anchor": None, "kind": "idle",
            "task": None,
            "workplace": workplace_anchor(self.npcs[name].card, spawn,
                                          self.anchors),
            # Daily schedule (P11): the synthetic home anchor is
            # built lazily by _home_anchor (a real "House N" anchor
            # is adopted instead whenever one stands near spawn);
            # at_home marks a dwell that is a night at home, which
            # earns the long sleep dwell.
            "home_synth": None, "at_home": False,
            "tx": spawn[0], "tz": spawn[1], "until": 0.0,
            "route": None,
            "partner": None, "last_social": -1e9, "cooldown_until": 0.0,
            "speed": self.rng.uniform(0.9, 1.4),
            "ox": self.rng.uniform(-1.6, 1.6),
            "oz": self.rng.uniform(-1.6, 1.6),
            "initialized": False,
        }

    def reset_npc(self, name: str) -> None:
        """Drop all itinerary state for one NPC back to _fresh_state.
        Also drops any pending conversation plan involving the NPC:
        plans are live situation, and a souls load must not resurrect
        (or keep) the pending-plan queue — the plan *memory* is the
        knowledge that survives, not the appointment itself."""
        if name in self.state:
            self.state[name] = self._fresh_state(name)
            self.directed_until.pop(name, None)
        self.pending_plans = [p for p in self.pending_plans
                              if name not in (p["a"], p["b"])]
        # Live conversation records are situation too: a restored NPC
        # is not mid-chat with anyone.
        self._conversations = {k: c for k, c in self._conversations.items()
                               if name not in k}
        # Work assignments are situation too: a restored NPC is not
        # mid-task (the task is cancelled, not carried over).
        task = self._task_by_npc.pop(name, None)
        if task is not None:
            task["state"] = "cancelled"

    # -- city growth (P9) --------------------------------------------------
    def set_expansions(self, expansions: int) -> None:
        """Rebuild the district portion of the anchor set for a
        city grown by `expansions` rings: the previous districts
        (tracked by name) are dropped and the deterministic set for
        the count is appended. Founding anchors and building
        anchors are never touched. Idempotent — the bridge calls
        this after every terrain restore and every expansion, so
        the planner's map always matches the terrain's bound."""
        expansions = max(0, min(int(expansions), MAX_EXPANSIONS))
        if self._district_names:
            self.anchors = [a for a in self.anchors
                            if a.name not in self._district_names]
            self._district_names = set()
        fresh = district_anchors(expansions)
        self.anchors.extend(fresh)
        self._district_names = {a.name for a in fresh}
        self.expansions = expansions

    def apply_expansion(self):
        """Grow the world one ring through this planner's terrain
        and rebuild the district anchors to match. Returns the
        terrain's expansion info {"bound", "ring", "previous"} —
        or None when there is no terrain wired or the land is
        already at the engine cap (nothing changes then)."""
        if self.terrain is None:
            return None
        info = self.terrain.expand()
        if info is None:
            return None
        self.set_expansions(self.terrain.expansions)
        return info

    def situation_of(self, name: str, position) -> dict:
        """The NPC's current situation (where + doing what), for soul
        snapshots: position, current anchor name, current activity kind."""
        st = self.state.get(name) or {}
        anchor = st.get("anchor")
        return {"position": [float(position[0]), float(position[1])],
                "anchor": anchor.name if anchor is not None else None,
                "activity": st.get("kind"),
                "state": st.get("state")}

    def apply_situation(self, name: str, situation: dict,
                        now: float) -> None:
        """Restore a saved anchor/activity onto (fresh) planner state —
        soul load, 'full' mode. The NPC resumes dwelling at the saved
        anchor for a normal dwell span. An unknown anchor name is
        ignored: the planner then settles at the anchor nearest the
        restored position on its first tick."""
        st = self.state.get(name)
        if st is None:
            return
        anchor = next((a for a in self.anchors
                       if a.name == situation.get("anchor")), None)
        if anchor is None:
            return
        kind = situation.get("activity")
        if kind not in DWELL_RANGES:
            kind = _activity_kind(self.npcs[name].card, anchor,
                                  st["workplace"])
        st["anchor"] = anchor
        st["kind"] = kind
        st["state"] = "dwell"
        st["tx"], st["tz"] = anchor.x + st["ox"], anchor.z + st["oz"]
        st["until"] = now + self._dwell_for(kind)
        st["partner"] = None
        st["initialized"] = True

    # -- brain-directed movement (move_to / follow from the LLM) -------------
    def note_directed(self, name: str, seconds: float | None, now: float) -> None:
        """The brain ordered this NPC somewhere; the planner stands down
        (45s for a move_to, indefinitely for follow, brief for interacts)."""
        if name in self.state:
            self.directed_until[name] = (
                float("inf") if seconds is None else now + seconds)

    def _directed(self, name: str, now: float) -> bool:
        return now < self.directed_until.get(name, 0.0)

    # -- daily schedule + location context (P11) ------------------------------
    def _daypart(self):
        """The current daypart name from the attached weather
        engine (weather.daypart: morning / afternoon / evening /
        night), or None when no sky is wired — schedules are then
        a no-op and the planner behaves exactly as before P11."""
        w = self.weather
        if w is None:
            return None
        try:
            from .weather import daypart

            return daypart(float(w.hour))
        except Exception:
            return None

    def _home_anchor(self, name: str, st: dict) -> ActivityAnchor:
        """This citizen's home: the "House N" anchor nearest their
        spawn when the city has built one there, else a synthetic
        anchor standing on the spawn point itself ("home"). The
        synthetic anchor is per-NPC state, never added to the
        shared anchor list — nobody else itineraries to it."""
        spawn = self.spawns.get(name, (0.0, 0.0))
        houses = [a for a in self.anchors if a.name.startswith("House ")]
        if houses:
            return min(houses, key=lambda a: _anchor_dist(a, spawn))
        if st.get("home_synth") is None:
            st["home_synth"] = ActivityAnchor(
                _HOME_SYNTH_NAME, "front", spawn[0], spawn[1])
        return st["home_synth"]

    def _schedule_bonus(self, card, part: str, kind: str,
                        anchor, st: dict, home) -> float:
        """The schedule's additive bend on one candidate anchor:
        the city-wide kind table for the daypart, the role-class
        extra, the pull of home, and a small extra discouragement
        for workplaces after dark. Guards on night watch are
        exempt from the pull of home — their class table bends
        them to plaza/errand anchors instead."""
        bonus = _SCHEDULE_KIND.get(part, {}).get(kind, 0.0)
        cls = _ROLE_CLASS.get(card.role, "resident")
        bonus += _SCHEDULE_CLASS.get(cls, {}).get(part, {}).get(kind, 0.0)
        if anchor is home:
            if not (cls == "watch" and part == "night"):
                bonus += _SCHEDULE_HOME.get(part, 0.0)
        elif anchor is st["workplace"] and part == "night":
            bonus -= 2.0
        return bonus

    def _surface_phrase(self, pos) -> str | None:
        """The ground under `pos` as a short locative phrase —
        "on the grass", "in the middle of the road", "on the
        pavement", "by the pond", "right up against a building's
        walls", "on rough ground" — from the same terrain /
        walk-grid facts the no-anchor branch of situation_text
        narrates in full sentences. None when neither terrain
        nor walk grid is wired."""
        if self.terrain is not None:
            try:
                from .terrain import WATER_Y

                if self.terrain.sample(pos[0], pos[1]) < WATER_Y + 0.15:
                    return "by the pond"
            except Exception:
                pass
        if self.walkgrid is None:
            return None
        cost = self.walkgrid.cost_at(pos[0], pos[1])
        if cost == 5:
            return "in the middle of the road"
        if cost == 1:
            return "on the pavement"
        if cost == 0:
            try:
                for fx, fz, hw, hd in self.walkgrid._footprints():
                    if abs(pos[0] - fx) <= hw and abs(pos[1] - fz) <= hd:
                        return "right up against a building's walls"
            except Exception:
                pass
            return "on rough ground"
        return "on the grass"

    def situation_text(self, name: str, pos) -> str | None:
        """Where this citizen factually is right now, as one or
        two plain sentences for dialogue prompts (P11): the clock
        when a sky is wired, then the nearest anchor within ~12m
        — claimed ("at Market Row East, your workplace") only
        within _AT_CLAIM_RADIUS of it, otherwise "near <name>"
        plus the surface under their feet — else that surface
        narrated in full (road / pavement / open ground /
        against a building), the water's edge from the terrain,
        and the district when a district anchor is the nearest
        landmark. None when the position is unknown."""
        if pos is None:
            return None
        npc = self.npcs.get(name)
        if npc is None:
            return None
        st = self.state.get(name) or {}
        head = ""
        if self.weather is not None:
            try:
                from .weather import clock_text, daypart

                head = (f"It is {clock_text(float(self.weather.hour))} "
                        f"({daypart(float(self.weather.hour))}). ")
            except Exception:
                head = ""
        home = self._home_anchor(name, st) if st else None
        cands = list(self.anchors)
        if home is not None and all(home is not a for a in cands):
            cands.append(home)
        near, near_d = None, float("inf")
        for a in cands:
            d = _anchor_dist(a, pos)
            if d < near_d:
                near, near_d = a, d
        if near is not None and near_d <= 12.0:
            surface = self._surface_phrase(pos)
            if near_d > _AT_CLAIM_RADIUS:
                # Near, not at: name the place, claim nothing —
                # the surface says where they really stand.
                clause = f", {surface}" if surface else ""
                if home is not None and near is home:
                    label = near.name if near.name.startswith("House ") \
                        else "home"
                    return head + f"You are near {label}{clause}."
                return head + f"You are near {near.name}{clause}."
            # At the anchor: ownership claims are honest here.
            # Front anchors sit outside the door by construction,
            # so for them the surface replaces the generic
            # "outside the homes there": a citizen at their front
            # is outdoors, on whatever ground is actually there.
            outdoors = ""
            if near.kind == "front" and surface:
                if surface.startswith("right up"):
                    outdoors = f" — outdoors, {surface}"
                else:
                    outdoors = f" — outdoors, {surface} in front of it"
            if home is not None and near is home:
                if near.name.startswith("House "):
                    return head + f"You are at home ({near.name})" \
                                  f"{outdoors}."
                return head + "You are at home, in your own corner " \
                              f"of the district{outdoors}."
            where = _ANCHOR_WHERE.get(near.kind, "")
            if near.kind == "front":
                where = ""
            if st and near is st.get("workplace"):
                return head + f"You are at {near.name}, your " \
                              f"workplace{(' — ' + where) if where else ''}" \
                              f"{outdoors}."
            return head + f"You are at {near.name}" \
                          f"{(', ' + where) if where else ''}{outdoors}."
        # No anchor near: the ground itself, then the district.
        district = ""
        dist_a, dist_d = None, float("inf")
        for a in self.anchors:
            if a.name in self._district_names:
                d = _anchor_dist(a, pos)
                if d < dist_d:
                    dist_a, dist_d = a, d
        if dist_a is not None and dist_d <= 60.0:
            district = f" You are in the {dist_a.name} district."
        if self.terrain is not None:
            try:
                from .terrain import WATER_Y

                # Waterside = at (or within a step of) the water
                # surface itself. The founding plain sits at 0.0,
                # only 0.35 above WATER_Y, so the band must hug
                # the waterline: a dug pond and its low banks land
                # inside it, dry flat ground does not.
                if self.terrain.sample(pos[0], pos[1]) < WATER_Y + 0.15:
                    return head + "You are by the pond, at the " \
                                  "water's edge." + district
            except Exception:
                pass
        cost = self.walkgrid.cost_at(pos[0], pos[1]) \
            if self.walkgrid is not None else None
        if cost == 5:
            return head + "You are in the middle of the road on a " \
                          "main street — not at any shop or home." \
                          + district
        if cost == 1:
            return head + "You are on the pavement by the road — " \
                          "not at any shop or home." + district
        if cost == 0:
            inside = False
            try:
                for fx, fz, hw, hd in self.walkgrid._footprints():
                    if abs(pos[0] - fx) <= hw and abs(pos[1] - fz) <= hd:
                        inside = True
                        break
            except Exception:
                inside = False
            if inside:
                return head + "You are at a building, right up " \
                              "against its walls." + district
            return head + "You are on rough ground off the " \
                          "paths." + district
        return head + "You are on open grass, away from the " \
                      "streets." + district

    # -- utility choice ---------------------------------------------------------
    def _choose(self, name: str, now: float, pos) -> tuple[ActivityAnchor, str]:
        npc = self.npcs[name]
        card, st = npc.card, self.state[name]
        part = self._daypart()
        home = self._home_anchor(name, st) if part is not None else None
        cands = list(self.anchors)
        if home is not None and all(home is not a for a in cands):
            cands.append(home)
        best, best_score = None, float("-inf")
        for anchor in cands:
            kind = _activity_kind(card, anchor, st["workplace"])
            score = self.rng.uniform(0.0, 0.6)
            if anchor is st["workplace"]:
                score += 3.0        # vendors/workers live at their workplace
            score += _ROLE_AFFINITY.get(card.role, {}).get(kind, 0.0)
            for trait in card.traits:
                score += _TRAIT_AFFINITY.get(trait, {}).get(kind, 0.0)
            score -= 0.015 * _anchor_dist(anchor, pos)
            if kind == "social":    # the longer since the last chat, the keener
                score += min(2.0, max(0.0, (now - st["last_social"]) / 90.0))
            if st["anchor"] is anchor:
                score -= 1.0        # variety: don't just re-pick the same spot
            if part is not None:
                score += self._schedule_bonus(card, part, kind,
                                              anchor, st, home)
            if score > best_score:
                best, best_score = (anchor, kind), score
        st["at_home"] = (home is not None and best[0] is home
                         and part in ("evening", "night"))
        return best

    def _dwell_for(self, kind: str) -> float:
        lo, hi = DWELL_RANGES.get(kind, (30.0, 90.0))
        return self.rng.uniform(lo, hi)

    def _intent(self, name: str, st: dict) -> dict:
        ev = {"type": "intent", "npc": name, "state": st["state"],
              "kind": st["kind"], "speed": round(st["speed"], 2)}
        if st["anchor"] is not None:
            ev["anchor"] = st["anchor"].name
        if st["state"] == "travel":
            ev["target"] = [round(st["tx"], 1), round(st["tz"], 1)]
            if st.get("route"):
                # Walk grid (P10): the waypoints around whatever
                # blocks the straight line, computed when this
                # travel leg began (see _route_for). Additive —
                # clients without route support walk direct.
                ev["route"] = st["route"]
        if st["state"] == "social":
            ev["partner"] = st["partner"]
        return ev

    def _route_for(self, pos, tx: float, tz: float):
        """The walk-grid route from `pos` to a fresh travel
        target, or None when there is nothing to route around:
        no grid wired, the straight line stays on walkable
        cells, or no route exists (the citizen then walks direct
        and the arrival rules sort it out, as they always have).
        Server-side arrival logic never reads the route — it is
        guidance for the client's locomotion only."""
        wg = self.walkgrid
        if wg is None or pos is None:
            return None
        try:
            if wg.line_clear(pos[0], pos[1], tx, tz):
                return None
            return wg.find_route(pos[0], pos[1], tx, tz)
        except Exception:
            return None

    def _start_activity(self, name: str, now: float, pos,
                        events: list[dict]) -> None:
        st = self.state[name]
        anchor, kind = self._choose(name, now, pos)
        st["anchor"], st["kind"] = anchor, kind
        st["tx"], st["tz"] = anchor.x + st["ox"], anchor.z + st["oz"]
        if ((st["tx"] - pos[0]) ** 2 + (st["tz"] - pos[1]) ** 2) ** 0.5 \
                <= self.ARRIVE_RADIUS:
            st["state"] = "dwell"
            st["until"] = now + self._dwell_for(
                "home" if st.get("at_home") else kind)
            st["route"] = None
        else:
            st["state"] = "travel"
            st["route"] = self._route_for(pos, st["tx"], st["tz"])
        events.append(self._intent(name, st))

    # -- main tick ---------------------------------------------------------------
    def tick(self, now: float, positions: dict, engaged: set,
             tiers: dict) -> list[dict]:
        events: list[dict] = []
        if self._last_tick is not None:
            dt = max(0.0, now - self._last_tick)
            for st in self.state.values():
                bucket = "travel" if st["state"] == "travel" else "dwell"
                self.time_in[bucket] += dt
        else:
            dt = 0.0
        self._tick_dt = dt
        self._last_tick = now

        # Ambient LLM exchanges that finished since the last tick:
        # deliver only if the pair is still socializing together at the
        # same anchor; otherwise drop them. Then service the live
        # conversation records (deferred starts, retries, continuations).
        events.extend(self._drain_generated(now, tiers))
        self._service_conversations(now, tiers)

        # Conversation-born plans: expire stale ones, and collect the
        # due ones into a drive map — a due plan outranks the normal
        # utility choice until it completes or expires (but never a
        # conversation lock or a brain-directed order).
        drive: dict[str, dict] = {}
        for plan in list(self.pending_plans):
            if now > plan["expires"]:
                self.pending_plans.remove(plan)
            elif now >= plan["due"]:
                plan["active"] = True
                drive[plan["a"]] = plan
                drive[plan["b"]] = plan

        # RTS labor (P2): advance the resource world's regrowth to
        # sim time, then let the foreman staff the baseline stocks.
        # Assigned citizens are driven inside the loop below, under
        # their plan — and never over the conversation lock, which
        # the loop's engaged branch handles before either.
        if self.resources is not None:
            self.resources.tick(now)
            self._run_foreman(now, positions, engaged, events)

        for name, st in self.state.items():
            pos = positions.get(name)
            if name in engaged:
                if st["state"] != "engaged":
                    st["state"] = "engaged"
                    st["partner"] = None
                    events.append(self._intent(name, st))
                continue
            if st["state"] == "engaged":      # lock just released: re-plan
                st["state"] = "dwell"
                st["until"] = now + 5.0
            if pos is None or self._directed(name, now):
                continue
            plan = drive.get(name)
            if plan is not None:
                self._drive_plan(name, plan, now, pos, events)
                continue
            task = self._task_by_npc.get(name)
            if task is not None:
                self._drive_task(name, task, now, pos, events)
                continue
            if not st["initialized"]:
                st["initialized"] = True
                nearest = min(self.anchors,
                              key=lambda a: _anchor_dist(a, pos))
                st["anchor"] = nearest
                st["kind"] = _activity_kind(self.npcs[name].card, nearest,
                                            st["workplace"])
                st["state"] = "dwell"
                st["until"] = now + self.rng.uniform(8.0, 40.0)
                events.append(self._intent(name, st))
                continue
            if st["state"] == "travel":
                d = ((st["tx"] - pos[0]) ** 2 + (st["tz"] - pos[1]) ** 2) ** 0.5
                if d <= self.ARRIVE_RADIUS:
                    st["state"] = "dwell"
                    st["route"] = None
                    st["until"] = now + self._dwell_for(
                        "home" if st.get("at_home") else st["kind"])
                    events.append(self._intent(name, st))
            elif st["state"] == "social":
                partner = self.state.get(st["partner"] or "")
                ppos = positions.get(st["partner"] or "")
                if (now >= st["until"] or partner is None
                        or partner["state"] != "social" or ppos is None
                        or ((ppos[0] - pos[0]) ** 2
                            + (ppos[1] - pos[1]) ** 2) ** 0.5 > self.SOCIAL_RANGE):
                    st["state"] = "dwell"
                    st["partner"] = None
                    st["until"] = now + self.rng.uniform(15.0, 45.0)
                    events.append(self._intent(name, st))
            elif st["state"] == "dwell" and now >= st["until"]:
                self._start_activity(name, now, pos, events)

        self._complete_plans(now, positions, engaged, events)
        self._pair_social(now, positions, engaged, tiers, events)

        # Civics (P5): proposals, debate and the all-citizen vote
        # advance on the same world tick; the engine's events
        # (proposal / debate / ballot / civic_result, plus the
        # normal build/terraform events of an adopted action) join
        # the stream here.
        if self.civics is not None:
            events.extend(self.civics.tick(now, positions))
        # Research + innovation (P6): terminal sessions land
        # (insight / no_connection events), citizens settle at the
        # terminal, adopted discoveries finish synthesizing
        # (innovation events) and the market trades — all on the
        # same tick tail.
        if self.research is not None:
            events.extend(self.research.tick(now, positions))
        if self.innovations is not None:
            events.extend(self.innovations.tick(now))
        # Living-world quality (P8): the sky advances (weather
        # events on change) and finished design work lands
        # (design events) on the same tick tail — before
        # persistence, so a snapshot assembled this tick carries
        # them.
        if self.weather is not None:
            events.extend(self.weather.tick(now))
        if self.designs is not None:
            events.extend(self.designs.tick(now))
        # World-data persistence (P7): interval snapshot saves +
        # repo pushes advance on the same tick tail. The service
        # never raises and never blocks (network work is threaded);
        # its events are quiet status notes for the god panel.
        if self.persistence is not None:
            events.extend(self.persistence.tick(now))
        return events

    # -- conversation-born plans ------------------------------------------------
    def _drive_plan(self, name: str, plan: dict, now: float, pos,
                    events: list[dict]) -> None:
        """Steer one NPC toward its rendezvous: travel to the plan
        anchor, then wait there (dwell, refreshed) for the other."""
        st = self.state[name]
        anchor = plan["anchor"]
        st["initialized"] = True
        st["at_home"] = False
        st["anchor"] = anchor
        st["kind"] = "social"
        st["tx"], st["tz"] = anchor.x + st["ox"], anchor.z + st["oz"]
        d = ((st["tx"] - pos[0]) ** 2 + (st["tz"] - pos[1]) ** 2) ** 0.5
        if d <= self.ARRIVE_RADIUS:
            if st["state"] != "dwell" or st["partner"] is not None:
                st["state"] = "dwell"
                st["partner"] = None
                st["route"] = None
                events.append(self._intent(name, st))
            st["until"] = now + 30.0    # wait for the other to arrive
        elif st["state"] != "travel" or st["partner"] is not None:
            st["state"] = "travel"
            st["partner"] = None
            st["route"] = self._route_for(pos, st["tx"], st["tz"])
            events.append(self._intent(name, st))

    def _complete_plans(self, now: float, positions: dict, engaged: set,
                        events: list[dict]) -> None:
        """A plan completes when both parties are at the anchor: drop
        it and leave both dwelling there with their social cooldown
        cleared, so the ordinary pairing machinery picks them up (and
        any follow-up exchange can recall the plan from memory). A
        participant mid-conversation with the player delays completion
        — the plan never interrupts the conversation lock."""
        for plan in list(self.pending_plans):
            if not plan["active"]:
                continue
            if plan["a"] in engaged or plan["b"] in engaged:
                continue
            anchor = plan["anchor"]
            pa, pb = positions.get(plan["a"]), positions.get(plan["b"])
            if pa is None or pb is None:
                continue
            if (_anchor_dist(anchor, pa) > self.PLAN_MEET_RADIUS
                    or _anchor_dist(anchor, pb) > self.PLAN_MEET_RADIUS):
                continue
            self.pending_plans.remove(plan)
            for name in (plan["a"], plan["b"]):
                st = self.state[name]
                st["state"] = "dwell"
                st["anchor"] = anchor
                st["kind"] = "social"
                st["partner"] = None
                st["cooldown_until"] = 0.0
                st["until"] = now + self._dwell_for("social")
                events.append(self._intent(name, st))

    def _maybe_form_plan(self, a_name: str, b_name: str, plan_anchor,
                         now: float, events: list[dict]) -> None:
        """A generated exchange ended with the pair agreeing to meet
        later. Validate and register the plan: the anchor must be a
        real place (and not the spot they're already standing at),
        neither NPC may already hold a plan, and the pair's plan
        cooldown must have passed. Both NPCs remember the plan
        (kind="plan") — that memory is the knowledge a souls snapshot
        carries; the queue entry itself is live situation only."""
        if not isinstance(plan_anchor, str):
            return
        anchor = next((a for a in self.anchors if a.name == plan_anchor),
                      None)
        if anchor is None:
            return
        current = self.state[a_name].get("anchor")
        if current is not None and current.name == anchor.name:
            return            # already here together: nothing to plan
        if any(a_name in (p["a"], p["b"]) or b_name in (p["a"], p["b"])
               for p in self.pending_plans):
            return            # one pending plan per NPC
        key = (a_name, b_name) if a_name < b_name else (b_name, a_name)
        if now - self._plan_pair_last.get(key, -1e9) < self.PLAN_PAIR_COOLDOWN:
            return
        self._plan_pair_last[key] = now
        due = now + self.rng.uniform(*self.PLAN_DELAY_RANGE)
        self.pending_plans.append({
            "a": a_name, "b": b_name, "anchor": anchor, "due": due,
            "expires": due + self.PLAN_EXPIRY_S, "active": False})
        for name, other in ((a_name, b_name), (b_name, a_name)):
            self.npcs[name].witness(
                f"Planned to meet {other} at {anchor.name}.", kind="plan")
        events.append({"type": "plan", "a": a_name, "b": b_name,
                       "where": anchor.name})

    # -- RTS labor: task queue, foreman, work execution (P2) ----------------
    # A task behaves like a plan: a strong per-NPC intent the tick
    # drives (travel to the target, dwell there working) that yields
    # to the conversation lock — the tick's engaged branch runs first,
    # so an engaged citizen never starts work, and one mid-task simply
    # pauses until released. Definitions (stations, recipes, targets,
    # affinities) live in work.py; harvesting itself is
    # ResourceWorld.try_harvest, the single harvest path.
    FOREMAN_FIRST_DELAY = 10.0   # sim-s after world start before the
                                 # foreman first staffs the baselines —
                                 # a fresh world settles before it works
    FOREMAN_INTERVAL = 15.0      # sim-s between foreman stock checks

    @staticmethod
    def task_event(task: dict, state: str, **extra) -> dict:
        ev = {"type": "task", "task": task["id"],
              "task_type": task["type"], "npc": task["npc"],
              "target": task["target"], "state": state}
        ev.update(extra)
        return ev

    def _public_task(self, task: dict) -> dict:
        place = task["target"]
        if task["target_kind"] == "node" and self.resources is not None:
            node = self.resources.nodes.get(task["target"])
            if node is not None:
                place = node.name
        elif task["target_kind"] == "site" \
                and self.construction is not None:
            site = self.construction.sites.get(task["target"])
            if site is not None:
                place = f"{site.name} site"
        elif task["target_kind"] == "patch" \
                and self.terrain is not None:
            job = self.terrain.jobs.get(task["target"])
            if job is not None:
                place = f"the {job['op']} patch"
        else:
            place = self._book().station_name(
                task["target"], self.construction) or place
        return {"id": task["id"], "type": task["type"],
                "npc": task["npc"], "target": task["target"],
                "place": place,
                "state": task["state"], "phase": task["phase"]}

    def active_tasks(self) -> list[dict]:
        """Public view of the underway tasks (status/world payloads)."""
        return [self._public_task(t) for t in self.tasks
                if t["state"] == "active"]

    def boost_targets(self, resource: str, extra: int, duration: float,
                      now: float) -> None:
        """Raise a foreman stock target for a while (P5 gathering
        drives, adopted by vote). Stacking adds to a live boost;
        expiry is silent — the baseline simply resumes."""
        cur = self.target_boosts.get(resource)
        base = cur["extra"] if cur and now < cur["until"] else 0
        self.target_boosts[resource] = {
            "extra": base + int(extra), "until": now + float(duration)}

    def effective_targets(self, now: float | None = None) -> dict:
        """The foreman's live targets: work.py baselines plus any
        unexpired civic gathering-drive boosts."""
        if now is None:
            now = self._last_tick or 0.0
        targets = {"wood": work_mod.TARGET_WOOD,
                   "food": work_mod.TARGET_FOOD}
        for key, boost in self.target_boosts.items():
            if key in targets and now < boost["until"]:
                targets[key] += boost["extra"]
        return targets

    def work_summary(self) -> dict | None:
        """The work picture for digests (the interpreter): baseline
        targets, current stocks, who's working on what."""
        if self.resources is None:
            return None
        return {"targets": self.effective_targets(),
                "inventory": dict(self.resources.inventory),
                "tasks": self.active_tasks()}

    def assign_task(self, task_type: str, now: float, positions: dict,
                    npc_name: str | None = None,
                    engaged=frozenset(), events: list | None = None):
        """Create and assign a work task. Returns (task, None) on
        success or (None, human_reason). With `npc_name` the order is
        for that citizen specifically (an engaged citizen may be
        ordered — the task waits for the lock, see tick); without
        one, the best idle citizen is picked by role affinity +
        distance, skipping the engaged, the brain-directed and the
        already-tasked. `events`, when given, receives the
        "assigned" task event."""
        book = self._book()
        if task_type not in work_mod.TASK_TYPES \
                and not book.is_craft(task_type) \
                and not book.is_gather(task_type):
            return None, (f"Unknown task {task_type!r} — want one of: "
                          + ", ".join(book.known_task_types()) + ".")
        if self.resources is None:
            return None, "This world has no resource economy to work in."
        underway = [t for t in self.tasks if t["state"] == "active"]
        if len(underway) >= work_mod.MAX_ACTIVE_TASKS:
            return None, (f"The city already has "
                          f"{work_mod.MAX_ACTIVE_TASKS} tasks underway — "
                          f"let some finish first.")
        # The target comes from the RecipeBook (P6): a craft task —
        # seed or innovation-registered — works at its recipe's
        # station, which must actually be standing (a dynamic
        # station exists only once its building is finished); a
        # gather task works a node of its kinds.
        recipe = book.get(task_type)
        station_pos = None
        if recipe is not None:
            target_kind, target = "station", recipe["station"]
            station_pos = book.station_position(target,
                                                self.construction)
            if station_pos is None:
                return None, (
                    f"The {book.station_name(target) or target} "
                    f"isn't standing yet — the city has to build it "
                    f"before anyone can work there.")
        else:
            target_kind = "node"
            target = book.gather_kinds(task_type)
        nodes = []
        if target_kind == "node":
            kinds = target
            nodes = [n for n in self.resources.nodes.values()
                     if n.kind in kinds and n.harvestable]
            if not nodes:
                return None, ("Nothing ripe to harvest right now — "
                              "the nodes need time to regrow.")

        def target_dist(name) -> float:
            p = positions.get(name)
            if p is None:
                return 0.0
            if target_kind == "node":
                return min(((n.x - p[0]) ** 2 + (n.z - p[1]) ** 2) ** 0.5
                           for n in nodes)
            return ((station_pos[0] - p[0]) ** 2
                    + (station_pos[1] - p[1]) ** 2) ** 0.5

        if npc_name is not None:
            if npc_name not in self.npcs:
                return None, f"Unknown citizen {npc_name!r}."
            if npc_name in self._task_by_npc:
                busy = work_mod.TASK_LABELS[
                    self._task_by_npc[npc_name]["type"]]
                return None, f"{npc_name} is already busy: {busy}."
            name = npc_name
        else:
            best, best_score = None, float("-inf")
            for cand in self.npcs:
                if cand in self._task_by_npc or cand in engaged \
                        or self._directed(cand, now):
                    continue
                score = self.rng.uniform(0.0, 0.3)
                score += work_mod.WORK_AFFINITY.get(task_type, {}).get(
                    self.npcs[cand].card.role, 0.0)
                score -= 0.01 * target_dist(cand)
                if score > best_score:
                    best, best_score = cand, score
            if best is None:
                return None, "No idle citizen can take that on right now."
            name = best

        if target_kind == "node":
            p = positions.get(name)
            node = min(nodes, key=lambda n: (
                ((n.x - p[0]) ** 2 + (n.z - p[1]) ** 2) ** 0.5
                if p is not None else 0.0, n.id))
            target_id, tx, tz = node.id, node.x, node.z
            duration = work_mod.GATHER_SECONDS
        else:
            target_id, tx, tz = target, station_pos[0], station_pos[1]
            duration = recipe.get("work_s", work_mod.CRAFT_SECONDS)
        if self.innovations is not None \
                and self.innovations.tool_active():
            # A registered tool the city holds makes hands faster
            # (innovations.py documents the simplification).
            duration *= self.innovations.tool_factor
        self._task_seq += 1
        task = {"id": f"task-{self._task_seq}", "type": task_type,
                "npc": name, "target": target_id,
                "target_kind": target_kind, "tx": tx, "tz": tz,
                "duration": duration, "state": "active",
                "phase": "travel", "work_until": None, "created": now}
        self.tasks.append(task)
        self._task_by_npc[name] = task
        if events is not None:
            events.append(self.task_event(task, "assigned"))
        return task, None

    def assign_build(self, site_id: str, now: float, positions: dict,
                     npc_name: str | None = None,
                     engaged=frozenset(), events: list | None = None):
        """Create and assign a "build" task for one construction
        site (P3). Returns (task, None) or (None, human_reason).
        Mirrors assign_task's contract: a named citizen may be
        ordered even while engaged (the task then waits for the
        conversation lock, see tick); unnamed, the best idle citizen
        is picked by build affinity + distance to the site. One
        builder per site; the global task cap still binds."""
        if self.construction is None or self.resources is None:
            return None, "This world has no construction to build."
        site = self.construction.sites.get(site_id)
        if site is None:
            return None, f"Unknown construction site {site_id!r}."
        if site.complete:
            return None, f"The {site.name.lower()} is already built."
        for t in self.tasks:
            if t["state"] == "active" and t["target_kind"] == "site" \
                    and t["target"] == site_id:
                return None, (f"{t['npc']} is already building "
                              f"the {site.name.lower()}.")
        underway = [t for t in self.tasks if t["state"] == "active"]
        if len(underway) >= work_mod.MAX_ACTIVE_TASKS:
            return None, (f"The city already has "
                          f"{work_mod.MAX_ACTIVE_TASKS} tasks underway — "
                          f"let some finish first.")
        if npc_name is not None:
            if npc_name not in self.npcs:
                return None, f"Unknown citizen {npc_name!r}."
            if npc_name in self._task_by_npc:
                busy = work_mod.TASK_LABELS[
                    self._task_by_npc[npc_name]["type"]]
                return None, f"{npc_name} is already busy: {busy}."
            name = npc_name
        else:
            best, best_score = None, float("-inf")
            for cand in self.npcs:
                if cand in self._task_by_npc or cand in engaged \
                        or self._directed(cand, now):
                    continue
                score = self.rng.uniform(0.0, 0.3)
                score += work_mod.WORK_AFFINITY.get(
                    work_mod.BUILD, {}).get(self.npcs[cand].card.role, 0.0)
                p = positions.get(cand)
                if p is not None:
                    score -= 0.01 * ((site.x - p[0]) ** 2
                                     + (site.z - p[1]) ** 2) ** 0.5
                if score > best_score:
                    best, best_score = cand, score
            if best is None:
                return None, "No idle citizen can take that on right now."
            name = best
        self._task_seq += 1
        task = {"id": f"task-{self._task_seq}", "type": work_mod.BUILD,
                "npc": name, "target": site.id,
                "target_kind": "site", "tx": site.x, "tz": site.z,
                "duration": None, "state": "active",
                "phase": "travel", "work_until": None, "created": now}
        self.tasks.append(task)
        self._task_by_npc[name] = task
        if events is not None:
            events.append(self.task_event(task, "assigned"))
        return task, None

    def assign_terraform(self, job_id: str, now: float, positions: dict,
                         npc_name: str | None = None,
                         engaged=frozenset(), events: list | None = None):
        """Create and assign a "terraform" task for one crew job
        (P4). Returns (task, None) or (None, human_reason). Mirrors
        assign_build's contract exactly: a named citizen may be
        ordered even while engaged (the task then waits for the
        conversation lock); unnamed, the best idle citizen is
        picked by terraform affinity + distance to the patch. One
        citizen per job; the global task cap still binds."""
        if self.terrain is None:
            return None, "This world has no land to reshape."
        job = self.terrain.jobs.get(job_id)
        if job is None:
            return None, f"Unknown terraform job {job_id!r}."
        for t in self.tasks:
            if t["state"] == "active" and t["target_kind"] == "patch" \
                    and t["target"] == job_id:
                return None, (f"{t['npc']} is already working "
                              f"the {job['op']} patch.")
        underway = [t for t in self.tasks if t["state"] == "active"]
        if len(underway) >= work_mod.MAX_ACTIVE_TASKS:
            return None, (f"The city already has "
                          f"{work_mod.MAX_ACTIVE_TASKS} tasks underway — "
                          f"let some finish first.")
        if npc_name is not None:
            if npc_name not in self.npcs:
                return None, f"Unknown citizen {npc_name!r}."
            if npc_name in self._task_by_npc:
                busy = work_mod.TASK_LABELS[
                    self._task_by_npc[npc_name]["type"]]
                return None, f"{npc_name} is already busy: {busy}."
            name = npc_name
        else:
            best, best_score = None, float("-inf")
            for cand in self.npcs:
                if cand in self._task_by_npc or cand in engaged \
                        or self._directed(cand, now):
                    continue
                score = self.rng.uniform(0.0, 0.3)
                score += work_mod.WORK_AFFINITY.get(
                    work_mod.TERRAFORM, {}).get(
                        self.npcs[cand].card.role, 0.0)
                p = positions.get(cand)
                if p is not None:
                    score -= 0.01 * ((job["x"] - p[0]) ** 2
                                     + (job["z"] - p[1]) ** 2) ** 0.5
                if score > best_score:
                    best, best_score = cand, score
            if best is None:
                return None, "No idle citizen can take that on right now."
            name = best
        self._task_seq += 1
        task = {"id": f"task-{self._task_seq}", "type": work_mod.TERRAFORM,
                "npc": name, "target": job["id"],
                "target_kind": "patch", "tx": job["x"], "tz": job["z"],
                "duration": None, "state": "active",
                "phase": "travel", "work_until": None, "created": now}
        self.tasks.append(task)
        self._task_by_npc[name] = task
        if events is not None:
            events.append(self.task_event(task, "assigned"))
        return task, None

    def _run_foreman(self, now: float, positions: dict, engaged: set,
                     events: list[dict]) -> None:
        """Keep the baseline stocks (work.py): when wood or food sits
        below target and no equivalent gather task is underway,
        enqueue one for the best idle citizen. The equivalent-task
        check is what makes this idempotent — it never duplicates a
        gather already running, and the global task cap still binds."""
        if self._foreman_next is None:
            self._foreman_next = now + self.FOREMAN_FIRST_DELAY
            return
        if now < self._foreman_next:
            return
        self._foreman_next = now + self.FOREMAN_INTERVAL
        inv = self.resources.inventory
        targets = self.effective_targets(now)
        underway = {t["type"] for t in self.tasks if t["state"] == "active"}
        for task_type, key, target in (
                (work_mod.GATHER_WOOD, "wood", targets["wood"]),
                (work_mod.GATHER_FOOD, "food", targets["food"])):
            if inv.get(key, 0) >= target or task_type in underway:
                continue
            task, _reason = self.assign_task(
                task_type, now, positions, engaged=engaged, events=events)
            if task is not None:
                underway.add(task_type)
        # Construction (P3): every open site without a builder gets
        # one, oldest first, under the same global task cap. A site
        # short on materials still gets its builder — the builder
        # waits on site (state "awaiting_materials") so work starts
        # the moment the stores suffice.
        if self.construction is not None:
            staffed = {t["target"] for t in self.tasks
                       if t["state"] == "active"
                       and t["target_kind"] == "site"}
            for site in self.construction.sites.values():
                if site.complete or site.id in staffed:
                    continue
                task, _reason = self.assign_build(
                    site.id, now, positions, engaged=engaged,
                    events=events)
                if task is None:
                    break       # cap reached / nobody idle: stop here
                staffed.add(site.id)
        # Terraforming (P4): staff open crew jobs the same way —
        # but NEVER create one. Crew jobs come from the player (crew
        # mode) and, from P5, from adopted civic proposals; the
        # foreman only puts hands on jobs that already exist.
        if self.terrain is not None:
            staffed_tf = {t["target"] for t in self.tasks
                          if t["state"] == "active"
                          and t["target_kind"] == "patch"}
            for job in self.terrain.jobs.values():
                if job["state"] != "active" or job["id"] in staffed_tf:
                    continue
                task, _reason = self.assign_terraform(
                    job["id"], now, positions, engaged=engaged,
                    events=events)
                if task is None:
                    break       # cap reached / nobody idle: stop here
                staffed_tf.add(job["id"])
        # Innovation manufacturing (P6): a sellable good whose
        # stalls stand empty gets one craft queued — the engine
        # decides what is needed (inputs in store, station built);
        # the ordinary cap and equivalent-task rules still bind.
        if self.innovations is not None:
            underway_now = {t["type"] for t in self.tasks
                            if t["state"] == "active"}
            for task_type in self.innovations.craft_needs():
                if task_type in underway_now:
                    continue
                task, _reason = self.assign_task(
                    task_type, now, positions, engaged=engaged,
                    events=events)
                if task is None:
                    break
                underway_now.add(task_type)

    def _drive_task(self, name: str, task: dict, now: float, pos,
                    events: list[dict]) -> None:
        """Steer one citizen through its task: travel to the node /
        station, then work there in place until the duration ends.
        Mirrors _drive_plan — state is re-derived from the task every
        tick, so a pause (conversation lock, brain direction) simply
        resumes where it left off."""
        st = self.state[name]
        st["initialized"] = True
        st["kind"] = "work"
        st["task"] = task["type"]
        st["partner"] = None
        st["anchor"] = None
        st["tx"], st["tz"] = task["tx"], task["tz"]
        d = ((task["tx"] - pos[0]) ** 2 + (task["tz"] - pos[1]) ** 2) ** 0.5
        if task["phase"] == "travel":
            if d <= self.ARRIVE_RADIUS:
                task["phase"] = "work"
                # Build tasks have no fixed duration: they run until
                # the site completes (work_until stays None).
                task["work_until"] = (None if task["duration"] is None
                                      else now + task["duration"])
                st["state"] = "dwell"
                st["route"] = None
                st["until"] = now + 30.0
                events.append(self._intent(name, st))
                events.append(self.task_event(task, "started"))
            elif st["state"] != "travel":
                st["state"] = "travel"
                st["route"] = self._route_for(pos, task["tx"],
                                              task["tz"])
                events.append(self._intent(name, st))
        else:
            if st["state"] != "dwell":
                st["state"] = "dwell"
                events.append(self._intent(name, st))
            if task["target_kind"] == "site":
                st["until"] = now + 30.0     # stay parked on the site
                self._work_build(name, task, now, events)
            elif task["target_kind"] == "patch":
                st["until"] = now + 30.0     # stay parked on the patch
                self._work_terraform(name, task, now, events)
            elif now >= task["work_until"]:
                self._complete_task(name, task, now, pos, events)

    def _work_build(self, name: str, task: dict, now: float,
                    events: list[dict]) -> None:
        """One tick of building work. The site's own rules
        (buildings.py) deliver the current stage's materials from
        the city inventory and accrue this tick's work — progress
        only while a builder is present AND the stage stands
        supplied. Stage/completion build events pass through to the
        world; completion registers the finished building's planner
        anchor (a home for a house, a market anchor for a stall)
        and frees the builder with a completed task event."""
        site = self.construction.sites.get(task["target"]) \
            if self.construction is not None else None
        if site is None:
            # The site vanished (a layout restore replaced the
            # construction world under the task): fail cleanly.
            self._release_worker(name, now, events)
            task["state"] = "failed"
            events.append(self.task_event(
                task, "failed", reason="the construction site is gone"))
            return
        built = self.construction.work_at(
            site.id, self._tick_dt, self.resources.inventory, by=name)
        for ev in built:
            events.append(ev)
            if ev.get("state") == "complete":
                anchor = ev.get("anchor")
                if anchor is not None:
                    self.anchors.append(ActivityAnchor(
                        anchor["name"], anchor["kind"],
                        float(anchor["x"]), float(anchor["z"])))
                self._release_worker(name, now, events)
                task["state"] = "completed"
                events.append(self.task_event(
                    task, "completed", building=site.kind,
                    inventory=dict(self.resources.inventory)))

    def _work_terraform(self, name: str, task: dict, now: float,
                        events: list[dict]) -> None:
        """One tick of crew terraforming. The job's own rules
        (terrain.py) advance its progress by this tick and lerp the
        patch heights from start to goal — the full delta lands
        over ~30 attended sim-seconds, so a citizen who is pulled
        away (conversation lock, a layout restore) simply stalls
        the job where it stands. The completion terraform event
        (mode "crew") passes through to the world, and the citizen
        is freed with a completed task event."""
        job = self.terrain.jobs.get(task["target"]) \
            if self.terrain is not None else None
        if job is None:
            # The job vanished (a layout restore replaced the
            # terrain under the task): fail cleanly.
            self._release_worker(name, now, events)
            task["state"] = "failed"
            events.append(self.task_event(
                task, "failed", reason="the land works are gone"))
            return
        done = self.terrain.work_at(job["id"], self._tick_dt, by=name)
        for ev in done:
            events.append(ev)
            if ev.get("state") == "complete":
                self._release_worker(name, now, events)
                task["state"] = "completed"
                events.append(self.task_event(task, "completed"))

    def _release_worker(self, name: str, now: float,
                        events: list[dict]) -> None:
        """Hand a citizen back to their itinerary after a task ends
        (the state cleanup _complete_task does for node/station
        tasks, shared by the build flow)."""
        st = self.state[name]
        self._task_by_npc.pop(name, None)
        st["task"] = None
        st["state"] = "dwell"
        st["until"] = now + self.rng.uniform(5.0, 15.0)
        events.append(self._intent(name, st))

    def _complete_task(self, name: str, task: dict, now: float, pos,
                       events: list[dict]) -> None:
        """Finish a task at its target. Gathering goes through
        ResourceWorld.try_harvest with the NPC as actor (failure —
        someone else took it, it hasn't regrown — ends the task with
        a failed event; the foreman re-queues if the stock is still
        low). Crafting validates its inputs NOW, at completion:
        inputs still in the stores are consumed and outputs added;
        a stock-out is a graceful failure, never a negative store."""
        st = self.state[name]
        del self._task_by_npc[name]
        st["task"] = None
        st["state"] = "dwell"
        st["until"] = now + self.rng.uniform(5.0, 15.0)
        events.append(self._intent(name, st))
        if task["target_kind"] == "node":
            ev, notice = self.resources.try_harvest(
                task["target"], pos, now, by=name)
            if ev is not None:
                events.append(ev)
                task["state"] = "completed"
                events.append(self.task_event(
                    task, "completed", amount=ev["amount"],
                    resource=ev["resource"]))
            else:
                task["state"] = "failed"
                events.append(self.task_event(task, "failed",
                                              reason=notice))
            return
        recipe = self._book().get(task["type"])
        inv = self.resources.inventory
        missing = {k: v - inv.get(k, 0)
                   for k, v in recipe["inputs"].items()
                   if inv.get(k, 0) < v}
        if missing:
            task["state"] = "failed"
            need = ", ".join(f"{v} more {k}" for k, v in missing.items())
            events.append(self.task_event(
                task, "failed",
                reason=f"the stores no longer hold the inputs "
                       f"(short {need})"))
            return
        for k, v in recipe["inputs"].items():
            inv[k] = inv.get(k, 0) - v
        outputs = dict(recipe["outputs"])
        if self.innovations is not None:
            # Sellable innovation goods route a share of the output
            # to the market stalls' stock (innovations.py).
            outputs = self.innovations.route_craft_outputs(outputs)
        for k, v in outputs.items():
            inv[k] = inv.get(k, 0) + v
        task["state"] = "completed"
        events.append(self.task_event(
            task, "completed", outputs=dict(recipe["outputs"]),
            inventory=dict(inv)))

    def _pair_social(self, now: float, positions: dict, engaged: set,
                     tiers: dict, events: list[dict]) -> None:
        """Two NPCs dwelling at the same anchor may pair into a brief
        in-place conversation. Both stay put for the duration. With no
        generator wired (demo/mock), the exchange is one canned line.
        In live mode the pairing instead opens a conversation record
        and requests an LLM-written exchange on a background thread —
        for EVERY pairing, witnessed or not — delivered by a later
        tick (see _drain_generated); a generation failure is silence,
        never a canned line. A pair still inside its generation
        cooldown pairs up quietly: they stand together, nothing said."""
        by_anchor: dict[str, list[str]] = {}
        for name, st in self.state.items():
            if (st["state"] == "dwell" and st["anchor"] is not None
                    and now > st["cooldown_until"] and name not in engaged
                    and name not in self._task_by_npc
                    and not self._directed(name, now)
                    and positions.get(name) is not None):
                by_anchor.setdefault(st["anchor"].name, []).append(name)
        for names in by_anchor.values():
            if len(names) < 2:
                continue
            a_name, b_name = names[0], names[1]
            dur = self.rng.uniform(20.0, 45.0)
            for name, other in ((a_name, b_name), (b_name, a_name)):
                st = self.state[name]
                st["state"] = "social"
                st["partner"] = other
                st["until"] = now + dur
                st["last_social"] = now
                st["cooldown_until"] = now + dur + self.rng.uniform(45.0, 90.0)
                events.append(self._intent(name, st))
            anchor_name = self.state[a_name]["anchor"].name
            if self.conversation_generator is None:
                self._emit_canned_line(a_name, b_name, tiers, events)
                continue
            key = (a_name, b_name) if a_name < b_name else (b_name, a_name)
            if key in self._conversations:
                continue        # an earlier record still owns this pair
            if now - self._gen_pair_last.get(key, -1e9) \
                    < self.GEN_PAIR_COOLDOWN:
                continue        # generated together too recently: quiet
            conv = {"key": key, "a": a_name, "b": b_name,
                    "anchor": anchor_name, "history": [], "started": now,
                    "witnessed": (tiers.get(a_name) == "active"
                                  or tiers.get(b_name) == "active"),
                    "state": "pending", "next_at": now, "attempt": 0,
                    # Location context (P11): where each speaker
                    # factually is, captured at pairing time (the
                    # pair dwells at the anchor, so it stays true
                    # for the conversation's life). Situation-aware
                    # generator hooks receive it as a fifth argument.
                    "situations": {
                        a_name: self.situation_text(
                            a_name, positions.get(a_name)),
                        b_name: self.situation_text(
                            b_name, positions.get(b_name))}}
            self._conversations[key] = conv
            # Fire the first exchange right away; if the rate
            # discipline is busy it stays "pending" and the service
            # step starts it on a later tick (or the pair drifts apart
            # and the record is pruned, unsaid).
            self._start_exchange(conv, now, "first")

    # -- social exchanges: canned lines (demo only) + LLM generation --------
    def _emit_canned_line(self, a_name: str, b_name: str, tiers: dict,
                          events: list[dict]) -> None:
        line = self.rng.choice(_SOCIAL_LINES)
        events.append({"type": "social", "a": a_name, "b": b_name,
                       "npc": a_name, "text": line})
        for name, other in ((a_name, b_name), (b_name, a_name)):
            if tiers.get(name) == "active" or tiers.get(other) == "active":
                self.npcs[name].witness(
                    f"{name} chatted with {other} at "
                    f"{self.state[name]['anchor'].name}: “{line}”",
                    kind="social")

    def _pair_still_social(self, a_name: str, b_name: str,
                           anchor_name: str) -> bool:
        """Are these two still paired together at the same anchor?"""
        sa, sb = self.state.get(a_name), self.state.get(b_name)
        return (sa is not None and sb is not None
                and sa["state"] == "social" and sb["state"] == "social"
                and sa["partner"] == b_name and sb["partner"] == a_name
                and sa["anchor"] is not None
                and sa["anchor"].name == anchor_name)

    def _start_exchange(self, conv: dict, now: float, kind: str) -> bool:
        """Fire one generation for a conversation record on a daemon
        thread — the world loop never waits on the LLM. `kind` is
        "first" (a new pairing's opening exchange: charges the per-pair
        cooldown), "continuation" (history is handed over), or "retry"
        (the one allowed re-ask of a failed request). Returns False —
        leaving the record for the service step — when the rate
        discipline says not now: circuit breaker tripped, too many
        generations in flight, or the global start gap hasn't passed."""
        if now < self._gen_paused_until:
            return False
        if now - self._gen_last_at < self.GEN_GLOBAL_GAP:
            return False
        with self._gen_lock:
            if self._gen_inflight >= self.GEN_MAX_INFLIGHT:
                return False
            self._gen_inflight += 1
        self._gen_last_at = now
        a_name, b_name = conv["a"], conv["b"]
        if kind == "first":
            self._gen_pair_last[conv["key"]] = now
        # An exchange counts as the conversation's first until some
        # lines have actually landed — so a retried opener can still
        # carry a plan agreement, and continuations never do.
        first = not conv["history"]
        history = list(conv["history"][-self.GEN_HISTORY_MAX:])
        conv["state"] = "awaiting"
        conv["attempt"] = 1 if kind == "retry" else 0
        gen = self.conversation_generator
        arity = self._gen_arity
        situations = conv.get("situations")
        card_a, card_b = self.npcs[a_name].card, self.npcs[b_name].card
        anchor_name = conv["anchor"]

        def work():
            lines, plan = None, None
            try:
                if arity >= 5:
                    result = gen(card_a, card_b, anchor_name, history,
                                 situations)
                elif arity >= 4:
                    result = gen(card_a, card_b, anchor_name, history)
                else:
                    result = gen(card_a, card_b, anchor_name)
                # Extended contract: (lines, plan_anchor | None). The
                # original plain-lines return stays valid (plan=None).
                if isinstance(result, tuple) and len(result) == 2:
                    lines, plan = result
                else:
                    lines = result
            except Exception:
                lines, plan = None, None
            try:
                self._gen_results.put({"key": conv["key"], "a": a_name,
                                       "b": b_name, "anchor": anchor_name,
                                       "lines": lines, "plan": plan,
                                       "first": first})
            finally:
                with self._gen_lock:
                    self._gen_inflight -= 1

        threading.Thread(target=work, daemon=True,
                         name="hermes-ambient-social").start()
        return True

    def _service_conversations(self, now: float, tiers: dict) -> None:
        """Per-tick upkeep for live conversation records: prune the
        ones whose pairing ended, start deferred openers and retries
        when due, and fire continuations for pairs that are still
        social AND witnessed (either participant ACTIVE). While a
        witnessed conversation is live, the pairing's dwell deadline
        is stretched toward the continuation so the chat is not cut
        mid-thread — never past started + CONT_CAP_S."""
        if self.conversation_generator is None:
            return
        for key, conv in list(self._conversations.items()):
            a_name, b_name = conv["a"], conv["b"]
            if not self._pair_still_social(a_name, b_name, conv["anchor"]):
                del self._conversations[key]
                continue
            witnessed = (tiers.get(a_name) == "active"
                         or tiers.get(b_name) == "active")
            conv["witnessed"] = witnessed
            state = conv["state"]
            if state in ("pending", "retry"):
                if now >= conv["next_at"] and not self._start_exchange(
                        conv, now,
                        "first" if state == "pending" else "retry"):
                    conv["next_at"] = now + 2.0   # discipline busy: soon
            elif state == "live" and now >= conv["next_at"]:
                if not witnessed:
                    # Nobody is listening right now; check again in a
                    # few seconds — if the player wanders back while
                    # the pair is still together, the chat resumes.
                    conv["next_at"] = now + 5.0
                elif now - conv["started"] >= self.CONT_CAP_S:
                    conv["state"] = "dead"        # a good long talk: done
                elif not self._start_exchange(conv, now, "continuation"):
                    conv["next_at"] = now + 2.0
            elif state == "live" and witnessed:
                cap_end = conv["started"] + self.CONT_CAP_S
                stretch = min(conv["next_at"] + 8.0, cap_end)
                for name in (a_name, b_name):
                    st = self.state[name]
                    if st["until"] < stretch:
                        st["until"] = stretch

    @staticmethod
    def _validate_lines(lines, a_name: str, b_name: str):
        """A generated exchange is 2-4 (speaker, text) pairs, speakers
        exactly the pairing, both speaking. Anything else -> None."""
        if not isinstance(lines, (list, tuple)) or not 2 <= len(lines) <= 4:
            return None
        out, speakers = [], set()
        for item in lines:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                return None
            speaker, text = str(item[0]).strip(), str(item[1]).strip()
            if speaker not in (a_name, b_name) or not text:
                return None
            speakers.add(speaker)
            out.append((speaker, text))
        return out if speakers == {a_name, b_name} else None

    def _drain_generated(self, now: float, tiers: dict) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                res = self._gen_results.get_nowait()
            except queue.Empty:
                break
            a_name, b_name = res["a"], res["b"]
            conv = self._conversations.get(res.get("key"))
            lines = self._validate_lines(res.get("lines"), a_name, b_name)
            if lines is None:
                # Generator failed / timed out / unparseable. Live mode
                # has no canned fallback: the pair simply says nothing
                # (a note goes to the server log). A witnessed pair gets
                # ONE retry in a few seconds; anything else ends this
                # pairing's conversation quietly. Consecutive failures
                # trip the circuit breaker, pausing generation city-wide
                # rather than hammering a struggling brain.
                self._gen_fail_streak += 1
                print(f"[hermes] ambient generation failed for {a_name} "
                      f"& {b_name} at {res['anchor']} "
                      f"(failure streak {self._gen_fail_streak})",
                      flush=True)
                if self._gen_fail_streak >= self.GEN_BREAKER_FAILURES:
                    self._gen_fail_streak = 0
                    self._gen_paused_until = now + self.GEN_BREAKER_PAUSE
                    print(f"[hermes] ambient generation paused for "
                          f"{self.GEN_BREAKER_PAUSE:.0f}s "
                          f"(circuit breaker)", flush=True)
                if conv is not None and self._pair_still_social(
                        a_name, b_name, res["anchor"]):
                    witnessed = (tiers.get(a_name) == "active"
                                 or tiers.get(b_name) == "active")
                    if witnessed and conv["attempt"] == 0:
                        conv["state"] = "retry"
                        conv["next_at"] = now + self.GEN_RETRY_DELAY
                    else:
                        conv["state"] = "dead"
                continue
            self._gen_fail_streak = 0
            if conv is None or not self._pair_still_social(
                    a_name, b_name, res["anchor"]):
                continue            # pair broke up meanwhile: drop silently
            for speaker, text in lines:
                events.append({"type": "social", "a": a_name, "b": b_name,
                               "npc": speaker, "text": text})
                listener = b_name if speaker == a_name else a_name
                for name, other in ((a_name, b_name), (b_name, a_name)):
                    if (tiers.get(name) == "active"
                            or tiers.get(other) == "active"):
                        self.npcs[name].witness(
                            f"{speaker} said to {listener} at "
                            f"{res['anchor']}: “{text}”", kind="social")
            conv["history"].extend(lines)
            # Plans are born only here — from a delivered, generated
            # FIRST exchange. Canned lines never plan, and neither do
            # continuations.
            if res.get("first") and res.get("plan"):
                self._maybe_form_plan(a_name, b_name, res["plan"], now,
                                      events)
            # The conversation stays alive while it is witnessed: the
            # next exchange is due after a natural pause, and the
            # pairing is stretched to cover it (bounded by the cap).
            # Unwitnessed, this exchange was the whole chat.
            conv["attempt"] = 0
            witnessed = (tiers.get(a_name) == "active"
                         or tiers.get(b_name) == "active")
            conv["state"] = "live"
            if witnessed:
                nxt = now + self.rng.uniform(*self.CONT_CADENCE)
                if nxt + 5.0 <= conv["started"] + self.CONT_CAP_S:
                    conv["next_at"] = nxt
                    stretch = min(nxt + 8.0, conv["started"] + self.CONT_CAP_S)
                    for name in (a_name, b_name):
                        st = self.state[name]
                        if st["until"] < stretch:
                            st["until"] = stretch
                else:
                    conv["state"] = "dead"    # at the cap: a natural end
            else:
                conv["next_at"] = now + 5.0  # re-check: may be overheard
        return events

    def stats(self) -> dict:
        total = self.time_in["dwell"] + self.time_in["travel"]
        return {"dwell_s": round(self.time_in["dwell"], 1),
                "travel_s": round(self.time_in["travel"], 1),
                "dwell_share": (round(self.time_in["dwell"] / total, 3)
                                if total > 0 else 0.0)}


# --------------------------------------------------------------------------
# Proximity activation manager — the core scaling system.
#
# Tiers (distances in world meters, from the player's position):
#   ACTIVE   enter at <= active_radius, exit only beyond
#            active_exit_radius (hysteresis), a slot is free, or the NPC
#            is conversation-locked: full agent loop.
#   AMBIENT  enter at <= ambient_radius, exit beyond ambient_exit_radius:
#            scripted by the world client, zero LLM.
#   DORMANT  otherwise: nothing but a position on a timer.
# At most `max_active` NPCs are ACTIVE; the nearest win, except engaged
# NPCs, who are pinned. A tier demotion additionally requires the NPC to
# have held its tier for `min_hold_s` — except past `hard_demote_m`,
# where demotion is immediate (teleports shouldn't strand anyone ACTIVE).
# Together with the itinerary planner (NPCs dwell instead of wandering),
# this keeps the ACTIVE set — and therefore the chat-target list —
# stable instead of flickering.
#
# Conversation lock: note_speech() engages an NPC. While engaged it is
# pinned ACTIVE regardless of distance bands, and its planner state is
# "engaged" (the client stops it and turns it to face the player). The
# lock releases when `engage_timeout_s` pass with no message, or the
# player moves more than `engage_range_m` away.
# --------------------------------------------------------------------------
_TIER_RANK = {"dormant": 0, "ambient": 1, "active": 2}


class NPCManager:
    def __init__(self, npcs: dict[str, NPC], active_radius: float = 15.0,
                 ambient_radius: float = 50.0, max_active: int = 4,
                 active_exit_radius: float = 20.0,
                 ambient_exit_radius: float = 60.0,
                 min_hold_s: float = 5.0, engage_timeout_s: float = 20.0,
                 engage_range_m: float = 25.0, hard_demote_m: float = 120.0,
                 planner: ActivityPlanner | None = None,
                 spawns: dict | None = None,
                 conversation_generator=None):
        self.npcs = npcs
        self.active_radius = active_radius
        self.ambient_radius = ambient_radius
        self.max_active = max_active
        self.active_exit_radius = active_exit_radius
        self.ambient_exit_radius = ambient_exit_radius
        self.min_hold_s = min_hold_s
        self.engage_timeout_s = engage_timeout_s
        self.engage_range_m = engage_range_m
        self.hard_demote_m = hard_demote_m
        self.tiers: dict[str, str] = {name: "dormant" for name in npcs}
        self.tier_since: dict[str, float] = {name: 0.0 for name in npcs}
        self.positions: dict[str, tuple[float, float]] = {}
        self.player_pos: tuple[float, float] = (0.0, 0.0)
        # Ghost mode: the player is an unseen listener. Tiers still run
        # from the ghost's position (nearby citizens go ACTIVE and their
        # ambient conversations still trigger — that's the point), but
        # engagement never does: no conversation lock, no greeting, and
        # the bridge does not answer ghost speech.
        self.ghost: bool = False
        self.engaged_until: dict[str, float] = {}
        self._engaged_now: set[str] = set()
        if planner is None and spawns is not None:
            planner = ActivityPlanner(
                npcs, spawns, conversation_generator=conversation_generator)
        self.planner = planner
        if spawns is not None:
            self.spawns = {n: (float(p[0]), float(p[1]))
                           for n, p in spawns.items()}
        elif planner is not None:
            self.spawns = dict(planner.spawns)
        else:
            self.spawns = {}

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # -- conversation lock -----------------------------------------------------
    def note_speech(self, name: str, now: float | None = None) -> None:
        """Engage (or refresh) the conversation lock on an NPC. A ghost
        is never heard, so a ghost's speech never engages anyone."""
        if self.ghost:
            return
        if name in self.npcs:
            t = time.monotonic() if now is None else float(now)
            self.engaged_until[name] = t + self.engage_timeout_s

    def is_engaged(self, name: str, now: float | None = None) -> bool:
        until = self.engaged_until.get(name)
        if until is None:
            return False
        t = time.monotonic() if now is None else float(now)
        return t < until

    def note_directed(self, name: str, seconds: float | None,
                      now: float | None = None) -> None:
        if self.planner is not None:
            t = time.monotonic() if now is None else float(now)
            self.planner.note_directed(name, seconds, t)

    # -- main update --------------------------------------------------------------
    def update(self, player_pos, positions: dict,
               now: float | None = None,
               ghost: bool | None = None) -> list[dict]:
        """Recompute tiers from positions; fire activate/deactivate hooks.

        Returns tier-change / engage / intent / social event dicts for the
        world client. `now` is injectable for deterministic tests and
        simulations; production callers get time.monotonic(). `ghost`
        (None = leave as-is) toggles ghost mode: while ghost, every
        conversation lock is released and none can form, but tiers are
        still computed from the player's position so nearby citizens
        stay ACTIVE and audible."""
        now = time.monotonic() if now is None else float(now)
        if ghost is not None:
            self.ghost = bool(ghost)
        self.player_pos = (float(player_pos[0]), float(player_pos[1]))
        for name, pos in positions.items():
            if name in self.npcs:
                self.positions[name] = (float(pos[0]), float(pos[1]))
        dists = {name: self._dist(self.player_pos, pos)
                 for name, pos in self.positions.items() if name in self.npcs}
        events: list[dict] = []

        # Conversation lock: release on timeout, player walking away —
        # or immediately when the player goes ghost (an unseen listener
        # holds nobody's attention).
        engaged_now: set[str] = set()
        for name, until in list(self.engaged_until.items()):
            d = dists.get(name)
            if self.ghost or now >= until \
                    or (d is not None and d > self.engage_range_m):
                del self.engaged_until[name]
                events.append({"type": "engage", "npc": name,
                               "engaged": False,
                               "reason": "ghost mode" if self.ghost
                                         else "timeout" if now >= until
                                         else "player moved away"})
            else:
                engaged_now.add(name)
        self._engaged_now = engaged_now

        # Desired ACTIVE set: engaged pinned first, then nearest entrants
        # (<= active_radius) and holders (currently ACTIVE, <= exit band).
        cand = [n for n, d in dists.items()
                if n in engaged_now or d <= self.active_radius
                or (self.tiers[n] == "active" and d <= self.active_exit_radius)]
        cand.sort(key=lambda n: (0 if n in engaged_now else 1, dists[n]))
        desired_active = set(cand[:self.max_active]) | engaged_now
        for name in cand:  # min-hold: a fresh ACTIVE doesn't lose its slot
            if (name not in desired_active and self.tiers[name] == "active"
                    and dists[name] <= self.hard_demote_m
                    and now - self.tier_since.get(name, 0.0) < self.min_hold_s):
                desired_active.add(name)

        for name, npc in self.npcs.items():
            d = dists.get(name)
            if d is None:
                continue            # no position reported: leave tier as-is
            old_tier = self.tiers[name]
            if name in desired_active:
                new_tier = "active"
            elif d <= self.ambient_radius or (
                    old_tier in ("ambient", "active")
                    and d <= self.ambient_exit_radius):
                new_tier = "ambient"
            else:
                new_tier = "dormant"
            if new_tier != old_tier:
                demotion = _TIER_RANK[new_tier] < _TIER_RANK[old_tier]
                if (demotion and d <= self.hard_demote_m
                        and now - self.tier_since.get(name, 0.0) < self.min_hold_s):
                    new_tier = old_tier     # too soon to demote: hold
            if new_tier == old_tier:
                continue
            if new_tier == "active":
                npc.activate()
                events.append({"type": "tier", "npc": name, "tier": "active",
                               "reason": "engaged" if name in engaged_now
                                         else "entered active radius / slot free",
                               "engaged": name in engaged_now,
                               "distance": round(d, 1)})
            elif old_tier == "active":
                summary = npc.deactivate()
                events.append({"type": "tier", "npc": name, "tier": new_tier,
                               "reason": "left active radius or lost slot",
                               "summary": summary})
            else:
                events.append({"type": "tier", "npc": name, "tier": new_tier,
                               "reason": "proximity"})
            self.tiers[name] = new_tier
            self.tier_since[name] = now

        if self.planner is not None:
            events.extend(self.planner.tick(now, self.positions,
                                            engaged_now, self.tiers))
        return events

    def situation_for(self, name: str) -> str | None:
        """This citizen's factual location context (P11), from
        the planner's world data and the last reported position.
        None when either is missing — prompts then simply carry
        no situation, exactly as before P11."""
        if self.planner is None:
            return None
        pos = self.positions.get(name)
        if pos is None:
            return None
        try:
            return self.planner.situation_text(name, pos)
        except Exception:
            return None

    def route_speech(self, text: str, speaker: str = "Player",
                     now: float | None = None) -> dict:
        """Player speech reaches only ACTIVE NPCs (nearest first); every
        NPC spoken to is conversation-locked. In ghost mode nobody
        hears the player at all: no replies, no locks."""
        if self.ghost:
            return {}
        actives = [n for n, t in self.tiers.items() if t == "active"]
        actives.sort(key=lambda n: self._dist(self.player_pos,
                                              self.positions.get(n, (1e9, 1e9))))
        for name in actives:
            self.note_speech(name, now=now)
        return {name: self.npcs[name].hear(
                    text, speaker=speaker,
                    situation=self.situation_for(name))
                for name in actives}

    def stats(self) -> dict:
        counts = {"active": 0, "ambient": 0, "dormant": 0}
        for tier in self.tiers.values():
            counts[tier] += 1
        actives = [n for n, t in self.tiers.items() if t == "active"]
        # Live lock state (time-valid right now, not just as of the last
        # update) so a just-spoken-to NPC shows as engaged immediately.
        now = time.monotonic()
        engaged_live = {n for n, u in self.engaged_until.items() if now < u}
        engaged = sorted(n for n in actives if n in engaged_live)
        return {**counts,
                "ghost": self.ghost,
                # RTS labor (P2): the underway task list rides the
                # status payload so clients can show who's working.
                "tasks": (self.planner.active_tasks()
                          if self.planner is not None else []),
                # Stable chatters ordering: engaged first, then by name.
                "active_names": engaged + [n for n in sorted(actives)
                                           if n not in engaged_live],
                "engaged_names": sorted(engaged_live),
                "max_active": self.max_active,
                "active_radius": self.active_radius,
                "active_exit_radius": self.active_exit_radius,
                "ambient_radius": self.ambient_radius,
                "ambient_exit_radius": self.ambient_exit_radius}

    # -- souls: save / load who the NPCs are + what they know ----------------
    # A *soul* is identity + knowledge: the character card, every episodic
    # memory in the NPC's namespace, and its encounter history. A
    # *situation* is where the NPC is and what it is doing right now
    # (position, anchor, activity). Snapshots carry both; loading in
    # "knowledge" mode restores souls only — everyone respawns at home
    # with a fresh itinerary — while "full" also restores situations.
    # Conversation-born plans follow the same split: the plan *memory*
    # (kind="plan") is knowledge and rides along in the snapshot, but
    # the planner's live pending-plan queue is situation — it is never
    # snapshotted, and a load (either mode) does not resurrect it.
    SOULS_FORMAT = "hermes-local/souls"
    SOULS_VERSION = 1

    def export_souls(self) -> dict:
        """Snapshot every NPC's soul + current situation as one
        JSON-serializable document (see SOULS_FORMAT)."""
        out = []
        for name, npc in self.npcs.items():
            memories = (npc.memory.list_namespace(npc.ns)
                        if npc.memory is not None else [])
            pos = self.positions.get(name) or self.spawns.get(
                name, (0.0, 0.0))
            if self.planner is not None:
                situation = self.planner.situation_of(name, pos)
            else:
                situation = {"position": [float(pos[0]), float(pos[1])],
                             "anchor": None, "activity": None,
                             "state": None}
            out.append({
                "name": name,
                "card": npc.card.to_dict(),
                "memories": memories,
                "history": {"stats": dict(npc.stats),
                            "encounter_log": list(npc.encounter_log)},
                "situation": situation,
            })
        return {"format": self.SOULS_FORMAT, "version": self.SOULS_VERSION,
                "exported_at": time.time(), "npc_count": len(out),
                "npcs": out}

    @classmethod
    def _validate_souls(cls, snapshot) -> list[dict]:
        """Check a snapshot end to end BEFORE anything is applied, so a
        malformed file can never half-load into a running world. Returns
        the cleaned per-NPC entries; raises ValueError on any problem."""
        if not isinstance(snapshot, dict):
            raise ValueError("souls snapshot must be a JSON object")
        if snapshot.get("format") != cls.SOULS_FORMAT:
            raise ValueError(
                f"not a souls snapshot (format={snapshot.get('format')!r})")
        if snapshot.get("version") != cls.SOULS_VERSION:
            raise ValueError(
                f"unsupported souls version: {snapshot.get('version')!r}")
        entries = snapshot.get("npcs")
        if not isinstance(entries, list):
            raise ValueError("snapshot.npcs must be a list")
        cleaned = []
        for i, entry in enumerate(entries):
            where = f"npcs[{i}]"
            if not isinstance(entry, dict) \
                    or not isinstance(entry.get("name"), str) \
                    or not entry["name"]:
                raise ValueError(f"{where} needs a non-empty 'name'")
            card = entry.get("card")
            if not isinstance(card, dict):
                raise ValueError(f"{where}.card must be an object")
            for f in ("role", "persona", "backstory", "voice"):
                if f in card and not isinstance(card[f], str):
                    raise ValueError(f"{where}.card.{f} must be a string")
            for f in ("traits", "goals"):
                if f in card and not isinstance(card[f], list):
                    raise ValueError(f"{where}.card.{f} must be a list")
            if "relationships" in card \
                    and not isinstance(card["relationships"], dict):
                raise ValueError(f"{where}.card.relationships must be an object")
            memories = entry.get("memories", [])
            if not isinstance(memories, list):
                raise ValueError(f"{where}.memories must be a list")
            clean_memories = []
            for j, m in enumerate(memories):
                if not isinstance(m, dict) \
                        or not isinstance(m.get("text"), str):
                    raise ValueError(f"{where}.memories[{j}] needs a 'text'")
                clean_memories.append({
                    "text": m["text"],
                    "kind": str(m.get("kind") or "note"),
                    "ts": m.get("ts") if isinstance(m.get("ts"), (int, float))
                    else None})
            situation = entry.get("situation")
            clean_situation = None
            if situation is not None:
                if not isinstance(situation, dict):
                    raise ValueError(f"{where}.situation must be an object")
                pos = situation.get("position")
                if pos is not None and not (
                        isinstance(pos, (list, tuple)) and len(pos) == 2
                        and all(isinstance(v, (int, float)) for v in pos)):
                    raise ValueError(
                        f"{where}.situation.position must be [x, z]")
                clean_situation = {
                    "position": ([float(pos[0]), float(pos[1])]
                                 if pos is not None else None),
                    "anchor": situation.get("anchor")
                    if isinstance(situation.get("anchor"), str) else None,
                    "activity": situation.get("activity")
                    if isinstance(situation.get("activity"), str) else None}
            history = entry.get("history")
            cleaned.append({
                "name": entry["name"], "card": card,
                "memories": clean_memories, "situation": clean_situation,
                "encounter_log": ([str(x) for x in history["encounter_log"]]
                                  if isinstance(history, dict)
                                  and isinstance(history.get("encounter_log"),
                                                 list) else []),
                "stats": ({str(k): int(v) for k, v in history["stats"].items()
                           if isinstance(v, (int, float))}
                          if isinstance(history, dict)
                          and isinstance(history.get("stats"), dict) else {}),
            })
        return cleaned

    def import_souls(self, snapshot, mode: str = "knowledge") -> dict:
        """Restore a snapshot from export_souls().

        mode="knowledge": cards + memories + history are restored;
        situations are NOT — each NPC is placed at its spawn with fresh
        planner state. mode="full": the situation block (position +
        anchor/activity) is applied too. Snapshot NPCs with no matching
        NPC in this world are skipped (reported in the summary); world
        NPCs missing from the snapshot are left untouched. Raises
        ValueError — before changing anything — on malformed input."""
        if mode not in ("knowledge", "full"):
            raise ValueError(f"unknown souls load mode: {mode!r} "
                             "(want 'knowledge' or 'full')")
        entries = self._validate_souls(snapshot)
        now = time.monotonic()
        applied, skipped, total_memories = 0, [], 0
        for entry in entries:
            name = entry["name"]
            npc = self.npcs.get(name)
            if npc is None:
                skipped.append(name)
                continue
            card = entry["card"]
            for f in ("role", "persona", "backstory", "voice"):
                if f in card:
                    setattr(npc.card, f, card[f])
            if "traits" in card:
                npc.card.traits = [str(t) for t in card["traits"]]
            if "goals" in card:
                npc.card.goals = [str(g) for g in card["goals"]]
            if "relationships" in card:
                npc.card.relationships = dict(card["relationships"])
            if npc.memory is not None:
                total_memories += npc.memory.replace_namespace(
                    npc.ns, entry["memories"])
            npc.history = []
            npc.encounter_log = list(entry["encounter_log"])
            for k, v in entry["stats"].items():
                if k in npc.stats:
                    npc.stats[k] = v
            npc.active = False
            npc.recalled_on_activation = []
            self.engaged_until.pop(name, None)
            self._engaged_now.discard(name)
            self.tiers[name] = "dormant"
            self.tier_since[name] = now
            if self.planner is not None:
                self.planner.reset_npc(name)
            situation = entry["situation"]
            if mode == "full" and situation is not None \
                    and situation["position"] is not None:
                self.positions[name] = (situation["position"][0],
                                        situation["position"][1])
                if self.planner is not None:
                    self.planner.apply_situation(name, situation, now)
            else:
                spawn = self.spawns.get(name, (0.0, 0.0))
                self.positions[name] = (float(spawn[0]), float(spawn[1]))
            applied += 1
        return {"ok": True, "mode": mode, "npcs": applied,
                "memories": total_memories, "skipped": skipped}
