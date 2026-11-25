# Hermes Local

An offline-first personal agent framework for Aaron's RV command center —
and a character engine for AI NPCs in 3D worlds. One shared local brain,
persistent semantic memory, dual-mode tool calling, and a proximity
activation system that keeps 50+ NPCs cheap.

```
python -m hermes_local                 # interactive REPL
python -m hermes_local "summarize my notes"   # one-shot task
python -m hermes_local --info          # resources / model ladder / embedding status
python -m hermes_local serve --mock    # 3D world bridge + web city demo
python tests/run_tests.py              # the test suite (mock LLMs, no GPU)
```

## Architecture

```
                ┌──────────────────── hermes_local ────────────────────┐
  you / world ──┤ agent.py   agent loop (native tools ⇄ ReAct fallback)│
                │ npc.py     characters, director, proximity tiers     │
                │ world_bridge.py   JSON protocol: socket + HTTP + SSE │
                ├──────────────────────────────────────────────────────┤
                │ llm.py     brain ladder (resource-detected local LLM)│
                │ embeddings.py  EmbeddingGemma 2 → v1 → nomic → TF-IDF│
                │ memory.py  SQLite vector store + notes.md            │
                │ tools.py   registry: shell, files, web, RAG, plan,   │
                │            + NPC body actions (say/move/follow/…)     │
                └──────────────────────────────────────────────────────┘
```

Core transports use only the Python standard library (urllib, sqlite3,
http.server, socketserver) — the framework runs with **zero installs**.
PyYAML is the one real dependency (config files + YAML character cards);
numpy is an optional accelerator for vector search.

### The brain ladder (llm.py)

EmbeddingGemma 2 is an encoder — it cannot generate text or call tools —
so the framework pairs it with a generative brain, auto-selected by
detected RAM/VRAM (`/proc/meminfo`, `nvidia-smi` if present), overridable
in config:

| Brain | Needs | Why |
|---|---|---|
| `gpt-oss:20b` | 15GB VRAM / 20GB RAM | Full agent brain, native tool calling, holds 64K context in 16GB VRAM with q8_0 KV cache |
| `qwen3.5:9b` | 6GB VRAM / 10GB RAM | Strong mid-size tool caller |
| `gemma4:12b` | 8GB VRAM / 14GB RAM | Native tools + vision; shares its tokenizer family with EmbeddingGemma 2 |
| `qwen3:4b` | 3GB VRAM / 6GB RAM | The low-resource workhorse |
| `qwen3:0.6b` | anything | Tiny mode: chat/routing on nearly dead hardware |

### Optional hosted brain: Nous Portal free models (LongCat 2.5)

Local stays the default and the design center — but the client is
provider-agnostic, and a named preset wires Nous Portal's free tier in as
an alternative brain (handy for trying a big cloud model before the local
rig exists, or as a quality reference for the ladder):

```yaml
llm:
  provider: nous-longcat   # https://inference-api.nousresearch.com/v1
  api_key_env: NOUS_API_KEY
```

That resolves to model `meituan/longcat-2.5:free` — Meituan's LongCat 2.5,
a 560B-class MoE with 1M context and strong agentic tool use, listed free
on Nous Portal since Nov 2026. Get a key at portal.nousresearch.com and
`export NOUS_API_KEY=...` (the key is read from the environment, never
stored in config). Preset values fill only fields you didn't set yourself;
an explicit `base_url`/`model` in your config wins. `provider: nous` uses
the same endpoint with a model of your choice. Honest caveats: free
availability, quotas, and any balance requirements are Nous's to change —
in Sept 2026 their "free" models demanded a funded balance, while the
current $0 Portal plan lists free models with rate limits; treat the free
tier as opportunistic, never as a dependency. If the endpoint doesn't
answer, `serve` falls back to the demo brain and local Ollama work is
unaffected.

### Shortfall → low-resource solution

| Shortfall | Solution in this build |
|---|---|
| EmbeddingGemma 2 can't generate or call tools | It's the retrieval layer only; a ladder-picked generative brain does the talking/acting |
| Embedding server down / model not pulled (offline) | Fallback chain EG2 → EG1 → nomic-embed-text → built-in zero-dependency TF-IDF hashing embedder. Semantic memory never hard-fails |
| Model/server has no native tool calling | Dual mode: try the OpenAI-style `tools` array; on rejection, degrade permanently (per session) to prompt-based ReAct with fenced ` ```tool ` JSON blocks — the pattern Aaron's Hermes economy agent already uses |
| Small local context windows (upstream-style frameworks demand 64K) | ContextManager: token estimation, sliding window, brain-written summarization of older turns, plus RAG recall from long-term memory instead of context stuffing. Capable at 8K–32K |
| Vector storage/search cost | SQLite store (same pattern as the economy ledger); vectors stored at 768d, searched at 256d via Matryoshka-style truncation; numpy if present, pure Python otherwise |
| 50+ NPCs would melt one GPU | Proximity activation tiers (below): at most ~4 NPCs ever touch the LLM at once |
| No GPU at all | Ladder bottoms out at a 0.6B tiny model on CPU; embeddings fall back to the CPU hash embedder |

## NPCs in 3D worlds

### Character cards

Characters are files in a `characters/` folder (JSON or YAML), one per NPC:

```json
{ "name": "Mara", "role": "mechanic",
  "persona": "…", "backstory": "…",
  "traits": ["blunt", "loyal"], "voice": "style notes for how they talk",
  "goals": ["…"], "relationships": {"Player": "a new face"} }
```

Two examples ship in `characters/` (Mara, Joss). A procedural roster
generator (`npc.generate_roster`) creates 54 more real cards from persona
templates — every NPC is a framework character with its own memory
namespace (`npc:<name>`), not a dummy.

### Memory per NPC

Each NPC's episodes — conversations, witnessed world events, encounter
summaries — are embedded (EmbeddingGemma 2 chain) into its own namespace
of the shared SQLite vector store. On activation and on each line of
dialogue, relevant episodes are recalled semantically into its prompt.
The short-term window is handled by the same ContextManager as the agent.

### Body actions are tools

`say`, `move_to`, `follow`, `interact_with`, `play_animation`, `emote` —
emitted through the dual tool-calling modes and **validated against the
world's action schema** (sent in `register_world`) before reaching the
world. An NPC-to-NPC `Director` schedules multi-NPC conversations in
turns so nobody talks over anyone.

### Proximity activation (the scaling system)

Implemented in `npc.NPCManager` (Python side owns the tiers; the world
client just reports positions):

| Tier | Radius | Behavior |
|---|---|---|
| **ACTIVE** | ≤ `active_radius` (15m) and a slot is free | Full agent loop: persona loaded, episodic recall, action tools live |
| **AMBIENT** | ≤ `ambient_radius` (50m) | Scripted by the client (waypoint wandering), **zero LLM calls** |
| **DORMANT** | beyond | Nothing but a position on a timer |

At most `max_active` (4) NPCs are ACTIVE; the nearest win. On activation
the framework recalls the NPC's memories; on deactivation it **summarizes
the encounter into episodic memory and unloads the context**. Player
speech routes only to ACTIVE NPCs. Tuning knobs live in config:
`agent.active_radius`, `agent.ambient_radius`, `agent.max_active`.

### Bridge protocol (engine-agnostic JSON)

Transports: JSON-lines TCP socket (engines/tools), HTTP `POST /api/message`
+ SSE `GET /api/events` (browsers, CORS enabled), plus `GET /api/roster`,
`GET /api/status`, `GET/POST /api/layout`.

World → framework: `register_world` (action schema), `perception`
(writes episodic memory), `speech`, `proximity` (player + NPC positions),
`npc_chat`, `world_state`, `save_layout` / `load_layout`.
Framework → world: `dialogue`, `action`, `tier`, `status`, `ack`, `error`,
`layout`. Full message shapes are documented in `world_bridge.py`.

## Launch portal

**The deployment, plainly:** the Hermes brain system runs on the host;
the world loads in a browser pointed at the host. One command starts the
host side —

```
python -m hermes_local serve --mock     # or plain `serve` for a live LLM
```

— and then two URLs matter:

| URL | What it is |
|---|---|
| `http://<host>:8765/portal` | **Launch portal** (`webworld/portal.html`): the front door |
| `http://<host>:8765/` | The web city itself (`webworld/demo.html`) |

The portal is a single static page (no build step) that leads with the
**live pairing**: real-time status from `GET /api/status` — bridge mode,
the actual brain model/provider serving, the detected-hardware ladder
pick, and live NPC tier counts (ACTIVE / AMBIENT / DORMANT + who is
ACTIVE now), refreshed every 5 seconds. Below that: two launch buttons
(**Enter City — Play Mode** and **Enter City — God Mode**, same-origin
links into the world, God Mode via `demo.html?mode=god`), the brain
options summary (ladder table + the Nous LongCat preset and how to enable
it), quickstart commands, and links into this README. The city page links
back to the portal from its HUD.

**Standalone is the fallback, not the headline.** If the bridge API is
unreachable — server not started, or the pages opened as static files —
two graceful degradations happen:

- The **portal** keeps working as documentation: its status card switches
  to "server not running" and shows the start command.
- The **city** runs fully in the browser: a JS roster generated from the
  same persona templates as `npc.generate_roster` (same 54 names, roles,
  traits, spawn grid), a client-side proximity tier engine with the same
  rules (ACTIVE ≤15m, nearest 4; AMBIENT ≤50m scripted wandering; DORMANT
  beyond; encounters summarized into per-NPC session memory on
  deactivation), god mode and layout save/load (already client-side), and
  a **template-based JS demo brain** for ACTIVE NPCs — in-character
  replies assembled from the character card, simple session memory (the
  player's name once given, stored facts, topics), and movement commands
  ("follow me", "go to X", "fetch X", "wave"). It is demo-grade by design:
  a stand-in for the real brain, not a fake LLM.

The HUD always says which brain is driving: **LIVE — Hermes brain on
server** in the primary pairing, **STANDALONE fallback — demo brain in
browser** otherwise. Standalone engages only when the bridge genuinely
doesn't answer (the roster fetch is attempted first, with a timeout);
when the bridge is reachable, behavior is exactly the live path.

Honesty ledger for this section: the portal route, the enriched
`/api/status` (brain/provider/ladder pick, attached by `build_world`),
and the standalone roster/tier/brain code are implemented and covered by
the test suite + endpoint smoke checks; the standalone demo brain's
dialogue quality is template-grade and unjudged beyond its mechanics, and
page *visuals* remain as described in the Tests section below.

## The web world (browser reference client)

`python -m hermes_local serve --mock` → open http://127.0.0.1:8765/

A full procedural city in one self-contained page (`webworld/demo.html`,
Three.js via CDN — if the CDN is unreachable the page says so instead of
breaking; the bridge is unaffected):

- Street grid, sidewalks, districts (commercial / suburban / industrial),
  parked + driving cars, day lighting. City geometry is built from real
  CC0 GLB models (below) with per-object procedural-box fallback.
- 54 NPCs with real animated bodies (KayKit Adventurers, below) driven
  by the planner states — walking between anchors, idling or sitting at
  them, stopping to face you in conversation — over the original capsule
  layer, which remains as the per-NPC fallback (DORMANT citizens stay
  capsules until their first promotion; a body that fails to load never
  swaps in). Tier rings — green ACTIVE, amber AMBIENT, none DORMANT —
  name labels that hide with distance, WASD + mouse or click-to-move
  player (also model-bodied), chat box routed through the real bridge.
- HUD shows live tier counts and ACTIVE names, plus which brain is
  driving (LIVE — Hermes brain on server, or the standalone fallback).
- **Standalone fallback + launch portal:** see *Launch portal* above —
  with no bridge answering, the page generates its roster, runs the tier
  engine and a template demo brain entirely client-side, and `/portal`
  serves as the system's front door.
- With `--mock` (or automatically when no LLM server answers) the bridge
  uses a small rule-based **demo brain** so the page works out of the box:
  NPCs greet, remember, follow, walk to places, and fetch things. Live
  mode swaps in the ladder-picked local LLM with the identical interface.

### God mode city designer (Play ⇄ God, key `G`)

- **God camera:** orbit + pan + zoom, separate from the player camera.
- **Palette:** every vendored asset by category (buildings, houses,
  industrial, roads, vehicles, props) + NPC spawn points. Click a palette
  item, click the ground to place it.
- **Editing:** click any placed object to select it (highlight box), drag
  to move it on the ground plane, `R` rotate, buttons for scale /
  duplicate / delete (`Del`), grid-snap toggle with 1/2/4/6m sizes.
- **NPC integration:** spawn-point objects set where NPCs appear; the
  ambient waypoint graph is rebuilt from placed road objects (falling
  back to the sidewalk grid when a layout has no roads).
- **Save/Load:** the city is a JSON layout document —
  `{version, objects: [{asset, position, rotation, scale, kind}]}` —
  saved to named slots in localStorage, mirrored to the framework bridge
  (`/api/layout`, the same document the bridge protocol saves/loads, so
  external tools can generate cities), and exportable/importable as a
  `.json` file. Play mode loads the saved layout, else the default city.
  The designer edits the same city the simulation runs in.

### Assets (all CC0 — see `webworld/assets/CREDITS.md`)

Kenney City Kits (Commercial, Roads, Suburban, Industrial) + Car Kit and
KayKit City Builder Bits were downloaded and vendored on 2026-11-23
(~5MB total, every file verified). NPC bodies are the **KayKit
Character Pack: Adventurers 1.0** (CC0, vendored 2026-11-24): five
rigged GLBs sharing a 76-clip animation library, wired per
`webworld/assets/characters/WIRING.md` — role-assigned bodies, per-NPC
tints and walk variants, weapons hidden, planner-driven animation
(walk / idle / sit / cheer one-shots), with the capsule layer kept as
the fallback. Quaternius animated characters could not be fetched
automatically (poly.pizza API needs a key; the pack page exposes no
direct link) and are no longer needed for bodies.

## Upstream note & attribution

This project was built after a read-only study of **Hermes Agent** by
**Nous Research** (https://github.com/NousResearch/hermes-agent, MIT
License, © 2025 Nous Research) — its architecture (agent loop + tool
registry, provider transports over OpenAI-compatible endpoints, dedicated
context-compression subsystem, curated memory with session search, CLI +
gateway split) informed this design, and the same architectural patterns
are re-implemented here in compact, stdlib-first form for low-resource
offline hardware.

No upstream source code is included in this repository: the upstream tree
was flagged by this environment's execution-risk controls as content that
may not be copied, imported, or executed, so it was studied by reading
only, and everything here is original code written for Aaron. The
MIT-licensed upstream remains the right choice when you want the full
framework and have the hardware for it; Hermes Local exists for the
opposite corner — offline, tiny, and NPC-aware. Aaron's additions that
upstream does not provide: the EmbeddingGemma 2 embedding layer with its
fallback chain, the resource-detected brain ladder, the ReAct
fenced-JSON tool-calling fallback, sub-64K context compression as the
default path, SQLite vector memory with per-NPC namespaces, the
proximity activation system, and the world bridge + web city.

## Tests

`python tests/run_tests.py` — 10 tests, all with mock LLMs (no GPU, no
downloads): agent loop in native **and** ReAct modes, real tool execution,
embedding fallback to TF-IDF + memory recall, context compression on a
260-token window, a mock 3D world over a real bridge socket (perception →
episodic recall → dialogue → `move_to`/`interact_with` actions), proximity
tier transitions with the concurrency cap and deactivation summaries,
city layout round-trip through the bridge protocol, roster sanity, the
Nous provider preset resolution, and the HTTP transport (portal route at
`/portal`, README serving for the portal's doc links, and the
brain/provider fields on `/api/status`).

Not verified: live-model behavior against a real Ollama server and real
EmbeddingGemma 2 embeddings (both need hardware/services this
environment doesn't have). The web pages' visuals and frame rate are
also still unseen: both pages' JS is syntax-checked with node, the
standalone JS roster was executed in node and matches the Python
generators exactly (names, roles, personas, spawn positions), and the
served endpoints (`/`, `/portal`, `/api/status`, `/api/roster`, a GLB,
`/README.md`) were smoke-tested live against `serve --mock` — including
a proximity → activation → speech → `follow` action round-trip over
HTTP. But headless screenshots could not be captured in this VM: its
Chromium build refuses startup without a WebGL domain policy, and the
puppeteer headless shell's GPU/WebGL stack fails (context creation
errors, no output file). So the city has still never been *seen*
rendering — first visual check happens when Aaron opens it.
