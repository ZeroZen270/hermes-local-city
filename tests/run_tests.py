"""Hermes Local test suite — plain asserts, mock LLMs, no GPU, no downloads.

Run:  python tests/run_tests.py
Covers:
  (a) agent loop completes tasks with a mock LLM in BOTH native and ReAct modes
  (b) real tools (file write/read, shell echo) execute through the loop
  (c) embedding fallback chain reaches the built-in TF-IDF embedder and
      memory recall returns the right note
  (d) context compression triggers on a small window
  (e) mock 3D world over the real bridge socket: perception -> episodic
      memory, proximity activation, in-character recall, body actions
  (f) proximity tiers: concurrency cap enforced; leaving range summarizes
      the encounter into episodic memory and unloads context
  (g) city layout JSON round-trip through the bridge protocol
  (h) procedural roster: 54 unique real character cards
  (i) nous provider preset resolves endpoint/model, key from the env
  (j) HTTP transport: /portal page, /README.md, status brain_info
  (k) tier hysteresis: oscillation across the old 15m line does not flap
      tiers; exit bands + 5s minimum hold govern demotions
  (l) conversation lock: a spoken-to NPC stays pinned ACTIVE (and its
      planner state is "engaged") until 20s of silence or the player
      walks beyond 25m
  (m) itinerary planner: over a simulated 10 minutes NPCs spend the
      majority of their time dwelling, not traveling
  (n) witnessed social pairing with a live generator hook uses the
      generated exchange (events + episodic memory), not canned lines
  (o) generator raising in live mode: NO line is emitted for the
      pairing (silence, never canned); a witnessed pair gets exactly
      one retry ~5 sim-s later, then stays quiet
  (p) per-pair cooldown: a second pairing inside 90s sim time does not
      call the generator again and emits nothing at all
  (q) souls: export -> fresh world -> knowledge load restores cards +
      memories but NOT situations (spawns + fresh planner state)
  (r) souls: full load restores positions + anchor/activity too
  (s) souls: malformed snapshots (and bad modes) are rejected with the
      running world untouched; the /api/souls HTTP routes behave
  (t) ghost mode: point-blank ghost proximity releases the conversation
      lock and ghost speech goes unanswered, while tiers still run from
      the ghost's position (the NPC stays ACTIVE and audible)
  (u) conversation-born plans: a generated exchange ending in a PLAN
      agreement is stored + announced + remembered by both NPCs, and at
      the due time both planners travel to the rendezvous anchor, where
      arrival completes the plan and pairs them up again
  (v) no plan without a marker: the plain-lines generator contract and
      an invalid plan anchor both deliver the exchange with no plan
  (w) no tier gate: a pairing far from the player (both DORMANT) in
      live mode still gets a generated exchange, and zero canned lines
  (x) witnessed conversation continues: while the pair stays social a
      second exchange is requested, and the continuation call receives
      the first exchange's lines as history
  (y) generator raising for an UNwitnessed pairing: attempted once, no
      retry, and no social line is emitted at all
  (z) soul-fed prompts: the bridge generator's prompt contains a
      distinctive memory seeded into the NPC's own namespace
  (aa) world interpreter, live: POST /api/ask hands the fake brain a
      digest containing the NPC's current activity and the pending
      plan, and returns the brain's answer verbatim
  (ab) world interpreter: a question whose terms match a seeded
      memory puts that memory (attributed) into the brain's context
  (ac) world interpreter, mock mode: "Where is <name>?" and the
      citizen count are answered from the digest with no LLM involved
  (ad) npc chat: a GENERIC rejection of the native tools call (the
      LongCat shape, not ToolUnsupportedError) degrades the NPC to
      ReAct and retries in-turn — real reply, no second tools attempt
  (ae) npc chat: tools call and plain retry both failing keeps the
      "(seems distracted.)" line as the last resort
  (af) resources P1: player interact in range harvests (inventory up,
      node stock down, harvest event); out of range and depleted
      interacts are refused with a notice and change nothing
  (ag) resources P1: a felled tree regrows stump -> sapling -> mature
      over its regrow window in sim ticks; plants regrow unit by
      unit; tending an animal never removes it
  (ah) resources P1: GET /api/world returns inventory + nodes with
      the expected fields, and /api/status carries the inventory
  (ai) labor P2: the foreman auto-assigns gather_wood below the
      baseline stock; the citizen fells a tree via the NPC harvest
      path and the city wood stock rises
  (aj) labor P2: craft tasks convert at stations (1 wood -> 1 plank,
      2 food -> 1 meal); a craft whose inputs vanished mid-task
      fails gracefully and consumes nothing
  (ak) labor P2: the order protocol assigns the named citizen, and
      an engaged (conversation-locked) citizen does not start work
      until the lock is released
  (al) building P3: a placed stall is foreman-staffed and raised
      frame -> complete; exact cost deducted; building in /api/world;
      a market anchor joins the planner's anchors
  (am) building P3: a house short on planks sits in
      awaiting_materials consuming nothing until the stores suffice,
      then completes at full cost and adds a home anchor
  (an) building P3: sites + buildings ride the layout document and
      reload with stage/progress/materials intact; a souls load
      leaves construction untouched
  (ao) terraforming P4: a god raise applies in one call with radial
      falloff (centre > edge > rim, outside untouched); the grid
      serialize -> restores exactly and rides /api/world
  (ap) terraforming P4: a crew flatten opens a job (no instant
      edit); a citizen is assigned, works the patch for ~30 attended
      sim-seconds, and the heights converge exactly to the goal map
      with a crew completion event
  (aq) terraforming P4: out-of-bounds / oversize / unknown ops are
      notices that change nothing (crew mode included); a malformed
      heightmap document is rejected whole; a valid god op is
      instant
  (ar) civics P5: a player proposal for a house runs the full cycle
      with a fake AI ballot caster — all 54 citizens vote through it
      (soul-fed: a seeded distinctive memory reaches the fake), the
      tally adopts, a real construction site exists afterwards, and
      every citizen holds a kind="civic" memory with their vote
  (as) civics P5: a majority AGAINST rejects — no site is created,
      and the outcome memory is still written for every citizen
  (at) civics P5: live failure paths — a ballot call that raises is
      ABSTAIN "couldn't be reached" (called once, NO rule ballot
      substituted) and garbage is retried once then ABSTAIN "no
      clear answer"; mock mode completes a full cycle on rule
      ballots alone (the demo path)
  (au) research P6: a fake fetcher serves DuckDuckGo-shaped HTML;
      the terminal's insight lands in city knowledge with sources
      preserved AND in the asking citizen's kind="research" memory;
      a fetcher that raises answers no_connection and stores NOTHING
  (av) innovation P6 flagship: masonry end-to-end — research hint
      -> adopt_innovation adopted on fake AI ballots -> the fake
      brain's bricks spec registers live (clay pits, kiln + brick
      house in the catalog, generalized recipe registry) -> kiln
      built -> clay gathered -> bricks crafted (share routed to
      market stock) -> a sale ticks stock down and trade up; the
      registered state round-trips through the world save
  (aw) innovation P6: an invalid spec (unknown recipe input + bad
      station archetype) gets exactly ONE revision call with the
      errors fed back, then files "under study" — nothing is
      registered and the proposer remembers the outcome
  (ax) persistence P7: the canonical world snapshot round-trips a
      mutated world (terrain, a construction site, a civic history
      entry, a city-knowledge entry, a registered innovation + its
      market, an NPC memory) into a fresh bridge; a document with
      one corrupted section applies NOTHING
  (ay) persistence P7: repo sync over a fake GitHub transport —
      first push creates (no sha), second updates with the fetched
      sha; startup pull applies a newer remote (and writes it
      local) and ignores an older one; a failing transport means
      status offline, the local file intact, and a world that
      keeps ticking
  (az) persistence P7: a proposal mid-voting snapshots and
      restores with its ballots intact and its state voting; only
      the citizens who had not cast are re-cast, the pre-snapshot
      ballots are byte-identical afterwards, and the vote completes
  (ba) designs P8: seeded interiors/assets exist pre-brain; valid
      building_style + interior specs register; an off-palette
      style is refused; a bad interior (unknown prop + prop
      outside the footprint) is revised exactly once with the
      errors fed back, then shelved with nothing registered
  (bb) voted assets P8: an asset proposal runs the civic cycle to
      adoption, the fake design brain's fountain composition
      registers with its props in the served layout document
      (designs slice + layout marker), and a persistence snapshot
      round-trip carries it into a fresh world
  (bc) weather P8: seeded engines transition identically; god
      set_weather/set_time answer with weather events and ride
      /api/status; the rule interpreter answers weather/time from
      the set state; a snapshot round-trip preserves clock+weather
  (bd) growth P9: god expansion 120 -> 180 preserves every core
      height at its coordinates, eases ring relief in with no seam
      cliff, adds the four ring-1 district anchors, and the terrain
      slice round-trips bound + grid; ring 2 follows, /api/world
      reports the live geometry, the interpreter knows the span
  (be) growth P9: an adopted "expand" civic proposal (fake AI
      ballots) grows the land at once, emits the expansion event,
      and queues exactly two flatten crew jobs inside the new ring
  (bf) growth P9: placement follows the live bound (a fence fits
      in the new ring at 180, refused at 200); at the ±360 cap an
      expansion attempt is a notice and nothing changes
  (bg) regression: the OpenAI client never doubles a /v1 prefix
      already present in the base URL (chat + embeddings)
  (bh) walk grid P10: dimensions follow the live bound; building
      footprints block with their 0.4m inflation margin; ponds
      and steep ground block
  (bi) walk grid P10: A* routes around a house (no blocked cell,
      no corner-cut diagonals), LOS smoothing collapses an open
      route, and the snapper pulls a point out of a footprint
  (bj) walk grid P10: /api/world carries the grid (base64
      round-trip); a build target behind a house puts a clean
      route on the travel intent; completion bumps the version
  (bk) location context P11: the situation string names the
      anchor + clock at a market stall, says "middle of the road"
      (and claims nothing) mid-road, finds the pond; the string
      reaches a mock-LLM player chat's system prompt verbatim
  (bl) canned lines P11: the demo brain's vendor greeting is
      location-gated — no cart pitch mid-road, the pitch returns
      at the stall
  (bm) schedules P11: at 23:30 a vendor's planner pick is home
      (at_home set, sleep dwell), at 09:00 it is the workplace;
      a guard at 23:30 is NOT sent home (night watch); the
      default day length is 2880s with the 60s floor intact
  (bn) version copies P12: a changed push archives one dated
      copy under versions/; the throttle suppresses a second
      inside the window and admits one after it; retention
      prunes to version_keep, oldest first; a version-side
      failure leaves the main push synced with a status note;
      local-only never attempts a version
  (bo) workplace pools + claim radius: a shopkeeper's workplace
      is a browse anchor (not a house front), an engineer's is
      errand; situation_text claims workplace/home only within
      5m — at ~8m it says "near <name>" plus the surface, with
      no ownership claim
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import hermes_local.llm as llm_mod                        # noqa: E402
from hermes_local.agent import Agent                      # noqa: E402
from hermes_local.embeddings import EmbeddingClient, truncate  # noqa: E402
from hermes_local.interpreter import WorldInterpreter     # noqa: E402
from hermes_local.memory import Memory                    # noqa: E402
from hermes_local.npc import (Director, NPC, NPCManager,  # noqa: E402
                              _SOCIAL_LINES, generate_roster, load_card)
from hermes_local.resources import (ResourceNode, ResourceWorld,  # noqa: E402
                                    default_resource_nodes)
from hermes_local.terrain import TerrainWorld           # noqa: E402
from hermes_local.world_bridge import (WorldBridge,      # noqa: E402
                                       city_spawn_positions,
                                       make_conversation_generator,
                                       make_socket_server, validate_layout)

DEAD = "http://127.0.0.1:9"  # nothing listens here: forces offline fallbacks


def dead_embedder():
    return EmbeddingClient(base_url=DEAD, timeout=2)


def tmp_memory(tmp):
    return Memory(str(Path(tmp) / "mem.db"), str(Path(tmp) / "notes.md"),
                  dead_embedder(), search_dim=256)


# --------------------------------------------------------------------------
# Mock LLMs
# --------------------------------------------------------------------------
class ScriptedMock:
    """Returns queued responses, then a default final answer."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        if self.script:
            return self.script.pop(0)
        return {"content": "Done.", "tool_calls": []}


class ReactMock:
    """ReAct-style model: emits fenced tool blocks, checks tool results."""

    def chat(self, messages, tools=None):
        blob = json.dumps(messages)
        if "Tool results:" in blob and "react-works" in blob:
            return {"content": "Confirmed: react-works", "tool_calls": []}
        return {"content": 'Running echo.\n```tool\n'
                           '{"name": "run_shell", "arguments": '
                           '{"command": "echo react-works"}}\n```',
                "tool_calls": []}


class LoopMock:
    """Keeps calling a tool with bulky output, then finishes."""

    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        sys0 = messages[0].get("content", "") if messages else ""
        if "Summarize" in sys0:
            return {"content": "SUMMARY-TEXT", "tool_calls": []}
        self.n += 1
        if self.n <= 5:
            return {"content": "", "tool_calls": [
                {"id": f"c{self.n}", "name": "run_shell",
                 "arguments": {"command": "echo " + "y" * 120}}]}
        return {"content": "finished", "tool_calls": []}


class SummaryMock:
    def chat(self, messages, tools=None):
        return {"content": "Encounter summary: a chat happened.", "tool_calls": []}


class ToolsAllergicMock:
    """Provider that rejects the native `tools` array with a GENERIC
    error (the Nous LongCat shape — not a ToolUnsupportedError) but
    answers plain chat calls in character."""

    def __init__(self):
        self.calls = []  # one bool per chat call: True if tools were passed

    def chat(self, messages, tools=None):
        self.calls.append(bool(tools))
        if tools:
            raise RuntimeError("HTTP 400: tools are not supported by this model")
        return {"content": "Aye — the gate roster is pinned by the door.",
                "tool_calls": []}


class AlwaysDownMock:
    """Every chat call fails, with or without tools."""

    def chat(self, messages, tools=None):
        raise RuntimeError("connection refused")


class WorldMock:
    """In-character mock for the bridge test: recalls, then acts."""

    def chat(self, messages, tools=None):
        sys0 = messages[0].get("content", "") if messages else ""
        if "Summarize" in sys0:
            return {"content": "Summary.", "tool_calls": []}
        blob = json.dumps(messages)
        last_user = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        low = last_user.lower()
        if "where" in low and "toolbox" in blob:
            return {"content": "Aye — it's by the red toolbox near the gate.",
                    "tool_calls": []}
        tool_names = [m.get("name") for m in messages if m.get("role") == "tool"]
        if "grab" in low or "go to" in low:
            if "move_to" not in tool_names:
                return {"content": "", "tool_calls": [
                    {"id": "w1", "name": "move_to",
                     "arguments": {"target": "red toolbox"}}]}
            if "interact_with" not in tool_names:
                return {"content": "", "tool_calls": [
                    {"id": "w2", "name": "interact_with",
                     "arguments": {"object": "wrench"}}]}
            return {"content": "Got it — wrench in hand.", "tool_calls": []}
        return {"content": "Aye?", "tool_calls": []}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_agent_native_and_tools():
    with tempfile.TemporaryDirectory() as tmp:
        mock = ScriptedMock([
            {"content": "", "tool_calls": [
                {"id": "c1", "name": "write_file",
                 "arguments": {"path": "hello.txt", "content": "hi there"}}]},
            {"content": "", "tool_calls": [
                {"id": "c2", "name": "read_file",
                 "arguments": {"path": "hello.txt"}}]},
            {"content": "File says: hi there", "tool_calls": []},
        ])
        agent = Agent(mock, config={"agent": {"tool_mode": "native"}},
                      workspace=tmp)
        out = agent.run("write hello.txt then read it back")
        assert "hi there" in out, out
        assert (Path(tmp) / "hello.txt").read_text() == "hi there"
        tool_msgs = [m for call in mock.calls for m in call
                     if m.get("role") == "tool"]
        assert any("hi there" in (m.get("content") or "") for m in tool_msgs), \
            "read_file result never reached the model"


def test_agent_react():
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(ReactMock(), config={"agent": {"tool_mode": "react"}},
                      workspace=tmp)
        out = agent.run("echo something for me")
        assert "react-works" in out, out


def test_embedding_fallback_and_memory():
    vec = truncate([1.0] + [0.0] * 767, 256)
    assert len(vec) == 256 and abs(vec[0] - 1.0) < 1e-6
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        mem.remember("The RV fridge runs on propane or 12V DC power.")
        mem.remember("The Focus has a Crimestopper remote start fitted.")
        mem.remember("Starlink is the RV internet uplink.")
        assert mem.embedder.active_name == "local-hash-tfidf", mem.embedder.active_name
        hits = mem.recall("fridge propane power", k=1)
        assert hits and "fridge" in hits[0]["text"], hits
        assert (Path(tmp) / "notes.md").is_file()


def test_context_compression():
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(LoopMock(),
                      config={"agent": {"tool_mode": "native",
                                        "context_tokens": 260,
                                        "keep_recent_turns": 4}},
                      workspace=tmp)
        out = agent.run("keep echoing")
        assert out == "finished", out
        assert agent.stats["compressions"] >= 1, agent.stats
        assert any("[Summary of earlier conversation]" in (m.get("content") or "")
                   for m in agent.messages)


class LineReader:
    """Buffered JSON-lines reader over a raw socket (select-based, so
    timeouts never corrupt the buffer the way makefile timeouts can)."""

    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def read_until(self, pred, timeout=4.0):
        import select
        got, deadline = [], time.time() + timeout
        while time.time() < deadline:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                if line.strip():
                    got.append(json.loads(line))
                    if pred(got):
                        return got
            ready, _, _ = select.select([self.sock], [], [], 0.25)
            if ready:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self.buf += chunk
        return got


def test_mock_world_over_socket():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        card = load_card(PROJECT / "characters" / "mara.json")
        npc = NPC(card, WorldMock(), memory=mem, embedder=mem.embedder)
        npcs = {card.name: npc}
        manager = NPCManager(npcs, max_active=4)
        bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
        server = make_socket_server(bridge, "127.0.0.1", 0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                reader = LineReader(s)

                def send(obj):
                    s.sendall((json.dumps(obj) + "\n").encode())

                send({"type": "register_world",
                      "actions": {"say": {"required": ["text"]},
                                  "move_to": {"required": ["target"]},
                                  "interact_with": {"required": ["object"]}}})
                got = reader.read_until(
                    lambda g: any(m.get("type") == "ack" for m in g))
                assert any(m.get("type") == "ack" for m in got), got

                send({"type": "perception", "npc": "Mara",
                      "event": "Mara left her wrench by the red toolbox near the gate."})
                got = reader.read_until(
                    lambda g: any(m.get("type") == "ack" for m in g))
                assert any(m.get("type") == "ack" for m in got), got

                send({"type": "proximity", "player": [0, 0],
                      "npcs": {"Mara": [3, 0]}})
                got = reader.read_until(lambda g: any(
                    m.get("type") == "tier" and m.get("tier") == "active"
                    for m in g))
                assert any(m.get("tier") == "active" for m in got), got
                assert npc.active

                send({"type": "speech", "to": "Mara", "from": "Player",
                      "text": "Where did you leave your wrench?"})
                got = reader.read_until(lambda g: any(
                    m.get("type") == "dialogue" for m in g))
                dlg = [m for m in got if m.get("type") == "dialogue"]
                assert dlg and "toolbox" in dlg[0]["text"], got

                send({"type": "speech", "to": "Mara", "from": "Player",
                      "text": "Go to the red toolbox and grab the wrench."})
                got = reader.read_until(lambda g: any(
                    m.get("type") == "dialogue" for m in g)
                    and any(m.get("action") == "interact_with" for m in g))
                actions = [m.get("action") for m in got if m.get("type") == "action"]
                assert "move_to" in actions, got
                assert "interact_with" in actions, got
        finally:
            server.shutdown()


def test_proximity_tiers_and_cap():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = generate_roster(6)
        npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
                for c in cards}
        manager = NPCManager(npcs, active_radius=15, ambient_radius=50,
                             max_active=4)
        positions = {name: (i * 2.0, 0.0) for i, name in enumerate(npcs)}
        manager.update((0, 0), positions)
        stats = manager.stats()
        assert stats["active"] == 4, stats          # cap enforced with 6 in radius
        assert stats["ambient"] == 2, stats
        active_names = list(stats["active_names"])
        assert all(npcs[n].active for n in active_names)
        # Give the encounters substance: something actually happens
        # while they are ACTIVE (a bare walk-past files nothing —
        # see the second half of this test).
        for name in active_names:
            npcs[name].witness(f"{name} traded words with the player "
                               f"at the market square.")

        manager.update((500, 500), positions)        # player leaves
        stats = manager.stats()
        assert stats["dormant"] == 6, stats
        for name in active_names:
            npc = npcs[name]
            assert not npc.active and npc.history == []
            hits = mem.recall("encounter", k=4, namespace=npc.ns)
            assert any(h["kind"] == "encounter" for h in hits), \
                f"no encounter summary stored for {name}"

        # A bare pass files NOTHING: re-activate one citizen with
        # no interaction at all, then leave — no new encounter
        # memory, no summary (the Lila spam fix, 2026-11-24).
        npc = npcs[active_names[0]]
        before = len([e for e in mem.list_namespace(npc.ns)
                      if e.get("kind") == "encounter"])
        manager.update((0, 0), positions)
        assert npc.active
        manager.update((500, 500), positions)
        after = len([e for e in mem.list_namespace(npc.ns)
                     if e.get("kind") == "encounter"])
        assert after == before, (before, after)


def test_layout_round_trip():
    bridge = WorldBridge({}, manager=None)
    doc = {"version": 1, "objects": [
        {"asset": "building_a", "position": [10, 0, -5], "rotation": 90,
         "scale": 2, "kind": "building"},
        {"asset": "road_straight", "position": [0, 0, 0], "rotation": [0, 0, 0],
         "scale": [1, 1, 1], "kind": "road"},
        {"asset": "spawn_mara", "position": [3, 0, 3], "kind": "spawn"},
    ]}
    resp = bridge.handle_message({"type": "save_layout", "layout": doc})
    assert resp[0].get("ok") is True, resp
    resp = bridge.handle_message({"type": "load_layout"})
    assert resp[0]["layout"]["objects"] == validate_layout(doc)["objects"], resp
    bad = bridge.handle_message({"type": "save_layout",
                                 "layout": {"objects": [{"position": [0, 0]}]}})
    assert bad[0].get("type") == "error", bad


def test_roster():
    cards = generate_roster(54)
    assert len(cards) == 54
    assert len({c.name for c in cards}) == 54
    assert len({c.namespace for c in cards}) == 54
    assert all(c.persona and c.role for c in cards)


def test_provider_preset():
    """(i) The nous-longcat preset resolves endpoint/model and pulls the
    API key from $NOUS_API_KEY; explicit user values beat the preset."""
    import yaml

    from hermes_local.config import PROVIDERS, load_config

    assert "nous-longcat" in PROVIDERS
    old = os.environ.get("NOUS_API_KEY")
    os.environ["NOUS_API_KEY"] = "test-key-not-real"
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "config.yaml"
            p.write_text(yaml.safe_dump({"llm": {"provider": "nous-longcat"}}))
            cfg = load_config(str(p))
            llm = cfg["llm"]
            assert llm["base_url"] == "https://inference-api.nousresearch.com/v1", llm
            assert llm["api"] == "openai", llm
            assert llm["model"] == "meituan/longcat-2.5:free", llm
            assert llm["api_key"] == "test-key-not-real", llm
            # An explicit base_url in the same file wins over the preset.
            p.write_text(yaml.safe_dump({"llm": {
                "provider": "nous-longcat",
                "base_url": "http://example.invalid/v1"}}))
            cfg2 = load_config(str(p))
            assert cfg2["llm"]["base_url"] == "http://example.invalid/v1", cfg2["llm"]
            assert cfg2["llm"]["model"] == "meituan/longcat-2.5:free", cfg2["llm"]
    finally:
        if old is None:
            os.environ.pop("NOUS_API_KEY", None)
        else:
            os.environ["NOUS_API_KEY"] = old


def test_portal_and_status_http():
    """(j) The HTTP transport serves the launch portal at /portal (and
    /portal.html), the demo at /, the project README for the portal's doc
    links, and a status payload carrying build_world's brain_info."""
    import urllib.request

    from hermes_local.world_bridge import make_http_server

    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        card = load_card(PROJECT / "characters" / "mara.json")
        npc = NPC(card, SummaryMock(), memory=mem, embedder=mem.embedder)
        npcs = {card.name: npc}
        manager = NPCManager(npcs, max_active=4)
        bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
        bridge.brain_info = {"brain": "test-brain", "provider": "mock",
                             "ladder_pick": "qwen3:4b"}
        server = make_http_server(bridge, "127.0.0.1", 0,
                                  static_dir=PROJECT / "webworld",
                                  positions={card.name: (0.0, 0.0)},
                                  mode="mock (test)")
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def get(path):
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
                    return resp.status, resp.read().decode("utf-8", "replace")

            status, body = get("/portal")
            assert status == 200 and "Launch Portal" in body, (status, body[:200])
            status, body = get("/portal.html")
            assert status == 200 and "Launch Portal" in body, status
            status, body = get("/")
            assert status == 200 and "Hermes Local" in body, status
            status, body = get("/README.md")
            assert status == 200 and "Hermes Local" in body, status
            status, body = get("/api/status")
            payload = json.loads(body)
            assert payload["npc_count"] == 1, payload
            assert payload["brain"] == "test-brain", payload
            assert payload["provider"] == "mock", payload
            assert payload["ladder_pick"] == "qwen3:4b", payload
            assert payload["mode"] == "mock (test)", payload
        finally:
            server.shutdown()


def test_tier_hysteresis_no_flap():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        card = generate_roster(1)[0]
        npc = NPC(card, SummaryMock(), memory=mem, embedder=mem.embedder)
        manager = NPCManager({card.name: npc}, active_radius=15,
                             ambient_radius=50, max_active=4)
        name = card.name

        def upd(t, d):
            return manager.update((0.0, 0.0), {name: (d, 0.0)}, now=t)

        upd(100.0, 14.0)
        assert manager.tiers[name] == "active", manager.tiers
        # Oscillate across the OLD 15m boundary for 30s: inside the 20m
        # exit band the tier must not flap at all.
        changes, t = 0, 101.0
        for i in range(30):
            ev = upd(t, 19.0 if i % 2 else 16.0)
            changes += sum(1 for e in ev if e.get("type") == "tier")
            assert manager.tiers[name] == "active", (t, manager.tiers)
            t += 1.0
        assert changes == 0, changes
        # Beyond the exit band (minimum hold long past): demote to ambient.
        upd(t, 22.0)
        assert manager.tiers[name] == "ambient", manager.tiers
        # Ambient hysteresis: 55m holds ambient, 65m drops to dormant.
        upd(t + 10.0, 55.0)
        assert manager.tiers[name] == "ambient", manager.tiers
        upd(t + 20.0, 65.0)
        assert manager.tiers[name] == "dormant", manager.tiers
        # Minimum hold: a fresh activation cannot be demoted within 5s...
        upd(500.0, 10.0)
        assert manager.tiers[name] == "active", manager.tiers
        upd(502.0, 30.0)
        assert manager.tiers[name] == "active", manager.tiers
        # ...but once the hold is served, the demotion lands.
        upd(506.0, 30.0)
        assert manager.tiers[name] == "ambient", manager.tiers


def test_conversation_lock():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = generate_roster(3)
        npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
                for c in cards}
        spawns = {c.name: (100.0, 100.0) for c in cards}
        name = cards[0].name
        manager = NPCManager(npcs, max_active=4, spawns=spawns)
        pos = {name: (10.0, 0.0), cards[1].name: (300.0, 300.0),
               cards[2].name: (300.0, 300.0)}
        manager.update((0.0, 0.0), pos, now=1000.0)
        assert manager.tiers[name] == "active", manager.tiers
        manager.note_speech(name, now=1001.0)
        assert manager.is_engaged(name, now=1002.0)
        # Player backs off to 22m — beyond the 20m exit band but inside
        # the 25m engage range: the lock pins the NPC ACTIVE and its
        # planner stands down to "engaged" (it stops and faces the player).
        pos[name] = (22.0, 0.0)
        manager.update((0.0, 0.0), pos, now=1005.0)
        assert manager.tiers[name] == "active", manager.tiers
        assert manager.planner.state[name]["state"] == "engaged", \
            manager.planner.state[name]
        # 19s of silence: still locked. 21s: released, back to ambient.
        manager.update((0.0, 0.0), pos, now=1020.0)
        assert manager.tiers[name] == "active", manager.tiers
        ev = manager.update((0.0, 0.0), pos, now=1022.0)
        assert manager.tiers[name] == "ambient", manager.tiers
        assert any(e.get("type") == "engage" and e.get("engaged") is False
                   for e in ev), ev
        # Distance release: re-engage, then the player steps to 30m.
        manager.note_speech(name, now=1030.0)
        pos[name] = (30.0, 0.0)
        manager.update((0.0, 0.0), pos, now=1031.0)
        assert not manager.is_engaged(name, now=1031.0)
        assert manager.tiers[name] == "ambient", manager.tiers


def _steer(pos, tx, tz, speed, dt):
    """Arrive steering, mirroring the web client's arriveMove."""
    dx, dz = tx - pos[0], tz - pos[1]
    d = (dx * dx + dz * dz) ** 0.5
    if d <= 0.35:
        return pos
    slow = min(1.0, max(0.22, (d - 0.35) / 3.0))
    step = min(d - 0.35, speed * slow * dt)
    return (pos[0] + dx / d * step, pos[1] + dz / d * step)


def test_planner_dwell_majority():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = generate_roster(12)
        npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
                for c in cards}
        spawns = city_spawn_positions([c.name for c in cards])
        manager = NPCManager(npcs, max_active=4, spawns=spawns)
        planner = manager.planner
        assert planner is not None
        positions = dict(spawns)
        player = (500.0, 500.0)  # far away: tiers stay dormant, planner runs
        for step in range(600):  # 10 simulated minutes at 1s steps
            manager.update(player, positions, now=float(step))
            for name, st in planner.state.items():
                if st["state"] == "travel":
                    positions[name] = _steer(positions[name], st["tx"],
                                             st["tz"], st["speed"], 1.0)
        stats = planner.stats()
        assert stats["travel_s"] > 0, stats      # they do travel...
        assert stats["dwell_share"] > 0.5, stats  # ...but dwell the majority


class FakeGen:
    """Conversation-generator hook double: records calls, returns fixed
    lines (or raises, when `error` is set)."""

    def __init__(self, lines=None, error=None):
        self.lines = lines
        self.error = error
        self.calls = []

    def __call__(self, card_a, card_b, anchor_name):
        self.calls.append((card_a.name, card_b.name, anchor_name))
        if self.error is not None:
            raise self.error
        return list(self.lines or [])


class PlanGen:
    """Generator double on the extended contract: returns the lines
    plus a plan anchor name (valid or not — the planner must check)."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def __call__(self, card_a, card_b, anchor_name):
        self.calls.append(anchor_name)
        return ([(card_a.name, "Lovely running into you here."),
                 (card_b.name, "Let's meet properly later — the plaza?")],
                self.plan)


class HistoryGen:
    """Generator double on the history-aware contract: records the
    history transcript of every call and answers with fresh, numbered
    lines each time, so continuations are distinguishable."""

    def __init__(self):
        self.calls = []      # one history snapshot per call

    def __call__(self, card_a, card_b, anchor_name, history=None):
        hist = list(history or [])
        self.calls.append(hist)
        n = len(self.calls)
        return ([(card_a.name, f"Beat {n} opener from {card_a.name}."),
                 (card_b.name, f"Beat {n} reply from {card_b.name}.")],
                None)


class FakeAmbientClient:
    """Stands in for the LLMClient that make_conversation_generator
    builds internally: captures the exact messages it is sent and
    replies with the canned dialogue in `reply_text`."""

    instances = []
    reply_text = ""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeAmbientClient.instances.append(self)

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return {"content": FakeAmbientClient.reply_text, "tool_calls": []}


def _social_lab(tmp, generator):
    """Two NPCs parked dwelling at the same anchor, player standing on
    top of them (both ACTIVE, so any pairing is witnessed). Returns
    (manager, planner, cards, positions, anchor)."""
    mem = tmp_memory(tmp)
    cards = generate_roster(2)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = {cards[0].name: (0.0, 0.0), cards[1].name: (2.0, 0.0)}
    manager = NPCManager(npcs, max_active=4, spawns=spawns,
                         conversation_generator=generator)
    planner = manager.planner
    anchor = planner.anchors[0]
    for c in cards:
        st = planner.state[c.name]
        st["initialized"] = True
        st["state"] = "dwell"
        st["anchor"] = anchor
        st["kind"] = "social"
        st["until"] = 5000.0
        st["cooldown_until"] = 0.0
    positions = {cards[0].name: (anchor.x, anchor.z),
                 cards[1].name: (anchor.x + 2.0, anchor.z)}
    return manager, planner, cards, positions, anchor


def _tick_for_social(manager, positions, now, tries=60, player=None):
    """Advance sim time 1s per tick until a social text event shows up
    (background-thread generation drains on a later tick)."""
    if player is None:
        player = positions[next(iter(positions))]
    events = []
    for _ in range(tries):
        now += 1.0
        events.extend(manager.update(player, positions, now=now))
        if any(e.get("type") == "social" for e in events):
            return events, now
        time.sleep(0.05)
    return events, now


def _tick_fixed(manager, positions, now, seconds, player=None):
    """Advance sim time a fixed number of 1s ticks, collecting every
    event (used where the assertion is that something does NOT happen,
    or must happen within a window)."""
    if player is None:
        player = positions[next(iter(positions))]
    events = []
    for _ in range(seconds):
        now += 1.0
        events.extend(manager.update(player, positions, now=now))
        time.sleep(0.05)
    return events, now


def test_generated_social_exchange():
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        gen = FakeGen(lines=[(a, "Generated opener about market prices."),
                             (b, "Generated reply, in character.")])
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        events, _now = _tick_for_social(manager, positions, 1000.0)
        social = [e for e in events if e.get("type") == "social"]
        assert len(gen.calls) == 1, gen.calls
        assert gen.calls[0][2] == anchor.name, gen.calls
        assert [e["text"] for e in social] == [
            "Generated opener about market prices.",
            "Generated reply, in character."], social
        assert [e["npc"] for e in social] == [a, b], social
        assert all(e["text"] not in _SOCIAL_LINES for e in social), social
        # Same memory path as canned: ACTIVE witnesses logged the exchange.
        npc_a = manager.npcs[a]
        assert any("Generated opener" in entry
                   for entry in npc_a.encounter_log), npc_a.encounter_log


def test_generated_social_fallback():
    # NEW CONTRACT (live mode): a failed generation is silence — the
    # canned line set exists only when no generator is wired. This lab
    # is witnessed (player on top of the pair), so the pair gets
    # exactly one retry ~5 sim-s after the failure, then stays quiet.
    with tempfile.TemporaryDirectory() as tmp:
        gen = FakeGen(error=RuntimeError("model server exploded"))
        manager, planner, cards, positions, anchor = _social_lab(tmp, gen)
        events, _now = _tick_fixed(manager, positions, 1000.0, 16)
        social = [e for e in events if e.get("type") == "social"]
        assert social == [], social                 # nothing was said
        assert len(gen.calls) == 2, gen.calls       # attempt + one retry


def test_generated_social_pair_cooldown():
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        gen = FakeGen(lines=[(a, "First generated line."),
                             (b, "Second generated line.")])
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        _events, now = _tick_for_social(manager, positions, 1000.0)
        assert len(gen.calls) == 1, gen.calls
        # The pair drifts apart and re-pairs 30 sim-seconds later: past
        # the 3s global gap, well inside the 90s per-pair cooldown.
        # NEW CONTRACT: the re-pairing does not call the generator
        # again — and with no canned fallback in live mode, the pair
        # stands together in silence (no social events at all).
        for c in cards2:
            st = planner.state[c.name]
            st["state"] = "dwell"
            st["partner"] = None
            st["until"] = 5000.0
            st["cooldown_until"] = 0.0
        events, _now = _tick_fixed(manager, positions, now + 29.0, 40)
        social = [e for e in events if e.get("type") == "social"]
        assert len(gen.calls) == 1, gen.calls       # no second generation
        assert social == [], social                 # and nothing canned


def _souls_lab(tmp):
    """Three NPCs under a manager with real (city-grid) spawns."""
    mem = tmp_memory(tmp)
    cards = generate_roster(3)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = city_spawn_positions([c.name for c in cards])
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    return mem, cards, npcs, spawns, manager


def _souls_snapshot(tmp):
    """A populated world, exported: memories written, NPCs moved off
    their spawns, one parked dwelling at a named anchor in a social
    activity. Returns (snapshot, moved_positions, anchor)."""
    mem, cards, npcs, spawns, manager = _souls_lab(tmp)
    a, b = cards[0].name, cards[1].name
    npcs[a].witness(f"{a} saw the ferry dock at dawn.", kind="episode")
    npcs[a].witness("The player's name is Aaron.", kind="fact")
    npcs[b].witness(f"{b} fixed the depot gate hinge.", kind="episode")
    moved = {name: (spawns[name][0] + 30.0, spawns[name][1] - 12.0)
             for name in npcs}
    manager.update((500.0, 500.0), moved, now=1000.0)
    planner = manager.planner
    anchor = planner.anchors[5]
    st = planner.state[a]
    st["initialized"] = True
    st["anchor"] = anchor
    st["kind"] = "social"
    st["state"] = "dwell"
    return manager.export_souls(), moved, anchor


def test_souls_knowledge_load():
    with tempfile.TemporaryDirectory() as tmp:
        snap, moved, anchor = _souls_snapshot(tmp)
        a = generate_roster(3)[0].name
        assert snap["format"] == "hermes-local/souls", snap.get("format")
        assert snap["version"] == 1 and snap["npc_count"] == 3, snap.keys()
        entry = next(e for e in snap["npcs"] if e["name"] == a)
        texts = [m["text"] for m in entry["memories"]]
        assert any("ferry dock" in t for t in texts), texts
        assert entry["situation"]["position"] == list(moved[a]), \
            entry["situation"]
        assert entry["situation"]["anchor"] == anchor.name, entry["situation"]
        assert entry["situation"]["activity"] == "social", entry["situation"]
        entry["card"]["persona"] = "A changed persona for the test."

        with tempfile.TemporaryDirectory() as tmp2:
            mem2, cards2, npcs2, spawns2, manager2 = _souls_lab(tmp2)
            assert mem2.count() == 0
            summary = manager2.import_souls(snap, mode="knowledge")
            assert summary["ok"] and summary["npcs"] == 3, summary
            assert summary["memories"] == 3, summary
            assert summary["skipped"] == [], summary
            # Knowledge restored: memories (with kinds) + the edited card.
            restored = mem2.list_namespace(npcs2[a].ns)
            assert any("ferry dock" in m["text"] for m in restored), restored
            assert any(m["kind"] == "fact" for m in restored), restored
            assert npcs2[a].card.persona == "A changed persona for the test."
            # Situations NOT restored: everyone home at their spawn with
            # fresh planner state — no anchor, no partner, uninitialized.
            for name in npcs2:
                assert manager2.positions[name] == spawns2[name], \
                    (name, manager2.positions)
                st = manager2.planner.state[name]
                assert st["anchor"] is None and st["initialized"] is False, \
                    (name, st)
                assert st["partner"] is None, (name, st)


def test_souls_full_load():
    with tempfile.TemporaryDirectory() as tmp:
        snap, moved, anchor = _souls_snapshot(tmp)
        a = generate_roster(3)[0].name
        with tempfile.TemporaryDirectory() as tmp2:
            mem2, cards2, npcs2, spawns2, manager2 = _souls_lab(tmp2)
            summary = manager2.import_souls(snap, mode="full")
            assert summary["ok"] and summary["npcs"] == 3, summary
            # Situations restored: positions exactly as saved...
            for name in npcs2:
                assert manager2.positions[name] == moved[name], \
                    (name, manager2.positions)
            # ...and the parked NPC resumes at its saved anchor/activity.
            st = manager2.planner.state[a]
            assert st["anchor"] is not None \
                and st["anchor"].name == anchor.name, st
            assert st["kind"] == "social", st
            assert st["state"] == "dwell" and st["initialized"] is True, st
            # Knowledge came along too.
            restored = mem2.list_namespace(npcs2[a].ns)
            assert any("ferry dock" in m["text"] for m in restored), restored


def test_souls_malformed_rejected():
    import urllib.error
    import urllib.request

    from hermes_local.world_bridge import make_http_server

    with tempfile.TemporaryDirectory() as tmp:
        mem, cards, npcs, spawns, manager = _souls_lab(tmp)
        a = cards[0].name
        npcs[a].witness("A memory that must survive bad loads.",
                        kind="episode")
        manager.update((500.0, 500.0), dict(spawns), now=100.0)
        before_positions = dict(manager.positions)
        before_persona = npcs[a].card.persona
        good = manager.export_souls()

        bad_snaps = [
            "not a dict at all",
            {"format": "something/else", "version": 1, "npcs": []},
            {"format": "hermes-local/souls", "version": 99, "npcs": []},
            {"format": "hermes-local/souls", "version": 1, "npcs": {}},
            {"format": "hermes-local/souls", "version": 1,
             "npcs": [{"name": a}]},                       # missing card
            {"format": "hermes-local/souls", "version": 1,
             "npcs": [{"name": a, "card": {},
                        "memories": [{"kind": "x"}]}]},    # memory w/o text
            {"format": "hermes-local/souls", "version": 1,
             "npcs": [{"name": a, "card": {},
                        "situation": {"position": [1, 2, 3]}}]},
        ]
        for bad in bad_snaps:
            try:
                manager.import_souls(bad, mode="knowledge")
                raise AssertionError(f"accepted malformed snapshot: {bad!r}")
            except ValueError:
                pass
        try:
            manager.import_souls(good, mode="banana")
            raise AssertionError("accepted a bogus load mode")
        except ValueError:
            pass
        # The running world is untouched by every rejected load.
        assert manager.positions == before_positions, manager.positions
        assert npcs[a].card.persona == before_persona
        assert mem.count() == 1, mem.count()
        assert any("must survive" in m["text"]
                   for m in mem.list_namespace(npcs[a].ns))

        # Same story over HTTP: bad POST -> 400, good POST -> summary,
        # GET returns the snapshot document.
        bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
        server = make_http_server(bridge, "127.0.0.1", 0,
                                  static_dir=PROJECT / "webworld",
                                  positions=dict(spawns), mode="mock (test)")
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            def post(payload):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/souls",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        return resp.status, json.loads(resp.read())
                except urllib.error.HTTPError as err:
                    return err.code, json.loads(err.read())

            status, out = post({"mode": "knowledge",
                                "snapshot": bad_snaps[3]})
            assert status == 400 and "error" in out, (status, out)
            status, out = post({"mode": "knowledge", "snapshot": good})
            assert status == 200 and out["ok"] and out["npcs"] == 3, \
                (status, out)
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/souls", timeout=5) as resp:
                got = json.loads(resp.read())
            assert got["format"] == "hermes-local/souls", got
            assert got["npc_count"] == 3, got
        finally:
            server.shutdown()


def test_ghost_mode():
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = generate_roster(2)
        npcs = {c.name: NPC(c, SummaryMock(), memory=mem,
                            embedder=mem.embedder) for c in cards}
        spawns = {cards[0].name: (2.0, 0.0), cards[1].name: (300.0, 300.0)}
        manager = NPCManager(npcs, max_active=4, spawns=spawns)
        bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
        a, far = cards[0].name, cards[1].name
        pos = {a: (2.0, 0.0), far: (300.0, 300.0)}

        def proximity(ghost):
            return bridge.handle_message(
                {"type": "proximity", "player": [0, 0], "npcs": pos,
                 "ghost": ghost})

        def speech():
            return bridge.handle_message(
                {"type": "speech", "to": a, "from": "Player",
                 "text": "Hello there."})

        # Solid player, point blank: proximity activates, speech is
        # answered, and the conversation lock engages the planner.
        proximity(False)
        assert manager.ghost is False and manager.tiers[a] == "active"
        resp = speech()
        assert any(m.get("type") == "dialogue" for m in resp), resp
        assert manager.is_engaged(a)
        proximity(False)
        assert manager.planner.state[a]["state"] == "engaged", \
            manager.planner.state[a]
        turns = npcs[a].stats["turns"]
        assert turns == 1, npcs[a].stats

        # Ghost at the SAME point-blank position: the lock is released
        # (engage event + planner stands down) — but the tier still runs
        # from the ghost's position, so Mara stays ACTIVE and audible.
        resp = proximity(True)
        assert manager.ghost is True
        assert not manager.is_engaged(a)
        assert any(m.get("type") == "engage" and m.get("engaged") is False
                   for m in resp), resp
        assert manager.planner.state[a]["state"] != "engaged", \
            manager.planner.state[a]
        assert manager.tiers[a] == "active", manager.tiers

        # Ghost speech is not answered: a notice, no dialogue, no turn,
        # and it cannot re-engage her.
        resp = speech()
        assert not any(m.get("type") == "dialogue" for m in resp), resp
        assert any(m.get("type") == "notice" for m in resp), resp
        assert npcs[a].stats["turns"] == turns, npcs[a].stats
        assert not manager.is_engaged(a)

        # Ghost off: normal service resumes.
        proximity(False)
        assert manager.ghost is False
        resp = speech()
        assert any(m.get("type") == "dialogue" for m in resp), resp
        assert npcs[a].stats["turns"] == turns + 1, npcs[a].stats


def test_conversation_plan_rendezvous():
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        gen = PlanGen("Central Plaza")
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        events, now = _tick_for_social(manager, positions, 1000.0)
        social = [e for e in events if e.get("type") == "social"]
        assert len(social) == 2, social        # exchange delivered as usual
        # The plan formed: stored, announced, and remembered by BOTH.
        plans = [e for e in events if e.get("type") == "plan"]
        assert len(plans) == 1, plans
        assert plans[0]["where"] == "Central Plaza", plans
        assert {plans[0]["a"], plans[0]["b"]} == {a, b}, plans
        assert len(planner.pending_plans) == 1, planner.pending_plans
        plan = planner.pending_plans[0]
        assert plan["anchor"].name == "Central Plaza", plan
        assert plan["due"] > now, plan
        for name, other in ((a, b), (b, a)):
            mems = manager.npcs[name].memory.list_namespace(
                manager.npcs[name].ns)
            assert any(m["kind"] == "plan" and other in m["text"]
                       and "Central Plaza" in m["text"] for m in mems), mems
        # Past the due time, both planners target the rendezvous anchor.
        player = positions[a]
        due_events = manager.update(player, positions, now=plan["due"] + 1.0)
        for name in (a, b):
            st = planner.state[name]
            assert st["anchor"] is not None \
                and st["anchor"].name == "Central Plaza", (name, st)
            assert st["state"] == "travel", (name, st)
        intents = [e for e in due_events if e.get("type") == "intent"
                   and e.get("anchor") == "Central Plaza"]
        assert {e["npc"] for e in intents} == {a, b}, due_events
        # Arrival completes the plan: the queue empties and the pair is
        # left together at the plaza — the ordinary pairing machinery
        # immediately picks them up for their follow-up chat.
        positions[a] = (8.0, 8.0)
        positions[b] = (10.0, 8.0)
        manager.update(player, positions, now=plan["due"] + 6.0)
        assert planner.pending_plans == [], planner.pending_plans
        for name, other in ((a, b), (b, a)):
            st = planner.state[name]
            assert st["anchor"] is not None \
                and st["anchor"].name == "Central Plaza", (name, st)
            assert st["state"] == "social" and st["partner"] == other, \
                (name, st)


def test_no_plan_without_marker():
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        # Original contract: a plain list of lines, no plan tuple.
        gen = FakeGen(lines=[(a, "Just a chat, nothing planned."),
                             (b, "Agreed — no plans at all.")])
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        events, _now = _tick_for_social(manager, positions, 1000.0)
        assert any(e.get("type") == "social" for e in events), events
        assert not any(e.get("type") == "plan" for e in events), events
        assert planner.pending_plans == [], planner.pending_plans
    with tempfile.TemporaryDirectory() as tmp:
        # Extended contract, but the named place doesn't exist: the
        # exchange is delivered and no plan is stored.
        gen = PlanGen("Atlantis")
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        events, _now = _tick_for_social(manager, positions, 1000.0)
        assert any(e.get("type") == "social" for e in events), events
        assert not any(e.get("type") == "plan" for e in events), events
        assert planner.pending_plans == [], planner.pending_plans


def test_generated_social_unwitnessed():
    # (w) No tier gate: the player is half a city away (both NPCs
    # DORMANT), yet in live mode the pairing still gets a fully
    # generated exchange — and not one canned line.
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        gen = FakeGen(lines=[(a, "Far-side generated opener."),
                             (b, "Far-side generated reply.")])
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        far = (500.0, 500.0)
        events, _now = _tick_for_social(manager, positions, 1000.0,
                                        player=far)
        assert manager.tiers[a] == "dormant" and manager.tiers[b] == "dormant", \
            manager.tiers
        social = [e for e in events if e.get("type") == "social"]
        assert len(gen.calls) == 1, gen.calls
        assert [e["text"] for e in social] == [
            "Far-side generated opener.", "Far-side generated reply."], social
        assert all(e["text"] not in _SOCIAL_LINES for e in social), social


def test_generated_social_continuation():
    # (x) A witnessed pair keeps talking: after the first exchange
    # lands, a continuation is requested while they are still paired,
    # and the continuation call receives the first exchange's lines as
    # its history transcript.
    with tempfile.TemporaryDirectory() as tmp:
        cards = generate_roster(2)
        a, b = cards[0].name, cards[1].name
        gen = HistoryGen()
        manager, planner, cards2, positions, anchor = _social_lab(tmp, gen)
        events, now = _tick_for_social(manager, positions, 1000.0)
        assert len(gen.calls) == 1, gen.calls
        assert gen.calls[0] == [], gen.calls       # opener: no history yet
        first = [(a, f"Beat 1 opener from {a}."),
                 (b, f"Beat 1 reply from {b}.")]
        assert [e["text"] for e in events if e.get("type") == "social"] == \
            [t for _s, t in first], events
        # Keep them paired and witnessed: the continuation lands inside
        # the cadence window (15-30 sim-s) even though a natural
        # pairing might have rolled a shorter duration.
        events2, _now = _tick_fixed(manager, positions, now, 45)
        assert len(gen.calls) >= 2, gen.calls
        assert gen.calls[1] == first, gen.calls    # history handed over
        texts2 = [e["text"] for e in events2 if e.get("type") == "social"]
        assert f"Beat 2 opener from {a}." in texts2, texts2
        assert f"Beat 2 reply from {b}." in texts2, texts2


def test_generated_social_unwitnessed_failure_silent():
    # (y) A failing generator for an UNwitnessed pairing: attempted
    # exactly once (the retry is a witnessed-pair privilege) and not a
    # single social line is emitted — silence, never canned.
    with tempfile.TemporaryDirectory() as tmp:
        gen = FakeGen(error=RuntimeError("model server exploded"))
        manager, planner, cards, positions, anchor = _social_lab(tmp, gen)
        events, _now = _tick_fixed(manager, positions, 1000.0, 14,
                                   player=(500.0, 500.0))
        social = [e for e in events if e.get("type") == "social"]
        assert social == [], social
        assert len(gen.calls) == 1, gen.calls


def test_bridge_generator_soul_prompt():
    # (z) The bridge generator writes from the soul: a distinctive
    # memory seeded into Mara's own namespace must show up in the
    # prompt the (fake) ambient client is sent.
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = generate_roster(2)
        card_a, card_b = cards
        distinctive = "ZQX turnip festival disaster of last spring"
        mem.remember(f"Mara still laughs about the {distinctive}.",
                     kind="episode", namespace=card_a.namespace)
        mem.remember("Joss once lent Mara his spare wrench at the depot.",
                     kind="episode", namespace=card_a.namespace)
        stub = type("Stub", (), {"base_url": DEAD, "model": "fake-model",
                                 "api": "ollama", "temperature": 0.7,
                                 "api_key": ""})()
        FakeAmbientClient.instances = []
        FakeAmbientClient.reply_text = (
            f"{card_a.name}: Lovely weather for the district.\n"
            f"{card_b.name}: Aye, that it is.")
        real_client = llm_mod.LLMClient
        llm_mod.LLMClient = FakeAmbientClient
        try:
            gen = make_conversation_generator(
                stub, anchor_names=["Central Plaza"], memory=mem)
            out = gen(card_a, card_b, "Central Plaza")
        finally:
            llm_mod.LLMClient = real_client
        assert out is not None and len(out[0]) == 2, out
        ambient = FakeAmbientClient.instances[-1]
        assert len(ambient.calls) == 1, ambient.calls
        prompt = ambient.calls[0][-1]["content"]
        assert distinctive in prompt, prompt
        assert "remembers:" in prompt, prompt


class FakeAskClient:
    """LLMClient double for the World Interpreter: captures the exact
    messages it is sent and returns a fixed oracle answer."""

    def __init__(self, answer="The interpreter has spoken."):
        self.answer = answer
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return {"content": self.answer, "tool_calls": []}


def _ask_lab(tmp):
    """A small world for interpreter tests: Mara + Joss parked
    dwelling at Central Plaza, a pending plan for the pair to meet
    at West Depot, Priya at her spawn, and the player standing on
    top of Mara (so she is ACTIVE). One proximity report goes
    through the real protocol path, so tiers run and the events land
    in the bridge's recent-events buffer exactly as a live client's
    would. Returns (bridge, manager, cards, positions, plaza, depot).
    """
    mem = tmp_memory(tmp)
    cards = generate_roster(3)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = city_spawn_positions([c.name for c in cards])
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    planner = manager.planner
    bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
    plaza = next(a for a in planner.anchors if a.name == "Central Plaza")
    depot = next(a for a in planner.anchors if a.name == "West Depot")
    a, b = cards[0].name, cards[1].name
    for c in cards[:2]:
        st = planner.state[c.name]
        st["initialized"] = True
        st["state"] = "dwell"
        st["anchor"] = plaza
        st["kind"] = "social"
        st["until"] = 1e12          # handle_message ticks on wall time
        st["cooldown_until"] = 0.0
    # A pending plan, seeded straight into the queue with a due time
    # far away so no tick activates it and drives the pair off —
    # forming plans through conversation is test (u)'s job; here the
    # interpreter only has to READ the queue.
    planner.pending_plans.append(
        {"a": a, "b": b, "anchor": depot, "due": 1e12,
         "expires": 2e12, "active": False})
    positions = {a: (plaza.x, plaza.z), b: (plaza.x + 2.0, plaza.z),
                 cards[2].name: spawns[cards[2].name]}
    bridge.handle_message({"type": "proximity",
                           "player": [plaza.x, plaza.z],
                           "npcs": {k: list(v)
                                    for k, v in positions.items()}})
    return bridge, manager, cards, positions, plaza, depot


def _post_ask(port, question):
    import urllib.request
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/ask",
        data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def _ask_server(bridge, positions):
    from hermes_local.world_bridge import make_http_server
    server = make_http_server(bridge, "127.0.0.1", 0,
                              static_dir=PROJECT / "webworld",
                              positions=positions, mode="live (test)")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def test_interpreter_live_ask():
    # (aa) Live path: the fake brain receives a digest containing
    # Mara's current activity (dwelling/social at Central Plaza) and
    # the pending plan (meet at West Depot); the API returns the
    # fake's answer text verbatim.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, positions, plaza, depot = _ask_lab(tmp)
        a = cards[0].name
        fake = FakeAskClient()
        bridge.interpreter = WorldInterpreter(
            bridge, client=fake, memory=bridge.npcs[a].memory)
        server, port = _ask_server(bridge, positions)
        try:
            status, payload = _post_ask(
                port, f"What is {a} doing, and does anyone have plans?")
        finally:
            server.shutdown()
        assert status == 200, (status, payload)
        assert payload == {"answer": "The interpreter has spoken."}, payload
        assert len(fake.calls) == 1, fake.calls
        system, user = fake.calls[0][0], fake.calls[0][-1]
        assert "World Interpreter" in system["content"], system
        content = user["content"]
        assert "WORLD KNOWLEDGE" in content, content
        assert a in content and "ACTIVE" in content, content
        assert plaza.name in content, content          # her activity/place
        assert depot.name in content, content          # the plan's place
        assert "plan to meet" in content, content      # the pending plan


def test_interpreter_memory_in_context():
    # (ab) A question whose terms match a memory seeded into Mara's
    # namespace puts that memory text, attributed to Mara, into the
    # context the fake brain is sent.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, positions, plaza, depot = _ask_lab(tmp)
        a = cards[0].name
        mem = bridge.npcs[a].memory
        distinctive = "ZQX turnip festival disaster of last spring"
        mem.remember(f"Mara still laughs about the {distinctive}.",
                     kind="episode", namespace=cards[0].namespace)
        fake = FakeAskClient(answer="She remembers the festival.")
        bridge.interpreter = WorldInterpreter(bridge, client=fake,
                                              memory=mem)
        server, port = _ask_server(bridge, positions)
        try:
            status, payload = _post_ask(
                port, "What do you know about the turnip festival?")
        finally:
            server.shutdown()
        assert status == 200, (status, payload)
        assert payload["answer"] == "She remembers the festival.", payload
        content = fake.calls[0][-1]["content"]
        assert "CITIZEN MEMORIES" in content, content
        assert distinctive in content, content
        assert f"[{a} · episode]" in content, content   # attributed


def test_interpreter_mock_where():
    # (ac) Mock mode (the bridge's default interpreter, no client):
    # "Where is <name>?" and the headcount are answered straight from
    # the digest — no LLM anywhere in the loop.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, positions, plaza, depot = _ask_lab(tmp)
        a = cards[0].name
        assert bridge.interpreter.client is None
        answer = bridge.ask(f"Where is {a}?")
        assert a in answer and plaza.name in answer, answer
        counts = bridge.ask("How many citizens are here?")
        assert f"There are {len(cards)} citizens" in counts, counts


def test_npc_tools_error_degrades_to_react():
    # (ad) A generic (non-ToolUnsupportedError) rejection of the native
    # tools call degrades the NPC to ReAct and retries in the same turn:
    # the player hears the real in-character reply, and the next turn
    # goes straight to plain chat (no second tools attempt).
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        card = load_card(PROJECT / "characters" / "mara.json")
        mock = ToolsAllergicMock()
        npc = NPC(card, mock, memory=mem, embedder=mem.embedder)
        assert npc.mode == "auto"
        reply = npc.hear("Morning — who is on gate duty today?")
        assert "distracted" not in reply, reply
        assert "gate roster" in reply, reply
        assert npc.mode == "react", npc.mode
        assert mock.calls == [True, False], mock.calls
        reply2 = npc.hear("Thanks — anything else I should know?")
        assert "distracted" not in reply2, reply2
        assert mock.calls == [True, False, False], mock.calls


def test_npc_total_failure_still_distracted():
    # (ae) When BOTH the tools call and the plain retry fail, hear()
    # keeps the distracted line as the last resort.
    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        card = load_card(PROJECT / "characters" / "mara.json")
        npc = NPC(card, AlwaysDownMock(), memory=mem, embedder=mem.embedder)
        reply = npc.hear("Hello? Can you hear me?")
        assert reply == f"({card.name} seems distracted.)", reply


def _res_lab(tmp):
    """A small world for resource tests: 3 NPCs + a bridge with the
    default resource set. Returns (bridge, manager, cards, spawns)."""
    mem = tmp_memory(tmp)
    cards = generate_roster(3)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = city_spawn_positions([c.name for c in cards])
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
    return bridge, manager, cards, spawns


def _place_player(bridge, spawns, pos):
    bridge.handle_message({"type": "proximity", "player": list(pos),
                           "npcs": {k: list(v)
                                    for k, v in spawns.items()}})


def test_player_harvest_in_range():
    # (af) In-range interact fells a mature tree into the city
    # inventory and emits the harvest event; out-of-range and
    # depleted interacts are notices that change nothing.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        res = bridge.resources
        tree = res.nodes["tree-01"]
        assert tree.stock == tree.max_stock == 3

        _place_player(bridge, spawns, tree.position)
        resp = bridge.handle_message({"type": "interact",
                                      "node": "tree-01"})
        assert len(resp) == 1 and resp[0]["type"] == "harvest", resp
        ev = resp[0]
        assert ev["node"] == "tree-01" and ev["kind"] == "tree", ev
        assert ev["amount"] == 3 and ev["by"] == "Player", ev
        assert tree.stock == 0 and tree.state == "stump", tree
        assert res.inventory == {"wood": 3, "food": 0, "planks": 0, "meals": 0}, res.inventory

        # Depleted: same spot, second swing — refused, nothing moves.
        resp = bridge.handle_message({"type": "interact",
                                      "node": "tree-01"})
        assert resp[0]["type"] == "notice", resp
        assert res.inventory == {"wood": 3, "food": 0, "planks": 0, "meals": 0}, res.inventory
        assert tree.stock == 0

        # Out of range: a garden plot across the city refuses too.
        plant = res.nodes["plant-01"]
        resp = bridge.handle_message({"type": "interact",
                                      "node": "plant-01"})
        assert resp[0]["type"] == "notice", resp
        assert "Too far" in resp[0]["text"], resp
        assert plant.stock == plant.max_stock
        assert res.inventory == {"wood": 3, "food": 0, "planks": 0, "meals": 0}, res.inventory

        # In range of the plot, one food lands in the stores.
        _place_player(bridge, spawns, plant.position)
        resp = bridge.handle_message({"type": "interact",
                                      "node": "plant-01"})
        assert resp[0]["type"] == "harvest", resp
        assert resp[0]["amount"] == 1 and resp[0]["resource"] == "food"
        assert plant.stock == plant.max_stock - 1
        assert res.inventory == {"wood": 3, "food": 1, "planks": 0, "meals": 0}, res.inventory


def test_node_regrowth():
    # (ag) Regrowth is tick-driven in sim time: a felled tree comes
    # back unit by unit (stump -> sapling -> mature), a plant regrows
    # one unit per window, and a tended animal is never removed.
    tree = ResourceNode(id="tree-t", kind="tree", name="Oak tree",
                        x=0.0, z=0.0, stock=3, max_stock=3,
                        regrow_seconds=10.0, resource="wood",
                        per_harvest=None)
    world = ResourceWorld(nodes=[tree])
    ev, notice = world.try_harvest("tree-t", (0.5, 0.0), now=1000.0)
    assert notice is None and ev["amount"] == 3, (ev, notice)
    assert tree.stock == 0 and tree.state == "stump", tree
    world.tick(1009.0)
    assert tree.stock == 0, tree                     # window not met
    world.tick(1010.0)
    assert tree.stock == 1 and tree.state == "sapling", tree
    world.tick(1029.0)
    assert tree.stock == 2, tree
    world.tick(1030.0)
    assert tree.stock == 3 and tree.state == "mature", tree
    ev, notice = world.try_harvest("tree-t", (0.0, 0.0), now=1031.0)
    assert notice is None and ev["amount"] == 3     # fellable again

    plant = ResourceNode(id="plant-t", kind="plant", name="Garden plot",
                         x=5.0, z=5.0, stock=2, max_stock=2,
                         regrow_seconds=20.0, resource="food",
                         per_harvest=1)
    goats = ResourceNode(id="animal-t", kind="animal", name="Goat",
                         x=9.0, z=9.0, stock=2, max_stock=2,
                         regrow_seconds=20.0, resource="food",
                         per_harvest=1)
    world2 = ResourceWorld(nodes=[plant, goats])
    ev, _ = world2.try_harvest("plant-t", (5.0, 5.0), now=0.0)
    assert ev["amount"] == 1 and plant.stock == 1
    ev, _ = world2.try_harvest("plant-t", (5.0, 5.0), now=1.0)
    assert plant.stock == 0 and plant.state == "bare"
    ev, notice = world2.try_harvest("plant-t", (5.0, 5.0), now=2.0)
    assert ev is None and notice                     # bare plot refuses
    world2.tick(21.0)
    assert plant.stock == 1 and plant.state == "ready"

    ev, _ = world2.try_harvest("animal-t", (9.0, 9.0), now=30.0)
    assert ev["amount"] == 1 and goats.stock == 1
    ev, _ = world2.try_harvest("animal-t", (9.0, 9.0), now=31.0)
    assert goats.stock == 0 and goats.state == "resting"
    assert "animal-t" in world2.nodes                # tended, not taken
    world2.tick(71.0)
    assert goats.stock == 2 and goats.state == "ready"


def test_world_endpoint():
    # (ah) GET /api/world shape: inventory + every node with the
    # fields the client renders from; /api/status carries inventory.
    import urllib.request
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        server, port = _ask_server(bridge, spawns)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/world",
                    timeout=5) as resp:
                assert resp.status == 200
                payload = json.loads(resp.read().decode())
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/status",
                    timeout=5) as resp:
                status = json.loads(resp.read().decode())
        finally:
            server.shutdown()
        assert payload["inventory"] == {"wood": 0, "food": 0, "planks": 0, "meals": 0}, payload
        nodes = payload["nodes"]
        assert len(nodes) == len(default_resource_nodes()), len(nodes)
        for n in nodes:
            assert {"id", "kind", "name", "position", "stock",
                    "max_stock", "resource", "state"} <= set(n), n
            assert len(n["position"]) == 2, n
            assert n["stock"] == n["max_stock"], n   # all start full
        assert {n["kind"] for n in nodes} == {"tree", "plant", "animal"}
        assert status["inventory"] == {"wood": 0, "food": 0, "planks": 0, "meals": 0}, status


def _work_lab(tmp, n=3):
    """A small world for P2 labor tests: NPCs + manager whose planner
    is wired to a fresh ResourceWorld (the way WorldBridge wires it).
    Returns (manager, planner, world, cards, spawns)."""
    mem = tmp_memory(tmp)
    cards = generate_roster(n)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = city_spawn_positions([c.name for c in cards])
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    world = ResourceWorld()
    manager.planner.resources = world
    return manager, manager.planner, world, cards, spawns


def _work_sim(manager, positions, now, seconds, player=(500.0, 500.0)):
    """Advance the world `seconds` sim-seconds at 1s steps, steering
    every traveling NPC toward its planner target (the test-(m)
    arrive steering). Returns (events, now)."""
    planner = manager.planner
    events = []
    for _ in range(seconds):
        now += 1.0
        events.extend(manager.update(player, positions, now=now))
        for name, st in planner.state.items():
            if st["state"] == "travel":
                positions[name] = _steer(positions[name], st["tx"],
                                         st["tz"], st["speed"], 1.0)
    return events, now


def test_foreman_gather_wood():
    # (ai) Wood below the foreman's baseline: a gather_wood task is
    # auto-assigned to the best-placed idle citizen, who walks to a
    # tree, fells it through the NPC harvest path, and the city wood
    # stock rises by the tree's yield.
    with tempfile.TemporaryDirectory() as tmp:
        manager, planner, world, cards, spawns = _work_lab(tmp)
        a = cards[0].name
        positions = dict(spawns)
        positions[a] = (-76.0, -78.0)          # ~6m from tree-01
        positions[cards[1].name] = (200.0, 200.0)   # far: cannot win
        positions[cards[2].name] = (240.0, 240.0)   # the wood scoring
        assert world.inventory["wood"] == 0    # below TARGET_WOOD
        # 12 sim-s: the foreman's first check (~10s in) has staffed
        # the task, but nobody has finished yet — exactly one wood
        # task exists, held by the citizen standing by the grove.
        events, now = _work_sim(manager, positions, 1000.0, 12)
        wood_tasks = [t for t in planner.tasks
                      if t["type"] == "gather_wood"]
        assert len(wood_tasks) == 1, planner.tasks
        task = wood_tasks[0]
        assert task["npc"] == a, task          # nearest citizen staffed
        assert any(e.get("type") == "task"
                   and e.get("state") == "assigned"
                   and e.get("npc") == a for e in events), events
        # Run until the felling lands in the stores.
        for _ in range(200):
            if world.inventory["wood"] >= 3:
                break
            events2, now = _work_sim(manager, positions, now, 1)
            events.extend(events2)
        assert world.inventory["wood"] == 3, world.inventory
        tree = world.nodes[task["target"]]
        assert tree.stock == 0 and tree.state == "stump", tree
        states = [e["state"] for e in events
                  if e.get("type") == "task"
                  and e.get("task") == task["id"]]
        assert states == ["assigned", "started", "completed"], states
        harvests = [e for e in events if e.get("type") == "harvest"
                    and e.get("by") == a]
        assert len(harvests) == 1 and harvests[0]["amount"] == 3, harvests


def test_crafting_recipes():
    # (aj) Craft tasks convert inputs to outputs at their stations:
    # 1 wood -> 1 plank at the sawbench, 2 food -> 1 meal at the
    # stove; a craft whose inputs vanished mid-task fails gracefully
    # with an event and consumes nothing.
    with tempfile.TemporaryDirectory() as tmp:
        manager, planner, world, cards, spawns = _work_lab(tmp, n=2)
        planner._foreman_next = 1e12   # controlled assignments only
        a, b = cards[0].name, cards[1].name
        world.inventory["wood"] = 10
        world.inventory["food"] = 8
        positions = dict(spawns)
        positions[a] = (-100.0, 12.0)          # at the sawbench
        positions[b] = (-31.0, 5.0)            # at the stove
        player = (500.0, 500.0)

        task, err = planner.assign_task("craft_planks", 1000.0,
                                        positions, npc_name=a)
        assert err is None and task["target"] == "sawbench", (task, err)
        events, now = _work_sim(manager, positions, 1000.0, 15,
                                player=player)
        done = [e for e in events if e.get("type") == "task"
                and e.get("task") == task["id"]
                and e.get("state") == "completed"]
        assert len(done) == 1, events
        assert done[0]["outputs"] == {"planks": 1}, done
        assert world.inventory["wood"] == 9, world.inventory
        assert world.inventory["planks"] == 1, world.inventory

        positions[b] = (-31.0, 5.0)            # B is at the stove
        task2, err = planner.assign_task("craft_meal", now, positions,
                                         npc_name=b)
        assert err is None and task2["target"] == "stove", (task2, err)
        events, now = _work_sim(manager, positions, now, 15,
                                player=player)
        done = [e for e in events if e.get("type") == "task"
                and e.get("task") == task2["id"]
                and e.get("state") == "completed"]
        assert len(done) == 1, events
        assert world.inventory["food"] == 6, world.inventory
        assert world.inventory["meals"] == 1, world.inventory

        # Inputs vanish before completion: graceful failure, no
        # consumption, no output.
        world.inventory["food"] = 1           # a meal needs 2
        positions[a] = (-31.0, 5.0)            # A takes the stove
        task3, err = planner.assign_task("craft_meal", now, positions,
                                         npc_name=a)
        assert err is None, err
        events, now = _work_sim(manager, positions, now, 15,
                                player=player)
        failed = [e for e in events if e.get("type") == "task"
                  and e.get("task") == task3["id"]
                  and e.get("state") == "failed"]
        assert len(failed) == 1, events
        assert "stores" in failed[0]["reason"], failed
        assert world.inventory["food"] == 1, world.inventory
        assert world.inventory["meals"] == 1, world.inventory


def test_player_order_and_engaged_deferral():
    # (ak) The god-mode order protocol assigns the NAMED citizen; an
    # engaged (conversation-locked) citizen does not start the work
    # until the lock is released — then the task drives them.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        planner = manager.planner
        x = cards[0].name
        positions = dict(spawns)

        resp = bridge.handle_message({"type": "order", "task": "dance"})
        assert resp[0]["type"] == "notice", resp     # unknown task

        resp = bridge.handle_message(
            {"type": "order", "task": "gather_wood", "npc": x})
        assert resp[0]["type"] == "task", resp
        assert resp[0]["state"] == "assigned", resp
        assert resp[0]["npc"] == x, resp             # the named citizen
        task_id = resp[0]["task"]
        assert planner.active_tasks()[0]["npc"] == x

        resp = bridge.handle_message({"type": "order",
                                      "task": "gather_food"})
        assert resp[0]["type"] == "task", resp       # best idle picked
        assert resp[0]["npc"] != x, resp             # (X is busy)

        # Engage X in conversation: the lock wins over the task.
        now = 1000.0
        manager.note_speech(x, now=now)
        events = manager.update(positions[x], positions, now=now)
        assert planner.state[x]["state"] == "engaged", planner.state[x]
        for step in range(1, 20):
            events = manager.update(positions[x], positions,
                                    now=now + step)
            assert planner.state[x]["state"] == "engaged", \
                planner.state[x]
            assert not any(e.get("type") == "task"
                           and e.get("task") == task_id
                           and e.get("state") == "started"
                           for e in events), events
        assert bridge.resources.inventory["wood"] == 0

        # Release the lock (silence past the engage timeout, player
        # gone): the very next tick hands X over to the task.
        events = manager.update((500.0, 500.0), positions, now=now + 25.0)
        assert planner.state[x]["state"] == "travel", planner.state[x]
        task = next(t for t in planner.tasks if t["id"] == task_id)
        assert task["state"] == "active" and task["phase"] == "travel", task


def _build_sim_until(manager, positions, now, done, limit=400):
    """Advance 1 sim-second at a time (the _work_sim steering) until
    done() holds or the limit runs out. Returns (events, now)."""
    events = []
    for _ in range(limit):
        if done():
            break
        step, now = _work_sim(manager, positions, now, 1)
        events.extend(step)
    return events, now


def test_build_stall_completes():
    # (al) Building P3: a placed stall is staffed by the foreman and
    # raised frame -> complete by a citizen builder; the exact
    # material cost leaves the city stores, the finished building
    # shows in /api/world, and its market anchor joins the planner.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        planner = manager.planner
        con = bridge.construction
        inv = bridge.resources.inventory
        inv["wood"], inv["food"] = 10, 8      # foreman baselines met
        inv["planks"] = 4                     # exactly one stall

        resp = bridge.handle_message(
            {"type": "place", "kind": "stall", "position": [-30, 24]})
        assert resp[0]["type"] == "build", resp
        assert resp[0]["state"] == "placed", resp
        site = con.sites[resp[0]["site"]]
        assert site.state == "awaiting_materials", site
        assert any(r.get("type") == "build"
                   and r.get("state") == "placed"
                   for r in bridge.recent_events()), bridge.recent_events()

        # A road band refuses placement with a reason, creating
        # nothing: x=0 is a road centerline.
        resp = bridge.handle_message(
            {"type": "place", "kind": "lamp", "position": [0, 24]})
        assert resp[0]["type"] == "notice", resp
        assert len(con.sites) == 1, con.sites

        positions = dict(spawns)
        events, now = _build_sim_until(manager, positions, 1000.0,
                                       lambda: site.complete)
        assert site.complete, site
        build_states = [e["state"] for e in events
                        if e.get("type") == "build"
                        and e.get("site") == site.id]
        assert build_states == ["frame", "complete"], build_states
        task_states = [e["state"] for e in events
                       if e.get("type") == "task"
                       and e.get("task_type") == "build"]
        assert task_states == ["assigned", "started", "completed"], \
            task_states

        # Materials were deducted exactly by the stall's cost.
        assert inv == {"wood": 10, "food": 8, "planks": 0,
                       "meals": 0}, inv

        payload = bridge.world_payload()
        assert payload["sites"] == [], payload["sites"]
        assert len(payload["buildings"]) == 1, payload["buildings"]
        b = payload["buildings"][0]
        assert b["kind"] == "stall" and b["id"] == site.id, b
        assert b["anchor"] == "Market stall 1", b
        anchors = {a.name: a for a in planner.anchors}
        assert "Market stall 1" in anchors, sorted(anchors)
        assert anchors["Market stall 1"].kind == "browse"


def test_build_house_waits_for_materials():
    # (am) Building P3: a house placed with too few planks sits in
    # awaiting_materials — a builder attends the site but nothing is
    # consumed and no work accrues — until the stores suffice; then
    # it completes at exactly the full cost and adds a home anchor.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        planner = manager.planner
        con = bridge.construction
        inv = bridge.resources.inventory
        inv["wood"], inv["food"] = 14, 8
        inv["planks"] = 3                  # a house needs 10 + 4 wood

        resp = bridge.handle_message(
            {"type": "place", "kind": "house", "position": [30, 30]})
        assert resp[0]["type"] == "build", resp
        site = con.sites[resp[0]["site"]]

        positions = dict(spawns)
        events, now = _work_sim(manager, positions, 1000.0, 150)
        assert site.state == "awaiting_materials", site
        assert site.work_done == 0.0, site
        assert not any(site.delivered.values()), site
        assert inv == {"wood": 14, "food": 8, "planks": 3,
                       "meals": 0}, inv
        # The builder was still assigned and attends the site.
        build_tasks = [t for t in planner.tasks if t["type"] == "build"]
        assert len(build_tasks) == 1, planner.tasks
        assert build_tasks[0]["state"] == "active", build_tasks

        inv["planks"] = 10                 # the stores now suffice
        events, now = _build_sim_until(manager, positions, now,
                                       lambda: site.complete)
        assert site.complete, site
        assert inv == {"wood": 10, "food": 8, "planks": 0,
                       "meals": 0}, inv
        anchors = {a.name: a for a in planner.anchors}
        assert "House 1" in anchors, sorted(anchors)
        assert anchors["House 1"].kind == "front"


def test_build_layout_round_trip():
    # (an) Building P3: sites (stage/progress/materials intact) and
    # finished buildings serialize into the layout document and
    # reload into a fresh world; a souls load — a reset path —
    # leaves the construction world untouched.
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "a").mkdir()
        Path(tmp, "b").mkdir()
        bridge, manager, cards, spawns = _res_lab(str(Path(tmp) / "a"))
        con = bridge.construction
        inv = bridge.resources.inventory
        inv["planks"] = 20

        resp = bridge.handle_message(
            {"type": "place", "kind": "stall", "position": [-30, 24]})
        stall = con.sites[resp[0]["site"]]
        resp = bridge.handle_message(
            {"type": "place", "kind": "house", "position": [30, 30]})
        house = con.sites[resp[0]["site"]]

        # Raise the stall past halfway by hand (the site's own work
        # rule): the stage-1 tranche lands, work crosses 50%.
        evs = con.work_at(stall.id, 20.0, inv, by="Test")
        assert stall.stage == "frame", stall
        assert stall.work_done == 20.0, stall
        assert any(e.get("state") == "frame" for e in evs), evs
        assert inv["planks"] == 18, inv    # stage-1 tranche: 2 planks

        doc = bridge.handle_message({"type": "load_layout"})[0]["layout"]
        assert len(doc["sites"]) == 2, doc
        assert doc["buildings"] == [], doc

        Path(tmp, "b").mkdir(exist_ok=True)
        bridge2, manager2, cards2, spawns2 = \
            _res_lab(str(Path(tmp) / "b"))
        resp = bridge2.handle_message({"type": "save_layout",
                                       "layout": doc})
        assert resp[0]["type"] == "ack", resp
        con2 = bridge2.construction
        stall2 = con2.sites[stall.id]
        assert stall2.stage == "frame", stall2
        assert stall2.work_done == 20.0, stall2
        assert stall2.delivered == stall.delivered, stall2
        house2 = con2.sites[house.id]
        assert house2.stage == "site" and house2.work_done == 0.0, house2
        payload = bridge2.world_payload()
        assert len(payload["sites"]) == 2, payload["sites"]

        # Souls are knowledge, not situation: a full souls load on
        # the new bridge must not disturb its construction world.
        manager2.import_souls(bridge2.export_souls(), mode="full")
        assert set(con2.sites) == {stall.id, house.id}, con2.sites
        assert con2.sites[stall.id].work_done == 20.0

        # Finish the restored stall by hand: the stage-2 tranche
        # lands and the building completes under its anchor name.
        bridge2.resources.inventory["planks"] = 2
        evs = con2.work_at(stall.id, 15.0,
                           bridge2.resources.inventory, by="Test")
        assert stall2.complete, stall2
        done = [e for e in evs if e.get("state") == "complete"]
        assert len(done) == 1 and done[0]["anchor"]["name"] == \
            "Market stall 1", evs

        # A second reload carries the FINISHED building (and its
        # anchor) across: the new world's planner regains the
        # market anchor for the restored stall.
        doc2 = bridge2.handle_message({"type": "load_layout"})[0]["layout"]
        assert len(doc2["buildings"]) == 1, doc2
        Path(tmp, "c").mkdir(exist_ok=True)
        bridge3, manager3, cards3, spawns3 = \
            _res_lab(str(Path(tmp) / "c"))
        resp = bridge3.handle_message({"type": "save_layout",
                                       "layout": doc2})
        assert resp[0]["type"] == "ack", resp
        payload = bridge3.world_payload()
        assert len(payload["buildings"]) == 1, payload["buildings"]
        assert payload["buildings"][0]["anchor"] == "Market stall 1", \
            payload["buildings"]
        assert len(payload["sites"]) == 1, payload["sites"]  # the house
        anchor_names = {a.name for a in manager3.planner.anchors}
        assert "Market stall 1" in anchor_names, sorted(anchor_names)


def test_terraform_raise_and_round_trip():
    # (ao) Terraforming P4: a god-mode raise applies in one call —
    # heights inside the radius grow with the radial falloff
    # (centre > edge > rim, outside untouched) — and the shaped
    # grid serialize -> restores exactly, via the layout document
    # and via /api/world.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        ter = bridge.terrain
        resp = bridge.handle_message(
            {"type": "terraform", "op": "raise", "position": [20, 20],
             "radius": 8, "amount": 1.2})
        assert resp[0]["type"] == "terraform", resp
        assert resp[0]["mode"] == "god" and resp[0]["op"] == "raise", resp

        centre = ter.sample(20, 20)        # a grid point: full amount
        edge = ter.sample(26, 20)          # 6m out: cosine falloff
        rim = ter.sample(28, 20)           # exactly at the radius
        outside = ter.sample(40, 20)       # well beyond it
        assert centre == 1.2, centre
        assert 0.0 < edge < centre, edge
        assert rim == 0.0 and outside == 0.0, (rim, outside)
        assert ter.heights[ter.index(70, 70)] == 1.2

        # The interpreter's recent-events buffer saw the shaping.
        assert any(r.get("type") == "terraform"
                   and r.get("mode") == "god"
                   for r in bridge.recent_events()), bridge.recent_events()

        # Serialize -> restore round-trips the grid EXACTLY.
        grid = list(ter.heights)
        doc = bridge.handle_message({"type": "load_layout"})[0]["layout"]
        assert "terrain" in doc, sorted(doc)
        fresh = TerrainWorld()
        fresh.restore(doc)
        assert fresh.heights == grid
        assert fresh.sample(20, 20) == 1.2

        # /api/world carries the same grid + grid geometry.
        payload = bridge.world_payload()
        assert payload["terrain"]["heights"] == grid, "world payload grid"
        assert payload["terrain"]["size"] == 121
        assert payload["terrain"]["step"] == 2.0
        assert payload["terrain"]["water_y"] == -0.35
        assert payload["terrain"]["jobs"] == []

        # The digest names the new high ground once the land is shaped.
        digest = bridge.interpreter.digest_text(
            bridge.interpreter.snapshot())
        assert "Terrain:" in digest and "1.2 m" in digest, digest


def test_terraform_crew_flatten():
    # (ap) Terraforming P4: a crew-mode flatten opens a terraform
    # JOB (not an instant edit); a citizen is assigned through the
    # task queue, walks to the patch, and over ~30 attended
    # sim-seconds the patch heights converge to the job's goal map
    # (flatten target = the height at the clicked point). The job
    # completes with a crew terraform event + a completed task.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        planner = manager.planner
        ter = bridge.terrain
        inv = bridge.resources.inventory
        inv["wood"], inv["food"] = 10, 8   # foreman baselines met

        # A god-raised mound near the spawns, then a crew flatten
        # clicked on its slope: target = the slope height there.
        bridge.handle_message(
            {"type": "terraform", "op": "raise", "position": [-40, -58],
             "radius": 10, "amount": 2.0})
        assert ter.sample(-40, -58) == 2.0
        slope_h = ter.sample(-34, -58)
        assert 0.0 < slope_h < 2.0, slope_h

        resp = bridge.handle_message(
            {"type": "terraform", "mode": "crew", "op": "flatten",
             "position": [-34, -58], "radius": 6, "amount": 1.0})
        assert resp[0]["type"] == "terraform", resp
        assert resp[0]["mode"] == "crew", resp
        assert resp[0]["state"] == "created", resp
        job_id = resp[0]["job"]
        job = ter.jobs[job_id]
        assert job["target"] == slope_h, job
        goal = dict(job["goal"])
        assert goal, job
        # Nothing has moved yet: a job is a promise, not an edit.
        assert ter.sample(-40, -58) == 2.0

        # A citizen holds the terraform task (assigned immediately).
        assert any(r.get("type") == "task"
                   and r.get("task_type") == "terraform"
                   and r.get("state") == "assigned" for r in resp), resp
        task = next(t for t in planner.tasks
                    if t["target_kind"] == "patch")
        assert task["type"] == "terraform" and task["target"] == job_id

        positions = dict(spawns)
        events, now = _build_sim_until(
            manager, positions, 1000.0, lambda: job_id not in ter.jobs)
        assert job_id not in ter.jobs, ter.jobs   # the job completed

        # The patch converged EXACTLY to the job's goal map, and the
        # clicked point sits at the flatten target.
        for idx, g in goal.items():
            assert ter.heights[idx] == g, (idx, ter.heights[idx], g)
        assert ter.sample(-34, -58) == slope_h

        done = [e for e in events if e.get("type") == "terraform"
                and e.get("mode") == "crew"
                and e.get("state") == "complete"
                and e.get("job") == job_id]
        assert len(done) == 1, done
        assert done[0]["by"] == task["npc"], done
        task_states = [e["state"] for e in events
                       if e.get("type") == "task"
                       and e.get("task") == task["id"]]
        assert task_states == ["started", "completed"], task_states
        assert planner.active_tasks() == [], planner.active_tasks()


def test_terraform_invalid_ops():
    # (aq) Terraforming P4: invalid ops — out of bounds, radius or
    # amount out of range, unknown op, crew mode included — are
    # refused with a notice and change NOTHING (no heights move, no
    # job opens, the terrain stays unmodified). A valid god op then
    # applies in a single call, no sim ticks involved.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        ter = bridge.terrain
        flat = list(ter.heights)
        bad = [
            {"op": "raise", "position": [200, 0], "radius": 8,
             "amount": 1.0},                                   # out of bounds
            {"op": "raise", "position": [0, -121], "radius": 8,
             "amount": 1.0},                                   # out of bounds
            {"op": "lower", "position": [10, 10], "radius": 40,
             "amount": 1.0},                                   # radius too big
            {"op": "lower", "position": [10, 10], "radius": 1,
             "amount": 1.0},                                   # radius too small
            {"op": "raise", "position": [10, 10], "radius": 8,
             "amount": 5.0},                                   # amount too big
            {"op": "raise", "position": [10, 10], "radius": 8,
             "amount": 0.1},                                   # amount too small
            {"op": "explode", "position": [10, 10], "radius": 8,
             "amount": 1.0},                                   # unknown op
        ]
        for msg in bad:
            resp = bridge.handle_message({"type": "terraform", **msg})
            assert resp[0]["type"] == "notice", (msg, resp)
        resp = bridge.handle_message(
            {"type": "terraform", "mode": "crew", "op": "raise",
             "position": [500, 500], "radius": 8, "amount": 1.0})
        assert resp[0]["type"] == "notice", resp
        assert ter.jobs == {}, ter.jobs
        assert ter.heights == flat and not ter.modified

        # A malformed heightmap in a layout document is rejected
        # whole (all-or-nothing), leaving the terrain untouched.
        resp = bridge.handle_message(
            {"type": "save_layout",
             "layout": {"version": 1, "objects": [],
                        "terrain": {"heights": [0.0] * 10, "jobs": []}}})
        assert resp[0]["type"] == "error", resp
        assert ter.heights == flat and not ter.modified

        # Contrast: a valid god op applies in ONE call.
        resp = bridge.handle_message(
            {"type": "terraform", "op": "raise", "position": [10, 10],
             "radius": 4, "amount": 0.5})
        assert resp[0]["type"] == "terraform", resp
        assert ter.sample(10, 10) == 0.5
        assert ter.modified


class FakeBallotCaster:
    """Civics ballot-caster double for the engine's LIVE path:
    records every (citizen, soul_text) it is handed and answers per
    `policy(card)` — a (choice, reason) tuple, None (garbage: the
    engine retries once, then ABSTAIN "no clear answer"), or an
    exception to raise (hard failure: ABSTAIN "couldn't be
    reached")."""

    def __init__(self, policy=None):
        self.calls = []            # [(name, soul_text)]
        self.policy = policy or (lambda card: ("FOR", "the city needs it"))

    def __call__(self, card, soul, proposal, debate):
        self.calls.append((card.name, soul))
        return self.policy(card)


def _civic_lab(tmp, n=54):
    """A world for civics tests: bridge + manager + roster, with the
    bridge's default (mock) civics engine. Returns (bridge, manager,
    cards, spawns)."""
    mem = tmp_memory(tmp)
    cards = generate_roster(n)
    npcs = {c.name: NPC(c, SummaryMock(), memory=mem, embedder=mem.embedder)
            for c in cards}
    spawns = city_spawn_positions([c.name for c in cards])
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    bridge = WorldBridge(npcs, manager=manager, director=Director(npcs))
    return bridge, manager, cards, spawns


def _civic_run(manager, positions, now, done, limit=900):
    """Advance 1 sim-second at a time — with real pauses so the
    ballot machinery's background threads can land their results
    between ticks (voting is asynchronous by design) — until done()
    holds. Returns (events, now)."""
    events = []
    for _ in range(limit):
        if done():
            break
        now += 1.0
        events.extend(manager.update((500.0, 500.0), positions, now=now))
        time.sleep(0.02)
    return events, now


def test_civics_full_cycle_adopted():
    # (ar) Civics P5: the flagship cycle. A player proposal to build
    # a house runs draft -> (no debate hook) -> voting with a fake
    # AI caster standing in for the live brain: ALL 54 citizens cast
    # through it, each handed their soul (one citizen's seeded
    # distinctive memory must reach the fake verbatim), the tally
    # adopts, the building system gains a real house site, and every
    # citizen remembers the vote as kind="civic".
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=54)
        engine = bridge.civics
        assert engine is not None and not engine.ai_ballots
        fake = FakeBallotCaster()
        engine.ballot_caster = fake
        engine.ai_ballots = True
        marked = cards[7].name
        bridge.npcs[marked].witness(
            "ZQX amber lighthouse blueprint folded in my coat pocket",
            kind="episode")
        resp = bridge.handle_message(
            {"type": "propose", "title": "A new house by North Green",
             "action": {"type": "build", "kind": "house",
                        "anchor": "North Green"}})
        assert resp[0]["type"] == "proposal", resp
        assert resp[0]["proposer"] == "The Keeper", resp
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: engine.current is None and len(engine.history) == 1)
        assert engine.current is None and len(engine.history) == 1, \
            engine.public_state()
        prop = engine.history[0]
        # Every citizen voted through the fake — and the proposer
        # (the player) holds no ballot of their own.
        assert len(fake.calls) == 54, len(fake.calls)
        assert "The Keeper" not in prop["ballots"], prop["ballots"]
        souls = dict(fake.calls)
        assert "ZQX amber lighthouse blueprint" in souls[marked], \
            souls[marked]
        assert prop["state"] == "adopted", prop["state"]
        assert prop["tally"] == {"for": 54, "against": 0,
                                 "abstain": 0, "cast": 54,
                                 "total": 54}, prop["tally"]
        ballots = [e for e in events if e.get("type") == "ballot"]
        assert len(ballots) == 54, len(ballots)
        results = [e for e in events if e.get("type") == "civic_result"]
        assert len(results) == 1 \
            and results[0]["outcome"] == "adopted", results
        # Adopted -> executed through the building system: a real
        # construction site for a house exists afterwards.
        sites = list(bridge.construction.sites.values())
        assert len(sites) == 1 and sites[0].kind == "house", sites
        # Every citizen holds the outcome as a civic memory, with
        # their own vote in it.
        mem = bridge.npcs[cards[0].name].memory
        for card in cards:
            civic = [e for e in mem.list_namespace(card.namespace)
                     if e.get("kind") == "civic"]
            assert len(civic) == 1, (card.name, civic)
            assert "ADOPTED" in civic[0]["text"], civic
            assert "I voted FOR" in civic[0]["text"], civic


def test_civics_rejection_executes_nothing():
    # (as) Civics P5: the fake returns a majority AGAINST — the
    # proposal is rejected, NO construction site is created, and
    # every citizen still gets the outcome memory with their vote.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        engine = bridge.civics
        fake = FakeBallotCaster(
            lambda card: ("AGAINST", "too costly by half"))
        engine.ballot_caster = fake
        engine.ai_ballots = True
        resp = bridge.handle_message(
            {"type": "propose", "title": "A market stall",
             "action": {"type": "build", "kind": "stall",
                        "anchor": "Market Row West"}})
        assert resp[0]["type"] == "proposal", resp
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: engine.current is None and len(engine.history) == 1)
        prop = engine.history[0]
        assert prop["state"] == "rejected", prop["state"]
        assert prop["tally"]["against"] == 6 \
            and prop["tally"]["for"] == 0, prop["tally"]
        assert bridge.construction.sites == {}, \
            bridge.construction.sites
        assert prop["executed"] is None, prop
        mem = bridge.npcs[cards[0].name].memory
        for card in cards:
            civic = [e for e in mem.list_namespace(card.namespace)
                     if e.get("kind") == "civic"]
            assert len(civic) == 1, (card.name, civic)
            assert "REJECTED" in civic[0]["text"], civic
            assert "I voted AGAINST" in civic[0]["text"], civic


def test_civics_live_failure_paths_and_mock_rules():
    # (at) Civics P5: the live failure discipline. A ballot call
    # that RAISES leaves that citizen ABSTAIN "couldn't be reached"
    # — called exactly once (no retry on exceptions) and with NO
    # rule ballot substituted (the reason is the failure path's own
    # words, which the rule path never produces). A call returning
    # garbage is retried exactly once, then ABSTAIN "no clear
    # answer". The adopted remainder still executes. Then, in mock
    # mode (no caster at all), a full cycle completes on rule
    # ballots — the documented demo path.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=4)
        engine = bridge.civics
        unreachable, unclear = cards[0].name, cards[1].name

        def policy(card):
            if card.name == unreachable:
                raise RuntimeError("model server down")
            if card.name == unclear:
                return None                    # garbage, both times
            return ("FOR", "aye from me")

        fake = FakeBallotCaster(policy)
        engine.ballot_caster = fake
        engine.ai_ballots = True
        bridge.handle_message(
            {"type": "propose", "title": "Plant oaks at Old Park",
             "action": {"type": "plant_trees", "count": 3,
                        "anchor": "Old Park"}})
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: engine.current is None and len(engine.history) == 1)
        prop = engine.history[0]
        ballots = prop["ballots"]
        assert ballots[unreachable] == {
            "choice": "ABSTAIN", "reason": "couldn't be reached"}, ballots
        called = [n for n, _soul in fake.calls]
        assert called.count(unreachable) == 1, fake.calls
        assert ballots[unclear] == {
            "choice": "ABSTAIN", "reason": "no clear answer"}, ballots
        assert called.count(unclear) == 2, fake.calls
        assert prop["tally"]["for"] == 2 \
            and prop["tally"]["abstain"] == 2, prop["tally"]
        assert prop["state"] == "adopted", prop["state"]   # 2-0 majority
        trees = [n for n in bridge.resources.nodes.values()
                 if n.id.startswith("tree-civic-")]
        assert len(trees) == 3, sorted(bridge.resources.nodes)

    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        engine = bridge.civics
        assert engine.ballot_caster is None and not engine.ai_ballots
        bridge.handle_message(
            {"type": "propose", "title": "A food drive",
             "action": {"type": "gather_drive", "resource": "food",
                        "extra": 4, "duration": 600}})
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: engine.current is None and len(engine.history) == 1,
            limit=60)
        assert len(engine.history) == 1, engine.public_state()
        prop = engine.history[0]
        assert len(prop["ballots"]) == 6, prop["ballots"]
        assert all(b["choice"] in ("FOR", "AGAINST", "ABSTAIN")
                   for b in prop["ballots"].values()), prop["ballots"]
        assert prop["state"] in ("adopted", "rejected"), prop["state"]
        debate = [e for e in events if e.get("type") == "debate"]
        assert len(debate) == 3, debate    # 2 supporters + 1 skeptic
        if prop["state"] == "adopted":
            targets = manager.planner.effective_targets(now)
            assert targets["food"] == 12, targets   # 8 + the drive's 4


# ---------------------------------------------------------------------------
# Research terminal + dynamic innovation (P6)
# ---------------------------------------------------------------------------
from hermes_local.buildings import BUILDINGS as _BUILDINGS  # noqa: E402
from hermes_local.innovations import MASONRY_EXAMPLE  # noqa: E402

DDG_HTML = """<html><body>
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fbricks">How bricks are made</a>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Bricks are dug clay, shaped and fired in a kiln until hard.</a>
<a rel="nofollow" class="result__a" href="https://example.org/masonry">Masonry basics for small towns</a>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=y">Brick walls resist fire and rot and outlast timber.</a>
</body></html>"""


class FakeSpecBrain:
    """Innovation spec-synthesizer double: records every ctx it is
    handed (so a test can inspect the revision errors) and returns
    a deep copy of its canned spec on every call."""

    def __init__(self, spec):
        self.spec = spec
        self.calls = []

    def __call__(self, ctx):
        self.calls.append(ctx)
        return json.loads(json.dumps(self.spec))


def test_research_flow_and_no_connection():
    # (au) Research P6: a fake fetcher serves DuckDuckGo-shaped
    # HTML; the terminal's insight lands in city knowledge with its
    # sources preserved AND in the asking citizen's memory as
    # kind="research". A fetcher that raises answers with an honest
    # no_connection event and stores NOTHING.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        engine = bridge.research
        fetched = []

        def fake_fetcher(url):
            fetched.append(url)
            return DDG_HTML

        engine.fetcher = fake_fetcher

        def fake_brain(citizen, question, ctx, results):
            assert results and results[0]["url"] == \
                "https://example.org/bricks", results
            return {"summary": "Bricks are clay shaped and fired in "
                               "a kiln — much stronger than wood.",
                    "innovation_hint": "clay bricks"}

        engine.brain = fake_brain
        mara = cards[0].name
        resp = bridge.handle_message(
            {"type": "research_ask", "npc": mara,
             "question": "What is stronger than wood to make my "
                         "home stronger?"})
        assert resp[0]["type"] == "research" \
            and resp[0]["state"] == "searching", resp
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: len(bridge.city_knowledge) >= 1)
        assert fetched, "the web search never ran"
        assert len(bridge.city_knowledge) == 1, bridge.city_knowledge
        entry = bridge.city_knowledge[0]
        assert entry["kind"] == "insight", entry
        assert entry["citizen"] == mara, entry
        assert entry["innovation_hint"] == "clay bricks", entry
        assert entry["sources"] == ["https://example.org/bricks",
                                    "https://example.org/masonry"], entry
        assert "kiln" in entry["summary"], entry
        insights = [e for e in events if e.get("type") == "research"
                    and e.get("state") == "insight"]
        assert len(insights) == 1, events
        mem = bridge.npcs[mara].memory
        research = [e for e in mem.list_namespace(cards[0].namespace)
                    if e.get("kind") == "research"]
        assert len(research) == 1, research
        assert "kiln" in research[0]["text"], research
    # Offline: the fetcher raises -> no_connection, nothing stored.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        engine = bridge.research

        def dead_fetcher(url):
            raise OSError("network is unreachable")

        engine.fetcher = dead_fetcher
        engine.brain = lambda citizen, q, ctx, results: {
            "summary": "must never be used", "innovation_hint": None}
        mara = cards[0].name
        resp = bridge.handle_message(
            {"type": "research_ask", "npc": mara,
             "question": "How do we keep food from spoiling?"})
        assert resp[0]["state"] == "searching", resp
        events, now = _civic_run(manager, dict(spawns), 1000.0,
                                 lambda: False, limit=40)
        offline = [e for e in events if e.get("type") == "research"
                   and e.get("state") == "no_connection"]
        assert len(offline) == 1, events
        assert bridge.city_knowledge == [], bridge.city_knowledge
        for card in cards:
            mem = bridge.npcs[card.name].memory
            research = [e for e in mem.list_namespace(card.namespace)
                        if e.get("kind") == "research"]
            assert research == [], (card.name, research)


def test_brick_loop_end_to_end():
    # (av) Innovation P6 flagship: research insight (masonry hint)
    # -> adopt_innovation proposal -> fake AI ballots adopt -> the
    # fake brain emits the bricks spec -> registered live (clay
    # pits, kiln + brick house in the catalog, generalized recipe
    # registry) -> kiln built -> clay gathered through the task
    # machinery -> bricks crafted (a share routed to market stock)
    # -> a market sale ticks stock down and the trade counter up.
    # The registered state round-trips through the world save into
    # a fresh bridge.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        positions = dict(spawns)
        inv = bridge.resources.inventory
        # --- research: Aaron's worked-example question ------------
        bridge.research.fetcher = lambda url: DDG_HTML
        bridge.research.brain = lambda citizen, q, ctx, results: {
            "summary": "Bricks are clay shaped and fired in a kiln "
                       "— much stronger than wood.",
            "innovation_hint": "clay bricks"}
        mara = cards[0].name
        bridge.handle_message(
            {"type": "research_ask", "npc": mara,
             "question": "What is stronger than wood to make my "
                         "home stronger?"})
        _events, now = _civic_run(
            manager, positions, 1000.0,
            lambda: len(bridge.city_knowledge) >= 1)
        assert bridge.city_knowledge[0]["innovation_hint"] == \
            "clay bricks", bridge.city_knowledge
        # --- civics: adopt the discovery --------------------------
        engine = bridge.civics
        fake = FakeBallotCaster()
        engine.ballot_caster = fake
        engine.ai_ballots = True
        spec_brain = FakeSpecBrain(MASONRY_EXAMPLE)
        bridge.innovations.spec_brain = spec_brain
        bridge.innovations.searcher = lambda q: [
            {"title": "How bricks are made",
             "url": "https://example.org/bricks",
             "snippet": "Dig clay, shape it, fire it in a kiln."}]
        resp = bridge.handle_message(
            {"type": "propose",
             "title": "Adopt the discovery: clay bricks",
             "action": {"type": "adopt_innovation",
                        "innovation_hint": "clay bricks"}})
        assert resp[0]["type"] == "proposal", resp
        events, now = _civic_run(
            manager, positions, now,
            lambda: engine.current is None and len(engine.history) == 1)
        assert engine.history[0]["state"] == "adopted", engine.history
        # --- synthesis lands: bricks are real ----------------------
        # (the registered event may drain in either run — assert
        # over both)
        synth_events, now = _civic_run(
            manager, positions, now,
            lambda: "bricks" in bridge.innovations.records)
        events = events + synth_events
        inno = bridge.innovations
        assert spec_brain.calls, "the synthesizer never ran"
        assert "bricks" in inv and "clay" in inv, inv
        pits = [n for n in bridge.resources.nodes.values()
                if n.kind == "src_clay"]
        assert len(pits) >= 3, [n.kind for n in
                                bridge.resources.nodes.values()]
        assert "kiln" in _BUILDINGS and "brick_house" in _BUILDINGS
        book = bridge.recipes
        assert book.is_craft("craft_bricks")
        assert book.is_gather("gather_clay")
        registered = [e for e in events
                      if e.get("type") == "innovation"
                      and e.get("state") == "registered"]
        assert registered, events
        # --- build the kiln ---------------------------------------
        inv["wood"] = 10
        inv["food"] = 8
        inv["planks"] = 10
        resp = bridge.handle_message(
            {"type": "place", "kind": "kiln", "position": [30, 30]})
        assert resp[0]["type"] == "build" \
            and resp[0]["state"] == "placed", resp
        events, now = _build_sim_until(
            manager, positions, now,
            lambda: any(b["kind"] == "kiln"
                        for b in bridge.construction.buildings),
            limit=900)
        assert any(b["kind"] == "kiln"
                   for b in bridge.construction.buildings), \
            bridge.construction.buildings
        # --- gather clay through the task machinery ----------------
        worker = cards[1].name
        resp = bridge.handle_message(
            {"type": "order", "task": "gather_clay", "npc": worker})
        assert resp[0]["type"] == "task", resp
        events, now = _build_sim_until(
            manager, positions, now, lambda: inv.get("clay", 0) >= 1,
            limit=900)
        assert inv["clay"] >= 1, inv
        inv["clay"] = 4   # further pit runs top the stores up
        # --- craft bricks: output splits stores / market stock -----
        crafter = cards[2].name
        resp = bridge.handle_message(
            {"type": "order", "task": "craft_bricks", "npc": crafter})
        assert resp[0]["type"] == "task", resp
        events, now = _build_sim_until(
            manager, positions, now,
            lambda: inv.get("bricks", 0)
            + inno.market_stock.get("bricks", 0) >= 2, limit=900)
        assert inv["bricks"] == 1, inv
        assert inno.market_stock.get("bricks") == 1, inno.market_stock
        assert inv["clay"] == 2, inv
        # --- a market sale -----------------------------------------
        inno._sale_next = -1.0
        events, now = _civic_run(manager, positions, now,
                                 lambda: inno.trade > 0, limit=60)
        assert inno.trade == 6, inno.trade
        assert inno.market_stock.get("bricks", 0) == 0
        sales = [e for e in events if e.get("type") == "sale"]
        assert sales and sales[0]["item"] == "bricks", events
        # --- the world save carries it all --------------------------
        doc = bridge.handle_message(
            {"type": "load_layout"})[0]["layout"]
        assert "innovations" in doc and "knowledge" in doc, doc.keys()
    with tempfile.TemporaryDirectory() as tmp2:
        bridge2, manager2, cards2, spawns2 = _res_lab(tmp2)
        resp = bridge2.handle_message(
            {"type": "save_layout", "layout": doc})
        assert resp[0].get("ok") is True, resp
        assert "bricks" in bridge2.innovations.records
        assert bridge2.recipes.is_craft("craft_bricks")
        assert any(n.kind == "src_clay"
                   for n in bridge2.resources.nodes.values())
        assert bridge2.innovations.trade == 6
        assert bridge2.city_knowledge, "knowledge did not restore"


def test_invalid_spec_goes_under_study():
    # (aw) Innovation P6: the fake brain emits a spec with an
    # unknown recipe input AND a station archetype that doesn't
    # exist. Validation rejects it; the errors are fed back for
    # exactly ONE revision; the revision is still invalid, so the
    # discovery is filed "under study" — nothing registers, and the
    # proposing citizen remembers the outcome.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=3)
        positions = dict(spawns)
        engine = bridge.civics
        fake = FakeBallotCaster()
        engine.ballot_caster = fake
        engine.ai_ballots = True
        bad_spec = {
            "id": "wonderstuff", "name": "Wonderstuff",
            "kind": "material", "source": None,
            "recipe": {"inputs": {"unobtanium": 3}, "out_qty": 2,
                       "station_id": "kiln", "work_s": 10},
            "station_building": {"name": "Wonderworks",
                                 "cost": {"planks": 3},
                                 "archetype": "castle"},
            "uses": ["sellable"], "price": 9}
        spec_brain = FakeSpecBrain(bad_spec)
        bridge.innovations.spec_brain = spec_brain
        bridge.innovations.searcher = lambda q: [
            {"title": "Wonderstuff, probably",
             "url": "https://example.org/wonder",
             "snippet": "Nobody is quite sure."}]
        mara = cards[0].name
        events, err = engine.open_proposal(
            mara, "Adopt the discovery: wonderstuff", "",
            {"type": "adopt_innovation",
             "innovation_hint": "wonderstuff"}, 1000.0)
        assert err is None, err
        events, now = _civic_run(
            manager, positions, 1000.0,
            lambda: engine.current is None and len(engine.history) == 1)
        assert engine.history[0]["state"] == "adopted", engine.history
        filed_events, now = _civic_run(
            manager, positions, now,
            lambda: len(bridge.innovations.under_study) >= 1)
        events = events + filed_events
        inno = bridge.innovations
        assert len(spec_brain.calls) == 2, spec_brain.calls
        assert not spec_brain.calls[0].get("errors"), \
            spec_brain.calls[0]
        assert spec_brain.calls[1].get("errors"), spec_brain.calls[1]
        assert inno.records == {}, inno.records
        assert "wonderstuff" not in bridge.resources.inventory
        assert len(inno.under_study) == 1, inno.under_study
        assert inno.under_study[0]["hint"] == "wonderstuff"
        filed = [e for e in events if e.get("type") == "innovation"
                 and e.get("state") == "under_study"]
        assert filed, events
        mem = bridge.npcs[mara].memory
        research = [e for e in mem.list_namespace(cards[0].namespace)
                    if e.get("kind") == "research"]
        assert any("under study" in e["text"].lower()
                   for e in research), research


from hermes_local.innovations import validate_spec as _validate_spec  # noqa: E402
from hermes_local.persistence import (ConflictError,  # noqa: E402
                                      PersistenceService,
                                      apply_snapshot,
                                      assemble_snapshot)


class FakeGitHubTransport:
    """In-memory GitHub contents-API double with real sha
    discipline: a store whose sha doesn't match the current file
    raises ConflictError, exactly like the real transport. With
    `fail` set, every call raises — the offline world."""

    def __init__(self):
        self.files = {}       # (repo, branch, path) -> {"doc", "sha"}
        self.puts = []        # [(path, sha_used, new_sha)]
        self.deletes = []     # [path]
        self.fail = False
        # P12: when set, writes under versions/ fail while the
        # live world file keeps working — the best-effort path.
        self.fail_versions = False
        self._seq = 0

    def fetch(self, repo, branch, path, token):
        if self.fail:
            raise OSError("network is unreachable")
        rec = self.files.get((repo, branch, path))
        if rec is None:
            return None, None
        return json.loads(json.dumps(rec["doc"])), rec["sha"]

    def store(self, repo, branch, path, token, doc, sha, message):
        if self.fail:
            raise OSError("network is unreachable")
        if self.fail_versions and path.startswith("versions/"):
            raise OSError("version write refused")
        key = (repo, branch, path)
        cur = self.files.get(key)
        cur_sha = cur["sha"] if cur else None
        if sha != cur_sha:
            raise ConflictError(f"sha mismatch: {sha!r} != {cur_sha!r}")
        self._seq += 1
        new_sha = f"sha-{self._seq}"
        self.files[key] = {"doc": json.loads(json.dumps(doc)),
                           "sha": new_sha}
        self.puts.append((path, sha, new_sha))
        return new_sha

    def list_dir(self, repo, branch, dir_path, token):
        if self.fail:
            raise OSError("network is unreachable")
        if self.fail_versions and dir_path.startswith("versions"):
            raise OSError("version listing refused")
        out = []
        prefix = dir_path + "/"
        for (r, b, p), rec in sorted(self.files.items()):
            if r == repo and b == branch and p.startswith(prefix) \
                    and "/" not in p[len(prefix):]:
                out.append({"name": p[len(prefix):], "path": p,
                            "sha": rec["sha"]})
        return out

    def delete_file(self, repo, branch, path, token, sha, message):
        if self.fail:
            raise OSError("network is unreachable")
        if self.fail_versions and path.startswith("versions/"):
            raise OSError("version delete refused")
        key = (repo, branch, path)
        cur = self.files.get(key)
        cur_sha = cur["sha"] if cur else None
        if sha != cur_sha:
            raise ConflictError(f"sha mismatch: {sha!r} != {cur_sha!r}")
        del self.files[key]
        self.deletes.append(path)


def _persist_run(service, now, done, limit=300):
    """Drive a PersistenceService's tick — with real pauses so its
    background push threads can land between ticks — until done()
    holds. Returns the advanced clock."""
    for _ in range(limit):
        if done():
            break
        now += 1.0
        service.tick(now)
        time.sleep(0.02)
    return now


def _knowledge_entry(eid, citizen, summary):
    return {"id": eid, "kind": "insight", "citizen": citizen,
            "question": "What lasts?", "summary": summary,
            "sources": ["https://example.org/x"],
            "innovation_hint": None, "innovation": None, "ts": 1.0}


def test_snapshot_round_trip():
    # (ax) Persistence P7: mutate a world every way a snapshot
    # must carry — terrain, construction, civic history, city
    # knowledge, a registered innovation + its market, an NPC
    # memory — then apply the assembled document into a FRESH
    # bridge and find every mutation present. A corrupted section
    # rejects the whole document and applies nothing.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        mara = cards[0].name
        resp = bridge.handle_message(
            {"type": "terraform", "op": "raise", "position": [10, 10],
             "radius": 8, "amount": 1.2})
        assert resp[0]["type"] == "terraform", resp
        assert bridge.terrain.sample(10, 10) == 1.2
        resp = bridge.handle_message(
            {"type": "place", "kind": "stall", "position": [30, 30]})
        assert resp[0]["type"] == "build", resp
        engine = bridge.civics
        _events, err = engine.open_proposal(
            mara, "A food drive for the city stores", "the shelves "
            "are thinning", {"type": "gather_drive",
                             "resource": "food"}, 1000.0)
        assert err is None, err
        _civic_run(manager, dict(spawns), 1000.0,
                   lambda: engine.current is None
                   and len(engine.history) == 1)
        assert len(engine.history) == 1, engine.history
        bridge.city_knowledge.append(_knowledge_entry(
            "know-t1", mara, "Stone lasts longer than timber."))
        spec = {"id": "pottery", "name": "Pottery", "kind": "good",
                "source": {"node_label": "River clay bank",
                           "yields": "clay",
                           "area_hint": "low wet ground near a pond"},
                "recipe": {"inputs": {"clay": 2}, "out_qty": 1,
                           "station_id": "pottery_shed",
                           "work_s": 15},
                "station_building": {"name": "Pottery shed",
                                     "cost": {"planks": 3},
                                     "archetype": "stall"},
                "variant_building": None,
                "uses": ["sellable"], "price": 4}
        clean, errors = _validate_spec(spec,
                                       bridge.innovations._ctx())
        assert clean is not None, errors
        rec = dict(clean)
        rec["proposer"] = mara
        rec["hint"] = "river pottery"
        bridge.innovations.restore(
            {"records": [rec], "under_study": [],
             "market": {"stock": {"pottery": 3}, "trade": 11}})
        assert "pottery" in bridge.innovations.records
        bridge.npcs[mara].witness(
            "Mara saw a heron over the north pond at dawn.",
            kind="episode")
        doc = assemble_snapshot(bridge)
        assert doc["format"] == "hermes-local/world", doc["format"]
        assert doc["version"] == 1
        json.dumps(doc)      # the document is JSON-safe end to end
    with tempfile.TemporaryDirectory() as tmp2:
        bridge2, manager2, cards2, spawns2 = _res_lab(tmp2)
        apply_snapshot(bridge2, doc)
        assert bridge2.terrain.sample(10, 10) == 1.2
        assert any(s.kind == "stall"
                   for s in bridge2.construction.sites.values()), \
            bridge2.construction.sites
        assert len(bridge2.civics.history) == 1
        assert bridge2.civics.history[0]["title"] == \
            "A food drive for the city stores"
        assert any("Stone lasts" in (e.get("summary") or "")
                   for e in bridge2.city_knowledge), \
            bridge2.city_knowledge
        assert "pottery" in bridge2.innovations.records
        assert bridge2.innovations.market_stock.get("pottery") == 3
        assert bridge2.innovations.trade == 11
        assert bridge2.recipes.is_craft("craft_pottery")
        mem = bridge2.npcs[mara].memory
        texts = [e["text"]
                 for e in mem.list_namespace(cards2[0].namespace)]
        assert any("heron" in t for t in texts), texts
    # One corrupted section -> the whole document is refused.
    bad = json.loads(json.dumps(doc))
    bad["world"]["knowledge"] = "not-a-list"
    with tempfile.TemporaryDirectory() as tmp3:
        bridge3, manager3, cards3, spawns3 = _res_lab(tmp3)
        try:
            apply_snapshot(bridge3, bad)
            raise AssertionError("corrupt snapshot applied cleanly")
        except ValueError:
            pass
        assert bridge3.terrain.sample(10, 10) == 0.0
        assert not bridge3.construction.sites
        assert bridge3.city_knowledge == []
        assert "pottery" not in bridge3.innovations.records
        assert bridge3.civics.history == []


def test_repo_sync_with_fake_transport():
    # (ay) Persistence P7: the repo sync against a fake GitHub
    # transport — create, update-with-sha, startup pull semantics,
    # and the offline path.
    repo = "ZeroZen270/hermes-world-data"
    key = (repo, "main", "world-state.json")
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        mara = cards[0].name
        fake = FakeGitHubTransport()
        local = Path(tmp) / "world-state.local.json"
        svc = PersistenceService(
            bridge, {"repo": repo, "interval_s": 10,
                     "local_path": str(local)}, transport=fake)
        svc.startup()
        assert svc.public_status()["state"] == "synced", \
            svc.public_status()
        # First push creates the remote file (no sha presented).
        # (P12 also archives a version copy on the same push —
        # the sha bookkeeping here tracks the live file's puts.)
        bridge.city_knowledge.append(_knowledge_entry(
            "know-a", mara, "First synced fact."))
        _persist_run(svc, 1000.0,
                     lambda: svc.public_status()["last_push"])

        def main_puts():
            return [p for p in fake.puts
                    if p[0] == "world-state.json"]

        assert len(main_puts()) == 1, fake.puts
        assert main_puts()[0][1] is None, fake.puts
        first_sha = main_puts()[0][2]
        local_doc = json.loads(local.read_text())
        assert local_doc["format"] == "hermes-local/world"
        # Second push updates, presenting the fetched sha.
        bridge.city_knowledge.append(_knowledge_entry(
            "know-b", mara, "Second synced fact."))
        _persist_run(svc, 2000.0, lambda: len(main_puts()) >= 2)
        assert main_puts()[1][1] == first_sha, fake.puts
        # A NEWER remote appears (another device pushed).
        remote_doc = json.loads(json.dumps(
            json.loads(local.read_text())))
        remote_doc["saved_at"] = time.time() + 5000
        remote_doc["world"]["knowledge"] = [_knowledge_entry(
            "know-c", mara, "A fact from another device.")]
        fake.files[key] = {"doc": remote_doc, "sha": "sha-remote"}
    # Startup pull applies the newer remote and writes it local.
    with tempfile.TemporaryDirectory() as tmp2:
        bridge2, manager2, cards2, spawns2 = _res_lab(tmp2)
        local2 = Path(tmp2) / "world-state.local.json"
        svc2 = PersistenceService(
            bridge2, {"repo": repo, "interval_s": 10,
                      "local_path": str(local2)}, transport=fake)
        svc2.startup()
        assert any("another device" in (e.get("summary") or "")
                   for e in bridge2.city_knowledge), \
            bridge2.city_knowledge
        pulled = json.loads(local2.read_text())
        assert pulled["saved_at"] == remote_doc["saved_at"]
        # An OLDER remote is ignored when the local file is newer.
        local_doc3 = json.loads(json.dumps(remote_doc))
        local_doc3["saved_at"] = remote_doc["saved_at"] + 9000
        local_doc3["world"]["knowledge"] = [_knowledge_entry(
            "know-d", mara, "The newest local fact.")]
    with tempfile.TemporaryDirectory() as tmp3:
        bridge3, manager3, cards3, spawns3 = _res_lab(tmp3)
        local3 = Path(tmp3) / "world-state.local.json"
        local3.write_text(json.dumps(local_doc3))
        svc3 = PersistenceService(
            bridge3, {"repo": repo, "interval_s": 10,
                      "local_path": str(local3)}, transport=fake)
        svc3.startup()
        summaries = [(e.get("summary") or "")
                     for e in bridge3.city_knowledge]
        assert any("newest local fact" in s for s in summaries), \
            summaries
        assert not any("another device" in s for s in summaries), \
            summaries
    # Transport failure: offline status, local file intact, and
    # the world keeps ticking — no exception escapes.
    with tempfile.TemporaryDirectory() as tmp4:
        bridge4, manager4, cards4, spawns4 = _res_lab(tmp4)
        dead = FakeGitHubTransport()
        dead.fail = True
        local4 = Path(tmp4) / "world-state.local.json"
        svc4 = PersistenceService(
            bridge4, {"repo": repo, "interval_s": 10,
                      "local_path": str(local4)}, transport=dead)
        svc4.startup()
        assert svc4.public_status()["state"] == "offline", \
            svc4.public_status()
        bridge4.city_knowledge.append(_knowledge_entry(
            "know-e", cards4[0].name, "Learned while offline."))
        _persist_run(svc4, 1000.0, lambda: local4.is_file())
        svc4.tick(2000.0)
        time.sleep(0.1)
        svc4.tick(2001.0)      # drains the failed push
        status4 = svc4.public_status()
        assert status4["state"] == "offline", status4
        assert status4["last_error"], status4
        saved = json.loads(local4.read_text())
        assert saved["world"]["knowledge"], saved
        events = manager4.update((500.0, 500.0), dict(spawns4),
                                 now=3000.0)
        assert isinstance(events, list)


def test_versioned_snapshots():
    # (bn) Version copies P12: a CHANGED push also archives a
    # dated copy under versions/ — throttled to one per interval,
    # pruned to the newest version_keep, and strictly
    # best-effort: a version-side failure leaves the main push
    # synced with only a status note. No sync (local-only) means
    # no version attempts at all.
    repo = "ZeroZen270/hermes-world-data"
    key = (repo, "main", "world-state.json")

    def version_paths(fake):
        return sorted(p for (_r, _b, p) in fake.files
                      if p.startswith("versions/"))

    def main_puts(fake):
        return [p for p in fake.puts if p[0] == "world-state.json"]

    def knows(fake, path, frag):
        doc = fake.files[(repo, "main", path)]["doc"]
        return any(frag in (e.get("summary") or "")
                   for e in doc["world"]["knowledge"])

    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        mara = cards[0].name
        fake = FakeGitHubTransport()
        local = Path(tmp) / "world-state.local.json"
        svc = PersistenceService(
            bridge, {"repo": repo, "interval_s": 10,
                     "local_path": str(local),
                     "version_interval_s": 1800,
                     "version_keep": 20}, transport=fake)
        svc.startup()
        # (a) First dirty push: the live file AND one version
        # copy carrying the same content.
        bridge.city_knowledge.append(_knowledge_entry(
            "know-v1", mara, "Versioned fact one."))
        _persist_run(svc, 1000.0,
                     lambda: svc.public_status()["last_push"])
        assert len(main_puts(fake)) == 1, fake.puts
        v1 = version_paths(fake)
        assert len(v1) == 1, v1
        assert knows(fake, v1[0], "Versioned fact one")
        assert svc.public_status()["last_version"], \
            svc.public_status()
        # (b) A second dirty push inside the throttle window
        # updates the live file but archives nothing new.
        bridge.city_knowledge.append(_knowledge_entry(
            "know-v2", mara, "Versioned fact two."))
        _persist_run(svc, 2000.0, lambda: len(main_puts(fake)) >= 2)
        time.sleep(0.3)
        assert version_paths(fake) == v1, version_paths(fake)
        # (c) Once the window has passed, the next dirty push
        # archives a second copy.
        svc._last_version_at = time.time() - 7200.0
        bridge.city_knowledge.append(_knowledge_entry(
            "know-v3", mara, "Versioned fact three."))
        _persist_run(svc, 3000.0, lambda: len(main_puts(fake)) >= 3)
        time.sleep(0.3)
        v3 = version_paths(fake)
        assert len(v3) == 2, v3
        # (d) Retention: with keep=2 and no throttle, the next
        # dirty push archives and prunes the OLDEST copy.
        svc.version_keep = 2
        svc.version_interval_s = 0
        bridge.city_knowledge.append(_knowledge_entry(
            "know-v4", mara, "Versioned fact four."))
        _persist_run(svc, 4000.0, lambda: len(main_puts(fake)) >= 4)
        time.sleep(0.3)
        v4 = version_paths(fake)
        assert len(v4) == 2, v4
        assert fake.deletes, "retention pruned nothing"
        assert v1[0] not in v4, (v1, v4)
        assert any(knows(fake, p, "Versioned fact four")
                   for p in v4), v4
    # (e) Version writes failing: the main push still lands, the
    # status stays synced, and the failure is a note.
    with tempfile.TemporaryDirectory() as tmp2:
        bridge2, manager2, cards2, spawns2 = _res_lab(tmp2)
        fake2 = FakeGitHubTransport()
        fake2.fail_versions = True
        local2 = Path(tmp2) / "world-state.local.json"
        svc2 = PersistenceService(
            bridge2, {"repo": repo, "interval_s": 10,
                      "local_path": str(local2)}, transport=fake2)
        svc2.startup()
        bridge2.city_knowledge.append(_knowledge_entry(
            "know-v5", cards2[0].name, "Versioned fact five."))
        _persist_run(svc2, 1000.0,
                     lambda: svc2.public_status()["last_push"])
        status2 = svc2.public_status()
        assert status2["state"] == "synced", status2
        assert status2["last_error"] is None, status2
        assert status2["version_error"], status2
        assert version_paths(fake2) == []
        assert key in fake2.files
    # Local-only (no transport, no token in env): ticking saves
    # the local file and never attempts a version.
    saved_env = {k: os.environ.pop(k, None)
                 for k in ("HERMES_WORLD_TOKEN", "GITHUB_TOKEN")}
    try:
        with tempfile.TemporaryDirectory() as tmp3:
            bridge3, manager3, cards3, spawns3 = _res_lab(tmp3)
            local3 = Path(tmp3) / "world-state.local.json"
            svc3 = PersistenceService(
                bridge3, {"repo": repo, "interval_s": 10,
                          "local_path": str(local3)})
            assert svc3.public_status()["state"] == "local-only"
            svc3.startup()
            svc3.tick(1000.0)
            svc3.tick(2000.0)
            assert local3.is_file()
            assert svc3.public_status()["last_version"] is None
    finally:
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v


def test_open_proposal_snapshot_restore():
    # (az) Persistence P7: a proposal caught MID-VOTE — some
    # ballots cast, the window still open — snapshots and restores
    # into a fresh world with its ballots intact and its state
    # voting. Only the citizens who had not yet cast are re-cast;
    # the pre-snapshot ballots survive byte-identical, and the
    # vote runs to completion.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        engine = bridge.civics

        def slow_for(card):
            time.sleep(0.15)
            return ("FOR", "the city needs it")

        engine.ballot_caster = FakeBallotCaster(policy=slow_for)
        engine.ai_ballots = True
        _events, err = engine.open_proposal(
            "The Keeper", "A gathering drive for the stores",
            "Put to the city by the Keeper.",
            {"type": "gather_drive", "resource": "food"}, 1000.0)
        assert err is None, err
        _civic_run(manager, dict(spawns), 1000.0,
                   lambda: engine.current is not None
                   and engine.current["state"] == "voting"
                   and 2 <= len(engine.current["ballots"]) < 6,
                   limit=120)
        prop = engine.current
        assert prop is not None and prop["state"] == "voting", prop
        cast_before = {n: dict(b)
                       for n, b in prop["ballots"].items()}
        assert 2 <= len(cast_before) < 6, cast_before
        doc = assemble_snapshot(bridge)
        open_doc = doc["world"]["open_proposal"]
        assert open_doc is not None
        assert len(open_doc["ballots"]) == len(cast_before), open_doc
    with tempfile.TemporaryDirectory() as tmp2:
        bridge2, manager2, cards2, spawns2 = _civic_lab(tmp2, n=6)
        engine2 = bridge2.civics
        fake2 = FakeBallotCaster()
        engine2.ballot_caster = fake2
        engine2.ai_ballots = True
        # The restore moment is named in THIS test's tick domain
        # (the synthetic clock _civic_run ticks from 2000.0) — a
        # monotonic default would land the resumed vote's deadline
        # before the test's sim start on a fresh host and close
        # the vote early.
        apply_snapshot(bridge2, doc, now=2000.0)
        prop2 = engine2.current
        assert prop2 is not None, "the open proposal did not restore"
        assert prop2["state"] == "voting", prop2["state"]
        assert {n: dict(b) for n, b in prop2["ballots"].items()} \
            == cast_before
        _civic_run(manager2, dict(spawns2), 2000.0,
                   lambda: engine2.current is None
                   and len(engine2.history) == 1)
        final = engine2.history[0]
        assert final["state"] == "adopted", final["state"]
        assert len(final["ballots"]) == 6, final["ballots"]
        for name, ballot in cast_before.items():
            assert final["ballots"][name] == ballot, \
                (name, final["ballots"][name], ballot)
        called = {name for name, _soul in fake2.calls}
        assert called == {c.name for c in cards2} - set(cast_before), \
            called


class FakeDesignBrain:
    """Design-brain double (P8): records every ctx it is handed
    (so a test can inspect the revision errors) and returns a deep
    copy of its canned spec on every call."""

    def __init__(self, spec):
        self.spec = spec
        self.calls = []

    def __call__(self, ctx):
        self.calls.append(ctx)
        return json.loads(json.dumps(self.spec))


def test_design_validation_and_shelving():
    # (ba) Designs P8: the seeded defaults exist before any brain
    # call; valid building_style + interior specs register through
    # the synchronous path; an off-palette style is refused; and
    # the async pipeline takes a bad interior (an unknown prop
    # kind AND a prop outside the footprint), feeds the errors
    # back for exactly ONE revision, then shelves it — nothing
    # registers.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        engine = bridge.designs
        assert engine.interior_for_kind("house") is not None
        assert engine.interior_for_kind("stall") is not None
        assert any(a["name"] == "Founders' fountain"
                   for a in engine.assets), engine.assets

        rec, errors = engine.register_spec({
            "kind": "building_style", "name": "Slate & Cream",
            "archetype": "house",
            "palette": {"wall": "#e8dcc4", "roof": "#5a6b7a",
                        "trim": "#8a4b2a"},
            "roof_style": "gable", "floors": 2}, origin="test")
        assert errors == [] and rec is not None, errors
        assert engine.style_for("house")["name"] == "Slate & Cream"

        before = len(engine.interiors)
        rec, errors = engine.register_spec({
            "kind": "interior", "name": "Scholar's rooms",
            "building_kind": "house",
            "rooms": [{"name": "Study", "props": [
                {"kind": "bookshelf", "dx": -4.0, "dz": -2.0,
                 "rot": 90},
                {"kind": "table", "dx": 0.0, "dz": 0.0, "rot": 0}]}]},
            origin="test")
        assert errors == [] and rec is not None, errors
        assert len(engine.interiors) == before + 1

        rec, errors = engine.register_spec({
            "kind": "building_style", "name": "Neon blare",
            "archetype": "house",
            "palette": {"wall": "#ff00ff", "roof": "#5a6b7a",
                        "trim": "#8a4b2a"},
            "roof_style": "gable", "floors": 2}, origin="test")
        assert rec is None and errors, errors
        assert engine.style_for("house")["name"] == "Slate & Cream"

        bad = {"kind": "interior", "name": "Dream annex",
               "building_kind": "house",
               "rooms": [{"name": "Spa", "props": [
                   {"kind": "jacuzzi", "dx": 0.0, "dz": 0.0,
                    "rot": 0},
                   {"kind": "table", "dx": 99.0, "dz": 0.0,
                    "rot": 0}]}]}
        fake = FakeDesignBrain(bad)
        engine.design_brain = fake
        interiors_before = len(engine.interiors)
        ok, _text = engine.request_design(
            "interior", "a spa annex for the houses", "The Keeper",
            1000.0)
        assert ok
        _civic_run(manager, dict(spawns), 1000.0,
                   lambda: len(engine.shelved) == 1)
        assert len(engine.shelved) == 1, engine.shelved
        assert len(fake.calls) == 2, len(fake.calls)
        errs2 = fake.calls[1].get("errors") or []
        assert any("jacuzzi" in e for e in errs2), errs2
        assert any("outside the footprint" in e for e in errs2), errs2
        assert len(engine.interiors) == interiors_before
        assert engine.shelved[0]["reason"], engine.shelved


def test_asset_vote_cycle():
    # (bb) Voted assets P8: a player asset proposal runs the full
    # civic cycle (fake AI ballots, all FOR) -> adopted -> the
    # design engine's fake brain draws a fountain composition ->
    # the asset registers with its props in the served layout
    # document (designs slice + layout marker) and survives a
    # persistence snapshot round-trip into a fresh world.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        resp = bridge.handle_message(
            {"type": "save_layout",
             "layout": {"version": 1, "objects": []}})
        assert resp[0].get("ok") is True, resp
        civics = bridge.civics
        fake = FakeBallotCaster()
        civics.ballot_caster = fake
        civics.ai_ballots = True
        fountain = {"kind": "asset", "name": "Voting fountain",
                    "footprint": [5.0, 5.0],
                    "props": [
                        {"kind": "fountain", "dx": 0.0, "dz": 0.0,
                         "rot": 0},
                        {"kind": "bench", "dx": -2.0, "dz": 1.5,
                         "rot": 45},
                        {"kind": "planter", "dx": 2.0, "dz": 2.0,
                         "rot": 0}]}
        dbrain = FakeDesignBrain(fountain)
        bridge.designs.design_brain = dbrain
        resp = bridge.handle_message(
            {"type": "propose", "title": "A fountain for the plaza",
             "action": {"type": "asset",
                        "asset_hint": "a fountain for the plaza"}})
        assert resp[0]["type"] == "proposal", resp
        _civic_run(manager, dict(spawns), 1000.0,
                   lambda: civics.current is None
                   and len(civics.history) == 1
                   and any(a["name"] == "Voting fountain"
                           for a in bridge.designs.assets))
        prop = civics.history[0]
        assert prop["state"] == "adopted", prop
        asset = next(a for a in bridge.designs.assets
                     if a["name"] == "Voting fountain")
        assert asset["origin"] == "brain", asset
        assert any(p["kind"] == "fountain" for p in asset["props"]), \
            asset
        # The layout document carries it both ways: the designs
        # slice (full composition) and the prop marker in objects.
        resp = bridge.handle_message({"type": "load_layout"})
        doc = resp[0]["layout"]
        names = [a["name"] for a in doc["designs"]["assets"]]
        assert "Voting fountain" in names, doc.get("designs")
        marker = f"design-asset:{asset['id']}"
        assert any(o.get("asset") == marker
                   for o in doc["objects"]), doc["objects"]
        payload = bridge.world_payload()
        assert any(a["name"] == "Voting fountain"
                   for a in payload["designs"]["assets"]), payload
        # Snapshot round-trip into a fresh world.
        snap = assemble_snapshot(bridge)
        tmp2 = Path(tmp) / "world2"
        tmp2.mkdir()
        bridge2, manager2, cards2, spawns2 = _civic_lab(
            str(tmp2), n=6)
        apply_snapshot(bridge2, snap)
        assert any(a["name"] == "Voting fountain"
                   for a in bridge2.designs.assets), \
            bridge2.designs.assets
        assert any(o.get("asset") == marker
                   for o in (bridge2.layout or {})["objects"]), \
            bridge2.layout


def test_weather_engine_and_rules():
    # (bc) Weather P8: two identically-seeded engines ticked in
    # lockstep trace identical skies with real transitions; the
    # god protocol forces state + clock and the status payload
    # carries them; the rule interpreter answers weather/time
    # questions with the set state; and a snapshot round-trip
    # preserves clock + weather.
    from hermes_local.weather import WeatherEngine

    eng_a, eng_b = WeatherEngine(), WeatherEngine()
    seq_a, seq_b = [], []
    now = 0.0
    for _ in range(240):
        now += 10.0
        eng_a.tick(now)
        eng_b.tick(now)
        seq_a.append(eng_a.state)
        seq_b.append(eng_b.state)
    assert seq_a == seq_b
    assert len(set(seq_a)) >= 2, set(seq_a)

    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        resp = bridge.handle_message(
            {"type": "set_weather", "state": "rain"})
        assert resp[0]["type"] == "weather" \
            and resp[0]["state"] == "rain", resp
        resp = bridge.handle_message(
            {"type": "set_time", "hour": 21.5})
        assert resp[0]["type"] == "weather" \
            and abs(resp[0]["hour"] - 21.5) < 0.3, resp
        status = bridge.status_payload()
        assert status["weather"]["state"] == "rain", \
            status["weather"]
        ans = bridge.ask("What's the weather like?")
        assert "rain" in ans.lower(), ans
        ans = bridge.ask("What time is it?")
        assert "21:" in ans, ans

        snap = assemble_snapshot(bridge)
        tmp2 = Path(tmp) / "world2"
        tmp2.mkdir()
        bridge2, manager2, cards2, spawns2 = _res_lab(str(tmp2))
        apply_snapshot(bridge2, snap)
        assert bridge2.weather.state == "rain", bridge2.weather.state
        assert abs(bridge2.weather.hour - bridge.weather.hour) < 1.0


def test_expansion_mechanics():
    # (bd) City growth P9: the god protocol expands the world
    # 120 -> 180 — every founding sample keeps its height at its
    # world coordinates, the new ring carries eased relief with no
    # cliff at the seam, four ring-1 district anchors join the
    # planner, and the terrain slice round-trips bound + grid
    # exactly. A second expansion opens ring 2, /api/world reports
    # the live geometry, and the rule interpreter knows the span.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        ter = bridge.terrain
        planner = manager.planner
        ter.apply_op("raise", 20, 20, 8, 1.2)
        ter.apply_op("lower", -60, 40, 10, 1.0)
        core_points = [(x, z) for x in (-120, -60, 0, 60, 119)
                       for z in (-120, -40, 0, 80, 120)]
        core_points += [(20, 20), (16, 24), (-60, 40), (-56, 36)]
        before = {p: ter.sample(*p) for p in core_points}
        assert any(v != 0.0 for v in before.values())
        resp = bridge.handle_message({"type": "expand"})
        assert resp[0]["type"] == "expansion", resp
        assert resp[0]["bound"] == 180 and resp[0]["ring"] == 1, resp
        assert resp[0]["district"] == ["East Fields", "North Woods",
                                       "West Plots",
                                       "South Meadows"], resp
        assert ter.bound == 180 and ter.expansions == 1
        assert len(ter.heights) == 181 * 181
        for p, h in before.items():
            assert ter.sample(*p) == h, (p, h, ter.sample(*p))
        ring_max = max(ter.sample(x, z)
                       for x in (130, 150, 170)
                       for z in (-150, -60, 0, 60, 150))
        assert ring_max > 0.2, ring_max
        for z in (-60.0, 0.0, 60.0):
            step = abs(ter.sample(121.0, z) - ter.sample(119.0, z))
            assert step < 0.35, (z, step)
        names = {a.name for a in planner.anchors}
        for name in ("East Fields", "North Woods", "West Plots",
                     "South Meadows"):
            assert name in names, name
        east = next(a for a in planner.anchors
                    if a.name == "East Fields")
        assert (east.x, east.z) == (150.0, 0.0), (east.x, east.z)
        doc = ter.serialize()
        ter2 = TerrainWorld()
        out = ter2.restore(doc)
        assert out["bound"] == 180 and ter2.bound == 180
        assert ter2.heights == ter.heights
        resp = bridge.handle_message({"type": "expand"})
        assert resp[0]["bound"] == 240 and resp[0]["ring"] == 2, resp
        names = {a.name for a in planner.anchors}
        for name in ("East Reaches", "North Pines", "West Farms",
                     "South Downs"):
            assert name in names, name
        payload = bridge.world_payload()
        assert payload["terrain"]["bound"] == 240
        assert payload["terrain"]["size"] == 241
        assert len(payload["terrain"]["heights"]) == 241 * 241
        ans = bridge.ask("How big is the city?")
        assert "480m across" in ans, ans
        assert "East Fields" in ans, ans
        ans = bridge.ask("Can the city grow?")
        assert "can grow" in ans and "±360m" in ans, ans


def test_civic_expansion_adopted():
    # (be) City growth P9: an "expand" proposal runs the full civic
    # cycle on fake AI ballots; adoption grows the land at once
    # (the vote is the act), emits the expansion event as a civic
    # act, and queues exactly two flatten crew jobs on plots
    # inside the new ring for the foreman to staff.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        engine = bridge.civics
        fake = FakeBallotCaster()
        engine.ballot_caster = fake
        engine.ai_ballots = True
        resp = bridge.handle_message(
            {"type": "propose", "title": "Room to grow",
             "action": {"type": "expand"}})
        assert resp[0]["type"] == "proposal", resp
        events, now = _civic_run(
            manager, dict(spawns), 1000.0,
            lambda: engine.current is None and len(engine.history) == 1)
        prop = engine.history[0]
        assert prop["state"] == "adopted", prop["state"]
        assert bridge.terrain.bound == 180, bridge.terrain.bound
        exps = [e for e in events if e.get("type") == "expansion"]
        assert len(exps) == 1 and exps[0]["bound"] == 180, exps
        assert exps[0]["by"] == "civic vote", exps
        jobs = list(bridge.terrain.jobs.values())
        assert len(jobs) == 2, jobs
        for job in jobs:
            assert job["op"] == "flatten", job
            assert 120 < max(abs(job["x"]), abs(job["z"])) <= 180, job
        names = {a.name for a in manager.planner.anchors}
        assert "East Fields" in names and "South Meadows" in names


def test_expansion_limits_and_placement():
    # (bf) City growth P9: the limits hold — placement follows the
    # LIVE bound (a fence fits in the new ring at bound 180, the
    # same spot is refused beyond it), and at the ±360 engine cap
    # an expansion attempt is a notice that changes nothing.
    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _civic_lab(tmp, n=6)
        ter = bridge.terrain
        con = bridge.construction
        resp = bridge.handle_message({"type": "expand"})
        assert resp[0]["bound"] == 180, resp
        # (150, 24): inside the new ring, clear of the road bands.
        site, reason = con.place("fence", 150.0, 24.0)
        assert site is not None, reason
        site2, reason2 = con.place("fence", 200.0, 24.0)
        assert site2 is None \
            and "outside the city bounds" in reason2, reason2
        for want in (240, 300, 360):
            resp = bridge.handle_message({"type": "expand"})
            assert resp[0]["type"] == "expansion" \
                and resp[0]["bound"] == want, resp
        assert ter.bound == 360 and ter.expansions == 4
        heights_at_cap = list(ter.heights)
        resp = bridge.handle_message({"type": "expand"})
        assert resp[0]["type"] == "notice", resp
        assert ter.bound == 360 and ter.heights == heights_at_cap
        bare = TerrainWorld()
        for _ in range(4):
            assert bare.expand() is not None
        assert bare.expand() is None and bare.bound == 360


def test_openai_path_no_doubled_v1():
    """(bg) Regression: a base URL already ending in /v1 must not gain
    a second /v1 prefix. The doubled /v1/v1/chat/completions 404'd
    EVERY Nous chat call on the tablet (2026-11-24) — the root cause
    of the "(seems distracted)" outage, invisible to the startup
    probe because /models still answered. Covers chat + embeddings
    against /v1-suffixed and bare bases, no network (posts stubbed)."""
    from hermes_local.embeddings import EmbeddingClient
    from hermes_local.llm import LLMClient

    seen = []

    def fake_post(self, path, payload, **_kw):
        seen.append(path)
        if path.endswith("/chat/completions"):
            return {"choices": [{"message": {"content": "ok"}}]}
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    orig_chat, orig_emb = LLMClient._post, EmbeddingClient._post
    LLMClient._post = fake_post
    EmbeddingClient._post = fake_post
    try:
        for base, want in (
                ("https://inference-api.nousresearch.com/v1",
                 "/chat/completions"),
                ("https://api.example.com", "/v1/chat/completions")):
            client = LLMClient(base_url=base, model="m", api="openai")
            out = client._chat_openai(
                [{"role": "user", "content": "hi"}], None)
            assert out["content"] == "ok", out
            assert seen[-1] == want, (base, seen[-1])
        for base, want in (
                ("https://inference-api.nousresearch.com/v1",
                 "/embeddings"),
                ("https://api.example.com", "/v1/embeddings")):
            client = EmbeddingClient(base_url=base, api="openai",
                                     models=["m"])
            vecs = client._embed_http("m", ["hello"])
            assert vecs == [[0.1, 0.2]], vecs
            assert seen[-1] == want, (base, seen[-1])
        assert all("/v1/v1/" not in p for p in seen), seen
    finally:
        LLMClient._post = orig_chat
        EmbeddingClient._post = orig_emb


def test_walkgrid_derivation():
    """(bh) Walk grid P10: dimensions follow the live bound
    (120x120 cells founding, 180x180 after one expansion);
    building footprints block — including the 0.4m inflation
    margin outside the raw footprint; dug ponds and steep
    ground block too."""
    from hermes_local.buildings import ConstructionWorld
    from hermes_local.terrain import TerrainWorld
    from hermes_local.walkgrid import WalkGrid

    ter = TerrainWorld()
    con = ConstructionWorld()
    wg = WalkGrid(ter, con)
    assert (wg.w, wg.h) == (120, 120), (wg.w, wg.h)
    assert len(wg.cells) == 120 * 120
    assert wg.cost_at(0, 20) == 5, wg.cost_at(0, 20)      # road band
    assert wg.cost_at(30, 32) == 2, wg.cost_at(30, 32)    # open grass

    con.restore({"sites": [], "buildings": [
        {"id": "b1", "kind": "house", "position": [30.4, 30.0],
         "rotation": 0, "anchor": "House 1"}]})
    wg.rebuild()
    assert wg.cost_at(30, 30) == 0                        # under the house
    # Cell centre (25, 31): 5.4m from the centre — outside the
    # raw 5m half-footprint, inside it once inflated by 0.4m.
    assert wg.cost_at(25, 30) == 0, wg.cost_at(25, 30)
    assert wg.cost_at(23, 30) != 0                        # beyond the margin
    assert wg.cost_at(30, 36) == 1                        # the door apron

    ter.apply_op("lower", 60, 60, 8, 2.0)                 # dig a pond
    ter.apply_op("raise", -60, -60, 2, 2.0)               # a steep spike
    wg.rebuild()
    assert wg.cost_at(60, 60) == 0                        # below waterline
    assert any(wg.cost_at(x, -60) == 0
               for x in range(-66, -53)), "spike slope must block"

    info = ter.expand()
    assert info is not None and wg.rebuild()
    assert (wg.w, wg.h) == (180, 180), (wg.w, wg.h)
    assert wg.cost_at(156, 20) == 5                       # ring road


def test_walkgrid_routing():
    """(bi) Walk grid P10 routing: A* bends around a building
    with no blocked cell on the path and no corner-cutting
    diagonal; LOS smoothing collapses an open-ground route to
    its endpoints; the snapper pulls a point inside a footprint
    out to nearby walkable ground."""
    from hermes_local.buildings import ConstructionWorld
    from hermes_local.terrain import TerrainWorld
    from hermes_local.walkgrid import WalkGrid

    ter = TerrainWorld()
    con = ConstructionWorld()
    con.restore({"sites": [], "buildings": [
        {"id": "b1", "kind": "house", "position": [30, 30],
         "rotation": 0, "anchor": "House 1"}]})
    wg = WalkGrid(ter, con)

    route = wg.find_route(10, 30, 50, 30)
    assert route is not None and len(route) >= 3, route
    pts = [(10.0, 30.0)] + [tuple(p) for p in route]
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        assert wg.line_clear(ax, az, bx, bz), ((ax, az), (bx, bz))
    for x, z in route:
        assert wg.cost_at(x, z) != 0, (x, z)

    # The raw A* path respects the corner-cut rule: every
    # diagonal step has both adjacent cardinal cells walkable.
    sc, gc = wg._cell_of(10, 30), wg._cell_of(50, 30)
    raw = wg._astar(sc, gc)
    assert raw is not None and len(raw) > 10, raw
    for (ax, az), (bx, bz) in zip(raw, raw[1:]):
        assert wg.cells[az * wg.w + ax] != 0
        if ax != bx and az != bz:
            assert wg.cells[az * wg.w + bx] != 0, ((ax, az), (bx, bz))
            assert wg.cells[bz * wg.w + ax] != 0, ((ax, az), (bx, bz))

    open_route = wg.find_route(10, 60, 50, 60)
    assert open_route is not None and len(open_route) <= 2, open_route

    snapped = wg.snap_to_walkable(30, 30)
    assert snapped is not None and wg.cost_at(*snapped) != 0, snapped
    assert ((snapped[0] - 30) ** 2 + (snapped[1] - 30) ** 2) ** 0.5 < 9


def test_walkgrid_bridge_and_planner_route():
    """(bj) Walk grid P10 wiring: /api/world carries the grid
    (base64 round-trips to the bytearray); a build assignment
    whose target hides behind a finished house gets a route on
    its travel intent whose segments never cross blocked cells;
    completing the building bumps the grid version (its door
    apron appears)."""
    import base64

    with tempfile.TemporaryDirectory() as tmp:
        bridge, manager, cards, spawns = _res_lab(tmp)
        planner = manager.planner
        con = bridge.construction
        inv = bridge.resources.inventory
        inv["wood"], inv["food"] = 10, 8
        inv["planks"] = 4
        con.restore({"sites": [], "buildings": [
            {"id": "b1", "kind": "house", "position": [20, 20],
             "rotation": 0, "anchor": "House 1"}]})
        bridge._refresh_walkgrid(force=True)
        wg = bridge.walkgrid
        assert planner.walkgrid is wg

        payload = bridge.world_payload()
        wgp = payload["walkgrid"]
        assert (wgp["w"], wgp["h"], wgp["cell"]) == (120, 120, 2.0), wgp
        assert base64.b64decode(wgp["data_b64"]) == bytes(wg.cells)
        v1 = wgp["version"]

        resp = bridge.handle_message(
            {"type": "place", "kind": "stall", "position": [36, 20]})
        assert resp[0]["state"] == "placed", resp
        site = con.sites[resp[0]["site"]]

        a = cards[0].name
        positions = dict(spawns)
        positions[a] = (4.0, 20.0)          # west of the house
        task, reason = planner.assign_build(
            site.id, 1000.0, positions, npc_name=a)
        assert task is not None, reason
        events = manager.update((500.0, 500.0), positions, now=1001.0)
        intents = [e for e in events if e.get("type") == "intent"
                   and e.get("npc") == a and e.get("state") == "travel"]
        assert intents, events
        route = intents[0].get("route")
        assert route and len(route) >= 2, intents[0]
        pts = [(4.0, 20.0)] + [tuple(p) for p in route]
        for (ax, az), (bx, bz) in zip(pts, pts[1:]):
            assert wg.line_clear(ax, az, bx, bz), ((ax, az), (bx, bz))

        events, now = _build_sim_until(manager, positions, 1001.0,
                                       lambda: site.complete)
        assert site.complete, site
        payload = bridge.world_payload()
        assert payload["walkgrid"]["version"] > v1, payload["walkgrid"]
        assert any(r.get("type") == "walkgrid"
                   for r in bridge.handle_message(
                       {"type": "place", "kind": "lamp",
                        "position": [60, 36]}))


class PromptCaptureMock:
    """Mock LLM that records the exact messages of every call and
    answers plainly in character (no tool calls)."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        return {"content": "Aye, noted.", "tool_calls": []}


def _p11_lab(tmp, vera_client=None):
    """Two citizens with hand-set roles/spawns on a bare manager:
    Vera the vendor (spawn beside Market Row East, her workplace)
    and Gus the guard (spawn on Central Plaza, his workplace).
    The planner is wired with a terrain, a walk grid and a
    weather engine, as the bridge wires them."""
    from hermes_local.npc import CharacterCard
    from hermes_local.terrain import TerrainWorld
    from hermes_local.walkgrid import WalkGrid
    from hermes_local.weather import WeatherEngine

    mem = tmp_memory(tmp)
    cards = [
        CharacterCard(name="Vera", role="vendor",
                      persona="Vera sells food from a street cart.",
                      traits=["cheerful"], voice="Bright and loud."),
        CharacterCard(name="Gus", role="guard",
                      persona="Gus patrols the streets.",
                      traits=["stoic"], voice="Brief and formal."),
    ]
    npcs = {
        "Vera": NPC(cards[0], vera_client or SummaryMock(),
                    memory=mem, embedder=mem.embedder),
        "Gus": NPC(cards[1], SummaryMock(), memory=mem,
                   embedder=mem.embedder),
    }
    spawns = {"Vera": (30.0, -8.0), "Gus": (8.0, 8.0)}
    manager = NPCManager(npcs, max_active=4, spawns=spawns)
    planner = manager.planner
    ter = TerrainWorld()
    planner.terrain = ter
    planner.walkgrid = WalkGrid(ter)
    planner.weather = WeatherEngine()
    return manager, planner, ter, spawns


def test_location_context_and_prompt():
    """(bk) Location context P11: at her market anchor the
    situation names the anchor, her workplace and the clock;
    mid-road it says "middle of the road" and claims no place;
    at a dug pond it finds the water. The situation (with the
    grounding rule) reaches a mock-LLM player chat's system
    prompt through route_speech."""
    with tempfile.TemporaryDirectory() as tmp:
        cap = PromptCaptureMock()
        manager, planner, ter, spawns = _p11_lab(tmp, vera_client=cap)
        planner.weather.set_time(14.25)

        at_stall = planner.situation_text("Vera", (36.0, -8.0))
        assert "14:15" in at_stall and "afternoon" in at_stall, at_stall
        assert "Market Row East" in at_stall, at_stall
        assert "workplace" in at_stall, at_stall

        mid_road = planner.situation_text("Vera", (0.0, 20.0))
        assert "middle of the road" in mid_road, mid_road
        assert "Market Row" not in mid_road, mid_road
        assert "workplace" not in mid_road, mid_road
        assert "at home" not in mid_road, mid_road

        ter.apply_op("lower", -60, 0, 8, 2.0)          # dig a pond
        planner.walkgrid.rebuild()
        at_pond = planner.situation_text("Vera", (-60.0, 0.0))
        assert "pond" in at_pond, at_pond

        positions = {"Vera": (36.0, -8.0), "Gus": (400.0, 400.0)}
        manager.update((36.0, -8.0), positions, now=1000.0)
        replies = manager.route_speech("Hello there", now=1001.0)
        assert replies.get("Vera") == "Aye, noted.", replies
        systems = [c[0]["content"] for c in cap.calls
                   if c and c[0].get("role") == "system"]
        assert systems, cap.calls
        joined = "\n".join(systems)
        assert "Where you are right now" in joined, joined
        assert "Market Row East" in joined, joined
        assert "14:15" in joined, joined
        assert "passing through" in joined, joined   # the grounding rule


def test_canned_lines_location_gated():
    """(bl) Canned lines P11: the demo (mock) brain's vendor
    greeting is gated on the situation in the system prompt —
    met mid-road the vendor claims no cart and is just passing
    through; at the market stall the cart pitch is allowed."""
    from hermes_local.world_bridge import DemoBrain

    with tempfile.TemporaryDirectory() as tmp:
        _manager, planner, _ter, _spawns = _p11_lab(tmp)
        planner.weather.set_time(10.0)
        vera = planner.npcs["Vera"]
        brain = DemoBrain()

        road_sys = vera.system_prompt(
            [], planner.situation_text("Vera", (0.0, 20.0)))
        reply = brain.chat([{"role": "system", "content": road_sys},
                            {"role": "user", "content": "Player: hello"}])
        text = reply["content"]
        assert "right cart" not in text, text
        assert "my shop" not in text, text
        assert "passing through" in text, text

        stall_sys = vera.system_prompt(
            [], planner.situation_text("Vera", (36.0, -8.0)))
        reply = brain.chat([{"role": "system", "content": stall_sys},
                            {"role": "user", "content": "Player: hello"}])
        assert "right cart" in reply["content"], reply


def test_daily_schedules_and_day_length():
    """(bm) Schedules P11: with a weather engine attached, the
    planner's own utility pick sends a vendor home at 23:30
    (at_home set — the sleep dwell) and to her workplace at
    09:00; a guard at 23:30 is NOT sent home (night watch bends
    to the plaza). The default day is 2880 sim-seconds (48 real
    minutes) and the 60s config floor still holds."""
    from hermes_local.weather import DEFAULT_DAY_LENGTH_S, WeatherEngine

    assert DEFAULT_DAY_LENGTH_S == 2880.0
    assert WeatherEngine().day_length_s == 2880.0
    assert WeatherEngine(day_length_s=30).day_length_s == 60.0

    with tempfile.TemporaryDirectory() as tmp:
        _manager, planner, _ter, spawns = _p11_lab(tmp)
        weather = planner.weather
        st_v = planner.state["Vera"]

        weather.set_time(23.5)
        anchor, _kind = planner._choose("Vera", 1000.0, spawns["Vera"])
        home = planner._home_anchor("Vera", st_v)
        assert anchor is home, (anchor, home)
        assert st_v["at_home"] is True, st_v

        weather.set_time(9.0)
        anchor, kind = planner._choose("Vera", 2000.0, spawns["Vera"])
        assert anchor is st_v["workplace"], (anchor, st_v["workplace"])
        assert kind == "work", kind
        assert st_v["at_home"] is False, st_v

        weather.set_time(23.5)
        st_g = planner.state["Gus"]
        anchor, kind = planner._choose("Gus", 3000.0, spawns["Gus"])
        home_g = planner._home_anchor("Gus", st_g)
        assert anchor is not home_g, (anchor, home_g)
        assert anchor.name == "Central Plaza" or kind == "social", \
            (anchor, kind)
        assert st_g["at_home"] is False, st_g


def test_workplace_pools_and_claim_radius():
    """(bo) Workplace pools + claim radius: a shopkeeper's
    workplace is a market (browse) anchor, not a house front —
    Viktor's bug: pooled to a front, his situation told him he
    was AT his workplace there, and his shop persona did the
    rest ("inside my shop", sitting on grass). An engineer
    pools to the depots (errand) like the other works roles;
    a medic keeps a front (no clinic anchor exists). And
    situation_text only CLAIMS a place within 5m: at ~8m it
    says "near <name>" with the surface and no ownership —
    for workplaces and for home alike."""
    from hermes_local.npc import (ActivityAnchor, CharacterCard,
                                  workplace_anchor)
    from hermes_local.terrain import TerrainWorld
    from hermes_local.walkgrid import WalkGrid
    from hermes_local.weather import WeatherEngine

    with tempfile.TemporaryDirectory() as tmp:
        mem = tmp_memory(tmp)
        cards = [
            CharacterCard(name="Viktor", role="shopkeeper",
                          persona="Viktor runs a small shop.",
                          traits=["proud"], voice="Warm and exact."),
            CharacterCard(name="Mina", role="medic",
                          persona="Mina staffs the clinic.",
                          traits=["calm"], voice="Calm and precise."),
            CharacterCard(name="Oskar", role="elder",
                          persona="Oskar has lived here for decades.",
                          traits=["stoic"], voice="Slow and dry."),
        ]
        npcs = {c.name: NPC(c, SummaryMock(), memory=mem,
                            embedder=mem.embedder) for c in cards}
        spawns = {"Viktor": (30.0, -8.0), "Mina": (13.0, -13.0),
                  "Oskar": (-59.0, 40.0)}
        manager = NPCManager(npcs, max_active=4, spawns=spawns)
        planner = manager.planner
        ter = TerrainWorld()
        planner.terrain = ter
        planner.walkgrid = WalkGrid(ter)
        planner.weather = WeatherEngine()
        planner.weather.set_time(10.0)
        # A built house near Oskar's spawn: his home anchor.
        planner.anchors.append(
            ActivityAnchor("House 1", "front", -59.0, 40.0))

        # (a) pools
        wp_v = planner.state["Viktor"]["workplace"]
        assert wp_v.kind == "browse", wp_v
        assert wp_v.name == "Market Row East", wp_v
        wp_m = planner.state["Mina"]["workplace"]
        assert wp_m.kind == "front", wp_m
        eng = CharacterCard(name="Petra", role="engineer",
                            persona="Petra keeps the power on.",
                            traits=["meticulous"], voice="Dry.")
        wp_e = workplace_anchor(eng, (100.0, -8.0), planner.anchors)
        assert wp_e.kind == "errand", wp_e

        # (b) workplace claim radius (Viktor: browse; Mina: front)
        at_stall = planner.situation_text("Viktor", (33.0, -8.0))
        assert "your workplace" in at_stall, at_stall
        near_stall = planner.situation_text("Viktor", (44.0, -8.0))
        assert "near Market Row East" in near_stall, near_stall
        assert "your workplace" not in near_stall, near_stall

        at_front = planner.situation_text("Mina", (16.0, -13.0))
        assert "your workplace" in at_front, at_front
        assert "outdoors" in at_front, at_front
        assert "grass" in at_front, at_front
        near_front = planner.situation_text("Mina", (21.0, -13.0))
        assert "near Block 24,-24 front 3" in near_front, near_front
        assert "your workplace" not in near_front, near_front
        assert "grass" in near_front, near_front

        # (c) home follows the same 5m rule
        at_home = planner.situation_text("Oskar", (-56.0, 40.0))
        assert "at home (House 1)" in at_home, at_home
        near_home = planner.situation_text("Oskar", (-51.0, 40.0))
        assert "near House 1" in near_home, near_home
        assert "at home" not in near_home, near_home


TESTS = [
    ("(a) agent loop, native mode + (b) real tools", test_agent_native_and_tools),
    ("(a) agent loop, ReAct mode", test_agent_react),
    ("(c) embedding fallback + memory recall", test_embedding_fallback_and_memory),
    ("(d) context compression on small window", test_context_compression),
    ("(e) mock world over bridge socket", test_mock_world_over_socket),
    ("(f) proximity tiers + concurrency cap", test_proximity_tiers_and_cap),
    ("(g) city layout round-trip", test_layout_round_trip),
    ("(h) procedural roster (54 NPCs)", test_roster),
    ("(i) nous provider preset (longcat free)", test_provider_preset),
    ("(j) portal route + status brain_info", test_portal_and_status_http),
    ("(k) tier hysteresis: no flap across the boundary", test_tier_hysteresis_no_flap),
    ("(l) conversation lock pins ACTIVE until timeout/distance", test_conversation_lock),
    ("(m) planner: dwell is the majority of NPC time", test_planner_dwell_majority),
    ("(n) witnessed pairing uses the generated exchange", test_generated_social_exchange),
    ("(o) generator failure in live mode: silence + one witnessed retry", test_generated_social_fallback),
    ("(p) per-pair cooldown blocks a second generation (silence)", test_generated_social_pair_cooldown),
    ("(q) souls: knowledge load restores minds, not situations", test_souls_knowledge_load),
    ("(r) souls: full load restores situations too", test_souls_full_load),
    ("(s) souls: malformed snapshots rejected, world untouched", test_souls_malformed_rejected),
    ("(t) ghost mode: unseen listener, no engagement, tiers still run", test_ghost_mode),
    ("(u) conversation plan: formed, remembered, rendezvous kept", test_conversation_plan_rendezvous),
    ("(v) no plan without a valid marker", test_no_plan_without_marker),
    ("(w) unwitnessed pairing still generated, zero canned", test_generated_social_unwitnessed),
    ("(x) witnessed conversation continues with history", test_generated_social_continuation),
    ("(y) unwitnessed generation failure: one attempt, silence", test_generated_social_unwitnessed_failure_silent),
    ("(z) bridge generator prompt is soul-fed (seeded memory)", test_bridge_generator_soul_prompt),
    ("(aa) world interpreter: live ask serves digest + plan, returns brain answer", test_interpreter_live_ask),
    ("(ab) world interpreter: matched memory reaches the brain, attributed", test_interpreter_memory_in_context),
    ("(ac) world interpreter: mock mode answers where/count from the digest", test_interpreter_mock_where),
    ("(ad) npc chat degrades to ReAct on any tools error", test_npc_tools_error_degrades_to_react),
    ("(ae) npc chat total failure keeps the distracted line", test_npc_total_failure_still_distracted),
    ("(af) resources: in-range harvest works, range/depleted refused", test_player_harvest_in_range),
    ("(ag) resources: regrowth stump->sapling->mature in sim ticks", test_node_regrowth),
    ("(ah) resources: /api/world shape + status inventory", test_world_endpoint),
    ("(ai) labor: foreman gathers wood to baseline via NPC harvest", test_foreman_gather_wood),
    ("(aj) labor: crafting converts at stations, stock-out fails clean", test_crafting_recipes),
    ("(ak) labor: order assigns named citizen; lock defers work", test_player_order_and_engaged_deferral),
    ("(al) building: stall staffed by foreman, raised to complete", test_build_stall_completes),
    ("(am) building: house waits in awaiting_materials, consumes nothing", test_build_house_waits_for_materials),
    ("(an) building: sites+buildings layout round-trip; souls-safe", test_build_layout_round_trip),
    ("(ao) terraforming: god raise falloff + exact grid round-trip", test_terraform_raise_and_round_trip),
    ("(ap) terraforming: crew flatten job converges + completes", test_terraform_crew_flatten),
    ("(aq) terraforming: invalid ops refused, god op instant", test_terraform_invalid_ops),
    ("(ar) civics: full cycle adopted — 54 AI ballots, site built, civic memories", test_civics_full_cycle_adopted),
    ("(as) civics: rejection executes nothing, outcome remembered", test_civics_rejection_executes_nothing),
    ("(at) civics: live failure abstains honestly; mock cycle on rule ballots", test_civics_live_failure_paths_and_mock_rules),
    ("(au) research: fake-search insight stored (memory + knowledge + sources); offline = no_connection, nothing stored", test_research_flow_and_no_connection),
    ("(av) innovation flagship: bricks end-to-end — adopt, synthesize, register, build kiln, gather, craft, sell, save/restore", test_brick_loop_end_to_end),
    ("(aw) innovation: invalid spec -> one revision -> under study, nothing registered", test_invalid_spec_goes_under_study),
    ("(ax) persistence: canonical snapshot round-trips a mutated world; corrupt section applies nothing", test_snapshot_round_trip),
    ("(ay) persistence: repo sync — create/update pushes with sha, startup pull newer/older, offline keeps local + ticking", test_repo_sync_with_fake_transport),
    ("(az) persistence: open proposal mid-vote restores with ballots intact; only the missing re-cast", test_open_proposal_snapshot_restore),
    ("(ba) designs: seeds + valid specs register; bad interior revised once, then shelved", test_design_validation_and_shelving),
    ("(bb) voted asset: propose -> adopted -> brain fountain in layout + snapshot round-trip", test_asset_vote_cycle),
    ("(bc) weather: seeded transitions, god set_weather/set_time, rule answers, snapshot round-trip", test_weather_engine_and_rules),
    ("(bd) growth: expansion preserves core, eased ring relief, district anchors, bound round-trip", test_expansion_mechanics),
    ("(be) growth: adopted expand proposal grows land + queues two ring flatten jobs", test_civic_expansion_adopted),
    ("(bf) growth: placement follows live bound; cap expansion is a no-change notice", test_expansion_limits_and_placement),
    ("(bg) regression: no doubled /v1 in OpenAI chat/embedding paths", test_openai_path_no_doubled_v1),
    ("(bh) walk grid: dims follow bound; footprints+inflation, water, slope block", test_walkgrid_derivation),
    ("(bi) walk grid: A* around a house, corner-cut rule, LOS smoothing, spawn snap", test_walkgrid_routing),
    ("(bj) walk grid: /api/world base64, planner route on intent, version bumps", test_walkgrid_bridge_and_planner_route),
    ("(bk) location context: situation strings + prompt injection", test_location_context_and_prompt),
    ("(bl) canned lines: vendor greeting location-gated", test_canned_lines_location_gated),
    ("(bm) daily schedules + 2880s day length", test_daily_schedules_and_day_length),
    ("(bn) version copies: changed push archives, throttle + retention, failure is a note", test_versioned_snapshots),
    ("(bo) workplace pools (shopkeeper->browse) + 5m claim radius for workplace/home", test_workplace_pools_and_claim_radius),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
