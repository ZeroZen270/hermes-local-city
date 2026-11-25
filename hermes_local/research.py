"""The research computer + world planner — Phase 6 of the massive
world upgrade. Design doc: files/world-upgrade-design.md,
Addendum 2.

A physical **research terminal** stands by Central Plaza (the
"Research Terminal" planner anchor in npc.py). Citizens travel to
it as a planner activity — curiosity-driven, and keener when the
city is short of something — and the player can put a question to
it directly ({"type": "research_ask", ...}). Using it invokes the
**world planner**: a Hermes agent with TRUE web search.

  * The query is framed from the asker's question plus live world
    state (stores, shortages against the foreman's targets, what
    the city has already learned and built).
  * The search is keyless — DuckDuckGo's HTML endpoint through
    urllib with a plain user agent, with Bing's HTML results as
    fallback (DDG began answering plain clients with a bot-check
    page in Nov 2026) — and INJECTABLE: `fetcher`
    (url -> html) is a seam tests drive with canned pages, and
    build_world leaves the real one in place. Result titles,
    snippets and URLs are parsed out of the results page.
  * The brain (a dedicated ~45s client, wired by build_world)
    synthesizes the findings into an **insight**: {summary,
    innovation_hint | None} — the hint naming a concrete thing the
    city could adopt, which is what civics turns into an
    adopt_innovation proposal (innovations.py makes it real).

Honesty rules, enforced here: if the search cannot run (offline,
blocked, garbage page) the session ends with a
{"type": "research", "state": "no_connection"} event and NOTHING
is stored — no insight, no memory, no city knowledge. A world with
no brain wired (mock/demo) cannot research at all and says so the
same way; the standalone web client shows the terminal dark rather
than faking findings. If the search works but the brain fails, the
insight falls back to the findings themselves (titles + snippets,
attributed) with no innovation hint — real sources, no invention.

Insights live two places: the asker's own episodic memory
(kind="research" — soul-fed prompts and civics' seed engine read
them) and the bridge's shared **city_knowledge** list, which rides
the world save.
"""
from __future__ import annotations

import html as html_mod
import queue
import random
import re
import threading
import urllib.parse
import urllib.request

from . import work as work_mod

DDG_HTML_URL = "https://html.duckduckgo.com/html/?q="
BING_URL = "https://www.bing.com/search?q="
_USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
FETCH_TIMEOUT_S = 12.0
RESULT_LIMIT = 5

TERMINAL_ANCHOR = "Research Terminal"
TERMINAL_POS = (18.0, 10.0)      # a few steps from Central Plaza

SESSION_GAP_S = 20.0             # min sim-s between citizen sessions
CITIZEN_COOLDOWN_S = 600.0       # per citizen, after their session
KNOWLEDGE_KEEP = 40              # city knowledge entries retained

# What citizens ask when the city gives them no sharper worry —
# role-flavoured curiosity, rotated. The worked example from the
# design doc (stronger-than-wood) is the housing worry below.
_ROLE_CURIOSITY = {
    "mechanic": "What simple machines make hauling heavy loads easier?",
    "engineer": "What materials and methods make buildings last longer?",
    "vendor": "What foods keep longest without spoiling, and how are they sold?",
    "shopkeeper": "How do market towns display and sell goods better?",
    "medic": "What simple remedies and herbs help common illnesses?",
    "student": "What new ideas are people excited about in the wider world?",
    "artist": "What pigments and materials do painters use for lasting color?",
    "elder": "What did builders in the old days use that we have forgotten?",
    "courier": "What makes a good road, and how are they built to last?",
    "guard": "How do towns keep watch and light their streets at night?",
}
_GENERIC_CURIOSITY = [
    "What is stronger than wood for building a house?",
    "How can food be preserved so it keeps for months?",
    "What tools make digging and farming easier?",
    "How do people in other towns make cloth, glass and pottery?",
]


def default_fetcher(url: str) -> str:
    """The real fetch: one GET with a browser user agent. Raises on
    any failure — the caller turns that into no_connection."""
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _clean_text(text: str) -> str:
    return " ".join(html_mod.unescape(
        re.sub(r"<[^>]+>", " ", text or "")).split())


def _real_url(href: str) -> str:
    """DuckDuckGo wraps result links in a redirect (//duckduckgo
    .com/l/?uddg=<encoded>); unwrap to the real destination."""
    href = html_mod.unescape(href or "")
    if href.startswith("//"):
        href = "https:" + href
    try:
        parsed = urllib.parse.urlparse(href)
        if "duckduckgo.com" in parsed.netloc:
            target = urllib.parse.parse_qs(
                parsed.query).get("uddg", [None])[0]
            if target:
                return target
    except Exception:
        pass
    return href


def parse_ddg_html(page: str, limit: int = RESULT_LIMIT) -> list[dict]:
    """Result {title, url, snippet} dicts out of a DuckDuckGo HTML
    results page. Tolerant of the page's exact markup: anchors with
    the result__a class pair with the next result__snippet."""
    results: list[dict] = []
    anchors = list(re.finditer(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>'
        r'(.*?)</a>', page or "", re.DOTALL))
    snippets = [m.group(1) for m in re.finditer(
        r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', page or "",
        re.DOTALL)]
    for i, m in enumerate(anchors):
        title = _clean_text(m.group(2))
        url = _real_url(m.group(1))
        if not title or not url.startswith("http"):
            continue
        results.append({
            "title": title, "url": url,
            "snippet": _clean_text(snippets[i]) if i < len(snippets)
            else ""})
        if len(results) >= limit:
            break
    return results


def _bing_real_url(href: str) -> str:
    """Bing wraps result links in a /ck/a redirect whose `u`
    parameter is 'a1' + base64 of the real destination; unwrap
    it, like _real_url does for DuckDuckGo."""
    href = html_mod.unescape(href or "")
    try:
        parsed = urllib.parse.urlparse(href)
        if "bing.com" in parsed.netloc and "/ck/a" in parsed.path:
            import base64
            u = urllib.parse.parse_qs(parsed.query).get("u", [None])[0]
            if u and u.startswith("a1"):
                real = base64.urlsafe_b64decode(
                    u[2:] + "=" * (-len(u[2:]) % 4)).decode(
                    "utf-8", errors="replace")
                if real.startswith("http"):
                    return real
    except Exception:
        pass
    return href


def parse_bing_html(page: str, limit: int = RESULT_LIMIT) -> list[dict]:
    """Result {title, url, snippet} dicts out of a Bing results
    page: one <li class="b_algo"> block per result, its <h2>
    anchor the title link, its first <p> the snippet."""
    results: list[dict] = []
    for block in (page or "").split('<li class="b_algo"')[1:]:
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>'
                      r'(.*?)</a>', block, re.DOTALL)
        if not m:
            continue
        title = _clean_text(m.group(2))
        url = _bing_real_url(m.group(1))
        if not title or not url.startswith("http"):
            continue
        p = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL)
        results.append({"title": title, "url": url,
                        "snippet": _clean_text(p.group(1)) if p
                        else ""})
        if len(results) >= limit:
            break
    return results


def web_search(query: str, fetcher=None, limit: int = RESULT_LIMIT):
    """One keyless web search. `fetcher(url) -> html` is the
    injection seam. Two keyless sources, in order: DuckDuckGo's
    HTML endpoint, then Bing — since 2026-11-24 DDG serves a
    bot-check page (HTTP 200, zero results) to plain urllib
    clients, which silently emptied every terminal search; the
    fallback keeps the terminal honest AND working: a source
    that yields nothing falls through to the next, and if every
    source fails the first error raises. No results are ever
    invented here."""
    fetch = fetcher or default_fetcher
    q = urllib.parse.quote_plus(str(query))
    first_err: Exception | None = None
    for base, parser in ((DDG_HTML_URL, parse_ddg_html),
                         (BING_URL, parse_bing_html)):
        try:
            results = parser(fetch(base + q), limit=limit)
        except Exception as err:
            if first_err is None:
                first_err = err
            continue
        if results:
            return results
    if first_err is not None:
        raise first_err
    return []


def validate_knowledge(doc) -> list:
    """Clean the city-knowledge slice of a world save: a list of
    insight/innovation entries with their text fields clamped.
    All-or-nothing, like the other world slices."""
    if not isinstance(doc, list):
        raise ValueError("knowledge must be a list")
    out = []
    for i, entry in enumerate(doc):
        if not isinstance(entry, dict):
            raise ValueError(f"knowledge[{i}] must be an object")
        out.append({
            "id": str(entry["id"]) if entry.get("id") else None,
            "kind": str(entry.get("kind") or "insight"),
            "citizen": str(entry.get("citizen") or ""),
            "question": str(entry.get("question") or "")[:200],
            "summary": str(entry.get("summary") or "")[:600],
            "text": str(entry.get("text") or "")[:600],
            "sources": [str(s)[:300] for s in
                        (entry.get("sources") or [])][:6],
            "innovation_hint": (str(entry["innovation_hint"])[:120]
                                if entry.get("innovation_hint")
                                else None),
            "innovation": entry.get("innovation"),
            "ts": entry.get("ts") if isinstance(
                entry.get("ts"), (int, float)) else 0.0})
    return out[-KNOWLEDGE_KEEP:]


class ResearchEngine:
    """Terminal sessions, the world planner, city knowledge.

    Wiring: WorldBridge constructs the engine with itself and the
    planner reaches it as `planner.research` (citizen sessions are
    noticed from planner state inside tick()). `brain` —
    callable(citizen, question, world_ctx, results) ->
    {"summary": str, "innovation_hint": str | None} — and
    `fetcher` are injected: build_world wires the live brain and
    the real fetcher; tests wire fakes. With brain None the
    terminal is dark: asks answer no_connection and citizens never
    session."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.brain = None
        self.fetcher = None
        self._results: queue.Queue = queue.Queue()
        self._inflight = False
        self._last_at = -1e9
        self._citizen_last: dict[str, float] = {}
        self._seq = 0
        self._rng = random.Random(7)

    # -- world framing -----------------------------------------------------
    def world_ctx(self) -> dict:
        """The live state a query/insight is framed against: the
        stores, the shortages (against the foreman's targets), the
        open civic question, and what the city already makes."""
        bridge = self.bridge
        inv = dict(bridge.resources.inventory)
        planner = None
        if bridge.manager is not None:
            planner = bridge.manager.planner
        targets = planner.effective_targets() if planner is not None \
            else {"wood": work_mod.TARGET_WOOD,
                  "food": work_mod.TARGET_FOOD}
        shortages = []
        if inv.get("wood", 0) < targets.get("wood", 0):
            shortages.append(
                f"wood ({inv.get('wood', 0)} of {targets['wood']})")
        food_have = inv.get("food", 0)
        innovations = getattr(bridge, "innovations", None)
        if innovations is not None:
            food_have += innovations.food_stock()
        if food_have < targets.get("food", 0):
            shortages.append(f"food ({food_have} of {targets['food']})")
        ctx = {"inventory": inv, "shortages": shortages,
               "houses": 0, "innovations": [], "civic_open": None}
        if bridge.construction is not None:
            ctx["houses"] = sum(
                1 for b in bridge.construction.buildings
                if b.get("kind") == "house")
        if innovations is not None:
            ctx["innovations"] = [r["name"] for r in
                                  innovations.records.values()]
        if bridge.civics is not None \
                and bridge.civics.current is not None:
            ctx["civic_open"] = bridge.civics.current["title"]
        return ctx

    def frame_query(self, question: str, ctx: dict) -> str:
        """The question as searched: itself, plus the world's
        sharpest context when the question doesn't already carry
        it (a shortage makes 'how do we fix this' concrete)."""
        query = " ".join(str(question).split())
        if ctx.get("shortages") and not any(
                w in query.lower() for w in
                ("food", "wood", "hungry", "store", "preserve")):
            query += f" (a small town is short of " \
                     f"{ctx['shortages'][0].split(' (')[0]})"
        return query

    # -- asking --------------------------------------------------------------
    def ask(self, question: str, by: str = "The Keeper",
            npc_name: str | None = None, now: float = 0.0) -> list[dict]:
        """Put a question to the terminal. Answers with a
        "searching" event (the insight lands through tick()) or —
        when the terminal cannot work at all — the honest
        no_connection event, immediately."""
        question = " ".join(str(question or "").split())[:240]
        if not question:
            return [{"type": "notice",
                     "text": "Ask the research terminal a question "
                             "first."}]
        if self.brain is None:
            return [{"type": "research", "state": "no_connection",
                     "question": question, "by": by,
                     "reason": "the terminal is dark — no live "
                               "world-planner is wired to it"}]
        if self._inflight:
            return [{"type": "notice",
                     "text": "The research terminal is busy with "
                             "another question — one at a time."}]
        self._inflight = True
        self._last_at = now
        citizen = npc_name if npc_name in self.bridge.npcs else None
        asker = citizen or by
        ctx = self.world_ctx()
        query = self.frame_query(question, ctx)
        brain, fetcher = self.brain, self.fetcher

        def work():
            try:
                results = web_search(query, fetcher=fetcher)
                if not results:
                    raise ValueError("the search returned nothing")
            except Exception as err:
                self._results.put({"ok": False, "question": question,
                                   "by": asker, "citizen": citizen,
                                   "reason": f"{type(err).__name__}: "
                                             f"{err}"})
                return
            insight = None
            try:
                insight = brain(asker, question, ctx, results)
            except Exception:
                insight = None
            if not insight or not str(
                    insight.get("summary") or "").strip():
                # The search worked; the synthesis didn't. Fall
                # back to the findings themselves — real sources,
                # no invented summary, no innovation hint.
                top = results[0]
                insight = {
                    "summary": (f"Search findings on '{question}': "
                                f"{top['title']} — {top['snippet']}"
                                )[:400],
                    "innovation_hint": None}
            self._results.put({
                "ok": True, "question": question, "by": asker,
                "citizen": citizen,
                "summary": str(insight["summary"]).strip()[:400],
                "innovation_hint": (str(insight["innovation_hint"])
                                    [:120] if insight.get(
                                        "innovation_hint") else None),
                "sources": [r["url"] for r in results]})

        threading.Thread(target=work, daemon=True,
                         name="hermes-research").start()
        return [{"type": "research", "state": "searching",
                 "question": question, "by": asker}]

    # -- citizen sessions ------------------------------------------------------
    def _citizen_question(self, npc, ctx: dict) -> str:
        """What this citizen asks the terminal: the city's sharpest
        shortage first, then housing, then their own curiosity."""
        shortages = " ".join(ctx.get("shortages") or [])
        if "food" in shortages:
            return ("How can a small town preserve food so it "
                    "keeps for months?")
        if "wood" in shortages:
            return ("What is stronger and longer-lasting than "
                    "wood for building?")
        if ctx.get("houses", 0) < 2:
            return _GENERIC_CURIOSITY[0]
        role_q = _ROLE_CURIOSITY.get(npc.card.role)
        pool = ([role_q] if role_q else []) + _GENERIC_CURIOSITY
        return self._rng.choice(pool)

    def _maybe_start_session(self, now: float, planner) -> None:
        """A citizen dwelling at the terminal, off cooldown, with
        the terminal free, sits down and asks something."""
        if self.brain is None or self._inflight:
            return
        if now - self._last_at < SESSION_GAP_S:
            return
        for name, st in planner.state.items():
            anchor = st.get("anchor")
            if st.get("state") != "dwell" or anchor is None \
                    or anchor.name != TERMINAL_ANCHOR \
                    or st.get("kind") != "research":
                continue
            if now - self._citizen_last.get(name, -1e9) \
                    < CITIZEN_COOLDOWN_S:
                continue
            npc = self.bridge.npcs.get(name)
            if npc is None:
                continue
            self._citizen_last[name] = now
            question = self._citizen_question(npc, self.world_ctx())
            self.ask(question, by=name, npc_name=name, now=now)
            return

    # -- tick / landing -----------------------------------------------------------
    def tick(self, now: float, positions: dict) -> list[dict]:
        """Advance research one world tick: land finished sessions
        (insight -> memory + city knowledge + event; failure ->
        no_connection and NOTHING stored), then notice a citizen
        settling at the terminal."""
        events: list[dict] = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            self._inflight = False
            if not res["ok"]:
                events.append({
                    "type": "research", "state": "no_connection",
                    "question": res["question"], "by": res["by"],
                    "reason": res["reason"]})
                continue
            self._seq += 1
            entry = {"id": f"know-{self._seq}", "kind": "insight",
                     "citizen": res["by"], "question": res["question"],
                     "summary": res["summary"],
                     "sources": res["sources"],
                     "innovation_hint": res["innovation_hint"],
                     "innovation": None, "ts": now}
            knowledge = self.bridge.city_knowledge
            knowledge.append(entry)
            del knowledge[:-KNOWLEDGE_KEEP]
            citizen = res.get("citizen")
            if citizen and citizen in self.bridge.npcs:
                self.bridge.npcs[citizen].witness(
                    f"Asked the research terminal "
                    f"\"{res['question']}\" and learned: "
                    f"{res['summary']}", kind="research")
            events.append({
                "type": "research", "state": "insight",
                "question": res["question"], "by": res["by"],
                "summary": res["summary"],
                "innovation_hint": res["innovation_hint"],
                "sources": res["sources"]})
        planner = None
        if self.bridge.manager is not None:
            planner = self.bridge.manager.planner
        if planner is not None:
            self._maybe_start_session(now, planner)
        return events

    # -- read models ------------------------------------------------------------------
    def public(self) -> dict:
        """The Research panel view: the terminal, the latest
        insight, and the city's shared knowledge (newest last)."""
        knowledge = self.bridge.city_knowledge
        insights = [e for e in knowledge if e.get("kind") == "insight"]
        return {"terminal": {"anchor": TERMINAL_ANCHOR,
                             "position": list(TERMINAL_POS),
                             "live": self.brain is not None},
                "knowledge": knowledge[-8:],
                "latest": insights[-1] if insights else None}
