"""World-data persistence — Phase 7 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, Addendum 4 ("a repo to
store all live data so it persists between loads everytime").

One canonical snapshot document carries the WHOLE living world:

    {"format": "hermes-local/world", "version": 1,
     "saved_at": <unix ts>,
     "layout": {objects, sites, buildings, anchor_counts, terrain},
     "souls":  <the full souls export — cards, memories, situations>,
     "world":  {"knowledge": [...],            (research.py)
                "innovations": {"records": [...], "under_study": [...]},
                "market": {"stock": {...}, "trade": n},
                "civics_history": [...],
                "open_proposal": {...} | None}}

Every section reuses its subsystem's own serialize/validate path
(layout documents, souls snapshots, validate_knowledge,
validate_innovations, CivicsEngine.validate_snapshot), so there is
exactly one definition of each slice's shape. Session state the
rest of the framework never persisted — resource inventory counts,
node stock, underway tasks and plans — stays session state here
too; what a snapshot restores is the world's *durable* truth: its
land, its buildings, its citizens' minds, its knowledge, its
inventions, its market and its civic record.

Two homes for the document:

  * **Local file** (`world-state.local.json`, beside the layout
    file unless configured) — written on every snapshot save and
    loaded at startup. The local world is authoritative offline.
  * **A GitHub repo** (default `ZeroZen270/hermes-world-data`,
    file `world-state.json` on `main`) via the contents API over
    urllib — no git binary. The token comes from the environment
    ($HERMES_WORLD_TOKEN, falling back to $GITHUB_TOKEN), entered
    on-device by Aaron like $NOUS_API_KEY; with NO token the sync
    is disabled and the service reports "local-only" — never an
    error. Startup pulls: a remote document newer than the local
    one is applied (and written local); a newer local one is kept
    (the next push overwrites the remote). Pushes run when the
    snapshot's content hash changes, on the configured interval
    and on a {"type": "sync_now"} protocol request, always on a
    background thread whose results the world tick drains — a
    network failure marks the status "offline", leaves local state
    untouched, and never escapes the tick.

Version copies (P12): a push that CHANGES the remote document
also leaves a dated copy at `versions/world-state-<UTC stamp>.json`
beside it — throttled to one per `version_interval_s` (default
30 min), capped at the newest `version_keep` (default 20) copies,
and strictly best-effort: version writes and retention prunes
happen after the real push lands, and any failure is a status
note, never a failed sync. A fresh run can therefore wipe the
live file knowing the repo still holds the road back.

The GitHub transport is injectable (tests drive a fake); nothing
in this module creates a repo or touches the network unless a
token (or an injected transport) says it may.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .designs import validate_designs
from .innovations import validate_innovations
from .npc import NPCManager
from .research import validate_knowledge
from .terrain import SAMPLES as _TERRAIN_SAMPLES
from .weather import validate_weather
from .world_bridge import validate_layout

FORMAT = "hermes-local/world"
VERSION = 1

DEFAULT_REPO = "ZeroZen270/hermes-world-data"
DEFAULT_BRANCH = "main"
DEFAULT_PATH = "world-state.json"
DEFAULT_INTERVAL_S = 180.0
DEFAULT_VERSION_INTERVAL_S = 1800.0
DEFAULT_VERSION_KEEP = 20
VERSIONS_DIRNAME = "versions"
# Automatic version copies only: world-state-<UTC stamp>.json,
# with a -N suffix when two pushes land in the same second.
# Retention prunes NOTHING else in the folder — a manual archive
# under any other name (e.g. an epoch-stamped fresh-run copy) is
# never deleted by the service.
_VERSION_NAME_RE = re.compile(
    r"^world-state-\d{8}T\d{6}Z(-\d+)?\.json$")
STARTUP_PULL_WAIT_S = 10.0
_API_ROOT = "https://api.github.com"
_USER_AGENT = "hermes-local/world-persistence"


class ConflictError(Exception):
    """The remote file moved under a push (sha mismatch). The
    service re-fetches once and retries the PUT once."""


# ---------------------------------------------------------------------------
# The canonical snapshot: assemble / validate / apply
# ---------------------------------------------------------------------------
def assemble_snapshot(bridge) -> dict:
    """The whole living world as one JSON-safe document. Souls are
    exported at assembly time, so citizens' memories are exactly
    as of this moment."""
    layout = dict(bridge.layout or {"version": 1, "objects": []})
    # The layout section is the designer's document plus the
    # construction and terrain slices at full fidelity (the same
    # merge the bridge's load_layout handler serves). Innovation
    # and knowledge slices live under "world" instead.
    layout.update(bridge.construction.serialize())
    if bridge.terrain.modified:
        layout.update(bridge.terrain.serialize())
    layout.pop("innovations", None)
    layout.pop("knowledge", None)
    layout.pop("designs", None)
    inno = bridge.innovations.serialize()
    civics = bridge.civics.serialize() \
        if bridge.civics is not None else {"history": [], "open": None}
    doc = {
        "format": FORMAT, "version": VERSION, "saved_at": time.time(),
        "layout": layout,
        "souls": bridge.export_souls(),
        "world": {
            "knowledge": [dict(e) for e in bridge.city_knowledge],
            "innovations": {"records": inno["records"],
                            "under_study": inno["under_study"]},
            "market": inno["market"],
            "civics_history": civics["history"],
            "open_proposal": civics["open"],
            # Living-world quality (P8): the design registry and
            # the sky ride the snapshot's world section too —
            # additive keys; a P7-era snapshot simply lacks them
            # and restores the engines' seeded state. The version
            # stays 1.
            "designs": bridge.designs.serialize(),
            "weather": bridge.weather.serialize(),
        },
    }
    # A JSON round-trip guarantees the document is serializable
    # and detaches it from every live structure it was read from.
    return json.loads(json.dumps(doc))


def validate_snapshot(bridge, doc) -> dict:
    """Validate the WHOLE document before anything is applied:
    format/version, then every section through its own subsystem
    validator. Returns the cleaned pieces apply_snapshot consumes.
    Raises ValueError — a bad document changes nothing."""
    if not isinstance(doc, dict):
        raise ValueError("world snapshot must be a JSON object")
    if doc.get("format") != FORMAT:
        raise ValueError(
            f"not a world snapshot (format={doc.get('format')!r})")
    if doc.get("version") != VERSION:
        raise ValueError(
            f"unsupported world snapshot version: {doc.get('version')!r}")
    saved_at = doc.get("saved_at")
    if not isinstance(saved_at, (int, float)):
        raise ValueError("world snapshot needs a numeric 'saved_at'")
    layout_in = doc.get("layout")
    if not isinstance(layout_in, dict):
        raise ValueError("world snapshot 'layout' must be an object")
    world_in = doc.get("world")
    if not isinstance(world_in, dict):
        raise ValueError("world snapshot 'world' must be an object")
    # Innovations + knowledge ride through the layout validator
    # when merged in (the bridge's established document shape);
    # separately they are also validated on their own so a missing
    # slice cleans to its empty form.
    inno_in = world_in.get("innovations") or {}
    if not isinstance(inno_in, dict):
        raise ValueError("world.innovations must be an object")
    inno_doc = {"records": inno_in.get("records") or [],
                "under_study": inno_in.get("under_study") or [],
                "market": world_in.get("market") or {}}
    clean_inno = validate_innovations(inno_doc)
    clean_knowledge = validate_knowledge(world_in.get("knowledge")
                                         or [])
    merged = dict(layout_in)
    merged["innovations"] = inno_doc
    merged["knowledge"] = world_in.get("knowledge") or []
    clean_layout = validate_layout(merged)
    # Souls validate through the manager's own all-or-nothing
    # checker (the cleaned entries are discarded — import_souls
    # re-validates on apply; this pass exists so a bad souls
    # section rejects the document BEFORE anything mutates).
    if bridge.manager is None:
        raise ValueError("no NPC manager configured")
    NPCManager._validate_souls(doc.get("souls"))
    clean_civics = None
    if bridge.civics is not None:
        clean_civics = bridge.civics.validate_snapshot({
            "history": world_in.get("civics_history") or [],
            "open": world_in.get("open_proposal")})
    # Designs + weather (P8): cleaned only when the snapshot
    # carries them — a P7-era snapshot leaves both engines at
    # their current (seeded) state rather than wiping them.
    clean_designs = validate_designs(world_in["designs"]) \
        if "designs" in world_in else None
    clean_weather = validate_weather(world_in["weather"]) \
        if "weather" in world_in else None
    return {"saved_at": float(saved_at), "layout": clean_layout,
            "innovations": clean_inno, "knowledge": clean_knowledge,
            "souls": doc.get("souls"), "civics": clean_civics,
            "designs": clean_designs, "weather": clean_weather}


def apply_snapshot(bridge, doc, now: float | None = None) -> dict:
    """Apply a world snapshot all-or-nothing: the whole document
    validates first (validate_snapshot), and only then does each
    section land — innovations first (dynamic building kinds must
    exist before construction records naming them can restore),
    then knowledge, layout/construction/terrain, souls (full:
    situations included), and the civic record.

    `now` is the restore moment IN THE CALLER'S TICK DOMAIN — the
    same clock the caller's manager/civics ticks are driven with.
    It only feeds the civics restore (a resumed vote's deadline is
    now + the vote window), and it must live in that one domain:
    the bridge ticks in time.monotonic(), so the default
    (now=None) is time.monotonic(); a caller ticking a synthetic
    clock (tests, simulations) passes its own now, or a restored
    vote's deadline lands in the wrong domain and the vote can
    close before it ever resumes."""
    if now is None:
        now = time.monotonic()
    clean = validate_snapshot(bridge, doc)
    summary = {"saved_at": clean["saved_at"]}
    summary["innovations"] = bridge.innovations.restore(
        clean["innovations"])
    bridge.city_knowledge = clean["knowledge"]
    summary["knowledge"] = len(clean["knowledge"])
    bridge.layout = clean["layout"]
    con_doc = clean["layout"] if ("sites" in clean["layout"]
                                  or "buildings" in clean["layout"]) \
        else {"sites": [], "buildings": []}
    summary["construction"] = bridge.construction.restore(con_doc)
    bridge._sync_building_anchors()
    if "terrain" in clean["layout"]:
        summary["terrain"] = bridge.terrain.restore(clean["layout"])
    else:
        # A snapshot from a never-shaped world restores flat land
        # (at the founding bound — the slice carries no bound).
        summary["terrain"] = bridge.terrain.restore(
            {"terrain": {"heights": [0.0] * _TERRAIN_SAMPLES,
                         "jobs": []}})
    # City growth (P9): the restored bound carries the districts —
    # rebuild the planner's district anchors to match the land.
    if hasattr(bridge, "_sync_district_anchors"):
        bridge._sync_district_anchors()
    summary["souls"] = bridge.import_souls(
        {"snapshot": clean["souls"], "mode": "full"})
    if clean["civics"] is not None:
        summary["civics"] = bridge.civics.restore(
            clean["civics"], now)
    # Designs + sky (P8): restored last — the layout is already
    # in place, so placed-asset markers re-sync into it, and the
    # clock/weather simply take over the sky.
    if clean["designs"] is not None:
        summary["designs"] = bridge.designs.restore(clean["designs"])
    if clean["weather"] is not None:
        summary["weather"] = bridge.weather.restore(clean["weather"])
    return summary


def _content_hash(doc: dict) -> str:
    """A stable fingerprint of a snapshot's CONTENT — everything
    but the saved_at stamp. Pushes happen when this changes."""
    body = {k: v for k, v in doc.items() if k != "saved_at"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def _version_name(saved_at: float) -> str:
    """The automatic version-copy filename for a snapshot: its
    saved_at as a UTC stamp (the manual fresh-run archive uses
    the same versions/ folder under its own naming)."""
    return time.strftime("world-state-%Y%m%dT%H%M%SZ.json",
                         time.gmtime(float(saved_at)))


def _version_sort_key(name: str):
    """Chronological order for automatic version names: the UTC
    stamp sorts as text; a same-second collision suffix (-2, -3…)
    is a LATER write than the bare name (suffix 1)."""
    stem = name[:-len(".json")] if name.endswith(".json") else name
    stamp, _, suffix = stem.rpartition("-")
    if stamp.endswith("Z") and suffix.isdigit():
        return (stamp, int(suffix))
    return (stem, 1)


# ---------------------------------------------------------------------------
# GitHub contents-API transport (urllib only — no git binary)
# ---------------------------------------------------------------------------
class GithubContentsTransport:
    """fetch/store one JSON file in a GitHub repo via the contents
    API. fetch -> (doc | None, sha | None): a 404 is (None, None),
    i.e. "the repo has no world file yet". store PUTs with the
    fetched sha (create when sha is None) and returns the new sha;
    a sha mismatch raises ConflictError."""

    def _request(self, method: str, url: str, token: str,
                 body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", _USER_AGENT)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code == 404 and method == "GET":
                return None
            if err.code in (409, 422) and method == "PUT":
                raise ConflictError(
                    f"GitHub refused the push (HTTP {err.code}) — "
                    f"the remote file changed") from err
            raise OSError(
                f"GitHub {method} failed: HTTP {err.code}") from err

    def _url(self, repo: str, path: str) -> str:
        return f"{_API_ROOT}/repos/{repo}/contents/{path}"

    def fetch(self, repo: str, branch: str, path: str, token: str):
        payload = self._request(
            "GET", f"{self._url(repo, path)}?ref={branch}", token)
        if payload is None:
            return None, None
        raw = base64.b64decode(
            str(payload.get("content") or "").encode()).decode("utf-8")
        doc = json.loads(raw) if raw.strip() else None
        return doc, payload.get("sha")

    def store(self, repo: str, branch: str, path: str, token: str,
              doc: dict, sha: str | None, message: str) -> str:
        body = {"message": message, "branch": branch,
                "content": base64.b64encode(
                    json.dumps(doc, indent=1).encode()).decode()}
        if sha:
            body["sha"] = sha
        payload = self._request("PUT", self._url(repo, path), token,
                                body)
        return (payload.get("content") or {}).get("sha") or ""

    def list_dir(self, repo: str, branch: str, dir_path: str,
                 token: str) -> list[dict]:
        """The files directly inside a repo folder: [{name, path,
        sha}]. A missing folder (404) is an empty list; a path
        that names a file is an empty list too."""
        payload = self._request(
            "GET", f"{self._url(repo, dir_path)}?ref={branch}",
            token)
        if not isinstance(payload, list):
            return []
        return [{"name": e.get("name"), "path": e.get("path"),
                 "sha": e.get("sha")}
                for e in payload
                if isinstance(e, dict) and e.get("type") == "file"]

    def delete_file(self, repo: str, branch: str, path: str,
                    token: str, sha: str, message: str) -> None:
        """Delete one repo file by its current sha (retention
        pruning of old version copies)."""
        self._request("DELETE", self._url(repo, path), token,
                      {"message": message, "branch": branch,
                       "sha": sha})


# ---------------------------------------------------------------------------
# The service: local file + repo sync on the world tick
# ---------------------------------------------------------------------------
class PersistenceService:
    """Owns snapshot saving for one bridge.

    Wiring: build_world constructs the service from the config's
    "persistence" section, attaches it as bridge.persistence (and
    planner.persistence, so the planner's tick tail drives it), and
    calls startup(). Tests may construct it directly with an
    injected transport and a temp local path.

    Status (public_status(), riding /api/status):
    {"state": "synced" | "offline" | "local-only" | "disabled",
     "repo": "owner/name" | None, "last_push": ts | None,
     "last_pull": ts | None, "last_error": str | None}
    """

    def __init__(self, bridge, cfg: dict | None = None,
                 transport=None, layout_path: str | None = None):
        self.bridge = bridge
        cfg = dict(cfg or {})
        self.repo = str(cfg.get("repo") or DEFAULT_REPO)
        self.branch = str(cfg.get("branch") or DEFAULT_BRANCH)
        self.path = str(cfg.get("path") or DEFAULT_PATH)
        try:
            self.interval_s = max(5.0, float(
                cfg.get("interval_s") or DEFAULT_INTERVAL_S))
        except (TypeError, ValueError):
            self.interval_s = DEFAULT_INTERVAL_S
        # Version copies (P12): on by default; the interval is a
        # wall-clock throttle between automatic copies, the keep
        # count a retention cap on the versions/ folder.
        self.keep_versions = cfg.get("keep_versions") is not False
        try:
            self.version_interval_s = max(0.0, float(
                cfg.get("version_interval_s")
                if cfg.get("version_interval_s") is not None
                else DEFAULT_VERSION_INTERVAL_S))
        except (TypeError, ValueError):
            self.version_interval_s = DEFAULT_VERSION_INTERVAL_S
        try:
            self.version_keep = max(1, int(
                cfg.get("version_keep") or DEFAULT_VERSION_KEEP))
        except (TypeError, ValueError):
            self.version_keep = DEFAULT_VERSION_KEEP
        local = cfg.get("local_path") or ""
        if local:
            self.local_path = Path(str(local)).expanduser()
        elif layout_path:
            self.local_path = Path(layout_path).expanduser().parent \
                / "world-state.local.json"
        else:
            self.local_path = Path(
                "~/.hermes_local/world-state.local.json").expanduser()
        self.token = os.environ.get("HERMES_WORLD_TOKEN") \
            or os.environ.get("GITHUB_TOKEN") or ""
        # An injected transport (tests) stands in for the token:
        # sync is enabled when either is present.
        self.transport = transport
        self.sync_enabled = bool(self.repo) and (
            transport is not None or bool(self.token))
        if cfg.get("enabled") is False:
            self.sync_enabled = False
            self._state = "disabled"
        elif self.sync_enabled:
            # Nothing has synced yet in this process; startup()
            # (or the first landed push) moves this to "synced".
            self._state = "offline"
        else:
            self._state = "local-only"
        self.last_push: float | None = None
        self.last_pull: float | None = None
        self.last_error: str | None = None
        # Version copies (P12): the last automatic copy this
        # session wrote (wall ts + the content hash it captured)
        # and the last version-side failure, if any — a note, kept
        # apart from last_error so a version hiccup never reads
        # as a sync failure.
        self.last_version: float | None = None
        self.last_version_error: str | None = None
        self._last_version_at: float | None = None
        self._last_version_hash: str | None = None
        self._local_hash: str | None = None
        self._remote_hash: str | None = None
        self._local_saved_at = 0.0
        self._results: queue.Queue = queue.Queue()
        self._inflight = False
        self._force = False
        self._next_due: float | None = None

    # -- status ------------------------------------------------------------
    def public_status(self) -> dict:
        return {"state": self._state,
                "repo": self.repo or None,
                "last_push": self.last_push,
                "last_pull": self.last_pull,
                "last_error": self.last_error,
                "keep_versions": self.keep_versions,
                "last_version": self.last_version,
                "version_error": self.last_version_error}

    def _event(self, action: str, **extra) -> dict:
        ev = {"type": "persistence", "action": action,
              "state": self._state, "repo": self.repo,
              "last_push": self.last_push, "last_pull": self.last_pull}
        ev.update(extra)
        return ev

    # -- local file ---------------------------------------------------------
    def _read_local(self) -> dict | None:
        try:
            if not self.local_path.is_file():
                return None
            doc = json.loads(self.local_path.read_text())
        except Exception:
            return None
        return doc if isinstance(doc, dict) \
            and doc.get("format") == FORMAT else None

    def _write_local(self, doc: dict) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(json.dumps(doc, indent=1))

    # -- transport plumbing ---------------------------------------------------
    def _transport(self):
        if self.transport is None:
            self.transport = GithubContentsTransport()
        return self.transport

    def _push_doc(self, doc: dict) -> str:
        """One push: fetch the current sha, PUT with it; on a sha
        conflict, re-fetch once and retry once. Returns the new
        sha. Raises whatever the transport raises otherwise."""
        transport = self._transport()
        _cur, sha = transport.fetch(self.repo, self.branch,
                                    self.path, self.token)
        try:
            return transport.store(
                self.repo, self.branch, self.path, self.token, doc,
                sha, f"world snapshot {int(doc['saved_at'])}")
        except ConflictError:
            _cur, sha = transport.fetch(self.repo, self.branch,
                                        self.path, self.token)
            return transport.store(
                self.repo, self.branch, self.path, self.token, doc,
                sha, f"world snapshot {int(doc['saved_at'])}")

    # -- version copies (P12) -------------------------------------------------
    def _versions_dir(self) -> str:
        """The versions folder beside the live world file."""
        if "/" in self.path:
            return self.path.rsplit("/", 1)[0] + "/" \
                + VERSIONS_DIRNAME
        return VERSIONS_DIRNAME

    def _maybe_write_version(self, doc: dict, digest: str,
                             wall: float) -> str | None:
        """Leave a dated copy of a just-pushed CHANGED snapshot in
        versions/. Returns the copy's filename, or None when the
        throttle / duplicate rules skip it. Raises whatever the
        transport raises — the caller treats any failure as a
        note, never as a failed push.

        Skips: this exact content already versioned this session;
        the last copy is younger than version_interval_s; or the
        repo already holds an identical copy under the stamped
        name (adopted as this session's copy instead of written
        twice)."""
        if digest == self._last_version_hash:
            return None
        if self._last_version_at is not None \
                and wall - self._last_version_at \
                < self.version_interval_s:
            return None
        transport = self._transport()
        vdir = self._versions_dir()
        base = _version_name(doc["saved_at"])
        stem = base[:-len(".json")]
        name, n = base, 1
        while True:
            existing, _sha = transport.fetch(
                self.repo, self.branch, f"{vdir}/{name}",
                self.token)
            if existing is None:
                break
            if isinstance(existing, dict) \
                    and _content_hash(existing) == digest:
                self._last_version_at = wall
                self._last_version_hash = digest
                self.last_version = wall
                return None
            n += 1
            name = f"{stem}-{n}.json"
        transport.store(self.repo, self.branch, f"{vdir}/{name}",
                        self.token, doc, None,
                        f"world snapshot version {name}")
        self._last_version_at = wall
        self._last_version_hash = digest
        self.last_version = wall
        self._trim_versions(transport, vdir)
        return name

    def _trim_versions(self, transport, vdir: str) -> None:
        """Retention: keep the newest version_keep automatic
        copies, delete the older ones (oldest first). Only names
        matching the automatic pattern are ever touched, and
        every failure is swallowed — pruning must never fail a
        push or a version write."""
        try:
            entries = transport.list_dir(self.repo, self.branch,
                                         vdir, self.token)
        except Exception:
            return
        names = sorted(
            (e["name"] for e in entries
             if e.get("name") and _VERSION_NAME_RE.match(e["name"])),
            key=_version_sort_key)
        for name in names[:max(0, len(names) - self.version_keep)]:
            try:
                _doc, sha = transport.fetch(
                    self.repo, self.branch, f"{vdir}/{name}",
                    self.token)
                if sha:
                    transport.delete_file(
                        self.repo, self.branch, f"{vdir}/{name}",
                        self.token, sha,
                        f"prune old world snapshot version {name}")
            except Exception:
                pass

    # -- startup ----------------------------------------------------------------
    def startup(self) -> list[dict]:
        """Load the local snapshot (when present), then pull the
        repo copy: a NEWER remote replaces the world (and the
        local file); a newer local is kept — the next push
        overwrites the remote. The pull waits at most
        STARTUP_PULL_WAIT_S for the network, then proceeds with
        whatever the local load produced. Never raises."""
        events: list[dict] = []
        local_doc = self._read_local()
        if local_doc is not None:
            try:
                # The bridge ticks in time.monotonic(); restores
                # here name that domain explicitly (apply_snapshot's
                # own default is the same clock).
                apply_snapshot(self.bridge, local_doc,
                               now=time.monotonic())
                self._local_saved_at = float(local_doc["saved_at"])
                self._local_hash = _content_hash(local_doc)
                events.append(self._event("load_local"))
            except Exception as err:   # a bad local file never bricks boot
                self.last_error = f"local snapshot unreadable: {err}"
        if not self.sync_enabled:
            return events
        box: dict = {}

        def pull():
            try:
                box["result"] = self._transport().fetch(
                    self.repo, self.branch, self.path, self.token)
            except Exception as err:
                box["error"] = err

        thread = threading.Thread(target=pull, daemon=True,
                                  name="hermes-persistence-pull")
        thread.start()
        thread.join(STARTUP_PULL_WAIT_S)
        if thread.is_alive():
            self._state = "offline"
            self.last_error = "startup pull timed out"
            return events
        if "error" in box:
            self._state = "offline"
            self.last_error = str(box["error"])
            return events
        remote_doc, _sha = box.get("result") or (None, None)
        self.last_pull = time.time()
        if isinstance(remote_doc, dict) \
                and remote_doc.get("format") == FORMAT \
                and isinstance(remote_doc.get("saved_at"),
                               (int, float)):
            # Whatever the remote holds is now the known remote
            # state: if it is older than the local world, the hash
            # mismatch makes the next interval push the local world
            # over it (the spec's startup rule).
            self._remote_hash = _content_hash(remote_doc)
            if float(remote_doc["saved_at"]) > self._local_saved_at:
                try:
                    apply_snapshot(self.bridge, remote_doc,
                                   now=time.monotonic())
                except Exception as err:
                    self._state = "offline"
                    self.last_error = \
                        f"remote snapshot unreadable: {err}"
                    return events
                self._local_saved_at = float(remote_doc["saved_at"])
                self._local_hash = _content_hash(remote_doc)
                try:
                    self._write_local(remote_doc)
                except Exception:
                    pass
                events.append(self._event("pull"))
        self._state = "synced"
        self.last_error = None
        return events

    # -- the tick ------------------------------------------------------------------
    def request_sync(self) -> dict:
        """Protocol {"type": "sync_now"}: force a snapshot + push
        on the next tick (assembly happens there, off this call's
        critical path beyond a flag). Returns the status event to
        answer with right away."""
        self._force = True
        return self._event("sync_requested")

    def tick(self, now: float) -> list[dict]:
        """Advance persistence one world tick: land finished
        background pushes, and on the interval cadence (or when
        forced) assemble a snapshot — writing the local file and
        pushing whenever the content actually changed. Network
        failures surface as status, never as exceptions."""
        events: list[dict] = []
        events.extend(self._drain_results())
        if self._next_due is None:
            self._next_due = now + self.interval_s
        if not self._force and now < self._next_due:
            return events
        self._next_due = now + self.interval_s
        forced, self._force = self._force, False
        try:
            doc = assemble_snapshot(self.bridge)
        except Exception as err:
            self.last_error = f"snapshot assembly failed: {err}"
            return events
        digest = _content_hash(doc)
        if digest != self._local_hash or forced:
            try:
                self._write_local(doc)
                self._local_hash = digest
                self._local_saved_at = float(doc["saved_at"])
                events.append(self._event("save_local"))
            except Exception as err:
                self.last_error = f"local save failed: {err}"
        if self.sync_enabled and not self._inflight \
                and (digest != self._remote_hash or forced):
            self._inflight = True

            def push(doc=doc, digest=digest):
                try:
                    # A version copy is owed only when this push
                    # actually CHANGES the remote document — a
                    # forced re-push of identical content is not
                    # a new state worth archiving.
                    changed = digest != self._remote_hash
                    self._push_doc(doc)
                    res = {"ok": True, "hash": digest, "doc": doc}
                    if changed and self.keep_versions:
                        try:
                            name = self._maybe_write_version(
                                doc, digest, time.time())
                            if name:
                                res["version"] = name
                        except Exception as err:
                            res["version_error"] = str(err)
                    self._results.put(res)
                except Exception as err:
                    self._results.put({"ok": False, "error": str(err)})

            threading.Thread(target=push, daemon=True,
                             name="hermes-persistence-push").start()
        return events

    def _drain_results(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            self._inflight = False
            if res.get("ok"):
                self._remote_hash = res["hash"]
                self._state = "synced"
                self.last_push = time.time()
                self.last_error = None
                events.append(self._event("push"))
                if res.get("version"):
                    self.last_version_error = None
                    events.append(self._event(
                        "version", version=res["version"]))
                if res.get("version_error"):
                    # Best-effort archiving failed: the push above
                    # still stands — this is a status note only.
                    self.last_version_error = res["version_error"]
                    events.append(self._event(
                        "version_failed",
                        error=res["version_error"]))
            else:
                self._state = "offline"
                self.last_error = res.get("error") or "push failed"
                events.append(self._event("push_failed",
                                          error=self.last_error))
        return events
