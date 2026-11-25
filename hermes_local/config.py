"""Configuration: sane offline-first defaults + optional YAML overrides.

The framework runs with zero config. If PyYAML is installed and a config
file exists (./config.yaml, ~/.hermes_local/config.yaml, or the path in
$HERMES_LOCAL_CONFIG), it is deep-merged over these defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULTS: dict = {
    "llm": {
        "base_url": "http://localhost:11434",
        "api": "ollama",            # "ollama" (native /api/chat) or "openai" (/v1/...)
        "model": "auto",            # "auto" = pick from the ladder by detected resources
        "timeout": 120,
        "retries": 3,
        "temperature": 0.7,
        "api_key": "",              # only needed for OpenAI-compatible hosted endpoints
        "provider": "",             # named preset, e.g. "nous-longcat" (see PROVIDERS)
        "api_key_env": "",          # env var holding the key, e.g. "NOUS_API_KEY"
    },
    "embeddings": {
        "base_url": "http://localhost:11434",
        "api": "ollama",
        # Fallback chain, tried in order; the last resort is a built-in
        # zero-dependency hashing embedder (see embeddings.py).
        "models": ["embeddinggemma-2", "embeddinggemma", "nomic-embed-text"],
        "timeout": 10,
        "search_dim": 256,          # Matryoshka-style truncation used at search time
        "local_dim": 512,           # dimension of the built-in fallback embedder
    },
    "memory": {
        "db_path": "~/.hermes_local/memory.db",
        "notes_path": "~/.hermes_local/notes.md",
        "recall_k": 4,
    },
    "agent": {
        "tool_mode": "auto",        # auto | native | react
        "max_iterations": 12,
        "context_tokens": 16000,    # works at small windows; compression handles the rest
        "keep_recent_turns": 6,
        "memory_recall": True,      # inject relevant long-term memories into each task
        # NPC proximity activation (web world / 3D worlds):
        "active_radius": 15,        # meters: full agent loop inside this radius
        "ambient_radius": 50,       # meters: scripted-only inside this radius
        "max_active": 4,            # director cap on simultaneously ACTIVE NPCs
        # Tier hysteresis + conversation lock (anti-flicker): an ACTIVE NPC
        # only drops out beyond the exit radius, an AMBIENT one beyond its
        # exit radius, demotions need a 5s minimum hold, and an NPC the
        # player is talking to is pinned ACTIVE until 20s of silence or
        # the player walks more than 25m away.
        "active_exit_radius": 20,
        "ambient_exit_radius": 60,
        "tier_min_hold_s": 5,
        "engage_timeout_s": 20,
        "engage_range_m": 25,
    },
    "tools": {
        "workspace": ".",
        "docs_dir": "docs",
        "shell_timeout": 30,
    },
    # World-data persistence (P7, see persistence.py): the whole
    # living world auto-saves to a local file and syncs to a
    # dedicated GitHub repo via the contents API. The repo token
    # comes from $HERMES_WORLD_TOKEN (fallback $GITHUB_TOKEN) in
    # the environment — never from this file; with no token the
    # world is local-only, which is a state, not an error.
    "persistence": {
        "repo": "ZeroZen270/hermes-world-data",
        "branch": "main",
        "path": "world-state.json",
        "interval_s": 180,
        "local_path": "",           # default: beside the layout file
        # Version copies (P12): each changed push also archives a
        # dated copy under versions/ in the repo — at most one per
        # version_interval_s, keeping the newest version_keep.
        "keep_versions": True,
        "version_interval_s": 1800,
        "version_keep": 20,
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Named provider presets. Local Ollama stays the default; a preset is an
# opt-in hosted endpoint. Preset values fill only the llm fields the user's
# config file did NOT set explicitly, so a hand-written base_url/model in
# the same file always wins over the preset.
# ---------------------------------------------------------------------------
PROVIDERS: dict = {
    # Nous Portal's free tier: Meituan LongCat 2.5 (1M context, agentic
    # tool use), OpenAI-compatible. Needs a Nous Portal API key in
    # $NOUS_API_KEY. Free availability/quotas are Nous's to change.
    "nous-longcat": {
        "base_url": "https://inference-api.nousresearch.com/v1",
        "api": "openai",
        "model": "meituan/longcat-2.5:free",
        "api_key_env": "NOUS_API_KEY",
    },
    # Any Nous Portal model: same endpoint/key, set llm.model yourself.
    "nous": {
        "base_url": "https://inference-api.nousresearch.com/v1",
        "api": "openai",
        "model": "meituan/longcat-2.5:free",
        "api_key_env": "NOUS_API_KEY",
    },
}


def load_config(path: str | None = None) -> dict:
    cfg = _deep_merge({}, DEFAULTS)
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get("HERMES_LOCAL_CONFIG"):
        candidates.append(Path(os.environ["HERMES_LOCAL_CONFIG"]))
    candidates.append(Path("config.yaml"))
    candidates.append(Path("~/.hermes_local/config.yaml").expanduser())
    raw_llm: dict = {}
    for cand in candidates:
        if cand.is_file():
            try:
                import yaml  # optional dependency

                with open(cand) as fh:
                    loaded = yaml.safe_load(fh) or {}
                if isinstance(loaded, dict):
                    if isinstance(loaded.get("llm"), dict):
                        raw_llm = loaded["llm"]
                    cfg = _deep_merge(cfg, loaded)
            except ImportError:
                pass  # no PyYAML: defaults still work, YAML file ignored
            except Exception:
                pass  # a broken config file must never brick the agent
            break
    _apply_provider(cfg, raw_llm)
    return cfg


def _apply_provider(cfg: dict, raw_llm: dict) -> None:
    """Resolve llm.provider presets + api_key_env into concrete values."""
    llm = cfg.get("llm") or {}
    preset = PROVIDERS.get(str(llm.get("provider") or "").strip())
    if preset:
        for key, val in preset.items():
            if key not in raw_llm:  # explicit user values always win
                llm[key] = val
    env_name = str(llm.get("api_key_env") or "").strip()
    if env_name and not llm.get("api_key"):
        llm["api_key"] = os.environ.get(env_name, "")
