"""Tool registry and core tools.

Decorator-registered tools with JSON-schema catalogs (for native tool
calling) and a text catalog (for the ReAct fallback prompt). Every tool
returns a string and NEVER raises into the agent loop — a broken tool is
feedback for the model, not a crash.
"""
from __future__ import annotations

import inspect
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import cosine, truncate

TOOLS: dict[str, dict] = {}


def tool(name: str, description: str, properties: dict,
         required: list[str] | None = None):
    """Register a tool. `properties` is a JSON-schema properties dict."""
    def deco(fn):
        TOOLS[name] = {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or []},
            "fn": fn,
        }
        return fn
    return deco


@dataclass
class ToolContext:
    workspace: Path = Path(".")
    docs_dir: Path = Path("docs")
    memory: object = None          # memory.Memory or None
    embedder: object = None        # embeddings.EmbeddingClient/HashEmbedder
    shell_timeout: int = 30
    plan: list = field(default_factory=list)
    _rag_index: list | None = None  # cached [(source, chunk, vector)]
    # NPC / 3D-world fields (used by the world action tools in npc.py):
    world: object = None           # sink with send_action(npc, action, args)
    action_schema: dict | None = None  # per-world allowed actions, or None = default
    npc_name: str = ""             # which NPC this context belongs to


def schemas() -> list[dict]:
    """OpenAI-style tool schemas for native tool calling."""
    return [{"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in TOOLS.values()]


def catalog_text() -> str:
    """Text catalog injected into the system prompt in ReAct mode."""
    lines = ["Available tools — call one by replying with a fenced block:",
             '```tool', '{"name": "<tool>", "arguments": {...}}', '```']
    for t in TOOLS.values():
        args = ", ".join(
            f"{k}: {v.get('type', 'string')}"
            for k, v in t["parameters"]["properties"].items())
        lines.append(f"- {t['name']}({args}): {t['description']}")
    return "\n".join(lines)


def run_tool(ctx: ToolContext, name: str, arguments: dict | None) -> str:
    """Execute one tool call; never raises."""
    entry = TOOLS.get(name)
    if entry is None:
        return f"TOOL ERROR: unknown tool '{name}'. Available: {', '.join(sorted(TOOLS))}"
    arguments = arguments or {}
    try:
        params = inspect.signature(entry["fn"]).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = dict(arguments)  # fn takes **kwargs: pass everything
        else:
            kwargs = {k: v for k, v in arguments.items() if k in params}
        result = entry["fn"](ctx, **kwargs)
        return result if isinstance(result, str) else str(result)
    except Exception as err:  # tools must never crash the loop
        return f"TOOL ERROR in {name}: {type(err).__name__}: {err}"


def _resolve(ctx: ToolContext, path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ctx.workspace / p)


# --------------------------------------------------------------------------
# Shell (denylisted destructive patterns; timeout; never raises)
# --------------------------------------------------------------------------
_DENY_PATTERNS = [
    r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/(?:\s|$)",   # rm -rf /
    r"\bmkfs\b", r"\bwipefs\b",
    r"\bdd\b[^|]*\bof=/dev/",
    r">\s*/dev/(sd|nvme|hd)[a-z]",
    r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b", r"\bhalt\b",
    r"chmod\s+-R\s+777\s+/",
    r":\(\)\s*\{",                                # fork bomb
    r"\bformat\s+[a-z]:",
]


@tool("run_shell", "Run a shell command in the workspace and return its output.",
      {"command": {"type": "string"}}, ["command"])
def run_shell(ctx: ToolContext, command: str = "") -> str:
    for pat in _DENY_PATTERNS:
        if re.search(pat, command):
            return f"TOOL ERROR: command blocked by safety denylist ({pat})"
    try:
        proc = subprocess.run(command, shell=True, cwd=str(ctx.workspace),
                              capture_output=True, text=True,
                              timeout=ctx.shell_timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        return (out.strip() or f"(exit {proc.returncode}, no output)")[:4000]
    except subprocess.TimeoutExpired:
        return f"TOOL ERROR: command timed out after {ctx.shell_timeout}s"
    except Exception as err:
        return f"TOOL ERROR: {err}"


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------
@tool("read_file", "Read a text file (workspace-relative or absolute).",
      {"path": {"type": "string"}}, ["path"])
def read_file(ctx: ToolContext, path: str = "") -> str:
    p = _resolve(ctx, path)
    if not p.is_file():
        return f"TOOL ERROR: no such file: {path}"
    return p.read_text(errors="replace")[:8000]


@tool("write_file", "Write (create or overwrite) a text file.",
      {"path": {"type": "string"}, "content": {"type": "string"}},
      ["path", "content"])
def write_file(ctx: ToolContext, path: str = "", content: str = "") -> str:
    p = _resolve(ctx, path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


@tool("edit_file", "Replace the first occurrence of `old` with `new` in a file.",
      {"path": {"type": "string"}, "old": {"type": "string"},
       "new": {"type": "string"}}, ["path", "old", "new"])
def edit_file(ctx: ToolContext, path: str = "", old: str = "", new: str = "") -> str:
    p = _resolve(ctx, path)
    if not p.is_file():
        return f"TOOL ERROR: no such file: {path}"
    text = p.read_text(errors="replace")
    if old not in text:
        return f"TOOL ERROR: text to replace not found in {path}"
    p.write_text(text.replace(old, new, 1))
    return f"Edited {path}"


@tool("list_dir", "List a directory (workspace-relative or absolute).",
      {"path": {"type": "string"}})
def list_dir(ctx: ToolContext, path: str = ".") -> str:
    p = _resolve(ctx, path)
    if not p.is_dir():
        return f"TOOL ERROR: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: e.name)
    return "\n".join(f"{'d' if e.is_dir() else 'f'} {e.name}"
                     for e in entries) or "(empty)"


# --------------------------------------------------------------------------
# Web (offline-safe: failures come back as strings, not exceptions)
# --------------------------------------------------------------------------
@tool("fetch_url", "Fetch a URL and return its text. Fails gracefully offline.",
      {"url": {"type": "string"}}, ["url"])
def fetch_url(ctx: ToolContext, url: str = "") -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-local/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode(errors="replace")
        text = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", text).strip()[:4000]
    except Exception as err:
        return f"TOOL ERROR: could not fetch {url} ({err}). The agent may be offline."


# --------------------------------------------------------------------------
# Memory + RAG
# --------------------------------------------------------------------------
@tool("remember", "Save a fact to long-term memory.",
      {"text": {"type": "string"}}, ["text"])
def remember(ctx: ToolContext, text: str = "") -> str:
    if ctx.memory is None:
        return "TOOL ERROR: memory is not configured"
    mid = ctx.memory.remember(text)
    return f"Remembered (id {mid}): {text}"


@tool("recall", "Search long-term memory for facts relevant to a query.",
      {"query": {"type": "string"}}, ["query"])
def recall(ctx: ToolContext, query: str = "") -> str:
    if ctx.memory is None:
        return "TOOL ERROR: memory is not configured"
    hits = ctx.memory.recall(query)
    if not hits:
        return "No memories stored yet."
    return "\n".join(f"[{h['score']:.2f}] {h['text']}" for h in hits)


def _build_rag_index(ctx: ToolContext) -> list:
    docs: list[tuple[str, str]] = []
    if ctx.docs_dir.is_dir():
        for path in sorted(ctx.docs_dir.rglob("*")):
            if path.is_file() and path.suffix in (".md", ".txt", ".rst", ".py"):
                try:
                    text = path.read_text(errors="replace")
                except OSError:
                    continue
                for chunk in re.split(r"\n\s*\n", text):
                    chunk = chunk.strip()
                    if 40 <= len(chunk) <= 1200:
                        docs.append((str(path), chunk))
    if not docs or ctx.embedder is None:
        return []
    vecs = ctx.embedder.embed([c for _, c in docs])
    return [(src, chunk, vec) for (src, chunk), vec in zip(docs, vecs)]


@tool("rag_search", "Semantic search over the local docs directory.",
      {"query": {"type": "string"}}, ["query"])
def rag_search(ctx: ToolContext, query: str = "") -> str:
    if ctx.embedder is None:
        return "TOOL ERROR: no embedder configured"
    if ctx._rag_index is None:
        ctx._rag_index = _build_rag_index(ctx)
    if not ctx._rag_index:
        return f"No documents indexed under {ctx.docs_dir}."
    q = ctx.embedder.embed_one(query)
    scored = sorted(
        ((cosine(truncate(q, 256), truncate(vec, 256)), src, chunk)
         for src, chunk, vec in ctx._rag_index),
        key=lambda r: r[0], reverse=True)[:3]
    return "\n\n".join(f"[{score:.2f}] {src}: {chunk}"
                       for score, src, chunk in scored)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
@tool("plan", "Maintain a simple task plan. action: set|add|done|show. "
              "For set, `text` is newline-separated items.",
      {"action": {"type": "string"}, "text": {"type": "string"}})
def plan(ctx: ToolContext, action: str = "show", text: str = "") -> str:
    if action == "set":
        ctx.plan = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
    elif action == "add" and text:
        ctx.plan.append(text)
    elif action == "done" and text:
        ctx.plan = [f"✓ {item}" if item == text or text in item else item
                    for item in ctx.plan]
    if not ctx.plan:
        return "Plan is empty."
    return "\n".join(f"{i}. {item}" for i, item in enumerate(ctx.plan, 1))
