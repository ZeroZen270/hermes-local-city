"""The agent loop, with dual tool-calling modes and a context manager.

Tool calling, two ways:
  * **native** — the OpenAI-style `tools` array, when the model/server
    supports it (gpt-oss, Qwen3.x, Gemma 4 via Ollama/vLLM/...).
  * **react** — prompt-based fallback: the model emits fenced ```tool JSON
    blocks (the pattern Aaron's Hermes economy agent already uses), for
    models/servers with no native tool support.
`auto` tries native first and degrades to ReAct on any tools-call failure
(providers reject the `tools` array with many error shapes, not just the
ToolUnsupportedError the client can positively identify).

Context shortfall: hosted agent frameworks demand 64K context; small local
models often have 8K-32K. The ContextManager below keeps the agent capable
anyway — token estimation, sliding window, brain-written summaries of older
turns, and RAG recall from long-term memory instead of context stuffing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import tools as tools_mod
from .llm import LLMError, ToolUnsupportedError, messages_tokens

TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)

BASE_SYSTEM = (
    "You are Hermes Local, an offline-first personal agent running entirely "
    "on the user's own hardware. Be direct and useful. Use tools when they "
    "help you act or verify. When the task is complete, give a plain final "
    "answer with no tool blocks."
)


def parse_tool_blocks(text: str) -> list[dict]:
    """Extract fenced ```tool JSON blocks from a model reply (ReAct mode)."""
    calls = []
    for i, raw in enumerate(TOOL_BLOCK_RE.findall(text or "")):
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments") or {}
            calls.append({"id": f"react_{i}", "name": obj["name"],
                          "arguments": args if isinstance(args, dict) else {}})
    return calls


class ContextManager:
    """Keeps a conversation inside a small token budget.

    Strategy: pin system messages, keep the most recent turns verbatim,
    fold everything older into a brain-written summary. If even that does
    not fit, drop the oldest recent turns one at a time.
    """

    def __init__(self, max_tokens: int = 16000, keep_recent: int = 6):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent

    def estimate(self, messages: list[dict]) -> int:
        return messages_tokens(messages)

    def compress(self, messages: list[dict], summarize) -> tuple[list[dict], bool]:
        if self.estimate(messages) <= self.max_tokens:
            return messages, False
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]
        if len(rest) <= self.keep_recent:
            return messages, False  # nothing safe to fold yet
        middle, recent = rest[:-self.keep_recent], rest[-self.keep_recent:]
        excerpt = "\n".join(
            f"{m.get('role')}: {m.get('content') or ''}" for m in middle)[:8000]
        try:
            summary = summarize(excerpt)
        except Exception:
            summary = "(earlier turns dropped to fit the context window)"
        marker = {"role": "system",
                  "content": f"[Summary of earlier conversation] {summary}"}
        out = system + [marker] + recent
        while self.estimate(out) > self.max_tokens and len(recent) > 2:
            recent = recent[1:]
            out = system + [marker] + recent
        return out, True


class Agent:
    def __init__(self, client, memory=None, embedder=None, config: dict | None = None,
                 workspace: str = ".", docs_dir: str = "docs"):
        cfg = config or {}
        acfg = cfg.get("agent", {})
        tcfg = cfg.get("tools", {})
        self.client = client
        self.memory = memory
        self.mode = acfg.get("tool_mode", "auto")
        self.max_iterations = int(acfg.get("max_iterations", 12))
        self.memory_recall = bool(acfg.get("memory_recall", True))
        self.recall_k = int(cfg.get("memory", {}).get("recall_k", 4))
        self.cm = ContextManager(
            max_tokens=int(acfg.get("context_tokens", 16000)),
            keep_recent=int(acfg.get("keep_recent_turns", 6)))
        ws = Path(workspace)
        dd = Path(docs_dir)
        self.ctx = tools_mod.ToolContext(
            workspace=ws,
            docs_dir=dd if dd.is_absolute() else ws / dd,
            memory=memory,
            embedder=embedder or (memory.embedder if memory else None),
            shell_timeout=int(tcfg.get("shell_timeout", 30)))
        self.messages: list[dict] = []
        self.stats = {"iterations": 0, "tool_calls": 0, "compressions": 0}

    # -- prompts -------------------------------------------------------------
    def _system_prompt(self) -> str:
        prompt = BASE_SYSTEM
        if self.mode == "react":
            prompt += "\n\n" + tools_mod.catalog_text()
        return prompt

    def _refresh_system(self) -> None:
        system = {"role": "system", "content": self._system_prompt()}
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = system
        else:
            self.messages.insert(0, system)

    def _summarize(self, text: str) -> str:
        resp = self.client.chat([
            {"role": "system",
             "content": "Summarize the conversation excerpt briefly. Keep "
                        "facts, decisions, file paths, and open items."},
            {"role": "user", "content": text}])
        return resp.get("content") or "(summary unavailable)"

    # -- chat with mode fallback ----------------------------------------------
    def _chat(self) -> dict:
        if self.mode in ("auto", "native"):
            try:
                return self.client.chat(self.messages, tools=tools_mod.schemas())
            except Exception as err:
                # Degrade once, permanently for this session: ReAct prompting
                # works on every chat model, native tools or not. Providers
                # reject the tools array with many error shapes, not just
                # ToolUnsupportedError, so ANY tools-call failure degrades;
                # a failure of the plain retry still propagates to run().
                print(f"[hermes] agent: native tools failed "
                      f"({type(err).__name__}: {str(err)[:200]}); "
                      "degrading to ReAct mode", flush=True)
                self.mode = "react"
                self._refresh_system()
                return self.client.chat(self.messages)
        return self.client.chat(self.messages)

    # -- main loop ---------------------------------------------------------------
    def run(self, task: str) -> str:
        if not self.messages:
            self.messages = [{"role": "system", "content": self._system_prompt()}]
        else:
            self._refresh_system()

        if self.memory is not None and self.memory_recall:
            try:
                hits = [h for h in self.memory.recall(task, k=self.recall_k)
                        if h["score"] > 0.05]
            except Exception:
                hits = []
            if hits:
                note = "\n".join(f"- {h['text']}" for h in hits)
                self.messages.append(
                    {"role": "system",
                     "content": f"Relevant long-term memory:\n{note}"})

        self.messages.append({"role": "user", "content": task})
        last_content = ""

        for _ in range(self.max_iterations):
            self.stats["iterations"] += 1
            self.messages, did = self.cm.compress(self.messages, self._summarize)
            if did:
                self.stats["compressions"] += 1
            try:
                resp = self._chat()
            except LLMError as err:
                return f"Agent stopped: the model could not be reached. {err}"
            content = resp.get("content") or ""
            calls = resp.get("tool_calls") or []
            native = bool(calls)
            if not calls:
                calls = parse_tool_blocks(content)  # ReAct blocks work in any mode
            if not calls:
                self.messages.append({"role": "assistant", "content": content})
                return content
            last_content = content
            self.stats["tool_calls"] += len(calls)

            if native:
                self.messages.append({"role": "assistant", "content": content,
                                      "tool_calls": calls})
                for call in calls:
                    result = tools_mod.run_tool(
                        self.ctx, call["name"], call.get("arguments"))
                    self.messages.append(
                        {"role": "tool", "name": call["name"],
                         "tool_call_id": call.get("id", ""),
                         "content": result})
            else:
                self.messages.append({"role": "assistant", "content": content})
                results = []
                for call in calls:
                    result = tools_mod.run_tool(
                        self.ctx, call["name"], call.get("arguments"))
                    results.append(f"- {call['name']}: {result}")
                self.messages.append(
                    {"role": "user",
                     "content": "Tool results:\n" + "\n".join(results) +
                                "\nContinue the task, or give the final "
                                "answer with no tool blocks."})

        return (last_content +
                f"\n(stopped after {self.max_iterations} iterations)").strip()
