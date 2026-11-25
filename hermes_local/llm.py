"""Provider-agnostic LLM client with a low-resource brain ladder.

Talks to Ollama's native API or any OpenAI-compatible endpoint using only
the standard library (urllib), so the framework runs with zero installs.

Shortfall engineering: the "best" local brain (a 20B tool-caller) does not
fit every machine, so the brain is auto-selected from a ladder based on
detected RAM/VRAM, from gpt-oss 20B down to a tiny model that runs anywhere.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request


class LLMError(Exception):
    """A chat failure the agent loop can surface and recover from."""


class ToolUnsupportedError(LLMError):
    """The server/model rejected the native `tools` array.

    The agent catches this and degrades to prompt-based ReAct tool calling.
    """


# --------------------------------------------------------------------------
# Resource detection (stdlib only)
# --------------------------------------------------------------------------
def detect_resources() -> dict:
    """Return {'ram_gb', 'vram_gb', 'has_gpu'} using /proc + nvidia-smi."""
    ram_gb = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)  # kB -> GB
                    break
    except OSError:
        pass
    if ram_gb is None:
        try:
            import os

            ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
        except (ValueError, OSError, AttributeError):
            ram_gb = 4.0

    vram_gb = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = []
        for tok in out.stdout.split():
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        if vals:
            vram_gb = max(vals) / 1024  # MiB -> GB
    except (OSError, subprocess.SubprocessError):
        vram_gb = None

    return {
        "ram_gb": round(float(ram_gb), 1),
        "vram_gb": (round(float(vram_gb), 1) if vram_gb else None),
        "has_gpu": vram_gb is not None,
    }


# --------------------------------------------------------------------------
# Brain ladder: (ollama tag, min VRAM GB, min RAM GB for CPU, note)
# Order is preference order; the first entry the machine can hold wins.
# --------------------------------------------------------------------------
BRAIN_LADDER = [
    ("gpt-oss:20b", 15.0, 20.0,
     "full agent brain: native tool calling; holds 64K context in 16GB VRAM "
     "with a q8_0 KV cache"),
    ("qwen3.5:9b", 6.0, 10.0,
     "strong mid-size tool caller; comfortable on 8GB GPUs"),
    ("gemma4:12b", 8.0, 14.0,
     "Gemma 4 with native tools + vision; shares its tokenizer family with "
     "EmbeddingGemma 2"),
    ("qwen3:4b", 3.0, 6.0,
     "small but genuinely tool-capable; the low-resource workhorse"),
    ("qwen3:0.6b", 0.0, 1.0,
     "tiny mode: chat and simple routing on almost any hardware"),
]


def pick_brain_model(resources: dict, override: str = "auto") -> tuple[str, str]:
    """Pick the best brain the detected resources can hold."""
    if override and override != "auto":
        return override, "config override"
    use_vram = resources.get("vram_gb") is not None
    avail = resources["vram_gb"] if use_vram else (resources.get("ram_gb") or 0.0)
    for name, min_vram, min_ram, note in BRAIN_LADDER:
        need = min_vram if use_vram else min_ram
        if avail >= need:
            return name, note
    return BRAIN_LADDER[-1][0], BRAIN_LADDER[-1][3]


# --------------------------------------------------------------------------
# Token estimation (rough, dependency-free: ~4 chars/token)
# --------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(str(msg.get("content") or ""))
        if msg.get("tool_calls"):
            total += estimate_tokens(json.dumps(msg["tool_calls"], default=str))
    return total


def _json_or_empty(text: str) -> dict:
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else {"value": val}
    except (ValueError, TypeError):
        return {}


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class LLMClient:
    """Chat client for Ollama native API or OpenAI-compatible endpoints.

    chat(messages, tools=None) -> {"content": str,
                                  "tool_calls": [{"id", "name", "arguments"}]}
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen3:4b", api: str = "ollama",
                 timeout: int = 120, retries: int = 3,
                 temperature: float = 0.7, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api = api
        self.timeout = timeout
        self.retries = max(1, retries)
        self.temperature = temperature
        self.api_key = api_key

    # -- public ------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self.api == "openai":
                    return self._chat_openai(messages, tools)
                return self._chat_ollama(messages, tools)
            except ToolUnsupportedError:
                raise  # not retryable: the agent degrades to ReAct instead
            except LLMError as err:
                last_err = err
                if attempt < self.retries - 1:
                    time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"chat failed after {self.retries} attempt(s): {last_err}")

    # -- transports ----------------------------------------------------------
    def _post(self, path: str, payload: dict, tools_in_payload: bool = False) -> dict:
        url = self.base_url + path
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")[:500]
            lowered = body.lower()
            if (tools_in_payload and err.code in (400, 422)
                    and ("tool" in lowered or "function" in lowered)):
                raise ToolUnsupportedError(
                    f"server rejected tools (HTTP {err.code}): {body}")
            raise LLMError(f"HTTP {err.code} from {url}: {body}")
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise LLMError(f"cannot reach {url}: {err}")

    # -- Ollama native -------------------------------------------------------
    @staticmethod
    def _ollama_msg(msg: dict) -> dict:
        out: dict = {"role": msg.get("role", "user"),
                     "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = [
                {"function": {"name": tc["name"], "arguments": tc.get("arguments") or {}}}
                for tc in msg["tool_calls"]]
        return out

    def _chat_ollama(self, messages: list[dict], tools) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [self._ollama_msg(m) for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        data = self._post("/api/chat", payload, tools_in_payload=bool(tools))
        msg = data.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = _json_or_empty(args)
            calls.append({"id": tc.get("id") or f"call_{i}",
                          "name": fn.get("name", ""), "arguments": args})
        return {"content": msg.get("content") or "", "tool_calls": calls}

    # -- OpenAI-compatible ---------------------------------------------------
    @staticmethod
    def _openai_msg(msg: dict) -> dict:
        out: dict = {"role": msg.get("role", "user"),
                     "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = [
                {"id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc.get("arguments") or {})}}
                for i, tc in enumerate(msg["tool_calls"])]
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            out["tool_call_id"] = msg["tool_call_id"]
        return out

    def _chat_openai(self, messages: list[dict], tools) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [self._openai_msg(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
        # Preset bases (e.g. Nous) already end in /v1, and Google's
        # OpenAI-compatible base ends in /openai (…/v1beta/openai);
        # both are complete API roots. A bare OpenAI-style base needs
        # the /v1 prefix. Doubling it 404s every chat call.
        path = ("/chat/completions"
                if self.base_url.endswith(("/v1", "/openai"))
                else "/v1/chat/completions")
        data = self._post(path, payload,
                          tools_in_payload=bool(tools))
        choices = data.get("choices") or []
        msg = (choices[0].get("message") if choices else {}) or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = _json_or_empty(args)
            calls.append({"id": tc.get("id") or f"call_{i}",
                          "name": fn.get("name", ""), "arguments": args})
        return {"content": msg.get("content") or "", "tool_calls": calls}
