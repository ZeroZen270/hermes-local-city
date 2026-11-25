"""Embedding layer with a never-fail fallback chain.

Built-in model: **EmbeddingGemma 2** (Google, Gemma 4-based, Apache 2.0) —
768-dim natively multimodal embeddings in ~567MB. It is an encoder only:
it cannot generate text or call tools, so it is paired with a generative
brain (see llm.py) and serves purely as the retrieval layer.

Fallback chain (semantic memory must NEVER hard-fail offline):
    EmbeddingGemma 2 -> EmbeddingGemma v1 -> nomic-embed-text
    -> built-in zero-dependency TF-IDF-style hashing embedder.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request


class EmbeddingError(Exception):
    """One embedding backend failed; the chain moves on to the next."""


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are",
    "it", "its", "for", "with", "as", "at", "by", "be", "this", "that",
}


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower())
             if w not in _STOPWORDS]
    toks = list(words)
    toks += [f"{a}_{b}" for a, b in zip(words, words[1:])]  # bigrams
    return toks


def truncate(vec: list[float], dim: int | None) -> list[float]:
    """Matryoshka-style truncation: keep the leading `dim` components and
    renormalize. Lets us store 768d vectors but search at 256d."""
    if not dim or len(vec) <= dim:
        return list(vec)
    out = list(vec[:dim])
    norm = math.sqrt(sum(x * x for x in out))
    if norm:
        out = [x / norm for x in out]
    return out


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Last-resort embedder: zero dependencies, deterministic across runs
# (stable MD5 feature hashing — Python's hash() is per-process randomized).
# --------------------------------------------------------------------------
class HashEmbedder:
    """TF-IDF-style hashing embedder. Word unigrams + bigrams hashed into a
    fixed-dimension space with sublinear TF weighting, L2-normalized.
    Not a neural model — but it ranks shared-vocabulary text sensibly and
    works with no server, no GPU, and no downloads."""

    name = "local-hash-tfidf"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._one(text)

    def _one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for tok in _tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        vec = [0.0] * self.dim
        for tok, count in counts.items():
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0 + math.log(count)
        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec


# --------------------------------------------------------------------------
# HTTP embedding client with the fallback chain
# --------------------------------------------------------------------------
class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:11434",
                 api: str = "ollama",
                 models: list[str] | None = None,
                 timeout: int = 10, local_dim: int = 512):
        self.base_url = base_url.rstrip("/")
        self.api = api
        self.models = models or ["embeddinggemma-2", "embeddinggemma",
                                 "nomic-embed-text"]
        self.timeout = timeout
        self.local = HashEmbedder(local_dim)
        self.active_model: str | None = None
        self._use_local = False

    @property
    def active_name(self) -> str:
        if self._use_local:
            return self.local.name
        return self.active_model or "undecided"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._use_local:
            candidates = list(self.models)
            if self.active_model in candidates:
                candidates.remove(self.active_model)
                candidates.insert(0, self.active_model)
            for model in candidates:
                try:
                    vecs = self._embed_http(model, texts)
                    self.active_model = model
                    return vecs
                except EmbeddingError:
                    continue
            # Every server model failed (offline, model not pulled, ...).
            # Commit to the local embedder for the rest of the session so
            # we don't pay a connection attempt on every recall.
            self._use_local = True
        return self.local.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # -- transports ----------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as err:
            raise EmbeddingError(f"{path}: {err}")

    def _embed_http(self, model: str, texts: list[str]) -> list[list[float]]:
        if self.api == "openai":
            # Same /v1 rule as the chat client: don't double the prefix
            # when the configured base already ends in /v1.
            path = ("/embeddings" if self.base_url.endswith("/v1")
                    else "/v1/embeddings")
            data = self._post(path, {"model": model, "input": texts})
            items = sorted(data.get("data") or [],
                           key=lambda d: d.get("index", 0))
            vecs = [it.get("embedding") or [] for it in items]
        else:
            data = self._post("/api/embed", {"model": model, "input": texts})
            vecs = data.get("embeddings") or []
        if len(vecs) != len(texts) or any(not v for v in vecs):
            raise EmbeddingError(f"bad embedding response from {model}")
        return [[float(x) for x in v] for v in vecs]
