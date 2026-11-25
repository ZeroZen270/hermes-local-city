"""Command line: REPL, one-shot tasks, diagnostics, and the world server.

    python -m hermes_local                    # interactive REPL
    python -m hermes_local "do X"             # one-shot task
    python -m hermes_local --info             # resources / ladder / embedding status
    python -m hermes_local serve [--mock]     # 3D world bridge + web world demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config


def _build_agent(cfg: dict):
    from .agent import Agent
    from .embeddings import EmbeddingClient
    from .llm import LLMClient, detect_resources, pick_brain_model
    from .memory import Memory

    resources = detect_resources()
    model, why = pick_brain_model(resources, cfg["llm"].get("model", "auto"))
    client = LLMClient(base_url=cfg["llm"]["base_url"], model=model,
                       api=cfg["llm"]["api"], timeout=int(cfg["llm"]["timeout"]),
                       retries=int(cfg["llm"]["retries"]),
                       temperature=float(cfg["llm"]["temperature"]),
                       api_key=cfg["llm"].get("api_key", ""))
    ecfg = cfg["embeddings"]
    embedder = EmbeddingClient(base_url=ecfg["base_url"], api=ecfg["api"],
                               models=ecfg["models"],
                               timeout=int(ecfg["timeout"]),
                               local_dim=int(ecfg["local_dim"]))
    mcfg = cfg["memory"]
    memory = Memory(mcfg["db_path"], mcfg["notes_path"], embedder,
                    search_dim=int(ecfg["search_dim"]),
                    recall_k=int(mcfg["recall_k"]))
    agent = Agent(client, memory=memory, embedder=embedder, config=cfg,
                  workspace=cfg["tools"]["workspace"],
                  docs_dir=cfg["tools"]["docs_dir"])
    return agent, resources, model, why


def _info(cfg: dict) -> int:
    from .embeddings import EmbeddingClient
    from .llm import detect_resources, pick_brain_model

    resources = detect_resources()
    model, why = pick_brain_model(resources, cfg["llm"].get("model", "auto"))
    print(f"Resources: {resources}")
    print(f"Brain (ladder pick): {model} — {why}")
    if cfg["llm"].get("provider"):
        key_state = "key found" if cfg["llm"].get("api_key") else "NO KEY SET"
        print(f"Provider: {cfg['llm']['provider']} -> {cfg['llm']['base_url']} "
              f"(model {model}, {key_state})")
    ecfg = cfg["embeddings"]
    emb = EmbeddingClient(base_url=ecfg["base_url"], api=ecfg["api"],
                          models=ecfg["models"], timeout=3,
                          local_dim=int(ecfg["local_dim"]))
    vec = emb.embed_one("resource check")
    print(f"Embeddings: active={emb.active_name}, dim={len(vec)} "
          f"(chain: {' -> '.join(ecfg['models'])} -> local-hash-tfidf)")
    return 0


def _serve(args) -> int:
    from .world_bridge import build_world, make_http_server, make_socket_server

    cfg = load_config(args.config)
    project = Path(__file__).resolve().parent.parent
    mock = args.mock
    if not mock:
        # Out-of-the-box rule: if no LLM server answers, fall back to the
        # demo brain so the web world still works as a smoke demo.
        import urllib.request
        base = cfg["llm"]["base_url"].rstrip("/")
        if cfg["llm"].get("api") == "openai":
            probe = urllib.request.Request(base + "/models")
            if cfg["llm"].get("api_key"):
                probe.add_header("Authorization",
                                 f"Bearer {cfg['llm']['api_key']}")
        else:
            probe = base + "/api/tags"
        try:
            urllib.request.urlopen(probe, timeout=2)
        except Exception:
            print("No LLM server reachable — starting in MOCK mode (demo brain).")
            mock = True
    bridge, manager, npcs, positions, mode = build_world(
        cfg, mock=mock, characters_dir=args.characters,
        roster_count=args.npcs, layout_path=args.layout)
    http = make_http_server(bridge, args.host, args.port,
                            static_dir=project / "webworld",
                            positions=positions, mode=mode)
    sock = make_socket_server(bridge, args.host, args.socket_port)
    import threading
    threading.Thread(target=sock.serve_forever, daemon=True).start()
    print(f"World mode: {mode}")
    print(f"NPCs loaded: {len(npcs)} (cap {manager.max_active} ACTIVE, "
          f"radii {manager.active_radius}m / {manager.ambient_radius}m)")
    print(f"Web world:  http://{args.host}:{args.port}/")
    print(f"Portal:     http://{args.host}:{args.port}/portal")
    print(f"JSON-lines socket: {args.host}:{args.socket_port}")
    print("Ctrl+C to stop.")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        parser = argparse.ArgumentParser(prog="hermes_local serve")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--socket-port", type=int, default=8766)
        parser.add_argument("--mock", action="store_true",
                            help="use the rule-based demo brain (no LLM server)")
        parser.add_argument("--characters", default=None,
                            help="directory of character cards (default: ./characters if present)")
        parser.add_argument("--npcs", type=int, default=54,
                            help="procedural roster size")
        parser.add_argument("--layout", default=None,
                            help="path to persist the city layout JSON")
        parser.add_argument("--config", default=None)
        args = parser.parse_args(argv[1:])
        if args.characters is None:
            default_chars = Path("characters")
            if default_chars.is_dir():
                args.characters = str(default_chars)
        return _serve(args)

    cfg = load_config()
    if argv and argv[0] == "--info":
        return _info(cfg)
    agent, resources, model, _why = _build_agent(cfg)

    if argv:  # one-shot
        from .llm import LLMError
        try:
            print(agent.run(" ".join(argv)))
        except LLMError as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        return 0

    print(f"Hermes Local — brain: {model} | resources: {resources}")
    print("Commands: /quit, /stats, /mode. Anything else is a task.")
    while True:
        try:
            line = input("hermes> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/stats":
            print(agent.stats)
            continue
        if line == "/mode":
            print(f"tool mode: {agent.mode}")
            continue
        from .llm import LLMError
        try:
            print(agent.run(line))
        except LLMError as err:
            print(f"Error: {err}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
