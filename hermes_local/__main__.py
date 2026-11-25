"""Entry point: `python -m hermes_local` (REPL) or
`python -m hermes_local "do X"` (one-shot)."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
