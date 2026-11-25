"""Doorway server: the Hugging Face Space as a pure doorway.

The city itself lives ONLY on Aaron's tablet, inside the Hermes app
shell, thinking with the app's own Nous/LongCat sign-in. This module
lets that tablet dial OUT to the Space over one WebSocket and serve
its localhost bridge (127.0.0.1:8765) to the public web through it:

    browser -> Space /world/* and /api/* -> WS tunnel -> tablet

Design rules (Aaron's, 2026-11-25):
  * The Space NEVER runs a world of its own. One world, one writer
    of world-state.json, no divergence.
  * Tablet down = world paused. /world serves a paused page, /api
    answers 503, nothing ticks, no hosted brain quota is burned.

Protocol (JSON text frames over the one socket):
  tablet -> {"auth": <world token>}            (first frame)
  space  -> {"type": "hello"}
  space  -> {"type": "request", "id", "method", "path",
             "headers": {...}, "body": <base64>}
  tablet -> {"type": "head", "id", "status", "headers": {...}}
  tablet -> {"type": "data", "id", "data": <base64>}   (0..n)
  tablet -> {"type": "end", "id"}  |  {"type": "error", "id", ...}

The tablet side is doorway.py at the repo root (stdlib only, so it
runs inside the Android app shell where native binaries cannot).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import time

from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute

PAUSED_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hermes Local City - world paused</title>
<style>body{background:#0b0e13;color:#dbe4f0;font-family:system-ui,sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh;
margin:0}.box{max-width:34rem;padding:2rem;text-align:center}
h1{font-size:1.4rem;margin:.2em 0}p{color:#93a4bb;line-height:1.55}
.dot{display:inline-block;width:.6em;height:.6em;border-radius:50%;
background:#e0a83c;margin-right:.45em}</style></head>
<body><div class="box"><h1><span class="dot"></span>Hermes Local City</h1>
<p>The world is <strong>paused</strong>. The city lives on Aaron's tablet
and thinks with the Hermes app's own brain - when the tablet is on with
the city running and the doorway open, this page becomes the city.</p>
<p>Nothing is lost: the world's saved state waits on the tablet and in
its world-data repo. This page checks again every 30 seconds.</p>
</div></body></html>"""

_HOP = {"host", "content-length", "connection", "transfer-encoding"}


class Doorway:
    """One tablet connection + the pending-request registry."""

    def __init__(self, token: str):
        self._token = token or ""
        self.ws = None
        self.since: float | None = None
        self.pending: dict[str, asyncio.Queue] = {}
        self._counter = 0

    # -- status (for the Space's status page) -------------------------
    @property
    def connected(self) -> bool:
        return self.ws is not None

    def status_text(self) -> str:
        if self.ws is None:
            return "World paused - tablet offline (no doorway connection)."
        mins = max(0, int((time.time() - (self.since or time.time())) // 60))
        return (f"Tablet connected - doorway open for {mins} min. "
                "The city is live at /world/.")

    # -- tablet side ----------------------------------------------------
    async def ws_endpoint(self, websocket):
        await websocket.accept()
        try:
            first = await asyncio.wait_for(websocket.receive_text(),
                                           timeout=20)
            auth = str(json.loads(first).get("auth") or "")
        except Exception:
            await websocket.close(code=4401)
            return
        if not self._token or not secrets.compare_digest(auth, self._token):
            await websocket.close(code=4401)
            return
        old = self.ws
        self.ws = websocket
        self.since = time.time()
        if old is not None and old is not websocket:
            try:
                await old.close()
            except Exception:
                pass
        print("Doorway: tablet connected.", flush=True)
        try:
            await websocket.send_text(json.dumps({"type": "hello"}))
            while True:
                msg = json.loads(await websocket.receive_text())
                if os.environ.get("DOORWAY_DEBUG"):
                    print(f"Doorway rx: {msg.get('type')} {msg.get('id')} "
                          f"pending={list(self.pending)}", flush=True)
                queue = self.pending.get(str(msg.get("id") or ""))
                if queue is not None:
                    await queue.put(msg)
        except Exception:
            pass
        finally:
            if self.ws is websocket:
                self.ws = None
                self.since = None
            for queue in self.pending.values():
                await queue.put({"type": "error",
                                 "error": "doorway closed"})
            print("Doorway: tablet disconnected.", flush=True)

    # -- browser side ---------------------------------------------------
    async def _tunnel(self, request, upstream_path: str):
        if self.ws is None:
            if upstream_path.startswith("/api/"):
                return JSONResponse(
                    {"error": "world paused - tablet offline"},
                    status_code=503)
            return HTMLResponse(PAUSED_HTML, status_code=503)
        body = await request.body()
        self._counter += 1
        rid = f"r{self._counter}"
        queue: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = queue
        query = f"?{request.url.query}" if request.url.query else ""
        frame = {
            "type": "request", "id": rid, "method": request.method,
            "path": upstream_path + query,
            "headers": {k: v for k, v in request.headers.items()
                        if k.lower() not in _HOP},
            "body": base64.b64encode(body).decode() if body else "",
        }
        try:
            await self.ws.send_text(json.dumps(frame))
            head = await asyncio.wait_for(queue.get(), timeout=90)
            if head.get("type") != "head":
                self.pending.pop(rid, None)
                return JSONResponse(
                    {"error": "tablet did not answer"}, status_code=502)
            headers = {k: v for k, v in (head.get("headers") or {}).items()
                       if k.lower() not in _HOP}
        except asyncio.TimeoutError:
            self.pending.pop(rid, None)
            return JSONResponse({"error": "tablet timed out"},
                                status_code=504)
        except Exception:
            self.pending.pop(rid, None)
            return JSONResponse({"error": "doorway failure"},
                                status_code=502)

        async def gen():
            # The pending entry must live until the STREAM ends -
            # popping it when this handler returns (the response
            # object is built before the body streams) starves the
            # body: every data frame gets dropped as unknown.
            try:
                while True:
                    msg = await queue.get()
                    if msg.get("type") == "data" and msg.get("data"):
                        yield base64.b64decode(msg["data"])
                    else:  # "end" or "error"
                        break
            finally:
                self.pending.pop(rid, None)

        return StreamingResponse(gen(),
                                 status_code=int(head.get("status", 200)),
                                 headers=headers)

    async def proxy_api(self, request):
        return await self._tunnel(request,
                                  f"/api/{request.path_params['path']}")

    async def proxy_world(self, request):
        return await self._tunnel(request,
                                  f"/{request.path_params['path']}")

    async def proxy_world_root(self, request):
        return await self._tunnel(request, "/")

    def routes(self) -> list:
        return [
            WebSocketRoute("/doorway/connect", self.ws_endpoint),
            Route("/api/{path:path}", self.proxy_api,
                  methods=["GET", "POST"]),
            Route("/world/{path:path}", self.proxy_world,
                  methods=["GET", "POST"]),
            Route("/world", self.proxy_world_root, methods=["GET"]),
        ]


def attach_doorway(app, token: str) -> Doorway:
    """Register the doorway routes FIRST on a FastAPI/Gradio app."""
    doorway = Doorway(token)
    for route in reversed(doorway.routes()):
        app.router.routes.insert(0, route)
    return doorway
