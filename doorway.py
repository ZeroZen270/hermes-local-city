#!/usr/bin/env python3
"""Doorway client: runs INSIDE the Hermes app shell on Aaron's tablet,
next to the city server, and dials OUT to the Hugging Face Space so
the public web can reach the tablet's city.

    python3 doorway.py            # uses $HERMES_WORLD_TOKEN or ~/.world_token

Why this shape: the Android app shell refuses native binaries
(cloudflared died on the linker) and the tablet's city serves a
localhost only that shell can see - but the shell has Python, and
Python needs no one's permission to open a TLS socket. This client
is standard library ONLY, on purpose.

Protocol: see hermes_local/doorway_server.py. One WebSocket, JSON
frames; browser requests arrive as frames and are forwarded to
http://127.0.0.1:8765 (the city's own bridge, unchanged). When this
client is not running, the Space serves its "world paused" page -
tablet down, world paused (Aaron's rule).
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import socket
import ssl
import struct
import sys
import threading
import time
import urllib.parse

WS_URL = os.environ.get(
    "DOORWAY_WS_URL",
    "wss://zerozen270-hermes-local-city.hf.space/doorway/connect")
LOCAL_HOST = os.environ.get("DOORWAY_LOCAL_HOST", "127.0.0.1")
LOCAL_PORT = int(os.environ.get("DOORWAY_LOCAL_PORT", "8765"))
HOP = {"host", "content-length", "connection", "transfer-encoding"}


def load_token() -> str:
    token = (os.environ.get("HERMES_WORLD_TOKEN") or "").strip()
    if token:
        return token
    try:
        with open(os.path.expanduser("~/.world_token")) as fh:
            return fh.read().strip()
    except OSError:
        return ""


# ----------------------------------------------------------------------
# Minimal WebSocket client (RFC 6455), stdlib only.
# ----------------------------------------------------------------------
class WSClient:
    def __init__(self, url: str):
        parts = urllib.parse.urlparse(url)
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "wss" else 80)
        self.path = parts.path or "/"
        if parts.query:
            self.path += "?" + parts.query
        raw = socket.create_connection((self.host, self.port), timeout=30)
        if parts.scheme == "wss":
            ctx = ssl.create_default_context()
            raw = ctx.wrap_socket(raw, server_hostname=self.host)
        self.sock = raw
        self.buf = b""
        self.send_lock = threading.Lock()
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(request.encode())
        head = self._read_until(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise ConnectionError(f"websocket upgrade refused: {status!r}")
        expect = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
        ).digest()).decode()
        if expect.encode() not in head:
            raise ConnectionError("bad Sec-WebSocket-Accept")

    def _read_until(self, marker: bytes) -> bytes:
        while marker not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        end = self.buf.index(marker) + len(marker)
        out, self.buf = self.buf[:end], self.buf[end:]
        return out

    def _read_exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        head = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self.send_lock:
            self.sock.sendall(bytes(head) + mask + masked)

    def send_json(self, obj: dict) -> None:
        self.send_frame(0x1, json.dumps(obj).encode())

    def recv_msg(self) -> tuple[int, bytes]:
        """One complete message; pings are ponged inline."""
        frags: list[bytes] = []
        opcode0 = None
        while True:
            b0, b1 = self._read_exact(2)
            fin = b0 & 0x80
            opcode = b0 & 0x0F
            masked = b1 & 0x80
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read_exact(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if masked else None
            payload = self._read_exact(n) if n else b""
            if mask:
                payload = bytes(b ^ mask[i % 4]
                                for i, b in enumerate(payload))
            if opcode == 0x9:            # ping -> pong
                self.send_frame(0xA, payload)
                continue
            if opcode == 0x8:            # close
                raise ConnectionError("websocket closed by server")
            if opcode == 0xA:            # pong: ignore
                continue
            if opcode != 0x0:
                opcode0 = opcode
            frags.append(payload)
            if fin:
                return opcode0 or 0x1, b"".join(frags)


# ----------------------------------------------------------------------
# Request forwarding to the city's own bridge.
# ----------------------------------------------------------------------
def forward(ws: WSClient, frame: dict) -> None:
    rid = frame["id"]
    try:
        body = base64.b64decode(frame.get("body") or "")
        headers = {k: v for k, v in (frame.get("headers") or {}).items()
                   if k.lower() not in HOP}
        conn = http.client.HTTPConnection(LOCAL_HOST, LOCAL_PORT,
                                          timeout=180)
        conn.request(frame.get("method", "GET"), frame["path"],
                     body=body or None, headers=headers)
        resp = conn.getresponse()
        ws.send_json({"type": "head", "id": rid, "status": resp.status,
                      "headers": {k: v for k, v in resp.getheaders()
                                  if k.lower() not in HOP}})
        while True:
            # read1, not read: read(32768) waits until the buffer
            # fills or EOF, which would park SSE events (small,
            # frequent) in this client for minutes. read1 returns
            # whatever one underlying read produced.
            chunk = resp.read1(32768)
            if not chunk:
                break
            ws.send_json({"type": "data", "id": rid,
                          "data": base64.b64encode(chunk).decode()})
        conn.close()
        ws.send_json({"type": "end", "id": rid})
    except Exception as err:  # bridge down or broke mid-answer
        try:
            ws.send_json({"type": "error", "id": rid,
                          "error": f"{type(err).__name__}: {err}"})
        except Exception:
            pass


def heartbeat(ws: WSClient, stop: threading.Event) -> None:
    while not stop.wait(25):
        try:
            ws.send_json({"type": "ping"})
        except Exception:
            return


def session(token: str) -> None:
    ws = WSClient(WS_URL)
    ws.send_json({"auth": token})
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat, args=(ws, stop),
                            daemon=True)
    beat.start()
    print("Doorway: connected - the city is public at "
          "https://zerozen270-hermes-local-city.hf.space/world/",
          flush=True)
    try:
        while True:
            opcode, payload = ws.recv_msg()
            if opcode != 0x1:
                continue
            try:
                frame = json.loads(payload.decode())
            except ValueError:
                continue
            if frame.get("type") == "request":
                threading.Thread(target=forward, args=(ws, frame),
                                 daemon=True).start()
            # "hello" and anything else: nothing to do.
    finally:
        stop.set()


def main() -> int:
    token = load_token()
    if not token:
        print("Doorway: no token found ($HERMES_WORLD_TOKEN / "
              "~/.world_token). Cannot authenticate - exiting.")
        return 1
    delay = 3
    while True:
        try:
            session(token)
        except Exception as err:
            print(f"Doorway: connection lost ({err}); "
                  f"retrying in {delay}s...", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 30)


if __name__ == "__main__":
    sys.exit(main())
