"""Browser viewer: serves web/index.html and pushes subtitles with Server-Sent Events (SSE).

SSE is a long-lived HTTP response where the server writes `data: <json>\n\n` per message;
browsers reconnect automatically and resend the last id they saw (Last-Event-ID), so a
phone that drops off Wi-Fi for a moment gets the missed lines on reconnect.
"""

import json
import logging
import queue
import socket
import threading
from collections import deque
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..types import Subtitle

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"  # repo/web (src/auto_translate/output → repo)
BACKLOG = 50  # lines kept for (re)connecting clients
PING_S = 15.0  # keep-alive comment so proxies / phones don't drop an idle connection


@dataclass
class _Event:
    id: int
    data: str  # JSON


class WebOutput:
    def __init__(self, host: str, port: int) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._backlog: deque[_Event] = deque(maxlen=BACKLOG)
        self._clients: set[queue.Queue[_Event]] = set()

        hub = self  # the handler class below needs to reach this object

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path.split("?")[0] in ("/", "/index.html"):
                    self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                elif self.path == "/events":
                    hub._serve_events(self)
                else:
                    self.send_error(404)

            def _send_file(self, path: Path, content_type: str) -> None:
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass  # silence the default per-request access log

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="web", daemon=True).start()
        log.info("web viewer: http://%s:%d/  (this PC: http://127.0.0.1:%d/)",
                 _lan_ip() if host == "0.0.0.0" else host, port, port)

    def show(self, sub: Subtitle) -> None:
        payload = {k: v for k, v in asdict(sub).items() if k != "timings"}
        with self._lock:
            event = _Event(self._next_id, json.dumps(payload, ensure_ascii=False))
            self._next_id += 1
            self._backlog.append(event)
            for client in self._clients:
                client.put(event)

    def drain(self) -> None:
        pass

    def close(self) -> None:
        self._server.shutdown()

    def _serve_events(self, handler: BaseHTTPRequestHandler) -> None:
        """Runs on the server's per-connection thread for as long as the browser is connected."""
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()

        last_id = int(handler.headers.get("Last-Event-ID") or 0)
        client: queue.Queue[_Event] = queue.Queue()
        with self._lock:
            # New viewer: the last few lines. Reconnecting viewer: whatever it missed.
            missed = [e for e in self._backlog if e.id > last_id]
            for event in missed[-BACKLOG if last_id else -3:]:
                client.put(event)
            self._clients.add(client)
        try:
            while True:
                try:
                    event = client.get(timeout=PING_S)
                    msg = f"id: {event.id}\ndata: {event.data}\n\n"
                except queue.Empty:
                    msg = ": ping\n\n"
                handler.wfile.write(msg.encode("utf-8"))
                handler.wfile.flush()
        except (ConnectionError, OSError):
            pass  # browser closed the tab / lost Wi-Fi
        finally:
            with self._lock:
                self._clients.discard(client)


def _lan_ip() -> str:
    """This PC's LAN address (UDP 'connect' sends no packets; it just picks the outgoing interface)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.0.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
