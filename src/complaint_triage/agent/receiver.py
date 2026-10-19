"""Récepteur local d'alertes : écoute sur la machine seule et enregistre chaque alerte reçue.

Rien ne sort du poste. Il remplace un salon Discord ou Slack pendant le développement et les
tests ; l'agent envoie la même requête dans les deux cas.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 10_000
DEFAULT_PORT = 8765


def make_handler(log_file: Path) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()

    class AlertHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (nom imposé par http.server)
            length = int(self.headers.get("Content-Length", "0"))
            if self.path != "/alerts" or not 0 < length <= MAX_BODY_BYTES:
                self.send_error(400, "Requête refusée")
                return
            try:
                alert = json.loads(self.rfile.read(length))
            except json.JSONDecodeError:
                self.send_error(400, "JSON illisible")
                return
            record = {"recue": datetime.now(UTC).isoformat(timespec="seconds"), "alerte": alert}
            with lock, log_file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return  # pas de journal sur la console : les alertes sont dans le fichier

    return AlertHandler


def start_receiver(log_file: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Démarre le récepteur sur 127.0.0.1 dans un fil d'exécution ; port 0 = port libre."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(log_file))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
