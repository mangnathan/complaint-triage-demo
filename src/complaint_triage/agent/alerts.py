"""Envoi d'une alerte à un webhook (récepteur local, Discord ou Slack plus tard)."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

Poster = Callable[[urllib.request.Request, float], int]


def check_webhook_url(url: str) -> str:
    """https partout, http seulement vers la machine elle-même (récepteur local)."""
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return url
    if parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
        return url
    raise ValueError(
        "Adresse de webhook refusée : https, ou http vers la machine locale seulement."
    )


def _post(request: urllib.request.Request, timeout: float) -> int:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        status: int = response.status
        return status


def send_alert(url: str, payload: dict[str, Any], *, post: Poster = _post) -> int:
    request = urllib.request.Request(  # noqa: S310
        check_webhook_url(url),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return post(request, 5.0)
