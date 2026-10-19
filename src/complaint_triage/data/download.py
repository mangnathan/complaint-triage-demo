"""Téléchargement idempotent et atomique des fichiers sources."""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import BinaryIO, cast

# Une fonction (url, timeout) -> flux binaire. Remplaçable dans les tests, sans accès réseau.
Opener = Callable[[str, float], AbstractContextManager[BinaryIO]]


class DownloadError(RuntimeError):
    pass


class TruncatedDownloadError(ConnectionError):
    """Le serveur a fermé la connexion avant d'envoyer la taille annoncée."""


_RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError)


def open_https(url: str, timeout: float) -> AbstractContextManager[BinaryIO]:
    if not url.startswith("https://"):
        raise ValueError(f"URL refusée, seul https est accepté : {url}")
    return cast(AbstractContextManager[BinaryIO], urllib.request.urlopen(url, timeout=timeout))  # noqa: S310


def _announced_size(response: BinaryIO) -> int | None:
    headers = getattr(response, "headers", None)
    length = headers.get("Content-Length") if headers is not None else None
    return int(length) if length is not None else None


def _is_client_error(error: Exception) -> bool:
    return isinstance(error, urllib.error.HTTPError) and 400 <= error.code < 500


def download(
    url: str,
    dest: Path,
    *,
    opener: Opener = open_https,
    retries: int = 3,
    timeout: float = 120.0,
    backoff_seconds: float = 2.0,
) -> Path:
    """Télécharge `url` vers `dest`, sauf si le fichier existe déjà.

    Le contenu est d'abord écrit dans un fichier `.part`, renommé seulement une fois complet et
    de la taille annoncée par le serveur : un fichier tronqué n'est jamais pris pour valide.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with opener(url, timeout) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
                expected = _announced_size(response)
            received = partial.stat().st_size
            if expected is not None and received != expected:
                raise TruncatedDownloadError(f"{received} octets reçus sur {expected} annoncés")
            partial.replace(dest)
            return dest
        except _RETRYABLE as error:
            partial.unlink(missing_ok=True)
            if _is_client_error(error):
                # Une 404 ou une 403 ne se corrige pas en réessayant.
                raise DownloadError(f"Fichier introuvable ou refusé : {url}") from error
            last_error = error
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    raise DownloadError(f"Échec du téléchargement après {retries} essais : {url}") from last_error
