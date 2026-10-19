"""Accès au modèle de langage, avec des garde-fous qui empêchent tout dépassement du quota gratuit.

Trois protections, dans cet ordre :
1. le cache : une requête déjà faite est relue sur disque, sans appel ;
2. le compteur : au-delà d'un plafond par minute et par jour, fixé sous les limites gratuites de
   Google, le code refuse l'appel avant de contacter l'API ;
3. une réponse 429 de Google (limite atteinte) arrête tout, sans nouvel essai.

Le projet Google utilisé n'a aucune facturation liée : même en cas d'erreur, rien ne peut être
facturé. Ces garde-fous évitent en plus de consommer le quota pour rien.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Limites gratuites de gemini-3.5-flash-lite relevées le 18 octobre 2026 : 15 requêtes par
# minute, 500 par jour. Les plafonds du code restent nettement en dessous.
DAILY_CAP = 300
MINUTE_CAP = 10
# Google remet le compteur quotidien à zéro à minuit, heure du Pacifique.
QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")


class QuotaExceededError(RuntimeError):
    """Plafond atteint : l'appel n'a pas été envoyé, ou Google a répondu 429."""


class LanguageModel(Protocol):
    name: str

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str: ...


@dataclass
class QuotaGuard:
    """Compteur d'appels enregistré sur disque, pour qu'il survive d'une exécution à l'autre."""

    state_file: Path
    daily_cap: int = DAILY_CAP
    minute_cap: int = MINUTE_CAP
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    def _load(self) -> dict[str, Any]:
        if self.state_file.exists():
            state: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
            return state
        return {"day": None, "count": 0, "recent": []}

    def _day(self, now: float) -> str:
        return datetime.fromtimestamp(now, UTC).astimezone(QUOTA_TIMEZONE).date().isoformat()

    def acquire(self) -> None:
        """Réserve un appel, ou lève QuotaExceededError si le plafond du jour est atteint."""
        state = self._load()
        now = self.clock()
        if state["day"] != self._day(now):
            state = {"day": self._day(now), "count": 0, "recent": []}
        if state["count"] >= self.daily_cap:
            raise QuotaExceededError(
                f"Plafond du jour atteint ({self.daily_cap} appels) : aucun appel envoyé."
            )
        recent = [t for t in state["recent"] if now - t < 60]
        if len(recent) >= self.minute_cap:
            # Attendre ne coûte rien : on laisse passer la minute plutôt que de dépasser.
            self.sleep(60 - (now - min(recent)) + 0.5)
            now = self.clock()
            recent = [t for t in recent if now - t < 60]
        state["count"] += 1
        state["recent"] = [*recent, now]
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def used_today(self) -> int:
        state = self._load()
        return int(state["count"]) if state["day"] == self._day(self.clock()) else 0


class CachedModel:
    """Relit sur disque les réponses déjà obtenues : relancer une évaluation ne coûte rien."""

    def __init__(self, model: LanguageModel, cache_dir: Path) -> None:
        self.model = model
        self.name = model.name
        self.cache_dir = cache_dir

    def _path(self, system: str, prompt: str, schema: dict[str, Any]) -> Path:
        key = json.dumps([self.name, system, prompt, schema], sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def is_cached(self, system: str, prompt: str, schema: dict[str, Any]) -> bool:
        """Vrai si la réponse est déjà sur disque : sert à estimer les appels avant de lancer."""
        return self._path(system, prompt, schema).exists()

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        path = self._path(system, prompt, schema)
        if path.exists():
            cached: str = json.loads(path.read_text(encoding="utf-8"))["response"]
            return cached
        response = self.model.generate(system, prompt, schema)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.name, "response": response}), encoding="utf-8")
        return response


Poster = Callable[[urllib.request.Request, float], bytes]


def _post(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body: bytes = response.read()
        return body


class GeminiModel:
    """Appel direct à l'API Gemini. La clé passe dans un en-tête, jamais dans l'adresse."""

    def __init__(
        self,
        guard: QuotaGuard,
        *,
        model: str | None = None,
        api_key: str | None = None,
        post: Poster = _post,
        timeout: float = 60.0,
    ) -> None:
        self.name = model or os.environ.get("GEMINI_MODEL", "")
        self._key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.name or not self._key:
            raise ValueError("GEMINI_MODEL et GEMINI_API_KEY doivent être définis dans .env.")
        self.guard = guard
        self.post = post
        self.timeout = timeout

    def generate(self, system: str, prompt: str, schema: dict[str, Any]) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        request = urllib.request.Request(  # noqa: S310
            API_URL.format(model=self.name),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
            method="POST",
        )
        self.guard.acquire()
        try:
            raw = self.post(request, self.timeout)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                raise QuotaExceededError(
                    "Google a répondu 429 : limite atteinte, arrêt."
                ) from error
            raise
        payload = json.loads(raw)
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""  # réponse bloquée ou vide : traitée comme un refus par l'appelant
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)


def load_dotenv(path: Path) -> None:
    """Charge les variables d'un fichier .env sans écraser celles déjà définies."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())
