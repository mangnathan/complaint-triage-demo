"""Adresses de connexion, lues dans l'environnement ou le fichier .env, jamais dans le code."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

from complaint_triage.db.roles import AGENT_READ, AGENT_WRITE, DATABASE, OWNER

ENV_KEYS = {
    OWNER: "DATABASE_URL_OWNER",
    AGENT_READ: "DATABASE_URL_AGENT_READ",
    AGENT_WRITE: "DATABASE_URL_AGENT_WRITE",
}


def database_url(role: str, password: str, host: str = "localhost", port: int = 5432) -> str:
    return f"postgresql://{role}:{quote(password, safe='')}@{host}:{port}/{DATABASE}"


def url_for(role: str) -> str:
    key = ENV_KEYS[role]
    url = os.environ.get(key, "")
    if not url:
        raise RuntimeError(f"{key} n'est pas défini : lancer scripts/db/setup_database.py.")
    return url


def write_env_values(path: Path, values: dict[str, str]) -> None:
    """Remplace ou ajoute des variables dans .env, sans toucher aux autres lignes."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(values)
    updated = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    if remaining:
        updated += ["", "# Base PostgreSQL (écrit par scripts/db/setup_database.py)"]
        updated += [f"{key}={value}" for key, value in remaining.items()]
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
