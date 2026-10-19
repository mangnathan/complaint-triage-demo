"""Traçabilité des expériences : version du code et des données associée à chaque résultat."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

TRACKING_DB = Path("mlflow.db")


def tracking_uri(db_path: Path = TRACKING_DB) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def git_commit() -> str:
    """Commit courant, résolu avec le chemin complet de git (un exécutable homonyme placé plus tôt
    dans le PATH ne peut pas être lancé à sa place)."""
    git = shutil.which("git")
    if git is None:
        return "inconnu"
    result = subprocess.run(  # noqa: S603  commande fixe, sans entrée extérieure
        [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "inconnu"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
