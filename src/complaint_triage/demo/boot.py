"""Démarrage de la démo : PostgreSQL propre à l'image, mis en place avec le code du projet.

Au lancement du conteneur, une base neuve est créée dans un dossier temporaire. Elle n'écoute
que sur la machine elle-même, avec un mot de passe administrateur tiré au hasard et jamais
écrit ailleurs qu'en mémoire. Les comptes et les droits sont ceux du projet. La base repart à
zéro à chaque redémarrage : ce qu'un visiteur modifie ne dure pas.
"""

from __future__ import annotations

import os
import secrets
import subprocess
from pathlib import Path

import pandas as pd

from complaint_triage.agent.demo import demo_rows, reset_fixtures
from complaint_triage.agent.evaluation_data import evaluation_rows
from complaint_triage.db.load import load_queue
from complaint_triage.db.roles import OWNER
from complaint_triage.db.settings import ENV_KEYS, database_url, url_for
from complaint_triage.db.setup import admin_password, create_database, random_passwords

HOST, PORT = "127.0.0.1", 5432
QUEUE_FILE = "demo-complaints.parquet"


def postgres_binary(name: str) -> str:
    """Programme PostgreSQL installé par le paquet Debian (la version la plus récente)."""
    candidates = sorted(Path("/usr/lib/postgresql").glob(f"*/bin/{name}"))
    if not candidates:
        raise RuntimeError(f"{name} introuvable : PostgreSQL n'est pas installé dans l'image.")
    return str(candidates[-1])


def start_postgres(data_dir: Path) -> str:
    """Initialise et démarre une base neuve ; renvoie le mot de passe administrateur."""
    password = secrets.token_urlsafe(24)
    data_dir.mkdir(parents=True, exist_ok=True)
    password_file = data_dir.parent / "pg-admin-password"
    password_file.write_text(password, encoding="utf-8")
    password_file.chmod(0o600)
    try:
        subprocess.run(  # noqa: S603
            [
                postgres_binary("initdb"),
                "--pgdata",
                str(data_dir),
                "--username",
                "postgres",
                "--auth",
                "scram-sha-256",
                "--pwfile",
                str(password_file),
                "--encoding",
                "UTF8",
                "--no-locale",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        password_file.unlink(missing_ok=True)  # le mot de passe ne reste que dans la mémoire
    # Mémoire réduite : la démo tient dans les 512 Mo de l'offre gratuite de l'hébergeur.
    options = (
        f"-c listen_addresses={HOST} -c port={PORT} -c unix_socket_directories=/tmp"
        " -c shared_buffers=32MB -c max_connections=20"
    )
    subprocess.run(  # noqa: S603
        [
            postgres_binary("pg_ctl"),
            "--pgdata",
            str(data_dir),
            "--options",
            options,
            "--log",
            str(data_dir / "postgres.log"),
            "--wait",
            "start",
        ],
        check=True,
        capture_output=True,
    )
    return password


def boot(data_dir: Path, pg_dir: Path) -> dict[str, int]:
    """Base prête pour la démo ; les adresses des comptes passent dans l'environnement."""
    password = start_postgres(pg_dir)
    passwords = random_passwords()
    with admin_password(password):
        create_database(passwords, host=HOST, port=PORT)
    for role, role_password in passwords.items():
        os.environ[ENV_KEYS[role]] = database_url(role, role_password, host=HOST, port=PORT)

    owner = url_for(OWNER)
    queue_path = data_dir / QUEUE_FILE
    loaded = load_queue(pd.read_parquet(queue_path), owner) if queue_path.exists() else 0
    fictitious = demo_rows() + evaluation_rows()
    reset_fixtures(owner, fictitious)
    return {"reclamations_reelles": loaded, "dossiers_fictifs": len(fictitious)}
