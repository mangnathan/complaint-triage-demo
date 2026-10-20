"""Base de la démo : PostgreSQL propre à l'image, mis en place avec le code du projet.

La base est préparée une fois, à la construction de l'image (prepare) : comptes, droits,
réclamations et dossiers fictifs. Au lancement du conteneur (start), il ne reste qu'à démarrer
PostgreSQL et à tirer de nouveaux mots de passe : un réveil de l'hébergeur ne recharge plus
rien. Le système de fichiers du conteneur repart de l'image à chaque lancement, donc la base
aussi : ce qu'un visiteur modifie ne dure pas.

Aucun mot de passe de la construction ne sert ensuite. L'administrateur n'est joignable que
par le socket local, pour l'utilisateur système du conteneur (authentification peer) ; les
comptes du projet, par mot de passe et depuis la machine elle-même seulement.
"""

from __future__ import annotations

import getpass
import os
import secrets
import subprocess
from pathlib import Path

import pandas as pd
import psycopg

from complaint_triage.agent.demo import demo_rows, reset_fixtures
from complaint_triage.agent.evaluation_data import evaluation_rows
from complaint_triage.db.load import load_queue
from complaint_triage.db.roles import OWNER, ROLES, role_statements
from complaint_triage.db.settings import ENV_KEYS, database_url
from complaint_triage.db.setup import admin_password, create_database, random_passwords

HOST, PORT = "127.0.0.1", 5432
SOCKET_DIR = "/tmp"  # noqa: S108 - socket de la base, dans le conteneur de la démo seulement
QUEUE_FILE = "demo-complaints.parquet"
# Mémoire réduite : la démo tient dans les 512 Mo de l'offre gratuite de l'hébergeur.
SERVER_OPTIONS = (
    f"-c listen_addresses={HOST} -c port={PORT} -c unix_socket_directories={SOCKET_DIR}"
    " -c shared_buffers=32MB -c max_connections=20"
)
HBA = """# Administrateur : socket local, utilisateur système du conteneur seulement.
local all postgres peer map=demo
# Comptes du projet : mot de passe, depuis la machine elle-même.
host all {roles} 127.0.0.1/32 scram-sha-256
"""


def postgres_binary(name: str) -> str:
    """Programme PostgreSQL installé par le paquet Debian (la version la plus récente)."""
    candidates = sorted(Path("/usr/lib/postgresql").glob(f"*/bin/{name}"))
    if not candidates:
        raise RuntimeError(f"{name} introuvable : PostgreSQL n'est pas installé dans l'image.")
    return str(candidates[-1])


def pg_ctl(pg_dir: Path, action: str) -> None:
    subprocess.run(  # noqa: S603
        [
            postgres_binary("pg_ctl"),
            "--pgdata",
            str(pg_dir),
            "--options",
            SERVER_OPTIONS,
            "--log",
            str(pg_dir / "postgres.log"),
            "--wait",
            action,
        ],
        check=True,
        capture_output=True,
    )


def initdb(pg_dir: Path, password: str) -> None:
    pg_dir.mkdir(parents=True, exist_ok=True)
    password_file = pg_dir.parent / "pg-admin-password"
    password_file.write_text(password, encoding="utf-8")
    password_file.chmod(0o600)
    try:
        subprocess.run(  # noqa: S603
            [
                postgres_binary("initdb"),
                "--pgdata",
                str(pg_dir),
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
        password_file.unlink(missing_ok=True)


def prepare(data_dir: Path, pg_dir: Path) -> dict[str, int]:
    """Construction de l'image : base complète, puis arrêt. Les mots de passe sont jetés."""
    admin = secrets.token_urlsafe(24)
    initdb(pg_dir, admin)
    pg_ctl(pg_dir, "start")
    passwords = random_passwords()
    with admin_password(admin):
        create_database(passwords, host=HOST, port=PORT)
    owner = database_url(OWNER, passwords[OWNER], host=HOST, port=PORT)
    queue_path = data_dir / QUEUE_FILE
    loaded = load_queue(pd.read_parquet(queue_path), owner) if queue_path.exists() else 0
    fictitious = demo_rows() + evaluation_rows()
    reset_fixtures(owner, fictitious)
    pg_ctl(pg_dir, "stop")
    (pg_dir / "pg_ident.conf").write_text(f"demo {getpass.getuser()} postgres\n", encoding="utf-8")
    (pg_dir / "pg_hba.conf").write_text(HBA.format(roles=",".join(ROLES)), encoding="utf-8")
    return {"reclamations_reelles": loaded, "dossiers_fictifs": len(fictitious)}


def start(pg_dir: Path) -> None:
    """Lancement du conteneur : démarre la base et donne aux comptes des mots de passe neufs."""
    pg_ctl(pg_dir, "start")
    passwords = random_passwords()
    with psycopg.connect(
        host=SOCKET_DIR, port=PORT, user="postgres", dbname="postgres", autocommit=True
    ) as conn:
        for role, password in passwords.items():
            conn.execute(role_statements(role, password, exists=True))
    for role, password in passwords.items():
        os.environ[ENV_KEYS[role]] = database_url(role, password, host=HOST, port=PORT)
