"""Lance la démo : python -m complaint_triage.demo (dans l'image de démonstration).

Avec --prepare, à la construction de l'image : prépare la base, puis s'arrête. Sans option :
démarre la base, puis remplace le processus par Streamlit, qui hérite des adresses de connexion
placées dans l'environnement.
"""

from __future__ import annotations

import argparse
import os
import shutil
from importlib.resources import files
from pathlib import Path

import psycopg

from complaint_triage.db.roles import OWNER
from complaint_triage.db.settings import url_for
from complaint_triage.demo.boot import prepare, start

APP_PORT = os.environ.get("PORT", "7860")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="préparer la base et s'arrêter")
    args = parser.parse_args()
    data_dir = Path(os.environ.get("DEMO_DATA_DIR", "/app/data"))
    pg_dir = Path(os.environ.get("DEMO_PG_DIR", "/app/pgdata"))
    if args.prepare:
        counts = prepare(data_dir, pg_dir)
        print(
            f"Base préparée : {counts['reclamations_reelles']} réclamations réelles, "
            f"{counts['dossiers_fictifs']} dossiers fictifs."
        )
        return

    start(pg_dir)
    with psycopg.connect(url_for(OWNER)) as conn:
        row = conn.execute("SELECT count(*) FROM triage.complaints").fetchone()
    print(f"Base de démonstration prête : {row[0] if row else 0} dossiers.", flush=True)
    app = str(files("complaint_triage.demo").joinpath("app.py"))
    streamlit = shutil.which("streamlit")
    if streamlit is None:
        raise SystemExit("streamlit introuvable : installer le groupe de dépendances demo.")
    os.execv(  # noqa: S606
        streamlit,
        [
            streamlit,
            "run",
            app,
            "--server.port",
            APP_PORT,
            "--server.address",
            # Dans le conteneur, l'interface doit être joignable par le proxy de l'hébergeur ;
            # la base, elle, n'écoute que sur 127.0.0.1.
            "0.0.0.0",  # noqa: S104
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
    )


if __name__ == "__main__":
    main()
