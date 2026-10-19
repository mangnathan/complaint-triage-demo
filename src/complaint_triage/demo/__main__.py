"""Lance la démo : python -m complaint_triage.demo (dans l'image de démonstration).

Démarre la base, puis remplace le processus par Streamlit, qui hérite des adresses de connexion
placées dans l'environnement par le démarrage.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

from complaint_triage.demo.boot import boot

APP_PORT = os.environ.get("PORT", "7860")


def main() -> None:
    data_dir = Path(os.environ.get("DEMO_DATA_DIR", "/app/data"))
    # Dossier temporaire unique et privé (droits 700), impossible à deviner pour un autre
    # processus de la machine.
    pg_dir = Path(tempfile.mkdtemp(prefix="demo-pg-")) / "data"
    counts = boot(data_dir, pg_dir)
    print(
        f"Base de démonstration prête : {counts['reclamations_reelles']} réclamations réelles, "
        f"{counts['dossiers_fictifs']} dossiers fictifs.",
        flush=True,
    )
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
