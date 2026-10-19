"""Charge la file de réclamations dans la base : python -m complaint_triage.db [--data-dir data]."""

from __future__ import annotations

import argparse
from pathlib import Path

from complaint_triage.db.load import build_queue, load_queue
from complaint_triage.db.roles import OWNER
from complaint_triage.db.settings import url_for
from complaint_triage.rag.llm import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    load_dotenv(Path(".env"))
    queue = build_queue(args.data_dir)
    count = load_queue(queue, url_for(OWNER))
    print(f"{count} réclamations chargées dans la file (reçues depuis le 1er janvier 2026).")


if __name__ == "__main__":
    main()
