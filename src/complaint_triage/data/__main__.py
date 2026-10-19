"""Construit le jeu de données : python -m complaint_triage.data [--refresh-export]."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from complaint_triage.data.pipeline import build_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument(
        "--refresh-export",
        action="store_true",
        help="retélécharge l'export du jour pour obtenir les issues les plus récentes",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args.data_dir, args.start, refresh_export=args.refresh_export)


if __name__ == "__main__":
    main()
