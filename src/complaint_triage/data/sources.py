"""Emplacement des sources officielles du CFPB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

EXPORT_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ARCHIVE_BASE_URL = "https://files.consumerfinance.gov/f/documents/"


@dataclass(frozen=True)
class ArchiveFile:
    """Un fichier de l'archive des récits et la période de réception qu'il couvre."""

    name: str
    first_day: date
    last_day: date

    @property
    def url(self) -> str:
        return f"{ARCHIVE_BASE_URL}{self.name}.zip"

    def overlaps(self, start: date) -> bool:
        return self.last_day >= start


# Liste publiée sur la page FOIA du CFPB ; l'archive est figée au 14/08/2026.
ARCHIVE_FILES: tuple[ArchiveFile, ...] = (
    ArchiveFile(
        "CCDB_Export_1_December_2011_through_April_2018", date(2011, 12, 1), date(2018, 4, 30)
    ),
    ArchiveFile("CCDB_Export_2_May_2018_through_April_2021", date(2018, 5, 1), date(2021, 4, 30)),
    ArchiveFile(
        "CCDB_Export_3_May_2021_through_October_2022", date(2021, 5, 1), date(2022, 10, 31)
    ),
    ArchiveFile(
        "CCDB_Export_4_November_2022_through_August_2023", date(2022, 11, 1), date(2023, 8, 31)
    ),
    ArchiveFile(
        "CCDB_Export_5_September_2023_through_March_2024", date(2023, 9, 1), date(2024, 3, 31)
    ),
    ArchiveFile("CCDB_Export_6_April_2024_through_July_2024", date(2024, 4, 1), date(2024, 7, 31)),
    ArchiveFile(
        "CCDB_Export_7_August_2024_through_October_2024", date(2024, 8, 1), date(2024, 10, 31)
    ),
    ArchiveFile(
        "CCDB_Export_8_November_2024_through_December_2024", date(2024, 11, 1), date(2024, 12, 31)
    ),
    ArchiveFile(
        "CCDB_Export_9_January_2025_through_February_2025", date(2025, 1, 1), date(2025, 2, 28)
    ),
    ArchiveFile(
        "CCDB_Export_10_March_2025_through_April_2025", date(2025, 3, 1), date(2025, 4, 30)
    ),
    ArchiveFile("CCDB_Export_11_May_2025_through_June_2025", date(2025, 5, 1), date(2025, 6, 30)),
    ArchiveFile(
        "CCDB_Export_12_July_2025_through_August_2025", date(2025, 7, 1), date(2025, 8, 31)
    ),
    ArchiveFile(
        "CCDB_Export_13_September_2025_through_October_2025", date(2025, 9, 1), date(2025, 10, 31)
    ),
    ArchiveFile(
        "CCDB_Export_14_November_2025_through_December_2025", date(2025, 11, 1), date(2025, 12, 31)
    ),
    ArchiveFile(
        "CCDB_Export_15_January_2026_through_February_2026", date(2026, 1, 1), date(2026, 2, 28)
    ),
    ArchiveFile("CCDB_Export_16_March_2026", date(2026, 3, 1), date(2026, 3, 31)),
    ArchiveFile("CCDB_Export_17_April_2026", date(2026, 4, 1), date(2026, 4, 30)),
    ArchiveFile("CCDB_Export_18_May_2026", date(2026, 5, 1), date(2026, 5, 31)),
    ArchiveFile("CCDB_Export_19_June_2026", date(2026, 6, 1), date(2026, 6, 30)),
    ArchiveFile("CCDB_Export_20_July_2026", date(2026, 7, 1), date(2026, 7, 31)),
    ArchiveFile("CCDB_Export_21_August_2026", date(2026, 8, 1), date(2026, 8, 14)),
)


def archive_files_since(start: date) -> list[ArchiveFile]:
    return [f for f in ARCHIVE_FILES if f.overlaps(start)]
