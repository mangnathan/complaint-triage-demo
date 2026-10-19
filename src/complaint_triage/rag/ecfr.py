"""Textes réglementaires publics de l'eCFR : sélection, téléchargement et lecture du XML.

Le corpus réunit les règles fédérales américaines qui encadrent les deux produits du projet :
Regulation E (erreurs sur les transferts électroniques, donc les comptes courants), Regulation Z
(contestations et règles propres aux cartes de crédit) et Regulation DD (comptes d'épargne). Le
texte est figé à une date inscrite ici : chaque citation reste vérifiable, et une mise à jour du
corpus est une décision explicite.
"""

from __future__ import annotations

import gzip
import urllib.request
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast
from xml.etree.ElementTree import (
    Element,  # seulement pour le typage ; la lecture passe par defusedxml
)

from defusedxml import ElementTree

from complaint_triage.data.download import Opener, download

ECFR_TITLE = 12
CORPUS_DATE = "2026-10-15"
API_BASE_URL = "https://www.ecfr.gov/api/versioner/v1/full"
READER_BASE_URL = "https://www.ecfr.gov/on"
_GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class RegulationPart:
    part: int
    regulation: str
    # Numéros de sections retenus, bornes comprises (1005.1 à 1005.20 -> (1, 20)).
    section_ranges: tuple[tuple[int, int], ...]

    def keeps(self, section_number: str) -> bool:
        part, _, rest = section_number.partition(".")
        if int(part) != self.part or not rest.isdigit():
            return False
        return any(low <= int(rest) <= high for low, high in self.section_ranges)


CORPUS_PARTS: tuple[RegulationPart, ...] = (
    # Sous-partie A : transferts électroniques de fonds, dont la résolution des erreurs (1005.11).
    RegulationPart(1005, "Regulation E", ((1, 20),)),
    # Sous-parties B (crédit renouvelable, dont les contestations, 1026.13) et G (cartes de crédit).
    RegulationPart(1026, "Regulation Z", ((5, 16), (51, 61))),
    # Toute la Regulation DD : information sur les comptes de dépôt.
    RegulationPart(1030, "Regulation DD", ((1, 11),)),
)


@dataclass(frozen=True)
class Section:
    number: str
    heading: str
    paragraphs: tuple[str, ...]


def part_url(part: int, on_date: str = CORPUS_DATE) -> str:
    return f"{API_BASE_URL}/{on_date}/title-{ECFR_TITLE}.xml?part={part}"


def reader_url(
    section_number: str, paragraph: str | None = None, on_date: str = CORPUS_DATE
) -> str:
    """Adresse de la section sur le site public de l'eCFR, ouverte au paragraphe cité."""
    url = f"{READER_BASE_URL}/{on_date}/title-{ECFR_TITLE}/section-{section_number}"
    return f"{url}#p-{section_number}{paragraph}" if paragraph else url


def open_ecfr(url: str, timeout: float) -> AbstractContextManager[BinaryIO]:
    """L'API de l'eCFR refuse les requêtes qui n'acceptent pas une réponse compressée."""
    if not url.startswith("https://"):
        raise ValueError(f"URL refusée, seul https est accepté : {url}")
    request = urllib.request.Request(  # noqa: S310
        url, headers={"Accept-Encoding": "gzip", "User-Agent": "complaint-triage"}
    )
    return cast(AbstractContextManager[BinaryIO], urllib.request.urlopen(request, timeout=timeout))  # noqa: S310


def download_parts(
    raw_dir: Path, *, on_date: str = CORPUS_DATE, opener: Opener = open_ecfr
) -> dict[int, Path]:
    """Télécharge chaque partie une fois, telle que reçue (souvent compressée)."""
    return {
        p.part: download(
            part_url(p.part, on_date), raw_dir / on_date / f"part-{p.part}.xml.raw", opener=opener
        )
        for p in CORPUS_PARTS
    }


def read_xml_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw.startswith(_GZIP_MAGIC) else raw


def _text(element: Element) -> str:
    return " ".join("".join(element.itertext()).split())


def parse_sections(xml: bytes, part: RegulationPart) -> list[Section]:
    """Sections retenues de la partie, avec leurs paragraphes dans l'ordre du texte."""
    root = ElementTree.fromstring(xml)
    sections = []
    for div in root.iter("DIV8"):
        number = div.get("N", "")
        if div.get("TYPE") != "SECTION" or not part.keeps(number):
            continue
        head = div.find("HEAD")
        paragraphs = tuple(
            text for element in div.iter() if element.tag in ("P", "FP") if (text := _text(element))
        )
        sections.append(Section(number, _text(head) if head is not None else number, paragraphs))
    return sections
