"""Assemblage du jeu de données : export structuré du CFPB et récits de l'archive."""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from complaint_triage.data.download import Opener, download, open_https
from complaint_triage.data.schema import EXPORT_COLUMNS, validate_complaints
from complaint_triage.data.sources import EXPORT_URL, archive_files_since
from complaint_triage.data.taxonomy import RAW_PRODUCTS_TO_READ, SCOPE_PRODUCTS, harmonize_product

logger = logging.getLogger(__name__)

NARRATIVE_COLUMN = "Consumer complaint narrative"
DATE_COLUMNS = ("date_received", "date_sent_to_company")
_CHUNK_ROWS = 250_000


@dataclass(frozen=True)
class SourceFile:
    url: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    built_at: str
    # Date de l'export utilisé, distincte de built_at : l'export est réutilisé depuis le cache,
    # et c'est elle qui détermine quelles issues étaient connues (maturité des labels).
    export_retrieved_at: str
    start: str
    rows: int
    rows_with_narrative: int
    rows_dropped_without_issue: int
    sources: list[SourceFile]


def _read_zipped_csv(path: Path, columns: list[str]) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(path) as archive:
        (member,) = [name for name in archive.namelist() if name.endswith(".csv")]
        with archive.open(member) as handle:
            yield from pd.read_csv(handle, usecols=columns, dtype=str, chunksize=_CHUNK_ROWS)


def read_export(path: Path, start: date) -> pd.DataFrame:
    """Lit l'export complet et ne garde que le périmètre, à partir de `start`."""
    kept = []
    for chunk in _read_zipped_csv(path, list(EXPORT_COLUMNS)):
        # Les dates sont au format AAAA-MM-JJ : la comparaison de chaînes suffit.
        in_scope = chunk["Product"].isin(RAW_PRODUCTS_TO_READ) & (
            chunk["Date received"] >= start.isoformat()
        )
        kept.append(chunk.loc[in_scope])
    df = pd.concat(kept, ignore_index=True).rename(columns=EXPORT_COLUMNS)

    df["product"] = harmonize_product(df["product"], df["sub_product"])
    df = df.loc[df["product"].isin(SCOPE_PRODUCTS)].reset_index(drop=True)
    df["complaint_id"] = df["complaint_id"].astype("int64")
    for column in DATE_COLUMNS:
        df[column] = pd.to_datetime(df[column], format="%Y-%m-%d").astype("datetime64[ns]")
    return df


def drop_rows_without_issue(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Écarte les fiches sans motif : le motif est indispensable à la priorisation.

    Observé dans l'export d'octobre 2026 : 5 réclamations transmises par une autre
    administration (Referral) entre mars et mai 2024, sans motif ni sous-produit.
    """
    missing = df["issue"].isna()
    if missing.any():
        logger.warning("%d réclamations écartées car sans motif", missing.sum())
    return df.loc[~missing].reset_index(drop=True), int(missing.sum())


def read_narratives(paths: Iterable[Path], complaint_ids: set[int]) -> pd.DataFrame:
    """Lit les récits de l'archive, uniquement pour les réclamations du périmètre."""
    kept = []
    for path in paths:
        for chunk in _read_zipped_csv(path, ["Complaint ID", NARRATIVE_COLUMN]):
            chunk = chunk.dropna(subset=[NARRATIVE_COLUMN])
            ids = chunk["Complaint ID"].astype("int64")
            kept.append(
                pd.DataFrame({"complaint_id": ids, "narrative": chunk[NARRATIVE_COLUMN]}).loc[
                    ids.isin(complaint_ids)
                ]
            )
    if not kept:
        # Aucun fichier d'archive ne couvre la période (début après le 14/08/2026).
        return pd.DataFrame(
            {"complaint_id": pd.Series(dtype="int64"), "narrative": pd.Series(dtype="str")}
        )
    narratives = pd.concat(kept, ignore_index=True)

    duplicated = narratives["complaint_id"].duplicated(keep="last")
    if duplicated.any():
        logger.warning("%d récits présents dans plusieurs fichiers d'archive", duplicated.sum())
    return narratives.loc[~duplicated].reset_index(drop=True)


def attach_narratives(complaints: pd.DataFrame, narratives: pd.DataFrame) -> pd.DataFrame:
    # validate="one_to_one" fait échouer la jointure si un identifiant apparaît deux fois.
    return complaints.merge(narratives, on="complaint_id", how="left", validate="one_to_one")


def _describe(url: str, path: Path) -> SourceFile:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return SourceFile(url, path.as_posix(), path.stat().st_size, digest.hexdigest())


def load_interim(data_dir: Path) -> tuple[pd.DataFrame, date]:
    """Relit le jeu de données construit et la date de l'export dont il est issu."""
    path = data_dir / "interim" / "complaints.parquet"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    export_date = datetime.fromisoformat(manifest["export_retrieved_at"]).date()
    return pd.read_parquet(path), export_date


def build_dataset(
    data_dir: Path,
    start: date,
    *,
    refresh_export: bool = False,
    opener: Opener = open_https,
) -> Path:
    """Télécharge les sources, assemble, contrôle et écrit `interim/complaints.parquet`."""
    raw_dir = data_dir / "raw"
    export_path = raw_dir / "export" / "complaints.csv.zip"
    if refresh_export and export_path.exists():
        # L'export change chaque jour : les issues des dossiers récents évoluent. Le nouveau
        # fichier remplace l'ancien seulement s'il est complet, sinon le cache reste utilisable.
        fresh = export_path.with_name("complaints.fresh.csv.zip")
        fresh.unlink(missing_ok=True)
        download(EXPORT_URL, fresh, opener=opener).replace(export_path)

    sources = [(EXPORT_URL, download(EXPORT_URL, export_path, opener=opener))]
    for archive_file in archive_files_since(start):
        path = raw_dir / "archive" / f"{archive_file.name}.zip"
        sources.append((archive_file.url, download(archive_file.url, path, opener=opener)))

    logger.info("Lecture de l'export")
    complaints, dropped_without_issue = drop_rows_without_issue(read_export(export_path, start))
    logger.info("Lecture des récits pour %d réclamations", len(complaints))
    narratives = read_narratives([path for _, path in sources[1:]], set(complaints["complaint_id"]))
    dataset = validate_complaints(attach_narratives(complaints, narratives))

    out_path = data_dir / "interim" / "complaints.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path, index=False)

    manifest = Manifest(
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        export_retrieved_at=datetime.fromtimestamp(export_path.stat().st_mtime, UTC).isoformat(
            timespec="seconds"
        ),
        start=start.isoformat(),
        rows=len(dataset),
        rows_with_narrative=int(dataset["narrative"].notna().sum()),
        rows_dropped_without_issue=dropped_without_issue,
        sources=[_describe(url, path) for url, path in sources],
    )
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    logger.info("%d réclamations écrites dans %s", manifest.rows, out_path)
    return out_path
