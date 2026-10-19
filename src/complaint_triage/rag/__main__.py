"""Construit le corpus réglementaire : python -m complaint_triage.rag [--data-dir data]."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from complaint_triage.rag.ecfr import (
    CORPUS_DATE,
    CORPUS_PARTS,
    download_parts,
    parse_sections,
    part_url,
    read_xml_bytes,
)
from complaint_triage.rag.passages import Passage, passages_frame, section_passages

PASSAGES_FILE = Path("rag") / "passages.parquet"
MANIFEST_FILE = Path("rag") / "passages.manifest.json"


def build_corpus(data_dir: Path) -> Path:
    raw = download_parts(data_dir / "raw" / "ecfr")
    passages: list[Passage] = []
    sources = []
    for part in CORPUS_PARTS:
        sections = parse_sections(read_xml_bytes(raw[part.part]), part)
        passages.extend(p for section in sections for p in section_passages(section, part))
        sources.append(
            {
                "regulation": part.regulation,
                "url": part_url(part.part),
                "sections": len(sections),
                "sha256": hashlib.sha256(raw[part.part].read_bytes()).hexdigest(),
            }
        )
    frame = passages_frame(passages)
    out = data_dir / PASSAGES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    manifest = {
        "corpus_date": CORPUS_DATE,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "passages": len(frame),
        "sources": sources,
    }
    (data_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    out = build_corpus(args.data_dir)
    print(f"Corpus écrit : {out}")


if __name__ == "__main__":
    main()
