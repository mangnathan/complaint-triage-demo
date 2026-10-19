"""Connexions de l'agent : un compte pour lire, un autre pour écrire, jamais le propriétaire."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from complaint_triage.db.roles import AGENT_READ, AGENT_WRITE
from complaint_triage.db.settings import url_for

Connection = psycopg.Connection[dict[str, Any]]


@dataclass(frozen=True)
class AgentDatabase:
    read_url: str
    write_url: str

    @classmethod
    def from_environment(cls) -> AgentDatabase:
        return cls(url_for(AGENT_READ), url_for(AGENT_WRITE))

    @contextmanager
    def reading(self) -> Iterator[Connection]:
        with psycopg.connect(self.read_url, row_factory=dict_row) as conn:
            yield conn

    @contextmanager
    def writing(self) -> Iterator[Connection]:
        """Une transaction : tout est validé à la sortie, ou tout est annulé en cas d'erreur."""
        with psycopg.connect(self.write_url, row_factory=dict_row) as conn, conn.transaction():
            yield conn
