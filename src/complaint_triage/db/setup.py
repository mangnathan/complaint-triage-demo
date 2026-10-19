"""Mise en place de la base : comptes, base, schéma et droits, en tant que super-utilisateur.

Partagée par le script du poste (scripts/db/setup_database.py) et par la démo, qui démarre son
propre PostgreSQL : les deux appliquent exactement les mêmes comptes et les mêmes droits.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files

import psycopg
from psycopg import sql

from complaint_triage.db.roles import (
    DATABASE,
    OWNER,
    ROLES,
    SCHEMA,
    privilege_statements,
    role_statements,
)


def random_passwords() -> dict[str, str]:
    """Un mot de passe aléatoire de 32 caractères par compte du projet."""
    return {role: secrets.token_urlsafe(24) for role in ROLES}


@contextmanager
def admin_password(password: str) -> Iterator[None]:
    """Le mot de passe passe par PGPASSWORD et non par la chaîne de connexion : sous Windows,
    libpq prend l'octet 0xA0 pour un espace et couperait un mot de passe contenant « à ». La
    variable est propre au processus et effacée à la sortie."""
    os.environ["PGPASSWORD"] = password
    try:
        yield
    finally:
        os.environ.pop("PGPASSWORD", None)


def connect_as_admin(
    dbname: str, *, host: str = "localhost", port: int = 5432
) -> psycopg.Connection[tuple[object, ...]]:
    return psycopg.connect(host=host, port=port, user="postgres", dbname=dbname, autocommit=True)


def create_database(
    passwords: dict[str, str], *, host: str = "localhost", port: int = 5432
) -> None:
    """Crée ou met à jour les comptes, la base, le schéma et les droits."""
    with connect_as_admin("postgres", host=host, port=port) as conn:
        for role, password in passwords.items():
            exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
            conn.execute(role_statements(role, password, exists is not None))
        if (
            conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)).fetchone()
            is None
        ):
            conn.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(DATABASE), sql.Identifier(OWNER)
                )
            )

    schema_sql = files("complaint_triage.db").joinpath("schema.sql").read_text(encoding="utf-8")
    with connect_as_admin(DATABASE, host=host, port=port) as conn:
        conn.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}").format(
                sql.Identifier(SCHEMA), sql.Identifier(OWNER)
            )
        )
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(OWNER)))
        conn.execute(schema_sql)  # fichier du dépôt, sans aucun paramètre extérieur
        conn.execute("RESET ROLE")
        for statement in privilege_statements():
            conn.execute(statement)
