"""Comptes de la base, selon le principe du moindre privilège.

- ct_owner crée les tables et charge les données ;
- ct_agent_read ne peut que lire, et ses transactions sont en lecture seule par défaut ;
- ct_agent_write ne peut modifier que le statut d'un dossier et écrire dans le journal.

Même si l'agent se trompait, il ne pourrait ni supprimer ni modifier autre chose qu'un statut.
"""

from __future__ import annotations

from psycopg import sql

DATABASE = "complaint_triage"
SCHEMA = "triage"
OWNER = "ct_owner"
AGENT_READ = "ct_agent_read"
AGENT_WRITE = "ct_agent_write"
ROLES = (OWNER, AGENT_READ, AGENT_WRITE)
# Une requête de l'agent ne doit jamais bloquer la base.
AGENT_STATEMENT_TIMEOUT = "5s"


def role_statements(role: str, password: str, exists: bool) -> sql.Composed:
    """Crée le compte, ou change son mot de passe s'il existe déjà."""
    verb = "ALTER" if exists else "CREATE"
    return sql.SQL("{} ROLE {} WITH LOGIN PASSWORD {}").format(
        sql.SQL(verb), sql.Identifier(role), sql.Literal(password)
    )


def privilege_statements() -> list[sql.SQL | sql.Composed]:
    """Droits accordés à chaque compte, une fois le schéma créé par le propriétaire."""
    schema, read, write = (
        sql.Identifier(SCHEMA),
        sql.Identifier(AGENT_READ),
        sql.Identifier(AGENT_WRITE),
    )
    complaints = sql.SQL("{}.complaints").format(schema)
    log = sql.SQL("{}.action_log").format(schema)
    statements: list[sql.SQL | sql.Composed] = [
        sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(sql.Identifier(DATABASE)),
        sql.SQL("REVOKE ALL ON SCHEMA public FROM PUBLIC"),
    ]
    for role in (read, write):
        statements += [
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(DATABASE), role),
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, role),
            sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
                role, sql.Literal(AGENT_STATEMENT_TIMEOUT)
            ),
            sql.SQL("ALTER ROLE {} SET search_path = {}").format(role, schema),
        ]
    statements += [
        sql.SQL("GRANT SELECT ON {}, {} TO {}").format(complaints, log, read),
        sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(read),
        # Lire le dossier pour le retrouver, ne modifier que son statut.
        sql.SQL("GRANT SELECT ON {} TO {}").format(complaints, write),
        sql.SQL("GRANT UPDATE (status, status_updated_at) ON {} TO {}").format(complaints, write),
        sql.SQL("GRANT INSERT ON {} TO {}").format(log, write),
        sql.SQL("GRANT USAGE ON SEQUENCE {}.action_log_id_seq TO {}").format(schema, write),
    ]
    return statements
