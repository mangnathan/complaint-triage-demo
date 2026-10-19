"""Outils de l'agent : chacun valide ses entrées, puis exécute une action fixée à l'avance.

L'agent ne peut jamais écrire de SQL : il choisit un outil et lui donne des valeurs, que pydantic
contrôle (type, bornes, valeurs permises) avant toute exécution. Les requêtes sont écrites ici,
et les valeurs y passent toujours en paramètres. Les outils qui écrivent utilisent le compte
d'écriture, limité au statut et au journal, et laissent une trace dans le journal dans la même
transaction.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from complaint_triage.agent.alerts import send_alert
from complaint_triage.agent.database import AgentDatabase
from complaint_triage.rag.answer import ANSWERED, Answer

PRODUCTS = ("Credit card", "Checking or savings account")
Product = Literal["Credit card", "Checking or savings account"]
Status = Literal["nouveau", "en_cours", "traite", "escalade"]
# L'agent ne peut pas remettre un dossier à « nouveau » : il ne recule jamais dans la file.
NewStatus = Literal["en_cours", "traite", "escalade"]
MAX_NARRATIVE_CHARS = 2000
PRIORITY_SHARE = 0.10


class ToolError(Exception):
    """Erreur expliquée à l'agent, qui doit alors corriger sa demande ou refuser."""


class _Input(BaseModel):
    # Champ inconnu ou de mauvais type : refus, plutôt qu'une conversion silencieuse.
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ComplaintRef(_Input):
    complaint_id: int = Field(gt=0, lt=10**10, description="Identifiant CFPB de la réclamation")


class FindCompany(_Input):
    name: str = Field(min_length=2, max_length=80, description="Tout ou partie du nom")


class PriorityQueue(_Input):
    company: str = Field(min_length=2, max_length=200, description="Nom exact de l'établissement")
    product: Product | None = Field(default=None, description="Produit, ou tous")
    status: Status = Field(default="nouveau", description="Statut des dossiers listés")
    limit: int = Field(default=10, ge=1, le=20, description="Nombre de dossiers")


class RegulationQuestion(_Input):
    question: str = Field(min_length=10, max_length=1000, description="Question en anglais")


class StatusUpdate(_Input):
    complaint_id: int = Field(gt=0, lt=10**10)
    status: NewStatus
    reason: str = Field(min_length=5, max_length=300, description="Raison, gardée au journal")


class Alert(_Input):
    complaint_id: int = Field(gt=0, lt=10**10)
    severity: Literal["info", "urgent"]
    message: str = Field(min_length=10, max_length=500)


@dataclass
class ToolContext:
    database: AgentDatabase
    ask_regulations: Callable[[str], Answer]
    alert_url: str
    actor: str = "agent"


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v.isoformat() if isinstance(v, dt.date) else v for k, v in row.items()}


def find_company(ctx: ToolContext, args: FindCompany) -> dict[str, Any]:
    # Les caractères spéciaux de LIKE sont neutralisés : « % » cherche un vrai « % ».
    pattern = "%" + args.name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    with ctx.database.reading() as conn:
        rows = conn.execute(
            "SELECT company, count(*) AS reclamations FROM complaints"
            " WHERE company ILIKE %s GROUP BY company ORDER BY reclamations DESC LIMIT 10",
            (pattern,),
        ).fetchall()
    return {"etablissements": rows}


def get_complaint(ctx: ToolContext, args: ComplaintRef) -> dict[str, Any]:
    with ctx.database.reading() as conn:
        row = conn.execute(
            "SELECT complaint_id, date_received, product, sub_product, issue, sub_issue, company,"
            " state, submitted_via, tags, left(narrative, %s) AS narrative, priority_score,"
            " status, status_updated_at FROM complaints WHERE complaint_id = %s",
            (MAX_NARRATIVE_CHARS, args.complaint_id),
        ).fetchone()
    if row is None:
        raise ToolError(f"Réclamation {args.complaint_id} introuvable.")
    return _jsonable(row)


PRIORITY_CONTEXT = """
WITH target AS (
    SELECT company, product, date_trunc('week', date_received) AS week
    FROM complaints WHERE complaint_id = %(id)s
), queue AS (
    SELECT c.complaint_id, c.priority_score,
           rank() OVER (ORDER BY c.priority_score DESC) AS rang,
           count(*) OVER () AS taille
    FROM complaints c JOIN target t
      ON c.company = t.company AND c.product = t.product
     AND date_trunc('week', c.date_received) = t.week
)
SELECT priority_score, rang, taille FROM queue WHERE complaint_id = %(id)s
"""


def priority_context(ctx: ToolContext, args: ComplaintRef) -> dict[str, Any]:
    """Place du dossier dans sa file : même banque, même produit, même semaine (règle v0.1)."""
    with ctx.database.reading() as conn:
        row = conn.execute(PRIORITY_CONTEXT, {"id": args.complaint_id}).fetchone()
    if row is None:
        raise ToolError(f"Réclamation {args.complaint_id} introuvable.")
    places = max(1, round(PRIORITY_SHARE * row["taille"]))
    return {**row, "places_prioritaires": places, "prioritaire": row["rang"] <= places}


def list_priority_queue(ctx: ToolContext, args: PriorityQueue) -> dict[str, Any]:
    with ctx.database.reading() as conn:
        known = conn.execute(
            "SELECT 1 FROM complaints WHERE company = %s LIMIT 1", (args.company,)
        ).fetchone()
        if known is None:
            raise ToolError(
                f"Établissement inconnu : « {args.company} ». Chercher son nom exact avec "
                "find_company."
            )
        rows = conn.execute(
            "SELECT complaint_id, date_received, product, issue, priority_score, status"
            " FROM complaints WHERE company = %s AND (%s::text IS NULL OR product = %s)"
            " AND status = %s ORDER BY priority_score DESC, date_received, complaint_id"
            " LIMIT %s",
            (args.company, args.product, args.product, args.status, args.limit),
        ).fetchall()
    return {"dossiers": [_jsonable(r) for r in rows]}


def search_regulations(ctx: ToolContext, args: RegulationQuestion) -> dict[str, Any]:
    answer = ctx.ask_regulations(args.question)
    if answer.status != ANSWERED:
        return {"repondu": False, "raison": answer.reason}
    sources = [{"citation": s.citation, "extrait": s.quote, "lien": s.url} for s in answer.sources]
    return {"repondu": True, "reponse": answer.text, "sources": sources}


def update_status(ctx: ToolContext, args: StatusUpdate) -> dict[str, Any]:
    with ctx.database.writing() as conn:
        row = conn.execute(
            "SELECT status FROM complaints WHERE complaint_id = %s FOR UPDATE",
            (args.complaint_id,),
        ).fetchone()
        if row is None:
            raise ToolError(f"Réclamation {args.complaint_id} introuvable.")
        if row["status"] == args.status:
            raise ToolError(f"Le dossier {args.complaint_id} a déjà le statut « {args.status} ».")
        conn.execute(
            "UPDATE complaints SET status = %s, status_updated_at = now() WHERE complaint_id = %s",
            (args.status, args.complaint_id),
        )
        details = {"avant": row["status"], "apres": args.status, "raison": args.reason}
        conn.execute(
            "INSERT INTO action_log (actor, action, complaint_id, details)"
            " VALUES (%s, 'changement de statut', %s, %s)",
            (ctx.actor, args.complaint_id, Jsonb(details)),
        )
    return {"complaint_id": args.complaint_id, **details}


def raise_alert(ctx: ToolContext, args: Alert) -> dict[str, Any]:
    payload = {"complaint_id": args.complaint_id, "gravite": args.severity, "message": args.message}
    with ctx.database.writing() as conn:
        if (
            conn.execute(
                "SELECT 1 FROM complaints WHERE complaint_id = %s", (args.complaint_id,)
            ).fetchone()
            is None
        ):
            raise ToolError(f"Réclamation {args.complaint_id} introuvable.")
        # L'alerte n'est inscrite au journal que si l'envoi réussit : sinon, tout est annulé.
        send_alert(ctx.alert_url, payload)
        conn.execute(
            "INSERT INTO action_log (actor, action, complaint_id, details)"
            " VALUES (%s, 'alerte', %s, %s)",
            (ctx.actor, args.complaint_id, Jsonb(payload)),
        )
    return {"envoyee": True, **payload}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[_Input]
    run: Callable[[ToolContext, Any], dict[str, Any]]
    writes: bool = False

    def schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in (
        Tool("find_company", "Find the exact name of an institution.", FindCompany, find_company),
        Tool("get_complaint", "Read one complaint by its id.", ComplaintRef, get_complaint),
        Tool(
            "priority_context",
            "Rank of a complaint in its weekly queue (same institution and product), and whether "
            "it falls in the 10% handled first.",
            ComplaintRef,
            priority_context,
        ),
        Tool(
            "list_priority_queue",
            "Highest-priority complaints of one institution, optionally for one product.",
            PriorityQueue,
            list_priority_queue,
        ),
        Tool(
            "search_regulations",
            "Answer a regulation question from Regulations E, Z and DD, with verified citations, "
            "or refuse.",
            RegulationQuestion,
            search_regulations,
        ),
        Tool(
            "update_status",
            "Change the status of a complaint (en_cours, traite, escalade), with a reason.",
            StatusUpdate,
            update_status,
            writes=True,
        ),
        Tool(
            "raise_alert",
            "Alert a supervisor about one complaint.",
            Alert,
            raise_alert,
            writes=True,
        ),
    )
}


def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Seul point d'entrée : outil connu, entrées validées, erreurs rendues lisibles."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"Outil inconnu : {name}.")
    try:
        args = tool.input_model.model_validate(arguments)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))} : {e['msg']}" for e in error.errors())
        raise ToolError(f"Entrées refusées pour {name} : {problems}.") from error
    return tool.run(ctx, args)
