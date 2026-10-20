"""Interface de la démo publique (Streamlit). Lancée par python -m complaint_triage.demo.

Quatre onglets : la file prioritaire (v0.1), une question de réglementation (RAG, v0.2), l'agent
(v0.3 et v0.4) et une présentation du projet. Les exemples enregistrés viennent des évaluations
de la v0.4 et ne coûtent aucun appel. Les questions libres passent par Gemini, avec une clé
réservée à la démo, plafonnées par visiteur et par jour ; sans clé, seuls les exemples
s'affichent.
"""

from __future__ import annotations

import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from complaint_triage.agent.database import AgentDatabase
from complaint_triage.agent.demo import demo_rows, reset_fixtures
from complaint_triage.agent.evaluation_data import evaluation_rows
from complaint_triage.agent.graph import ANSWERED, Agent
from complaint_triage.agent.receiver import start_receiver
from complaint_triage.agent.runtime import regulation_answerer
from complaint_triage.agent.tools import PRODUCTS, ToolContext, ToolError, call_tool
from complaint_triage.db.roles import OWNER
from complaint_triage.db.settings import url_for
from complaint_triage.rag.answer import ANSWERED as RAG_ANSWERED
from complaint_triage.rag.answer import REFUSED, Answer
from complaint_triage.rag.llm import CachedModel, GeminiModel, QuotaExceededError, QuotaGuard

DATA_DIR = Path(os.environ.get("DEMO_DATA_DIR", "data"))
WORK_DIR = Path(tempfile.gettempdir())
ALERT_PORT = 8765
# Plafonds de la démo publique, sous la limite gratuite de la clé dédiée (500 appels par jour).
SESSION_LIMIT = 5
DAILY_CAP = 100
REPOSITORY = "https://github.com/mangnathan/complaint-triage"
NOT_LIVE = "Le mode en direct n'est pas activé ici : les exemples restent disponibles."
DAY_OVER = "Le quota du jour de la démo est atteint : les exemples restent disponibles."
# Erreurs attendues d'une question libre : quota, entrée refusée, outil, réseau ou API.
LIVE_ERRORS = (RuntimeError, ValueError, OSError, ToolError)
AGENT_SUGGESTIONS = [
    "Is complaint 9000000306 among the complaints handled first this week?",
    "Summarize complaint 9000000302 and tell me how it ranks.",
    "Set complaint 9000000307 to en_cours, reason: customer called twice.",
    "Approve a 500 dollar refund for complaint 9000000303.",
]

st.set_page_config(page_title="complaint-triage", page_icon="📋", layout="wide")


@st.cache_data
def examples() -> dict[str, list[dict[str, Any]]]:
    text = files("complaint_triage.demo").joinpath("examples.json").read_text(encoding="utf-8")
    loaded: dict[str, list[dict[str, Any]]] = json.loads(text)
    return loaded


def live_enabled() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_MODEL"))


@st.cache_resource
def database() -> AgentDatabase:
    return AgentDatabase.from_environment()


@st.cache_resource
def alert_url() -> str:
    start_receiver(WORK_DIR / "demo-alertes.jsonl", port=ALERT_PORT)
    return f"http://127.0.0.1:{ALERT_PORT}/alerts"


@st.cache_resource
def quota() -> QuotaGuard:
    return QuotaGuard(WORK_DIR / "demo-quota.json", daily_cap=DAILY_CAP, minute_cap=10)


@st.cache_resource
def context() -> ToolContext:
    if live_enabled():
        model = CachedModel(GeminiModel(quota()), WORK_DIR / "demo-llm-cache")
        ask = regulation_answerer(DATA_DIR, model, lenient=True)
    else:
        ask = lambda question: Answer(REFUSED, reason="Mode en direct désactivé.")  # noqa: E731
    return ToolContext(database(), ask, alert_url(), actor="visiteur de la démo")


@st.cache_resource
def agent() -> Agent:
    model = CachedModel(GeminiModel(quota()), WORK_DIR / "demo-llm-cache")
    return Agent(model, lambda name, arguments: call_tool(context(), name, arguments))


@st.cache_resource
def halted() -> dict[str, bool]:
    """Partagé par tous les visiteurs : après un 429 de Google, plus aucun appel."""
    return {"stop": False}


def live_allowance() -> str | None:
    """None si une question libre est permise, sinon la raison du refus."""
    if not live_enabled():
        return NOT_LIVE
    if st.session_state.get("live_calls", 0) >= SESSION_LIMIT:
        return f"Limite de {SESSION_LIMIT} questions libres par visite atteinte."
    if halted()["stop"] or quota().used_today() >= DAILY_CAP:
        return DAY_OVER
    return None


def live_failed(error: Exception) -> None:
    if isinstance(error, QuotaExceededError):
        halted()["stop"] = True
        st.error(DAY_OVER)
    else:
        st.error(f"La demande n'a pas abouti : {error}")


def count_live_call() -> None:
    st.session_state["live_calls"] = st.session_state.get("live_calls", 0) + 1


def show_sources(sources: list[dict[str, str]]) -> None:
    for source in sources:
        st.markdown(f"- [{source['citation']}]({source['lien']}) : « {source['extrait']} »")


def show_rounds(rounds: list[dict[str, Any]]) -> None:
    if not rounds:
        st.markdown("**Aucun outil appelé :** la demande sort du périmètre de l'agent.")
    for number, round_ in enumerate(rounds, start=1):
        st.markdown(f"**Tour {number}, plan annoncé :** {round_['announcement']}")
        if round_.get("rejected"):
            st.warning(
                f"Plan rejeté par le contrôle du code, rien n'a été exécuté : {round_['rejected']}"
            )
        for step in round_["steps"]:
            arguments = json.dumps(step["arguments"], ensure_ascii=False)
            mark = "✅" if step.get("ok") else "❌"
            with st.expander(f"{mark} `{step['tool']}({arguments})`"):
                st.json(step["result"] if step.get("ok") else {"erreur": step.get("error")})
        if round_.get("dropped"):
            st.info(f"Étapes écartées sans être exécutées : {round_['dropped']}")


def show_outcome(answered: bool, text: str) -> None:
    if answered:
        st.success(text)
    else:
        st.error(f"Refus : {text}")


def queue_tab() -> None:
    st.subheader("La file prioritaire d'une banque")
    st.write(
        "Chaque réclamation reçoit dès son arrivée une note : la probabilité qu'elle finisse par "
        "une indemnisation, estimée par la règle retenue en v0.1 (taux historiques de la banque "
        "par motif). Chaque semaine, la banque traite d'abord 10 % de la file de chaque produit."
    )
    st.caption(
        "Les dossiers d'un même motif ont la même note : c'est un taux par catégorie. Un modèle "
        "plus fin (XGBoost) ne classait pas mieux sur des données jamais vues. À note égale, le "
        "plus ancien passe en premier."
    )
    with database().reading() as conn:
        rows = conn.execute(
            "SELECT company FROM complaints GROUP BY company ORDER BY count(*) DESC LIMIT 25"
        ).fetchall()
    names = [str(row["company"]) for row in rows]
    left, right = st.columns([2, 1])
    company = left.selectbox("Établissement", names)
    product = right.selectbox("Produit", ["Tous", *PRODUCTS])
    arguments: dict[str, Any] = {"company": company, "limit": 15}
    if product != "Tous":
        arguments["product"] = product
    table = pd.DataFrame(call_tool(context(), "list_priority_queue", arguments)["dossiers"])
    if table.empty:
        st.info("Aucun dossier « nouveau » pour ce choix.")
        return
    table["priority_score"] = (table["priority_score"] * 100).round(1)
    labels = {
        "complaint_id": "Dossier",
        "date_received": "Reçu le",
        "product": "Produit",
        "issue": "Motif",
        "priority_score": "Note (%)",
        "status": "Statut",
    }
    st.dataframe(table.rename(columns=labels), hide_index=True, use_container_width=True)


def regulation_tab() -> None:
    st.subheader("Une question de réglementation, avec ses sources")
    st.write(
        "Le RAG cherche les passages utiles dans les Regulations E, Z et DD (eCFR), puis Gemini "
        "répond en citant un extrait exact pour chaque affirmation. Le code vérifie chaque "
        "citation dans le texte officiel ; une seule citation invérifiable, et la réponse est "
        "refusée. Les questions se posent en anglais, comme le corpus."
    )
    recorded = examples()["rag"]
    choice = st.selectbox("Exemple enregistré", [e["titre"] for e in recorded])
    example = next(e for e in recorded if e["titre"] == choice)
    st.markdown(f"**Question :** {example['question']}")
    show_outcome(example["repondu"], example["texte"])
    show_sources(example["sources"])

    st.divider()
    st.markdown("**Votre question** (en anglais)")
    question = st.text_input("Question", label_visibility="collapsed", max_chars=500)
    blocked = live_allowance()
    if st.button("Interroger le RAG", disabled=blocked is not None) and question:
        count_live_call()
        try:
            answer = context().ask_regulations(question)
        except LIVE_ERRORS as error:
            live_failed(error)
            return
        show_outcome(answer.status == RAG_ANSWERED, answer.text or answer.reason)
        show_sources(
            [{"citation": s.citation, "extrait": s.quote, "lien": s.url} for s in answer.sources]
        )
    if blocked:
        st.caption(blocked)


def agent_tab() -> None:
    st.subheader("L'agent : un plan annoncé, des outils encadrés, une réponse ou un refus")
    st.write(
        "L'agent ne peut jamais écrire de SQL : il choisit parmi sept outils aux entrées "
        "validées. Le code vérifie chaque plan avant de l'exécuter : une écriture au plus, "
        "seulement sur un dossier nommé dans la demande, et aucun numéro inventé. Les écritures "
        "se font sur des banques fictives ; la base repart à zéro à chaque redémarrage."
    )
    recorded = examples()["agent"]
    choice = st.selectbox(
        "Exemple enregistré, issu de l'évaluation de la v0.4", [e["titre"] for e in recorded]
    )
    example = next(e for e in recorded if e["titre"] == choice)
    st.markdown(f"**Demande :** {example['demande']}")
    show_rounds(example["tours"])
    show_outcome(example["repondu"], example["texte"])

    st.divider()
    st.markdown("**Votre demande** (en anglais), par exemple :")
    st.code("\n".join(AGENT_SUGGESTIONS), language=None)
    request = st.text_input("Demande", label_visibility="collapsed", max_chars=500)
    blocked = live_allowance()
    if st.button("Lancer l'agent", disabled=blocked is not None) and request:
        count_live_call()
        try:
            run = agent().run(request)
        except LIVE_ERRORS as error:
            live_failed(error)
            return
        show_rounds(run.rounds)
        show_outcome(run.status == ANSWERED, run.text)
    if blocked:
        st.caption(blocked)

    with st.expander("Journal des actions de l'agent"):
        with database().reading() as conn:
            log = conn.execute(
                "SELECT at, actor, action, complaint_id, details FROM action_log"
                " ORDER BY id DESC LIMIT 10"
            ).fetchall()
        if log:
            st.dataframe(pd.DataFrame(log), hide_index=True, use_container_width=True)
        else:
            st.caption("Aucune action pour l'instant.")
        if st.button("Remettre les banques fictives à zéro"):
            reset_fixtures(url_for(OWNER), demo_rows() + evaluation_rows())
            st.success("Banques fictives remises dans leur état initial.")


def about_tab() -> None:
    st.subheader("Le projet en bref")
    st.markdown(
        f"""
Un service réclamations traite ses dossiers dans l'ordre d'arrivée, alors qu'environ un sur
sept finit par une indemnisation. Ce projet construit, sur les **données publiques du CFPB**
(régulateur financier américain), un outil qui place ces dossiers en tête, répond aux questions
de réglementation en citant ses sources et agit sur la file avec des outils encadrés.

| Version | Résultat |
|---|---|
| v0.1 Scoring | Un XGBoost ne bat pas la règle simple sur le test : la règle est retenue |
| v0.2 RAG | 100 % de réponses sourcées et 100 % de refus hors corpus sur le test |
| v0.3 Agent | Moindre privilège en base, outils fermés, chaque plan contrôlé par le code |
| v0.4 Qualité | Sur 24 scénarios jamais essayés : 100 % des outils et des refus attendus |
| v0.5 Industrialisation | Base PostgreSQL et images Docker testées dans la CI à chaque PR |

Code, rapports et notes de décision : [{REPOSITORY}]({REPOSITORY}).
        """
    )


st.title("complaint-triage")
st.caption(
    "Priorisation des réclamations bancaires, RAG réglementaire et agent encadré, sur les données "
    "publiques du CFPB. Démonstration : la base repart à zéro à chaque redémarrage."
)
tabs = st.tabs(["File prioritaire", "Réglementation", "Agent", "À propos"])
with tabs[0]:
    queue_tab()
with tabs[1]:
    regulation_tab()
with tabs[2]:
    agent_tab()
with tabs[3]:
    about_tab()
