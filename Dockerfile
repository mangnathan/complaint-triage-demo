# Image de la démo publique (Render, offre gratuite) : l'application, l'interface Streamlit et
# un PostgreSQL propre à l'image, démarré et mis en place au lancement par le code du projet.
# Construite et vérifiée par la CI ; publiée par scripts/demo/deploy_demo.py.

FROM python:3.12.15-slim-trixie

# PostgreSQL du système (Debian 13). Pas de serveur lancé ici : la démo l'initialise au démarrage.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends postgresql \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group demo --no-install-project

COPY src ./src
RUN uv sync --locked --no-default-groups --group demo

# L'application ne tourne jamais en root.
RUN useradd --create-home --uid 1000 user && mkdir -p /app/data && chown -R user:user /app
USER user
ENV PATH="/app/.venv/bin:$PATH" \
    DEMO_DATA_DIR=/app/data

# Données de la démo (échantillon public, corpus du RAG), ajoutées seulement au déploiement.
COPY --chown=user:user demo/data/ /app/data/

# Le modèle d'embeddings, téléchargé à la construction : la démo n'en dépend pas au démarrage.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/data/models/fastembed')"

# La base est préparée ici, une fois : au lancement, il ne reste qu'à la démarrer.
ENV DEMO_PG_DIR=/app/pgdata
RUN python -m complaint_triage.demo --prepare

EXPOSE 7860
CMD ["python", "-m", "complaint_triage.demo"]
