-- Schéma de la base, exécuté par le compte propriétaire (ct_owner).
-- Seules les informations connues à la réception sont stockées : ni l'issue de la réclamation,
-- qui est la cible du modèle, ni le code postal, dont l'agent n'a pas besoin.

CREATE TABLE IF NOT EXISTS triage.complaints (
    complaint_id      bigint PRIMARY KEY,
    date_received     date NOT NULL,
    product           text NOT NULL,
    sub_product       text,
    issue             text NOT NULL,
    sub_issue         text,
    company           text NOT NULL,
    state             text,
    submitted_via     text,
    tags              text,
    narrative         text,
    priority_score    double precision NOT NULL CHECK (priority_score BETWEEN 0 AND 1),
    status            text NOT NULL DEFAULT 'nouveau'
                      CHECK (status IN ('nouveau', 'en_cours', 'traite', 'escalade')),
    status_updated_at timestamptz
);

CREATE INDEX IF NOT EXISTS complaints_queue_idx
    ON triage.complaints (company, product, date_received);

-- Journal des actions : chaque écriture de l'agent y laisse une trace, avec son auteur.
CREATE TABLE IF NOT EXISTS triage.action_log (
    id           bigserial PRIMARY KEY,
    at           timestamptz NOT NULL DEFAULT now(),
    actor        text NOT NULL,
    action       text NOT NULL,
    complaint_id bigint REFERENCES triage.complaints (complaint_id),
    details      jsonb NOT NULL DEFAULT '{}'::jsonb
);
