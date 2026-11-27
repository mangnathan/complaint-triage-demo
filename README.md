# complaint-triage : démonstration

**Démo publique de [complaint-triage](https://github.com/mangnathan/complaint-triage) : priorisation des réclamations clients en
banque de détail, réponses réglementaires avec citations vérifiées, et un agent aux outils
encadrés.**

**[Ouvrir la démo : complaint-triage-zj94.onrender.com](https://complaint-triage-zj94.onrender.com)**
(offre gratuite de Render : si personne ne l'a ouverte depuis un quart d'heure, le premier
chargement peut prendre jusqu'à deux minutes).

Ce dépôt sert uniquement au déploiement. Il est produit par `scripts/demo/deploy_demo.py` à
partir du dépôt principal, commit `49b7eb2`, et ne se modifie pas à la main : le code, les
résultats, les rapports et les notes de décision sont dans [complaint-triage](https://github.com/mangnathan/complaint-triage).

## Ce que montre la démo

- **File prioritaire** : les réclamations publiques reçues depuis août 2026 par les 25
  établissements les plus visés, notées dès leur arrivée.
- **Réglementation** : une question en anglais, une réponse qui cite le texte officiel, chaque
  citation vérifiée par le code ; sinon, un refus.
- **Agent** : une demande en langage courant ; l'agent annonce son plan, utilise des outils
  encadrés, puis répond ou refuse. Ses écritures ne touchent que des banques fictives, remises à
  zéro à chaque redémarrage.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `Dockerfile` | Image de la démo : application, interface Streamlit et base PostgreSQL |
| `src/`, `pyproject.toml`, `uv.lock` | Code du projet et dépendances figées |
| `demo/data/` | Échantillon des réclamations publiques du CFPB et corpus réglementaire de l'eCFR |
| `render.yaml` | Service Render, reconstruit à chaque commit |

## Lancer la démo chez soi

Avec Docker :

```bash
docker build -t complaint-triage-demo .
docker run --rm -p 7860:7860 complaint-triage-demo
```

Puis ouvrir http://localhost:7860. Sans clé, les exemples enregistrés fonctionnent et les
questions libres sont désactivées : c'est le cas de la démo en ligne, car les conditions de
l'API Gemini réservent aux services payants les applications ouvertes à des visiteurs en
Europe. Pour essayer les questions libres sur son propre poste, passer sa propre clé :
`-e GEMINI_API_KEY=... -e GEMINI_MODEL=...`.

## Données et licence

Données publiques uniquement : la Consumer Complaint Database du CFPB, que le CFPB décrit comme
« freely available for anyone to use, analyze, and build on » (ce n'est pas un échantillon
statistique de l'expérience des consommateurs), et les Regulations E, Z et DD de l'eCFR,
compilation éditoriale non officielle du CFR. Ce projet est indépendant : il n'est ni soutenu ni
validé par le CFPB.

Projet personnel de Nathan Mang. Code sous licence MIT (voir `LICENSE`).
