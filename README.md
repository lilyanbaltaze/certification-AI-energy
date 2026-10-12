# Qualité et coût énergétique des modèles d'IA : pipeline de données

Projet de certification **RNCP37827BC01** « Réaliser la collecte, le stockage et la mise à disposition des données d'un projet en intelligence artificielle » (Développeur en intelligence artificielle, Simplon.co).

**Problématique** : comment fournir à une équipe technique une donnée fiable et à jour pour choisir un modèle d'IA selon sa qualité et son impact énergétique ?

- Quels modèles offrent le meilleur compromis qualité / énergie ?
- Peut-on se fier aux estimations énergétiques affichées au public (énergie estimée vs mesurée) ?

## Architecture : ETL et médaillon

Le pipeline suit une logique **ETL** (*Extract, Transform, Load*) : les données sont **extraites** des sources, **transformées** (nettoyées) en Python, puis **chargées** dans BigQuery. Une dernière transformation, l'agrégation, est faite **dans la base**, en SQL.
Les données traversent trois couches de qualité (**architecture médaillon**) :

| Étape | Étape ETL | Couche | Contenu | Où | Script |
|---|---|---|---|---|---|
| 1 | **Extract** | 🥉 Bronze | données brutes, telles que collectées (anomalies comprises) | `data/raw/` | `src/collect.py` |
| 2 | **Transform** | 🥈 Silver | données nettoyées et homogénéisées (5 tables) | `data/processed/` | `src/clean.py` |
| 3 | **Load** | 🥈 Silver | les 5 tables chargées dans la base | BigQuery (`ai_energy`) | `src/load_bigquery.py` |
| 4 | Transform (dans la base) | 🥇 Gold | données agrégées prêtes à l'emploi (4 vues SQL) | BigQuery | `sql/queries/*.sql` |
| 5 | Mise à disposition | — | API REST sécurisée qui lit la couche gold | — | `api/main.py` |
| 6 | Utilisation | — | dashboard qui interroge l'API | — | `streamlit/app.py` |

| Compétence | Fichiers |
|---|---|
| C1 Collecter les données de manière automatisée | `src/collect.py` |
| C2 Préparer les données pour leur première utilisation | `src/clean.py` |
| C3 Mettre en œuvre des règles d'agrégation | `references/model_mapping.csv`, `sql/queries/`, `src/check_gold.py` |
| C4 Créer une base de données | `docs/modele_donnees.md`, `sql/create_tables.sql`, `src/load_bigquery.py`, `docs/rgpd.md` |
| C5 Mettre à disposition les données via une API | `api/main.py`, consommée par `streamlit/app.py` |

Documentation : [modèle de données (MCD, MLD, MPD)](docs/modele_donnees.md) · [tables et vues SQL illustrées](docs/schemas_sql.md) · [conformité RGPD](docs/rgpd.md)

## Sources de données

| Source | Contenu | Licence |
|---|---|---|
| [ComparIA](https://www.data.gouv.fr/datasets/compar-ia) (ministère de la Culture) | conversations en duel, votes, énergie **estimée** | Licence Ouverte 2.0 |
| [AI Energy Score](https://huggingface.co/spaces/AIEnergyScore/Leaderboard) (Hugging Face) | énergie GPU **mesurée** (H100), modèles open-weight | données publiques |

Aucun texte écrit par les utilisateurs ni aucun identifiant n'est collecté (minimisation RGPD, voir `docs/rgpd.md`).

## Installation

Prérequis : Python 3.11, un projet Google Cloud avec BigQuery.

```bash
git clone https://github.com/lilyanbaltaze/certification-AI-energy.git
cd certification-AI-energy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ⚠️ **Toujours activer l'environnement du projet** (`source .venv/bin/activate`) dans chaque nouveau terminal.
> Si un autre environnement Python est actif (pyenv, conda…), les commandes `uvicorn` ou `streamlit` peuvent échouer avec `No module named 'google.cloud'`.
> Vérification : `which uvicorn` doit pointer vers `.venv/bin/uvicorn`.

### Google Cloud (BigQuery)

1. Créer un projet et activer l'API BigQuery.
2. Créer un dataset `ai_energy`, **région EU**.
3. Créer un compte de service avec les rôles **BigQuery Data Editor** et **BigQuery Job User**, puis télécharger sa clé JSON **hors du dépôt** (par exemple `~/.gcp/`).

### Fichier `.env` (à la racine, jamais commité)

```
GCP_PROJECT=identifiant-du-projet-gcp
BQ_DATASET=ai_energy
GOOGLE_APPLICATION_CREDENTIALS=/chemin/absolu/vers/cle-compte-de-service.json
API_KEY=une-cle-secrete-de-votre-choix
API_URL=http://localhost:8000
```

| Variable | Rôle |
|---|---|
| `GCP_PROJECT` | identifiant du projet Google Cloud |
| `BQ_DATASET` | dataset BigQuery (`ai_energy`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | chemin **complet** de la clé du compte de service (sans `~`) |
| `API_KEY` | clé exigée par l'API (en-tête `X-API-Key`) ; l'API refuse de démarrer sans elle |
| `API_URL` | adresse de l'API utilisée par le dashboard (facultatif, `http://localhost:8000` par défaut) |

## Utilisation

**1. Pipeline complet** (collecte → nettoyage → chargement BigQuery → vues gold) :
```bash
python main.py
```
Idempotent : les fichiers bruts de moins de 30 jours ne sont pas retéléchargés, les tables BigQuery sont recréées à chaque lancement. Les logs sont écrits dans `logs/pipeline.log`.

**2. API** (terminal 1) :
```bash
uvicorn api.main:app --reload
```
Documentation interactive (Swagger) : http://localhost:8000/docs (bouton **Authorize** pour saisir la clé).

| Endpoint | Accès | Contenu |
|---|---|---|
| `GET /` | public | présentation |
| `GET /health` | public | état de l'API et volumes en base |
| `GET /ranking?min_duels=200` | clé | classement par score ajusté avec intervalle de confiance à 95 % |
| `GET /energy?min_conversations=200&min_avg_kwh=0.00001` | clé | énergie estimée par modèle (kWh/conversation, kWh/requête, mWh/token) |
| `GET /categories?min_conversations=30&category=…` | clé | modèles par catégorie de conversation |
| `GET /estimated-vs-measured?task=text_generation` | clé | énergie estimée (ComparIA) face à l'énergie mesurée (AI Energy Score) |

**3. Dashboard** (terminal 2, l'API doit tourner) :
```bash
streamlit run streamlit/app.py
```
Puis http://localhost:8501.

**4. Contrôles et analyse** (l'API doit tourner pour l'analyse) :
```bash
python -m src.check_gold                  # les vues SQL donnent les mêmes chiffres que pandas
python -m src.analyse_estime_vs_mesure    # corrélation de Spearman estimé / mesuré + figure
```

## Structure du dépôt

```
main.py                     lance le pipeline complet
src/
  config.py                 chemins, URLs, seuils, logs
  collect.py                C1 : collecte (SQL DuckDB sur les parquet distants, CSV)
  clean.py                  C2 : nettoyage et homogénéisation → couche silver
  load_bigquery.py          C4 : création des tables, chargement, contrôles, vues gold
  check_gold.py             C3 : contrôle SQL = pandas
  analyse_estime_vs_mesure.py  question d'analyse n° 2 (Spearman)
sql/
  create_tables.sql         modèle physique (5 tables)
  queries/                  couche gold (4 vues)
api/main.py                 C5 : API FastAPI sécurisée
streamlit/app.py            dashboard (consomme l'API)
references/model_mapping.csv  correspondance des noms ComparIA ↔ Hugging Face
docs/                       modèle de données, schémas SQL, RGPD, figures
notebooks/exploration.ipynb analyse exploratoire d'origine
```

## Limites et perspectives

- L'énergie ComparIA est **estimée** (tokens × coefficient par modèle) ; la comparaison avec des mesures réelles ne porte que sur 11 modèles open-weight.
- Projet BigQuery en mode bac à sable : les tables expirent après 60 jours (recréées par `python main.py`).
- Perspectives : planifier la collecte mensuelle, ajouter des tests automatisés, intégrer les prix des modèles (API OpenRouter), générer automatiquement les commentaires du dashboard.
