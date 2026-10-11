"""API REST : met la couche gold à disposition des autres outils (compétence C5).

Lancement (depuis la racine du projet) : uvicorn api.main:app --reload
Documentation interactive (Swagger) : http://localhost:8000/docs

Sécurité :
- clé obligatoire dans l'en-tête X-API-Key (sauf / et /health), sinon erreur 403 ;
- uniquement des requêtes GET : l'API lit les données, elle ne peut pas les modifier ;
- requêtes SQL paramétrées : les valeurs envoyées par l'utilisateur ne sont jamais collées
  dans le texte SQL, ce qui empêche l'injection SQL ;
- paramètres validés par FastAPI (type, bornes) : une valeur invalide renvoie une erreur 422.
"""
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from google.cloud import bigquery

from src import config

# Sans clé définie, l'API serait ouverte à tous : on refuse de démarrer
if not config.API_KEY:
    raise RuntimeError("Variable API_KEY absente du fichier .env : l'API refuse de démarrer sans clé")

client = bigquery.Client(project=config.GCP_PROJECT)
DATASET = config.BQ_DATASET  # vient de la configuration, jamais de l'utilisateur

app = FastAPI(
    title="API énergie et qualité des modèles d'IA",
    description=(
        "Données ComparIA (qualité perçue, énergie estimée) et AI Energy Score (énergie mesurée), "
        "nettoyées et agrégées dans BigQuery (couche gold). "
        "Les endpoints protégés exigent l'en-tête X-API-Key."
    ),
    version="1.0",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: str | None = Security(api_key_header)):
    """Refuse la requête (403) si la clé est absente ou fausse."""
    if key != config.API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide ou manquante")


def run_query(sql, parameters=None):
    """Exécute une requête BigQuery paramétrée et renvoie une liste de dictionnaires (→ JSON)."""
    job_config = bigquery.QueryJobConfig(query_parameters=parameters or [])
    rows = client.query(sql, job_config=job_config).result()
    return [dict(row) for row in rows]


@app.get("/", tags=["Public"], summary="Présentation de l'API")
def home():
    return {
        "projet": "Qualité et coût énergétique des modèles d'IA",
        "documentation": "/docs",
        "endpoints": ["/health", "/ranking", "/energy", "/categories", "/estimated-vs-measured"],
    }


@app.get("/health", tags=["Public"], summary="État de l'API et de la base")
def health():
    try:
        result = run_query(f"""
            SELECT
                (SELECT COUNT(*) FROM {DATASET}.models) AS nb_models,
                (SELECT COUNT(*) FROM {DATASET}.conversations) AS nb_conversations,
                (SELECT COUNT(*) FROM {DATASET}.votes) AS nb_votes,
                (SELECT MAX(timestamp) FROM {DATASET}.conversations) AS last_conversation
        """)
    except Exception:
        raise HTTPException(status_code=503, detail="Base BigQuery inaccessible")
    return {"status": "ok", **result[0]}


@app.get("/ranking", tags=["Qualité"], dependencies=[Depends(check_api_key)],
         summary="Classement des modèles par score ajusté (victoire 1, égalité 0,5)")
def ranking(min_duels: int = Query(config.MIN_DUELS, ge=1, description="Nombre minimum de duels")):
    return run_query(
        f"""SELECT * FROM {DATASET}.gold_classement_score_ajuste
            WHERE duels >= @min_duels
            ORDER BY score DESC""",
        [bigquery.ScalarQueryParameter("min_duels", "INT64", min_duels)],
    )


@app.get("/energy", tags=["Énergie"], dependencies=[Depends(check_api_key)],
         summary="Énergie estimée par ComparIA, par modèle (kWh/conversation, kWh/requête, mWh/token)")
def energy(
    min_conversations: int = Query(config.MIN_CONVERSATIONS, ge=1, description="Nombre minimum de conversations"),
    min_avg_kwh: float = Query(config.MIN_AVG_KWH, ge=0, description="Énergie moyenne minimum (kWh/conversation)"),
):
    return run_query(
        f"""SELECT * FROM {DATASET}.gold_energy_global
            WHERE nb_conversations >= @min_conversations AND avg_kwh >= @min_avg_kwh
            ORDER BY avg_kwh DESC""",
        [bigquery.ScalarQueryParameter("min_conversations", "INT64", min_conversations),
         bigquery.ScalarQueryParameter("min_avg_kwh", "FLOAT64", min_avg_kwh)],
    )


@app.get("/categories", tags=["Énergie"], dependencies=[Depends(check_api_key)],
         summary="Modèles par catégorie de conversation, avec leur énergie")
def categories(
    min_conversations: int = Query(config.MIN_CONV_PER_CATEGORY, ge=1,
                                   description="Nombre minimum de conversations du modèle dans la catégorie"),
    category: str | None = Query(None, description="Filtrer sur une catégorie (ex. Education)"),
):
    # « @category IS NULL » : si aucun filtre n'est demandé, toutes les catégories sont renvoyées
    return run_query(
        f"""SELECT * FROM {DATASET}.gold_ranking_per_cat
            WHERE nb_conv_cat >= @min_conversations
              AND (@category IS NULL OR categories = @category)
            ORDER BY categories, mwh_per_token""",
        [bigquery.ScalarQueryParameter("min_conversations", "INT64", min_conversations),
         bigquery.ScalarQueryParameter("category", "STRING", category)],
    )


@app.get("/estimated-vs-measured", tags=["Énergie"], dependencies=[Depends(check_api_key)],
         summary="Énergie estimée (ComparIA) face à l'énergie mesurée (AI Energy Score)")
def estimated_vs_measured(
    task: Literal["text_generation", "reasoning"] = Query("text_generation",
                                                         description="Tâche mesurée : ne comparer qu'à tâche égale"),
):
    return run_query(
        f"""SELECT * FROM {DATASET}.gold_estime_vs_mesure
            WHERE task = @task
            ORDER BY gpu_energy_wh""",
        [bigquery.ScalarQueryParameter("task", "STRING", task)],
    )
