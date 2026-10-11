"""Étape 3 du pipeline : chargement dans BigQuery (compétences C4 et C3).

1. crée les tables avec sql/create_tables.sql (le modèle physique) ;
2. charge les 5 fichiers parquet de data/processed/ (couche silver) ;
3. vérifie que BigQuery contient autant de lignes que les fichiers locaux ;
4. crée les vues de la couche gold (sql/queries/), qui agrègent la couche silver.

Relancer le script ne crée jamais de doublons : les tables sont recréées vides à chaque fois.
"""
import logging

import pandas as pd
from google.cloud import bigquery

from src import config

# Ordre de chargement : models d'abord, car les autres tables y font référence.
# Pour chaque table : les colonnes de sa clé primaire (voir sql/create_tables.sql).
PRIMARY_KEYS = {
    "models": ["model_name"],
    "conversations": ["conversation_pair_id"],
    "conversation_categories": ["conversation_pair_id", "category"],
    "votes": ["conversation_pair_id"],
    "energy_measures": ["hf_model_id", "task", "reasoning_level"],
}


def create_tables(client):
    """Exécute le script SQL du modèle physique (suppression puis création des 5 tables)."""
    sql = (config.ROOT_DIR / "sql" / "create_tables.sql").read_text(encoding="utf-8")
    client.query(sql).result()  # result() attend la fin de l'exécution
    logging.info("BigQuery : 5 tables créées d'après sql/create_tables.sql")


def check_primary_key(df, name):
    """Vérifie que la clé primaire est unique avant le chargement.
    BigQuery ne vérifie pas les clés (« NOT ENFORCED ») : sans ce contrôle,
    un doublon entrerait dans la base sans que personne ne le voie."""
    key = PRIMARY_KEYS[name]
    duplicates = df.duplicated(subset=key).sum()
    if duplicates > 0:
        raise ValueError(f"{name} : {duplicates} doublons sur la clé primaire {key}")
    logging.info(f"{name} : clé primaire {key} unique")


def load_table(client, name):
    """Charge un fichier parquet dans la table BigQuery du même nom."""
    path = config.DATA_PROCESSED_DIR / f"{name}.parquet"
    table_id = f"{config.GCP_PROJECT}.{config.BQ_DATASET}.{name}"

    local_df = pd.read_parquet(path)
    check_primary_key(local_df, name)

    # WRITE_APPEND dans une table qui vient d'être créée vide, avec le schéma de cette table
    # (celui de create_tables.sql). Si un type ne correspond pas au MPD, ou si une colonne
    # NOT NULL contient une valeur vide, BigQuery refuse le chargement :
    # c'est une vérification gratuite que les données respectent le modèle.
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=client.get_table(table_id).schema,
    )
    with open(path, "rb") as file:
        client.load_table_from_file(file, table_id, job_config=job_config).result()

    # Contrôle : autant de lignes dans BigQuery qu'en local
    local_rows = len(local_df)
    bq_rows = client.get_table(table_id).num_rows
    if bq_rows != local_rows:
        raise ValueError(f"{name} : {bq_rows} lignes dans BigQuery, {local_rows} en local")
    logging.info(f"BigQuery : {name} chargée ({bq_rows} lignes, identique au fichier local)")


# Ordre imposé : gold_ranking_per_cat et gold_estime_vs_mesure lisent gold_energy_global
GOLD_VIEWS = ["gold_classement_score_ajuste", "gold_energy_global",
              "gold_ranking_per_cat", "gold_estime_vs_mesure"]


def create_gold_views(client):
    """Crée (ou remplace) les vues de la couche gold.
    Une vue est une requête enregistrée : BigQuery la recalcule à chaque lecture,
    elle est donc toujours à jour avec la couche silver, sans copie des données."""
    for name in GOLD_VIEWS:
        sql = (config.ROOT_DIR / "sql" / "queries" / f"{name}.sql").read_text(encoding="utf-8")
        client.query(sql).result()
        rows = client.query(f"SELECT COUNT(*) AS n FROM {config.BQ_DATASET}.{name}").result()
        logging.info(f"BigQuery : vue {name} créée ({next(iter(rows)).n} lignes)")


def run_load():
    """Lance tout le chargement (appelée par main.py) : silver local → BigQuery."""
    client = bigquery.Client(project=config.GCP_PROJECT)
    create_tables(client)
    for name in PRIMARY_KEYS:
        load_table(client, name)
    create_gold_views(client)


if __name__ == "__main__":
    config.setup_logging()
    run_load()
