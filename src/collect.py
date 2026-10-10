"""Étape 1 du pipeline : collecte des données brutes (compétences C1 et C2).

Les données sont gardées telles quelles dans data/raw/ : le nettoyage est fait
dans clean.py, pour pouvoir compter et justifier chaque ligne supprimée.

reactions.parquet n'est volontairement PAS collecté : aucune analyse ne l'utilise.
Principe RGPD de minimisation : on ne collecte que ce qui sert à la finalité.
"""
import logging
import time

import duckdb
import requests

from src import config

# Requêtes SQL DuckDB (C2) : on ne lit que les colonnes utiles, ce qui évite
# de télécharger les 2,6 Go. Aucun identifiant ni texte utilisateur (RGPD).
# CAST explicites : le schéma de la source a déjà changé entre deux versions,
# on impose donc nous-mêmes le type attendu de chaque colonne.
CONVERSATIONS_QUERY = f"""
    SELECT
        CAST(conversation_pair_id AS VARCHAR)       AS conversation_pair_id,
        CAST(timestamp AS TIMESTAMP)                AS timestamp,
        CAST(model_a_name AS VARCHAR)               AS model_a_name,
        CAST(model_b_name AS VARCHAR)               AS model_b_name,
        CAST(total_conv_a_kwh AS DOUBLE)            AS total_conv_a_kwh,
        CAST(total_conv_b_kwh AS DOUBLE)            AS total_conv_b_kwh,
        CAST(total_conv_a_output_tokens AS BIGINT)  AS total_conv_a_output_tokens,
        CAST(total_conv_b_output_tokens AS BIGINT)  AS total_conv_b_output_tokens,
        CAST(conv_turns AS BIGINT)                  AS conv_turns,
        CAST(categories AS VARCHAR[])               AS categories,
        CAST(model_a_total_params AS DOUBLE)        AS model_a_total_params,
        CAST(model_b_total_params AS DOUBLE)        AS model_b_total_params,
        CAST(model_a_active_params AS DOUBLE)       AS model_a_active_params,
        CAST(model_b_active_params AS DOUBLE)       AS model_b_active_params
    FROM read_parquet('{config.COMPARIA_CONVERSATIONS_URL}')
"""

VOTES_QUERY = f"""
    SELECT
        CAST(conversation_pair_id AS VARCHAR)  AS conversation_pair_id,
        CAST(timestamp AS TIMESTAMP)           AS timestamp,
        CAST(model_a_name AS VARCHAR)          AS model_a_name,
        CAST(model_b_name AS VARCHAR)          AS model_b_name,
        CAST(chosen_model_name AS VARCHAR)     AS chosen_model_name,
        CAST(both_equal AS BOOLEAN)            AS both_equal
    FROM read_parquet('{config.COMPARIA_VOTES_URL}')
"""


def collect_comparia(query, output_path):
    """Exécute une requête DuckDB sur un parquet distant et sauvegarde le résultat."""
    # Idempotence : si le fichier existe déjà, on ne relance pas la collecte
    if output_path.exists():
        logging.info(f"{output_path.name} déjà présent, collecte ignorée")
        return

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            start = time.time()
            con = duckdb.connect()
            con.execute("INSTALL httpfs; LOAD httpfs;")  # extension pour lire des fichiers via HTTP
            df = con.execute(query).df()
            df.to_parquet(output_path, index=False)
            logging.info(f"{output_path.name} : {len(df)} lignes collectées en {time.time() - start:.0f} s")
            return
        except Exception as error:
            logging.warning(f"{output_path.name} : échec tentative {attempt}/{config.MAX_RETRIES} ({error})")
            # Un fichier à moitié écrit serait pris pour un fichier valide par l'idempotence
            output_path.unlink(missing_ok=True)
            time.sleep(config.RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Collecte impossible après {config.MAX_RETRIES} tentatives : {output_path.name}")


def collect_csv(url, output_path):
    """Télécharge un CSV AI Energy Score tel quel (les anomalies sont traitées dans clean.py)."""
    if output_path.exists():
        logging.info(f"{output_path.name} déjà présent, collecte ignorée")
        return

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # déclenche une erreur si le serveur répond 404, 500…
            output_path.write_bytes(response.content)
            nb_lines = response.text.count("\n") - 1  # -1 pour la ligne d'en-tête
            logging.info(f"{output_path.name} : {nb_lines} lignes collectées")
            return
        except Exception as error:
            logging.warning(f"{output_path.name} : échec tentative {attempt}/{config.MAX_RETRIES} ({error})")
            output_path.unlink(missing_ok=True)
            time.sleep(config.RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Collecte impossible après {config.MAX_RETRIES} tentatives : {output_path.name}")


def run_collect():
    """Lance toutes les collectes (appelée par main.py)."""
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    collect_comparia(CONVERSATIONS_QUERY, config.DATA_RAW_DIR / "comparia_conversations.parquet")
    collect_comparia(VOTES_QUERY, config.DATA_RAW_DIR / "comparia_votes.parquet")
    collect_csv(config.AIES_TEXT_GENERATION_URL, config.DATA_RAW_DIR / "aies_text_generation.csv")
    collect_csv(config.AIES_REASONING_URL, config.DATA_RAW_DIR / "aies_reasoning.csv")


if __name__ == "__main__":
    config.setup_logging()
    run_collect()
