"""Contrôle de la couche gold : les requêtes SQL donnent-elles les mêmes chiffres que pandas ?

On recalcule le score ajusté et l'énergie avec la méthode du notebook (pandas), à partir
des fichiers silver locaux, puis on compare aux vues BigQuery. Deux implémentations
indépendantes qui donnent le même résultat : c'est la preuve que les requêtes SQL sont justes.
Lancement : python -m src.check_gold
"""
import logging

import pandas as pd
from google.cloud import bigquery

from src import config


def pandas_score(votes):
    """Score ajusté, méthode du notebook : victoire 1 pt, égalité 0,5 pt chacun."""
    ties = votes[votes["both_equal"]]
    wins = votes[~votes["both_equal"]]
    points = wins["chosen_model_name"].value_counts()
    points = points.add(ties["model_a_name"].value_counts() * 0.5, fill_value=0)
    points = points.add(ties["model_b_name"].value_counts() * 0.5, fill_value=0)
    duels = votes["model_a_name"].value_counts().add(votes["model_b_name"].value_counts(), fill_value=0)
    result = pd.DataFrame({"points": points, "duels": duels}).fillna(0)
    result["score"] = result["points"] / result["duels"] * 100
    return result


def pandas_energy(conv):
    """Énergie par modèle, méthode du notebook, avec les mêmes exclusions que la vue gold."""
    side_a = conv[["model_a_name", "total_conv_a_kwh", "total_conv_a_output_tokens"]]
    side_a.columns = ["model", "kwh", "tokens"]
    side_b = conv[["model_b_name", "total_conv_b_kwh", "total_conv_b_output_tokens"]]
    side_b.columns = ["model", "kwh", "tokens"]
    sides = pd.concat([side_a, side_b])
    sides = sides[sides["kwh"].notna() & (sides["tokens"] > 0)]
    result = sides.groupby("model").agg(avg_kwh=("kwh", "mean"), total_kwh=("kwh", "sum"), total_tokens=("tokens", "sum"))
    result["mwh_per_token"] = result["total_kwh"] / result["total_tokens"] * 1e6
    return result


def compare(name, sql_df, pandas_df, columns):
    """Compare colonne par colonne ; arrête tout si un écart dépasse l'arrondi des calculs."""
    merged = sql_df.set_index("model")[columns].join(pandas_df[columns], rsuffix="_pandas", how="outer")
    for column in columns:
        gap = (merged[column] - merged[f"{column}_pandas"]).abs().max()
        if not gap < 1e-9:
            raise ValueError(f"{name} : écart SQL / pandas de {gap} sur {column}")
    logging.info(f"{name} : SQL = pandas pour {len(merged)} modèles ({', '.join(columns)})")


def run_check():
    client = bigquery.Client(project=config.GCP_PROJECT)
    votes = pd.read_parquet(config.DATA_PROCESSED_DIR / "votes.parquet")
    conv = pd.read_parquet(config.DATA_PROCESSED_DIR / "conversations.parquet")

    sql_score = client.query(f"SELECT * FROM {config.BQ_DATASET}.gold_classement_score_ajuste").to_dataframe()
    compare("gold_classement_score_ajuste", sql_score, pandas_score(votes), ["points", "duels", "score"])

    sql_energy = client.query(f"SELECT * FROM {config.BQ_DATASET}.gold_energy_global").to_dataframe()
    compare("gold_energy_global", sql_energy, pandas_energy(conv), ["avg_kwh", "mwh_per_token"])


if __name__ == "__main__":
    config.setup_logging()
    run_check()
