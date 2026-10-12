"""Question d'analyse n° 2 : peut-on se fier aux estimations énergétiques affichées au public ?

On compare, sur les modèles présents dans les deux sources, l'énergie ESTIMÉE par ComparIA
et l'énergie MESURÉE par AI Energy Score (GPU H100), à tâche égale (text_generation).
Les unités et les conditions diffèrent : on compare donc des CLASSEMENTS (corrélation de rangs
de Spearman), pas des valeurs.

Analyse de sensibilité : on refait le calcul avec les seules correspondances « exactes »,
pour vérifier que l'approximation version de base / version instruct ne change pas la conclusion.

Ce script consomme l'API (comme le dashboard) : l'API doit être lancée.
Lancement : python -m src.analyse_estime_vs_mesure
"""
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

from src import config

API_URL = os.getenv("API_URL", "http://localhost:8000")
FIGURE_PATH = config.ROOT_DIR / "docs" / "figures" / "estime_vs_mesure.png"


def load_pairs():
    """Récupère les couples estimé / mesuré et garde les modèles comparables."""
    response = requests.get(f"{API_URL}/estimated-vs-measured", params={"task": "text_generation"},
                            headers={"X-API-Key": config.API_KEY}, timeout=60)
    response.raise_for_status()
    pairs = pd.DataFrame(response.json())
    logging.info(f"{len(pairs)} mesures text_generation reliées à un modèle ComparIA")

    # Mêmes règles que le reste de l'analyse : énergie ComparIA disponible et au moins 200 conversations
    comparable = pairs[pairs["comparia_wh_per_req"].notna() & (pairs["nb_conversations"] >= config.MIN_CONVERSATIONS)]
    excluded = sorted(set(pairs["model"]) - set(comparable["model"]))
    logging.info(f"{len(comparable)} modèles comparables ; exclus : {', '.join(excluded)}")
    return comparable


def spearman(pairs, estimated_column, label):
    """Corrélation de rangs entre une colonne ComparIA (estimée) et la mesure AI Energy Score."""
    rho, p_value = spearmanr(pairs[estimated_column], pairs["gpu_energy_wh"])
    logging.info(f"Spearman {label} ({len(pairs)} modèles) : rho = {rho:.2f}, p = {p_value:.3f}")
    return rho, p_value


def save_figure(pairs, rho):
    """Nuage de points estimé / mesuré (échelles log), couleur selon le type de correspondance."""
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"exact": "#2A9D8F", "nom_different": "#E9C46A", "version_base": "#E76F51"}
    for match_type, group in pairs.groupby("match_type"):
        ax.scatter(group["gpu_energy_wh"], group["comparia_wh_per_req"], s=70,
                   color=colors.get(match_type, "#999"), label=match_type, edgecolors="white")
    for _, row in pairs.iterrows():
        ax.annotate(row["model"], (row["gpu_energy_wh"], row["comparia_wh_per_req"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Énergie MESURÉE — AI Energy Score (Wh GPU par requête, H100)")
    ax.set_ylabel("Énergie ESTIMÉE — ComparIA (Wh par requête)")
    ax.set_title(f"Estimé vs mesuré, text_generation ({len(pairs)} modèles) — Spearman ρ = {rho:.2f}")
    ax.legend(title="Correspondance")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    logging.info(f"Figure enregistrée : {FIGURE_PATH.relative_to(config.ROOT_DIR)}")


def run_analysis():
    pairs = load_pairs()
    print(pairs[["model", "match_type", "gpu_energy_wh", "comparia_wh_per_req", "comparia_mwh_per_token"]]
          .sort_values("gpu_energy_wh").to_string(index=False))

    rho, _ = spearman(pairs, "comparia_wh_per_req", "Wh/requête, toutes correspondances")
    spearman(pairs, "comparia_mwh_per_token", "mWh/token, toutes correspondances")

    # Analyse de sensibilité : seulement les correspondances exactes
    exact = pairs[pairs["match_type"] == "exact"]
    spearman(exact, "comparia_wh_per_req", "Wh/requête, correspondances exactes")
    spearman(exact, "comparia_mwh_per_token", "mWh/token, correspondances exactes")

    # Pourquoi pas Pearson ? Il travaille sur les valeurs : ici, deux modèles extrêmes (glm-4.5, glm-4.6)
    # suffisent à le tirer vers le haut. On le calcule pour le montrer, avec et sans ces deux modèles.
    r, p_value = pearsonr(pairs["comparia_wh_per_req"], pairs["gpu_energy_wh"])
    logging.info(f"Pour comparaison, Pearson sur les valeurs brutes : r = {r:.2f}, p = {p_value:.3f}")
    without_glm = pairs[~pairs["model"].str.startswith("glm")]
    r, p_value = pearsonr(without_glm["comparia_wh_per_req"], without_glm["gpu_energy_wh"])
    logging.info(f"Pearson sans glm-4.5 et glm-4.6 ({len(without_glm)} modèles) : r = {r:.2f}, p = {p_value:.3f}")

    save_figure(pairs, rho)


if __name__ == "__main__":
    config.setup_logging()
    run_analysis()
