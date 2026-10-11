"""Configuration centrale : tous les chemins, URLs et seuils du projet."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Racine calculée depuis ce fichier (src/config.py → on remonte d'un dossier) :
# les scripts marchent quel que soit le dossier depuis lequel on les lance.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"

# Les secrets sont dans .env, jamais dans le code
load_dotenv(ROOT_DIR / ".env")
GCP_PROJECT = os.getenv("GCP_PROJECT")
BQ_DATASET = os.getenv("BQ_DATASET", "ai_energy")
API_KEY = os.getenv("API_KEY")  # clé exigée par l'API (en-tête X-API-Key)

# Sources
COMPARIA_CONVERSATIONS_URL = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/COMPARIA/conversations.parquet"
COMPARIA_VOTES_URL = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/COMPARIA/votes.parquet"
AIES_TEXT_GENERATION_URL = "https://huggingface.co/spaces/AIEnergyScore/Leaderboard/resolve/main/data/energy/text_generation.csv"
AIES_REASONING_URL = "https://huggingface.co/spaces/AIEnergyScore/Leaderboard/resolve/main/data/energy/reasoning.csv"

# Collecte : une source distante peut être momentanément indisponible,
# on réessaie donc plusieurs fois avant d'abandonner.
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 10
# Un fichier local plus vieux que ça est retéléchargé : ComparIA est mis à jour chaque mois
MAX_FILE_AGE_DAYS = 30

# Seuils d'analyse repris du notebook (à réexaminer sur les données complètes)
MIN_DUELS = 200
MIN_CONVERSATIONS = 200
MIN_AVG_KWH = 0.00001
MIN_CONV_PER_CATEGORY = 30


def setup_logging():
    """Affiche les logs dans la console ET les garde dans logs/pipeline.log (trace pour le rapport)."""
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")],
    )
