"""Lance tout le pipeline de données, dans l'ordre : python main.py"""
from src import config
from src.clean import run_clean
from src.collect import run_collect
from src.load_bigquery import run_load

config.setup_logging()
run_collect()  # étape 1 : bronze (data/raw/)
run_clean()    # étape 2 : silver (data/processed/)
run_load()     # étape 3 : silver chargée dans BigQuery
