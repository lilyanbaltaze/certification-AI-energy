"""Étape 2 du pipeline : nettoyage et homogénéisation (C3).

Architecture médaillon :  couche bronze (data/raw/, données brutes)
à silver (data/processed/, données propres sur 5 tables).

Règle : ici on corrige ou supprime ce qui est FAUX dans la donnée.
Les choix d'analyse (seuils de 200 duels, exclusion des réponses vides des moyennes…)
sont faits plus tard en SQL (couche gold), pour rester modifiables.
"""
import logging
import pandas as pd
from src import config

def normalize_model_name(names):
    """Met les noms de modèles en minuscules, sans espaces autour.
    Pour éviter que le même modèle pourrait apparaître sous deux noms et que les jointures échouent."""
    return names.str.strip().str.lower()

def get_provider(model):
    #Fournisseur du modèle à partir du début de son nom.
    m = model.lower()
    if m.startswith(("gemini", "gemma")):                              return "Google"
    if m.startswith("deepseek"):                                       return "DeepSeek"
    if m.startswith(("mistral", "magistral", "ministral", "mixtral")): return "Mistral"
    if m.startswith("glm"):                                            return "Zhipu"
    if m.startswith(("qwen", "qwq")):                                  return "Alibaba"
    if m.startswith("grok"):                                           return "xAI"
    if "nemotron" in m:                                                return "NVIDIA"
    if m.startswith(("llama", "hermes")):                              return "Meta"
    if m.startswith("kimi"):                                           return "Moonshot"
    if m.startswith("claude"):                                         return "Anthropic"
    if m.startswith(("gpt", "o3", "o4")):                              return "OpenAI"
    if m.startswith("minimax"):                                        return "MiniMax"
    if m.startswith("phi"):                                            return "Microsoft"
    if m.startswith("eurollm"):                                        return "EuroLLM"
    if m.startswith("olmo"):                                           return "Allen AI"
    # Règles ajoutées le 11/10 : ces fournisseurs n'existaient pas dans l'échantillon du notebook
    if m.startswith(("aya", "c4ai", "command")):                       return "Cohere"
    if m.startswith("apertus"):                                        return "Swiss AI"
    if m.startswith("lfm"):                                            return "Liquid AI"
    if m.startswith("jamba"):                                          return "AI21 Labs"
    if m.startswith("trinity"):                                        return "Arcee AI"
    if m.startswith("yi-"):                                            return "01.AI"
    if m.startswith("chocolatine"):                                    return "Jpacifico"
    return "Autre"


def clean_conversations(conv):
    """Nettoie les conversations et sépare les catégories dans leur propre table."""
    logging.info(f"conversations : {len(conv)} lignes en entrée")

    conv["model_a_name"] = normalize_model_name(conv["model_a_name"])
    conv["model_b_name"] = normalize_model_name(conv["model_b_name"])

    # Une conversation sans aucun tour est impossible : entrée corrompue.
    # (Elle ferait aussi diviser par zéro le calcul du kWh par requête.)
    corrupted = conv["conv_turns"] <= 0
    logging.info(f"conversations : {corrupted.sum()} supprimées (conv_turns <= 0)")
    conv = conv[~corrupted]

    # Une conversation peut avoir plusieurs catégories : on les met dans une table à part,
    # une ligne par (conversation, catégorie). Une base de données ne stocke pas de liste dans une case.
    categories = conv[["conversation_pair_id", "categories"]].explode("categories")
    categories = categories.rename(columns={"categories": "category"})
    categories["category"] = categories["category"].str.strip()
    empty = categories["category"].isna() | (categories["category"] == "")
    logging.info(f"catégories : {empty.sum()} catégories vides retirées")
    categories = categories[~empty]
    # Certaines conversations citent deux fois la même catégorie : doublon sans information,
    # et le couple (conversation, catégorie) est la clé primaire de la table → on le retire
    duplicates = categories.duplicated()
    logging.info(f"catégories : {duplicates.sum()} doublons (même catégorie citée deux fois) retirés")
    categories = categories[~duplicates]
    logging.info(f"catégories : {len(categories)} lignes, {categories['category'].nunique()} catégories distinctes")

    # Les nombres de paramètres décrivent le modèle, pas la conversation : ils iront dans la table models
    conversations = conv.drop(columns=["categories", "model_a_total_params", "model_b_total_params",
                                       "model_a_active_params", "model_b_active_params"])

    # Réponses vides (0 token, 0 kWh) : gardées volontairement, ce sont des faits réels.
    # Elles seront exclues des moyennes d'énergie en SQL (choix d'analyse, couche gold).
    empty_a = (conversations["total_conv_a_output_tokens"] <= 0).sum()
    empty_b = (conversations["total_conv_b_output_tokens"] <= 0).sum()
    logging.info(f"conversations : {empty_a + empty_b} réponses vides conservées (exclues des moyennes en SQL)")

    logging.info(f"conversations : {len(conversations)} lignes en sortie")
    return conversations, categories


def clean_votes(votes):
    #Supprime les votes corrompus et corrige le champ both_equal.
    logging.info(f"votes : {len(votes)} lignes en entrée")

    for column in ["model_a_name", "model_b_name", "chosen_model_name"]:
        votes[column] = normalize_model_name(votes[column])

    # Un gagnant qui n'a pas participé au duel est impossible : vote corrompu, supprimé
    has_winner = votes["chosen_model_name"].notna()
    winner_in_duel = (votes["chosen_model_name"] == votes["model_a_name"]) | (votes["chosen_model_name"] == votes["model_b_name"])
    corrupted = has_winner & ~winner_in_duel
    logging.info(f"votes : {corrupted.sum()} supprimés (gagnant hors du duel)")
    votes = votes[~corrupted].copy()

    # both_equal vide alors qu'un gagnant est renseigné (votes d'oct. à déc. 2024) :
    # s'il y a un gagnant, ce n'est pas une égalité → on corrige en False.
    # Sans ça, ces votes comptaient comme duels mais jamais comme victoires.
    to_fix = votes["both_equal"].isna() & votes["chosen_model_name"].notna()
    votes.loc[to_fix, "both_equal"] = False
    logging.info(f"votes : {to_fix.sum()} valeurs both_equal corrigées en False")

    # Cas restant : ni gagnant ni égalité → le vote ne dit rien, on le supprime
    unusable = votes["both_equal"].isna()
    logging.info(f"votes : {unusable.sum()} supprimés (ni gagnant ni égalité)")
    votes = votes[~unusable]
    votes["both_equal"] = votes["both_equal"].astype(bool)

    logging.info(f"votes : {len(votes)} lignes en sortie")
    return votes


def clean_aies(path, task):
    """Nettoie un CSV AI Energy Score (text_generation ou reasoning)."""
    # utf-8-sig : retire le BOM (caractère invisible) présent en tête de text_generation.csv,
    # sinon la première colonne s'appellerait "﻿model" au lieu de "model"
    aies = pd.read_csv(path, encoding="utf-8-sig")
    logging.info(f"{path.name} : {len(aies)} lignes en entrée")

    # Astérisque parasite dans certains noms (ex. ibm-granite/*granite-4.0-h-tiny)
    with_star = aies["model"].str.contains("*", regex=False)
    aies["model"] = aies["model"].str.replace("*", "", regex=False)
    logging.info(f"{path.name} : {with_star.sum()} astérisques retirés")

    # Dans reasoning.csv, le niveau de raisonnement est collé au nom ("openai/gpt-oss-20b low") :
    # on le sépare dans sa propre colonne pour que le nom serve de clé de jointure
    split = aies["model"].str.split(" ", n=1, expand=True)
    aies["hf_model_id"] = split[0]
    aies["reasoning_level"] = split[1] if split.shape[1] > 1 else None
    logging.info(f"{path.name} : {aies['reasoning_level'].notna().sum()} niveaux de raisonnement séparés du nom")
    # Sans niveau précisé, le modèle a été testé dans sa configuration par défaut (méthodologie AIES).
    # On écrit "default" plutôt que de laisser vide : cette colonne fait partie de la clé primaire,
    # et une clé ne peut pas contenir de valeur vide.
    aies["reasoning_level"] = aies["reasoning_level"].fillna("default")

    # Dates au format "Feb 25" en vraie date (1er du mois), triable et comparable
    aies["test_date"] = pd.to_datetime(aies["test date"], format="%b %y").dt.date

    # text_generation et reasoning sont deux tâches différentes (énergie ×10 à ×100) :
    # la colonne task empêche de les mélanger par erreur
    aies["task"] = task

    # total_gpu_energy : Wh GPU par requête (le leaderboard l'affiche ×1000, en Wh pour 1 000 requêtes)
    energy = aies.rename(columns={"total_gpu_energy": "gpu_energy_wh", "class": "size_class"})
    energy = energy[["hf_model_id", "task", "reasoning_level", "gpu_energy_wh",
                     "energy_score", "size_class", "test_date"]]
    logging.info(f"{path.name} : {len(energy)} lignes en sortie")
    return energy


def build_models(conv, votes):
    """Construit la table models : une ligne par modèle, avec fournisseur et paramètres."""
    # Paramètres : on empile les côtés a et b (même méthode que dans mon notebook pour l'énergie)
    side_a = conv[["model_a_name", "model_a_total_params", "model_a_active_params"]]
    side_a.columns = ["model_name", "total_params", "active_params"]
    side_b = conv[["model_b_name", "model_b_total_params", "model_b_active_params"]]
    side_b.columns = ["model_name", "total_params", "active_params"]
    # first() prend la première valeur non vide de chaque modèle
    models = pd.concat([side_a, side_b]).groupby("model_name").first().reset_index()

    # 3 modèles n'apparaissent que dans les votes : on les ajoute pour que chaque vote ait son modèle
    vote_models = pd.concat([votes["model_a_name"], votes["model_b_name"]]).unique()
    missing = [m for m in vote_models if m not in set(models["model_name"])]
    models = pd.concat([models, pd.DataFrame({"model_name": missing})], ignore_index=True)
    logging.info(f"models : {len(missing)} modèles présents uniquement dans les votes ajoutés ({', '.join(missing)})")

    models["provider"] = models["model_name"].apply(get_provider)
    others = models.loc[models["provider"] == "Autre", "model_name"].tolist()
    logging.info(f"models : {len(others)} modèles sans fournisseur reconnu (« Autre ») : {', '.join(others)}")

    logging.info(f"models : {len(models)} modèles")
    return models


def link_energy_to_models(energy, mapping):
    """Ajoute à chaque mesure AI Energy Score le nom du modèle ComparIA correspondant
    (vide si le modèle n'existe pas dans ComparIA), d'après la table de correspondance.
    Le lien est rangé ici et non dans models : dans le MCD, une mesure concerne 0 ou 1 modèle
    (cardinalité 0,1), donc la clé étrangère va du côté de la mesure."""
    mapping = mapping.rename(columns={"comparia_model": "model_name"})
    energy = energy.merge(mapping[["hf_model_id", "model_name", "match_type"]], on="hf_model_id", how="left")
    linked = energy["model_name"].notna()
    logging.info(f"energy_measures : {linked.sum()} mesures reliées à {energy.loc[linked, 'model_name'].nunique()} modèles ComparIA")
    return energy


def check_mapping(mapping, models, energy):
    """Vérifie que chaque ligne de la table de correspondance existe dans les deux sources.
    Une faute de frappe dans le CSV ferait sinon disparaître un modèle en silence."""
    unknown_comparia = set(mapping["comparia_model"]) - set(models["model_name"])
    unknown_hf = set(mapping["hf_model_id"]) - set(energy["hf_model_id"])
    if unknown_comparia or unknown_hf:
        raise ValueError(f"model_mapping.csv : noms introuvables → ComparIA {unknown_comparia}, AI Energy Score {unknown_hf}")
    logging.info(f"model_mapping.csv : {len(mapping)} correspondances vérifiées dans les deux sources")


def run_clean():
    """Lance tout le nettoyage (appelée par main.py) : bronze → silver."""
    raw = config.DATA_RAW_DIR
    out = config.DATA_PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    conv_raw = pd.read_parquet(raw / "comparia_conversations.parquet")
    votes_raw = pd.read_parquet(raw / "comparia_votes.parquet")
    mapping = pd.read_csv(config.ROOT_DIR / "references" / "model_mapping.csv")

    conversations, categories = clean_conversations(conv_raw)
    votes = clean_votes(votes_raw)
    energy = pd.concat([
        clean_aies(raw / "aies_text_generation.csv", "text_generation"),
        clean_aies(raw / "aies_reasoning.csv", "reasoning"),
    ], ignore_index=True)

    # Les paramètres viennent des conversations brutes (noms déjà normalisés par clean_conversations)
    conv_raw["model_a_name"] = normalize_model_name(conv_raw["model_a_name"])
    conv_raw["model_b_name"] = normalize_model_name(conv_raw["model_b_name"])
    models = build_models(conv_raw, votes)
    check_mapping(mapping, models, energy)
    energy = link_energy_to_models(energy, mapping)

    # Parquet : garde les types (dates, booléens) et se charge directement dans BigQuery
    models.to_parquet(out / "models.parquet", index=False)
    conversations.to_parquet(out / "conversations.parquet", index=False)
    categories.to_parquet(out / "conversation_categories.parquet", index=False)
    votes.to_parquet(out / "votes.parquet", index=False)
    energy.to_parquet(out / "energy_measures.parquet", index=False)
    logging.info(f"couche silver écrite dans {out} (5 tables)")


if __name__ == "__main__":
    config.setup_logging()
    run_clean()
