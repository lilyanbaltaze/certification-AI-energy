-- Modèle physique de données (MPD) : couche silver dans BigQuery, dataset ai_energy.
-- Traduction du MCD (docs/diagrammes/mcd.svg) : une table par entité,
-- et la table conversation_categories pour l'association n-n APPARTENIR.
--
-- BigQuery accepte les clés primaires et étrangères, mais ne les VÉRIFIE PAS
-- (« NOT ENFORCED ») : elles documentent les liens et aident l'optimiseur de requêtes.
-- La cohérence des données est garantie en amont par clean.py.
-- On ne déclare donc une clé que si les données la respectent vraiment :
-- une clé déclarée mais fausse peut faire renvoyer des résultats faux à BigQuery.
--
-- Ordre : on supprime d'abord les tables qui pointent vers les autres,
-- puis on recrée models en premier (les autres tables y font référence).

DROP TABLE IF EXISTS ai_energy.conversation_categories;
DROP TABLE IF EXISTS ai_energy.votes;
DROP TABLE IF EXISTS ai_energy.energy_measures;
DROP TABLE IF EXISTS ai_energy.conversations;
DROP TABLE IF EXISTS ai_energy.models;

-- Entité MODELE : un modèle d'IA de ComparIA
CREATE TABLE ai_energy.models (
    model_name     STRING NOT NULL,   -- nom normalisé (minuscules)
    total_params   FLOAT64,           -- milliards de paramètres (vide pour 10 modèles)
    active_params  FLOAT64,           -- milliards de paramètres actifs (modèles « mixture of experts »)
    provider       STRING NOT NULL,   -- fournisseur, calculé par get_provider
    PRIMARY KEY (model_name) NOT ENFORCED
);

-- Entité CONVERSATION : un duel entre deux modèles (associations OPPOSER_A et OPPOSER_B)
CREATE TABLE ai_energy.conversations (
    conversation_pair_id        STRING NOT NULL,
    timestamp                   TIMESTAMP NOT NULL,
    model_a_name                STRING NOT NULL,
    model_b_name                STRING NOT NULL,
    total_conv_a_kwh            FLOAT64,          -- énergie ESTIMÉE par ComparIA (vide pour 4 modèles)
    total_conv_b_kwh            FLOAT64,
    total_conv_a_output_tokens  INT64 NOT NULL,   -- 0 = réponse vide (gardée, exclue des moyennes en SQL)
    total_conv_b_output_tokens  INT64 NOT NULL,
    conv_turns                  INT64 NOT NULL,   -- toujours >= 1 après nettoyage
    PRIMARY KEY (conversation_pair_id) NOT ENFORCED,
    FOREIGN KEY (model_a_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED,
    FOREIGN KEY (model_b_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED
);

-- Association APPARTENIR (n-n) : une ligne par couple (conversation, catégorie).
-- L'entité CATEGORIE n'a qu'un libellé : pas besoin d'une table à part.
CREATE TABLE ai_energy.conversation_categories (
    conversation_pair_id  STRING NOT NULL,
    category              STRING NOT NULL,
    PRIMARY KEY (conversation_pair_id, category) NOT ENFORCED,
    FOREIGN KEY (conversation_pair_id) REFERENCES ai_energy.conversations (conversation_pair_id) NOT ENFORCED
);

-- Entité VOTE (associations COMPARER_A, COMPARER_B et GAGNER).
-- Pas de clé étrangère vers conversations : 2,9 % des votes n'ont pas de conversation publiée
-- par ComparIA. C'est pour ça que le vote porte lui-même les deux modèles du duel.
CREATE TABLE ai_energy.votes (
    conversation_pair_id  STRING NOT NULL,
    timestamp             TIMESTAMP NOT NULL,
    model_a_name          STRING NOT NULL,
    model_b_name          STRING NOT NULL,
    chosen_model_name     STRING,           -- vide = égalité
    both_equal            BOOL NOT NULL,
    PRIMARY KEY (conversation_pair_id) NOT ENFORCED,
    FOREIGN KEY (model_a_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED,
    FOREIGN KEY (model_b_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED,
    FOREIGN KEY (chosen_model_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED
);

-- Entité MESURE_ENERGIE : une mesure AI Energy Score (énergie GPU MESURÉE).
-- Association MESURER (0,1 côté mesure) : la clé étrangère model_name va dans cette table,
-- avec l'attribut de l'association, match_type. Vide si le modèle n'existe pas dans ComparIA.
CREATE TABLE ai_energy.energy_measures (
    hf_model_id      STRING NOT NULL,   -- identifiant Hugging Face (organisation/nom)
    task             STRING NOT NULL,   -- text_generation ou reasoning, à bien différencier
    reasoning_level  STRING NOT NULL,   -- effort de raisonnement réglé pour le test : low / medium / high (modèles gpt-oss), ou "default" (réglage par défaut)
    gpu_energy_wh    FLOAT64 NOT NULL,  -- Wh GPU pour UNE requête (le leaderboard affiche cette valeur × 1000 = Wh pour 1 000 requêtes)
    energy_score     INT64 NOT NULL,    -- étoiles 1 à 5 : efficacité relative à la tâche et à la classe de taille, recalculée à chaque mise à jour
    size_class       STRING NOT NULL,   -- A < 20B, B 20–66B, C > 66B
    test_date        DATE NOT NULL,
    model_name       STRING,            -- modèle ComparIA correspondant (table de correspondance)
    match_type       STRING,            -- exact / nom_different / version_base
    PRIMARY KEY (hf_model_id, task, reasoning_level) NOT ENFORCED,
    FOREIGN KEY (model_name) REFERENCES ai_energy.models (model_name) NOT ENFORCED
);
