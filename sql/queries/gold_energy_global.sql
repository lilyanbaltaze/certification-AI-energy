-- Couche gold : énergie ESTIMÉE par ComparIA, par modèle (méthode du notebook).
-- Mêmes colonnes que l'ancien energy_global.csv.
-- Choix d'analyse appliqués ici (et non au nettoyage) :
--   - les côtés sans énergie (4 modèles de 2024) sont ignorés ;
--   - les réponses vides (0 token) sont exclues : elles tireraient les moyennes vers le bas.
-- Les seuils (200 conversations, 0,00001 kWh) sont appliqués par l'API.
CREATE OR REPLACE VIEW ai_energy.gold_energy_global AS
WITH sides AS (
    -- Côtés A et B empilés : une ligne par (modèle, conversation)
    SELECT model_a_name AS model, total_conv_a_kwh AS kwh, total_conv_a_output_tokens AS tokens, conv_turns
    FROM ai_energy.conversations
    UNION ALL -- ici on empile
    SELECT model_b_name AS model, total_conv_b_kwh AS kwh, total_conv_b_output_tokens AS tokens, conv_turns
    FROM ai_energy.conversations
),
valid_sides AS (
    SELECT * FROM sides
    WHERE kwh IS NOT NULL AND tokens > 0
),
per_model AS ( -- agregation par modele
    SELECT
        model,
        AVG(kwh) AS avg_kwh,
        COUNT(*) AS nb_conversations,
        AVG(conv_turns) AS avg_turns,
        SUM(tokens) AS total_tokens,
        SUM(kwh) AS total_kwh
    FROM valid_sides
    GROUP BY model
)
SELECT
    model,
    avg_kwh,
    nb_conversations,
    avg_turns,
    avg_kwh / avg_turns AS avg_kwh_per_req,        -- kWh moyen par conv / nb moyen de requête = kwh moy par requete (un req = une question du user)
    total_tokens,
    total_kwh / total_tokens AS kwh_per_token,     -- somme kwh / somme tokens totaux , plus robuste qu'une moyenne de ratios
    total_kwh / total_tokens * 1000000 AS mwh_per_token -- on transforme les kwh en mwh
FROM per_model;
