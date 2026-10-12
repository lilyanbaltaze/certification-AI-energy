-- Couche gold : modèles par catégorie de conversation (méthode de mon notebook).
-- Mêmes colonnes que mon fichier ranking_per_cat.csv (dont le nom « categories »).
-- Le seuil de 30 conversations par modèle et par catégorie est appliqué plus tard via l'API.
CREATE OR REPLACE VIEW ai_energy.gold_ranking_per_cat AS
WITH model_categories AS (
    -- Une ligne par (modèle, catégorie, conversation) : côtés A et B empilés
    SELECT c.model_a_name AS model, cc.category
    FROM ai_energy.conversation_categories AS cc
    JOIN ai_energy.conversations AS c USING (conversation_pair_id)
    UNION ALL
    SELECT c.model_b_name AS model, cc.category
    FROM ai_energy.conversation_categories AS cc
    JOIN ai_energy.conversations AS c USING (conversation_pair_id)
),
counts AS (
    SELECT model, category, COUNT(*) AS nb_conv_cat
    FROM model_categories
    GROUP BY model, category
)
SELECT
    counts.model,
    counts.category AS categories, -- renommé avec le même nom qu'utilisé dans le dashboard
    counts.nb_conv_cat,
    energy.mwh_per_token,
    energy.avg_kwh,
    energy.avg_kwh_per_req
FROM counts
-- Jointure interne : seuls les modèles qui ont une énergie apparaissent (comme dans le notebook)
JOIN ai_energy.gold_energy_global AS energy USING (model);
