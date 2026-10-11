-- Couche gold : classement des modèles par score ajusté (méthode du notebook).
-- Victoire = 1 point, égalité = 0,5 point, défaite = 0 point.
-- Mêmes colonnes que l'ancien classement_score_ajuste.csv, pour que le dashboard reste identique.
-- Pas de seuil ici : le minimum de 200 duels est appliqué par l'API (paramètre min_duels),
-- pour que l'utilisateur puisse le changer.
CREATE OR REPLACE VIEW ai_energy.gold_classement_score_ajuste AS
WITH participations AS (
    -- Chaque vote compte pour les deux modèles du duel : on empile le côté A et le côté B
    SELECT model_a_name AS model, chosen_model_name, both_equal FROM ai_energy.votes
    UNION ALL
    SELECT model_b_name AS model, chosen_model_name, both_equal FROM ai_energy.votes
),
model_points AS (
    SELECT
        model,
        SUM(CASE
                WHEN both_equal THEN 0.5                -- égalité : un demi-point chacun
                WHEN chosen_model_name = model THEN 1.0 -- victoire
                ELSE 0.0                                -- défaite
            END) AS points,
        COUNT(*) AS duels
    FROM participations
    GROUP BY model
)
SELECT
    model,
    points,
    duels,
    points / duels * 100 AS score,
    -- Intervalle de confiance à 95 % : 1,96 × √(p(1−p)/n), exprimé en points de pourcentage
    1.96 * SQRT((points / duels) * (1 - points / duels) / duels) * 100 AS ic,
    (points / duels - 1.96 * SQRT((points / duels) * (1 - points / duels) / duels)) * 100 AS score_min,
    (points / duels + 1.96 * SQRT((points / duels) * (1 - points / duels) / duels)) * 100 AS score_max
FROM model_points;
