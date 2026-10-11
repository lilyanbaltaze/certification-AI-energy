-- Couche gold : énergie ESTIMÉE (ComparIA) face à l'énergie MESURÉE (AI Energy Score).
-- C'est la règle d'agrégation qui combine les deux sources, via le lien
-- energy_measures.model_name → models (rempli grâce à la table de correspondance).
CREATE OR REPLACE VIEW ai_energy.gold_estime_vs_mesure AS
SELECT
    m.model_name AS model,
    m.provider,
    m.total_params,
    e.match_type,                           -- exact / nom_different / version_base
    e.hf_model_id,
    e.task,                                 -- ne comparer qu'à tâche égale
    e.reasoning_level,
    e.size_class,
    e.gpu_energy_wh,                        -- MESURÉ : Wh GPU par requête (H100)
    g.avg_kwh_per_req * 1000 AS comparia_wh_per_req,   -- ESTIMÉ : Wh par requête
    g.mwh_per_token AS comparia_mwh_per_token,          -- ESTIMÉ : coefficient par token
    g.nb_conversations
FROM ai_energy.energy_measures AS e
JOIN ai_energy.models AS m ON e.model_name = m.model_name
-- LEFT JOIN : on garde aussi les modèles appariés sans énergie ComparIA (ex. ministral),
-- pour que la perte soit visible plutôt que silencieuse
LEFT JOIN ai_energy.gold_energy_global AS g ON g.model = m.model_name;
