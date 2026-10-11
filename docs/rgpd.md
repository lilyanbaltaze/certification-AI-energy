# Conformité RGPD (C4)

Registre simplifié du traitement mis en œuvre par le pipeline.

## Finalité
Fournir à une équipe « numérique responsable » une donnée fiable et à jour pour comparer des modèles d'IA selon leur qualité perçue (votes) et leur impact énergétique.
Le traitement porte sur **des modèles d'IA, pas sur des personnes** : aucune analyse ne vise les utilisateurs de ComparIA.

## Sources et licences
| Source | Producteur | Licence |
|---|---|---|
| ComparIA (conversations, votes) | ministère de la Culture, via data.gouv.fr | Licence Ouverte 2.0 (réutilisation libre, avec mention de la source) |
| AI Energy Score | Hugging Face | données publiques du leaderboard ; ne concernent que des modèles |

## Minimisation (article 5.1.c) : appliquée dès la collecte
`collect.py` sélectionne les colonnes une par une dans sa requête SQL DuckDB. Les autres colonnes **ne quittent jamais le serveur source**.

| Données exclues | Pourquoi |
|---|---|
| `session_hash`, `visitor_id` | identifiants pseudonymes : ils permettraient de relier les conversations d'une même personne |
| `opening_msg`, `conversation_a/b`, `system_prompt_a/b`, `short_summary`, `keywords` | texte écrit par les utilisateurs ou dérivé de ce texte : peut contenir des informations personnelles |
| `conv_comments_a/b` (votes) | commentaires libres des utilisateurs |
| `reactions.parquet` (fichier entier) | aucune analyse ne l'utilise |

**Données conservées** : noms de modèles, énergie, nombre de tokens, nombre de tours, catégories thématiques, date, résultat du vote.
Aucune de ces données ne permet d'identifier une personne. Les catégories (« Education », « Sports »…) sont des thèmes générés par ComparIA, pas du texte utilisateur.

## Hébergement et sécurité
- **Hébergement dans l'UE** : dataset BigQuery en région EU.
- **Moindre privilège** : un compte de service dédié, limité à deux rôles (BigQuery Data Editor et BigQuery Job User), sans rôle d'administration.
- **Secrets hors du code** : clé du compte de service rangée hors du dépôt, chemin et identifiant du projet dans `.env`, ignoré par Git.
- **Accès à l'API protégé** par une clé (`X-API-Key`).

## Durée de conservation
Les données sont **remplacées** à chaque collecte : fichiers bruts rafraîchis tous les 30 jours, tables BigQuery recréées à chaque lancement de `main.py`.
Aucun historique n'est accumulé au-delà de la dernière version publiée par les sources.

## Droits des personnes
Le traitement ne porte sur aucune donnée personnelle collectée par le projet. Les demandes relatives aux données d'origine relèvent du producteur, ComparIA (ministère de la Culture).
