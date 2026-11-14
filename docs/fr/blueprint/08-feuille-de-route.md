# Feuille de route

## Phase 1 : Produit minimum viable (MVP)
- [x] Support des sources de données en répertoire local
- [x] Support d'un seul modèle de base (Qwen 3.5 0.8B)
- [x] Support d'une seule édition (normal)
- [x] Construction de corpus basique (format conversation)
- [x] Pipeline d'entraînement QLoRA
- [x] Outils CLI (ego init / train / run)

## Phase 2 : Support multi-versions
- [ ] Quatre éditions par défaut (lite/normal/plus/super)
- [ ] Support multi-bases (Qwen, Gemma, Phi)
- [ ] Éditions personnalisées
- [ ] Gestion des versions et retour en arrière

## Phase 3 : Expansion des sources de données
- [ ] Support des sources de données URL
- [ ] Sources de données en stockage cloud (S3/OSS)
- [ ] Mécanisme de plugins pour les sources de données
- [ ] Collecte incrémentale des données et entraînement incrémental

## Phase 4 : Intelligence
- [ ] Évaluation automatique de l'alignement
- [ ] Auto-ajustement des paramètres d'entraînement
- [ ] Entraînement automatique planifié (scheduler)
- [ ] Interface de gestion Web UI

## Phase 5 : Intégration iFay
- [ ] Adaptateur de format de données iFay
- [ ] Intégration du protocole de chargement Fayger
- [ ] Mécanisme de distribution des packages Ego
