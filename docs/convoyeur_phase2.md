# Convoyeur — phase 2 : détection d'anomalie non supervisée

## Le problème que cela résout

Les couches A (vision classique) et B (YOLO segmentation) ne détectent que
ce qu'on leur a décrit : une trace claire allongée. Elles ne verront pas un
défaut d'un type nouveau — jonction qui s'ouvre, cloque, arrachement de
revêtement, corps étranger de forme inhabituelle.

La détection d'anomalie non supervisée inverse le problème : on apprend à
quoi ressemble une bande **saine**, et on signale tout ce qui s'en écarte.
Or de la bande saine, vous en avez des heures — c'est la seule donnée
abondante du projet.

## Méthode conseillée : PatchCore

PatchCore (bibliothèque `anomalib`) extrait les caractéristiques locales
d'un réseau pré-entraîné, en constitue une banque mémoire à partir des
images saines, et mesure pour chaque zone d'une nouvelle image sa distance
à cette banque. Il produit une carte de chaleur, pas seulement un score, ce
qui permet de montrer **où** est l'anomalie.

Pourquoi celui-ci :

- pas de rétropropagation : l'« entraînement » est un simple passage sur les
  images saines, quelques minutes même sans GPU ;
- très bon sur les textures régulières, ce qu'est exactement une bande ;
- une centaine d'images saines suffisent.

## Mise en œuvre

```bash
pip install anomalib
```

Organisation des données :

```
data/convoyeur_anomalie/
├── train/bon/          200 à 500 images de bande saine, éclairages variés
└── test/
    ├── bon/            50 images saines non vues
    └── defaut/         toutes les images de défaut disponibles, même peu
```

Les images doivent être **recadrées sur la bande** (utilisez la ROI du
module `convoyeur_cv`) : sans cela, le modèle apprend surtout la structure
métallique environnante.

Points d'attention :

- ne mettez **aucun** défaut dans `train/bon` : une seule image de déchirure
  suffit à faire considérer les déchirures comme normales ;
- couvrez toute la variabilité normale dans `train/bon` — jour, nuit, bande
  chargée, bande vide, bande poussiéreuse. Une condition absente sera
  signalée comme anomalie, et vous noierez l'opérateur sous les fausses
  alertes ;
- réglez le seuil sur le lot `test/bon` : visez au plus une fausse alerte
  par heure, sinon le système sera désactivé par les opérateurs dès la
  première semaine.

## Architecture cible

Les trois couches sont complémentaires, pas concurrentes :

| Couche | Détecte | Coût de mise en place |
|--------|---------|------------------------|
| A — vision classique | Traces claires allongées | Nul, opérationnel immédiatement |
| B — YOLO segmentation | Types de défauts appris, avec leur nom | Annotation de 200 à 400 images |
| C — PatchCore | Tout écart au normal, y compris inconnu | Collecte d'images saines seulement |

Règle de fusion conseillée : **alarme si A ou B détecte**, et **avertissement
de vérification si seul C détecte**. C signale beaucoup de choses
inhabituelles mais sans les nommer ; le traiter comme une alarme ferme
dégraderait la confiance dans le système.

## Pour le rapport de stage

Cette phase 2 est un bon sujet de section « perspectives » : elle montre
que vous avez identifié la limite structurelle de l'approche supervisée
(l'absence de données de défaut) et que vous proposez une réponse adaptée,
sans prétendre l'avoir déjà mise en production.
