# MLOps : traçabilité, promotion, surveillance

Un projet de vision ne meurt pas d'un mauvais modèle, il meurt d'un
`best_final_v3.pt` dont personne ne sait sur quelles données il a été
entraîné. Les trois mécanismes ci-dessous coûtent une journée à mettre en
place et évitent ce scénario.

---

## 1. Empreinte de dataset

```bash
python -m src.mlops.registre --empreinte convoyeur
```

```
Dataset 'convoyeur'
  empreinte : 3ef9715e9234edc5
  images    : 600  {'train': 420, 'val': 120, 'test': 60}
  instances : 733
      dechirure               527
      corps_etranger          113
      perforation              93
```

L'empreinte est un SHA-256 des noms d'images et du **contenu des
annotations**. Recompresser une image ne la change pas ; corriger une
annotation la change. C'est exactement le comportement voulu : on suit la
vérité terrain, pas l'encodage.

Son usage est diagnostique. Deux entraînements donnant des résultats
différents :

- **même empreinte** → seule la configuration a changé (hyperparamètres,
  augmentation, résolution). Cherchez là.
- **empreinte différente** → la donnée a changé. Comparer les métriques n'a
  alors aucun sens : vous ne mesurez pas la même chose.

Sans cette distinction, on passe des jours à optimiser des hyperparamètres
alors que la vraie cause est une ré-annotation partielle.

---

## 2. Registre des expériences

Chaque entraînement s'inscrit automatiquement dans `models/registre.json` :
empreinte du dataset, hyperparamètres retenus, métriques, chemin des poids.

```bash
python -m src.mlops.registre --lister
```

```
modele      ver   date              dataset            imgs   mAP50  mAP50-95  prod
----------------------------------------------------------------------------------
convoyeur   v1    2026-08-21T16:27  3ef9715e9234edc5    600   0.613     0.353 <=
```

Les métriques de **validation** servent à choisir le meilleur epoch : elles
sont optimistes par construction, puisque c'est sur elles qu'on a
sélectionné. Seules celles du lot de **test** sont présentables comme
performance réelle, et le registre les stocke séparément :

```bash
python -m src.train.evaluer --modele convoyeur          # remplit "evaluations.test"
```

C'est le chiffre à mettre dans le rapport de stage.

---

## 3. Promotion en production

```bash
python -m src.mlops.registre --promouvoir convoyeur --version v1
```

Les poids sont copiés vers `models/convoyeur/production.pt`, et le pipeline
ne charge **que** ce fichier :

1. `models/<modele>/production.pt` — promu explicitement ;
2. sinon le chemin de `configs/pipeline.yaml` — mode mise au point, signalé
   dans la console ;
3. sinon rien — le module bascule en repli (vision classique, photométrie,
   COCO générique).

L'intérêt est la séparation nette entre expérimenter et déployer : un
entraînement en cours dans `runs/` ne peut pas se retrouver en production
par accident. Le passage en production devient une décision datée et tracée.

---

## 4. Surveillance en production

```bash
python -m src.mlops.surveiller --jours 7
python -m src.mlops.surveiller --jours 30 --exporter runs/rapport.csv
```

Un modèle déployé ne prévient pas quand il se dégrade : il continue de
produire des sorties d'apparence normale. Le script analyse le journal
d'alarmes et signale trois défaillances :

| Signal | Interprétation | Réaction |
|--------|----------------|----------|
| **Dérive** du taux d'alarme (±50 % par défaut) | Caméra déplacée, éclairage modifié, bande remplacée : les conditions ne sont plus celles de l'entraînement | Ré-échantillonner des images dans les nouvelles conditions et affiner le modèle |
| **Silence** d'une caméra pendant plus de 2 jours | Presque toujours une panne de flux, pas une usine devenue parfaite | Vérifier le RTSP **avant** de conclure à l'absence d'incident |
| **Rafale** (20 alarmes en 10 min) | L'opérateur va cesser de lire les alertes, puis désactiver le système | Remonter le seuil, ou corriger la cause (soleil rasant, nettoyage de bande) |

Le silence est le mode de défaillance le plus dangereux : il ressemble
exactement au bon fonctionnement. C'est la raison d'être de ce script.

---

## 5. Ce qui n'est volontairement pas fait

Pour un stage, ajouter les outils suivants coûterait plus de temps qu'il
n'en rapporterait. Ils se justifieront quand le système sera industrialisé,
et le mentionner dans le rapport montre que le périmètre est un choix :

- **MLflow / Weights & Biases** : le registre JSON couvre le besoin actuel
  (moins de vingt entraînements). Ultralytics s'intègre nativement à MLflow
  le jour où l'équipe en aura besoin.
- **DVC** : le versionnement des données par empreinte suffit tant que les
  datasets restent sur un seul poste. DVC devient utile à plusieurs.
- **Ré-entraînement automatique** : dangereux ici. Un ré-entraînement
  déclenché sur des données de production non relues apprendrait ses propres
  faux positifs. La boucle doit rester : dérive détectée → annotation
  humaine → entraînement → promotion explicite.
- **Conteneurisation** : utile pour le déploiement final sur le serveur de
  l'usine, pas pendant la phase de mise au point.
