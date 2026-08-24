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

---

## 6. Le modèle ne s'améliore pas tout seul

C'est le malentendu le plus fréquent sur les systèmes de vision déployés,
et il vaut la peine d'être écrit noir sur blanc.

**Un modèle YOLO n'apprend rien de ce qu'il voit en production.** Il applique
ce qu'il a appris à l'entraînement et n'en garde aucune trace. Mettre le
système en service pendant six mois ne le rendra pas meilleur d'un iota.

Et il ne faut **surtout pas** le réentraîner automatiquement sur ses propres
sorties. Une fausse alarme réinjectée comme vérité terrain devient une règle
apprise ; elle se reproduit alors plus souvent, elle est réapprise, et la
dérive s'installe. Le piège est que les métriques internes du modèle
**s'améliorent** pendant que ses résultats réels se dégradent : il devient
de plus en plus sûr de ses erreurs.

### La boucle qui fonctionne

```
production  ->  collecte des cas UTILES  ->  correction humaine
            ->  reentrainement  ->  evaluation  ->  promotion explicite
```

Une seule étape est humaine, et elle est irremplaçable. Les autres sont
outillées.

### Choisir quelles images annoter

C'est l'étape qui coûte le plus de temps, et celle où l'on en gagne le plus.
Annoter 500 images de bande saine tirées au hasard n'apporte rien ; annoter
50 images bien choisies change le modèle.

```bash
python scripts/collecter_pour_reannotation.py --source runs/alarmes --nombre 100
```

Trois signaux, du plus fort au plus faible :

1. **Désaccord entre la couche A (vision classique) et la couche B (YOLO).**
   Quand deux méthodes indépendantes ne disent pas la même chose, l'une des
   deux se trompe — et c'est précisément là que l'annotation humaine
   tranche. C'est le signal le plus rentable du projet, et il ne coûte rien
   puisque les deux couches tournent déjà.
2. **Confiance faible.** Une détection à 0,30 est une hésitation.
3. **Classe rare.** Une `jonction_defectueuse` vaut plus qu'une millième
   `dechirure` : le modèle en a vu peu.

Le script exporte les images retenues avec des **pré-annotations** issues de
la couche A, dans un dossier prêt à ouvrir dans CVAT ou LabelImg. Ces
pré-annotations sont une aide à la saisie, pas une vérité : les entraîner
sans relecture reviendrait à apprendre au modèle les erreurs de la vision
classique, ce qui n'apporterait rien.

### Ce qui est déjà automatique

- le pipeline **enregistre une capture** de chaque alarme, avec son type,
  sa gravité et sa taille en millimètres ;
- `src/mlops/surveiller.py` **signale la dérive**, le silence d'une caméra
  et les rafales ;
- le registre **trace** quelle version du dataset a produit quel modèle ;
- la promotion **refuse** un modèle incohérent avec les classes déclarées.

Il ne reste donc qu'à relire les images sélectionnées et à relancer
l'entraînement. En pratique, une demi-journée par mois suffit à faire
progresser un modèle déployé.

---

## 7. Pourquoi l'apprentissage par renforcement ne s'applique pas

La question revient naturellement : pourquoi ne pas laisser le modèle
apprendre seul, par renforcement, comme un agent qui s'améliore à l'usage ?

Deux raisons de fond, indépendantes des moyens disponibles.

### Il n'y a pas de récompense

L'apprentissage par renforcement suppose qu'après chaque action,
l'environnement renvoie une **récompense** indiquant si elle était bonne.
C'est ce signal qui remplace l'annotation humaine.

Ici, il n'existe pas. Quand le modèle annonce une déchirure, rien dans
l'image suivante ne confirme ni n'infirme. La bande ne répond pas. Le seul
juge est un humain qui va voir.

Autrement dit : **la récompense, c'est l'annotation**. Le renforcement ne
supprime pas la boucle humaine, il la renomme.

### Ce n'est pas un problème séquentiel

Le renforcement s'applique aux problèmes où une action **change l'état** du
monde et influence les décisions suivantes — un robot qui se déplace, une
partie de jeu, une politique de maintenance.

La détection ne fonctionne pas ainsi : une image entre, une réponse sort,
et déclarer une déchirure ne modifie pas l'image suivante. C'est de la
perception supervisée. Y appliquer le renforcement serait une erreur de
catégorie, pas seulement un choix coûteux.

### Ce qui, en revanche, apprend sans annotation

Deux approches méritent d'être connues, et l'une est déjà prévue :

- **Détection d'anomalie non supervisée** (PatchCore, voir
  [convoyeur_phase2.md](convoyeur_phase2.md)). On entraîne **uniquement sur
  de la bande saine**, dont vous avez des heures, et le modèle signale tout
  écart. Aucune annotation de défaut n'est nécessaire. C'est ce qui se
  rapproche le plus de l'intuition « il apprend tout seul ».
- **Pré-entraînement auto-supervisé** sur des images non annotées. Le
  modèle apprend la structure visuelle d'une bande, pas ce qu'est un
  défaut : il faut toujours des étiquettes pour la tête de détection.

### Et le seul vrai signal de retour de l'usine

Il existe pourtant un retour automatique, produit par l'usine sans travail
supplémentaire : le **registre de maintenance**. Si une bande a été réparée
le 12 mars, une alarme du 11 mars était probablement vraie ; une semaine
sans intervention après une alarme la rend douteuse.

```bash
python scripts/confronter_maintenance.py --maintenance maintenance.csv \
       --exporter data/a_annoter_confirmees
```

Le script en tire trois choses que rien d'autre ne donne :

| Sortie | Intérêt |
|---|---|
| **Précision estimée** | Part des alarmes suivies d'une réparation — mesurée sur vos installations, pas sur un lot de test |
| **Rappel estimé** | Part des réparations précédées d'une alarme |
| **Détections manquées** | Une réparation sans alarme préalable est un défaut que le système n'a pas vu. **C'est la seule façon de mesurer le rappel en exploitation** |

Chaque détection manquée désigne des enregistrements précis à récupérer :
ce sont les images les plus précieuses du projet, puisqu'elles contiennent
un défaut réel que le modèle actuel ne voit pas.

L'appariement est **un pour un** : une réparation ne confirme qu'une alarme,
la plus proche. Sans cette contrainte, la précision afficherait 100 % dès
que les interventions sont fréquentes — on mesurerait la densité du planning
de maintenance, pas la qualité du modèle. Mesuré sur un jeu d'essai :
57 % de précision avec l'appariement un pour un, contre 100 % sans.

C'est une supervision **faible, différée et imparfaite**. Ce n'est pas du
renforcement, et les chiffres sont à lire comme des tendances. Mais c'est
gratuit, et un rappel qui chute d'un mois sur l'autre est un signal fiable.
