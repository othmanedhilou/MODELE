# Détection d'objets sur vidéosurveillance — cimenterie

Trois modèles de vision par ordinateur ajoutés au système de caméras de
l'usine, en complément des modèles déjà entraînés par l'équipe (fumée, feu,
gilet, casque) :

| # | Modèle | Question à laquelle il répond | Approche |
|---|--------|-------------------------------|----------|
| 1 | **Éclairage** | Quel luminaire est en panne ? Quelle zone est sous-éclairée la nuit ? | YOLO (détection des luminaires) + photométrie |
| 2 | **Véhicules** | Quels engins circulent, où, combien, depuis quand ? | YOLO fine-tuné + suivi ByteTrack + règles de zone |
| 3 | **Convoyeur** | La bande présente-t-elle une anomalie ? (10 classes : déchirure, fissure, perforation, jonction, cloque, usure, bord, corps étranger, déversement, désalignement) | Vision classique **puis** YOLO segmentation |
| + | **Plaque** | Quel véhicule est entré, à quelle heure ? | Greffé sur le modèle 2 : localisation + OCR |

Chacun est indépendant : on peut en déployer un sans attendre les autres.

---

## Ce qui existe aujourd'hui, et ce qui n'existe pas

Pour éviter tout malentendu :

| | État |
|---|---|
| Code des trois modèles, pipeline, outils | **Écrit et testé** |
| Jeu de données **synthétique** convoyeur | **600 images annotées**, générées par `scripts/generer_dataset_convoyeur.py` |
| Jeu de données **réel** (usine) | **Inexistant** — aucune image de vos caméras |
| Modèle éclairage | Entraîné sur données synthétiques, en production |
| Modèle convoyeur | **À entraîner sur Colab** — la taxonomie est passée à 10 classes, l'ancien modèle a été retiré |
| Jeu de données **synthétique** éclairage | **600 images, 2945 luminaires annotés** |
| Modèle véhicules | **Non entraîné** — repli COCO (voiture/camion/bus) |
| Module plaque | **Opérationnel sans entraînement** — localisation + OCR EasyOCR |
| `yolo11s.pt` à la racine | Modèle générique COCO téléchargé chez Ultralytics, **pas** un modèle de ce projet |
| Fichiers dans `data/raw/` | Images et vidéo **synthétiques**, dessinées par `scripts/generer_*.py` pour tester le code. Ce ne sont pas des images de l'usine. |

Le seul composant qui **détecte réellement quelque chose aujourd'hui** est la
couche A du convoyeur : c'est un algorithme de vision classique, pas un
modèle entraîné, et il ne nécessite donc aucune donnée. Les modèles 1 et 2
tournent en mode de repli (photométrie, COCO générique) en attendant leurs
données.

Le modèle convoyeur a été entraîné et évalué sur des **données
synthétiques** : cela prouve que la chaîne technique fonctionne de bout en
bout, mais ne dit rien de la performance sur vos vraies caméras. Voir
[docs/sans_donnees.md](docs/sans_donnees.md) pour les quatre façons de
démarrer sans données d'usine, et ce que chacune permet vraiment.

---

## Point important avant de commencer

Ce projet a été développé dans un chemin contenant des caractères arabes,
du type :

```
C:\Users\<utilisateur>\OneDrive\سطح المكتب\MODELE
```

Sous Windows, `cv2.imread` et `cv2.imwrite` passent par l'API ANSI et
**échouent silencieusement** sur ce type de chemin — ils renvoient `None` ou
`False` sans lever d'exception, ce qui donne des bugs très difficiles à
diagnostiquer. Le code du projet contourne le problème (`lire_image` et
`ecrire_image` dans `src/utils/common.py`), mais des bibliothèques tierces
peuvent encore trébucher dessus.

**Recommandation : déplacez le projet vers un chemin sans accents ni
caractères non latins**, par exemple `C:\projets\MODELE`. Le projet
fonctionne tel quel, mais vous vous épargnerez des heures de débogage,
notamment avec OneDrive qui synchronise en arrière-plan pendant
l'entraînement.

---

## Installation

```bash
pip install -r requirements.txt
```

Vérification (aucune donnée nécessaire) :

```bash
python scripts/generer_convoyeur_synthetique.py
python scripts/generer_eclairage_synthetique.py
python scripts/test_regles_vehicules.py
```

---

## Ce qui fonctionne dès aujourd'hui, sans un seul jour d'entraînement

C'est le point clé du projet : chaque modèle a un **mode de repli** qui donne
un résultat exploitable immédiatement, pendant que vous collectez et annotez
les données réelles.

```bash
# Convoyeur : détection de déchirure par vision classique (aucun entraînement)
python -m src.detect.convoyeur_cv --source data/raw/convoyeur_synthetique.mp4

# Éclairage : diagnostic des zones sous-éclairées (aucun entraînement)
python -m src.detect.eclairage --source data/raw/parc_nuit_synthetique.jpg

# Véhicules : détection générique via COCO (car / truck / bus déjà connus)
python -m src.detect.vehicules --source 0

# Pipeline complet sur un flux
python -m src.pipeline.run_stream --source data/raw/convoyeur_synthetique.mp4 \
       --modeles convoyeur --fps-analyse 15
```

### Lire les plaques au portail

**À faire en premier** — le test qui décide si c'est possible avec votre caméra :

```bash
python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4 --images 30
```

La lecture de plaque n'est pas un problème de modèle mais de **pixels** :
sous 80 px de largeur de plaque, aucun entraînement ne fera apparaître
l'information. Le script rend un verdict et indique quoi corriger. Détails
dans [docs/plaque.md](docs/plaque.md).

### Entraîner : sur Colab ou Kaggle, pas ici

**Ce poste n'a pas de GPU.** Les entraînements menés en local (3 époques,
320 px) ne sont que des tests de chaîne : ils prouvent que le code
fonctionne, pas que les modèles sont bons.

Le projet est sur GitHub : <https://github.com/othmanedhilou/MODELE>

Ouvrez [notebooks/entrainement_colab.ipynb](notebooks/entrainement_colab.ipynb)
dans Colab, activez le GPU (*Exécution → Modifier le type d'exécution →
GPU T4*), puis *Exécution → Tout exécuter*. **Il n'y a rien à téléverser** :
le notebook clone le dépôt et régénère les datasets sur place.

Si vous préférez ne rien passer par GitHub :

```bash
python scripts/preparer_envoi.py     # archive de 0,5 Mo, à déposer dans Drive
```

L'archive **exclut les images synthétiques** : elles se régénèrent en une
minute sur le GPU distant, alors que les téléverser prendrait une heure. Vos
images réelles, elles, sont toujours incluses.

Puis, dans [notebooks/entrainement_colab.ipynb](notebooks/entrainement_colab.ipynb)
ou [notebooks/entrainement_kaggle.ipynb](notebooks/entrainement_kaggle.ipynb) :

```bash
python scripts/generer_dataset_convoyeur.py --nombre 1200
python -m src.prepare.check_dataset --modele convoyeur
python -m src.train.train --modele convoyeur --epochs 120
python -m src.train.evaluer --modele convoyeur
python -m src.mlops.registre --promouvoir convoyeur --version v1
```

| Plateforme | Session max | Convoyeur (200 époques, 1024 px) |
|---|---|---|
| Colab T4 | quelques heures | 3 à 4 h — **risque de coupure** |
| Kaggle P100 | 12 h | 2 h 30 — recommandé |

Pour Colab, réduisez à `--epochs 120`, ou travaillez depuis Google Drive
pour pouvoir reprendre avec `--reprendre`.

Résultats mesurés sur les lots de test, entraînements de démonstration de
3 époques sur processeur :

| Modèle | Classe | mAP50 | Rappel |
|---|---|---|---|
| Convoyeur | dechirure | **0,869** | 0,855 |
| Convoyeur | corps_etranger | 0,786 | 0,775 |
| Éclairage | luminaire | **0,931** | 0,905 |

Ces chiffres portent sur des **données entièrement synthétiques** : ils
prouvent que la chaîne fonctionne, pas que les modèles marcheront sur vos
caméras. Dès que vous aurez des images de votre convoyeur — même sans aucun
défaut — passez à la méthode hybride, nettement plus proche du réel :

```bash
python -m src.prepare.extract_frames --source data/raw/convoyeur.mp4        --sortie data/frames/convoyeur --intervalle 1
python scripts/generer_dechirures_sur_reel.py --source data/frames/convoyeur --par-image 3
```

Tout est alors réel sauf la déchirure elle-même. Détails et limites dans
[docs/sans_donnees.md](docs/sans_donnees.md).

Les alarmes sont écrites dans `runs/alarmes/journal.jsonl`, avec une capture
d'écran horodatée pour chaque événement.

---

## Arborescence

```
MODELE/
├── configs/                    Toute la configuration, aucun paramètre en dur
│   ├── eclairage.yaml          Seuils photométriques, zones, jour/nuit
│   ├── vehicules.yaml          Lignes de comptage, zones interdites
│   ├── convoyeur.yaml          ROI bande, seuils CV, gravité des déchirures
│   ├── plaque.yaml             Localisation, OCR, seuils de lisibilité
│   ├── data_*.yaml             Datasets YOLO (chemins + classes)
│   └── pipeline.yaml           Caméras, flux RTSP, cadences d'analyse
├── data/
│   ├── raw/                    Vidéos brutes de l'usine
│   ├── frames/                 Images extraites, avant annotation
│   └── <modele>/images|labels  Datasets prêts pour YOLO (train/val/test)
├── src/
│   ├── prepare/                Extraction d'images, découpage, contrôle qualité
│   ├── train/                  Entraînement et évaluation
│   ├── detect/                 Les trois détecteurs, utilisables seuls
│   ├── pipeline/               Orchestration temps réel multi-caméras
│   ├── mlops/                  Registre, promotion, surveillance production
│   └── utils/                  Configuration, ROI, journal d'alarmes, dessin
├── scripts/                    Outils : tracé de zones, données de test
├── docs/                       Guides : démarrage sans données, annotation,
│                               calibration, MLOps, phase 2
├── models/                     Registre et poids promus en production
├── notebooks/                  Entraînement GPU : Colab et Kaggle
└── runs/                       Sorties : poids, métriques, alarmes
```

---

## Chaîne de travail complète

### 1. Récupérer les vidéos

Demandez à l'entreprise des enregistrements couvrant la **variabilité
réelle** — c'est ce qui détermine la qualité finale, bien plus que le choix
du modèle :

- jour, nuit, aube, crépuscule ;
- temps clair, pluie, brouillard, **poussière de ciment** (spécifique au site) ;
- toutes les caméras concernées, pas une seule ;
- au moins une semaine complète, week-end compris.

Pour le convoyeur, demandez en priorité **les enregistrements des incidents
passés**. Une déchirure réelle vaut cent images de bande saine.

### 2. Extraire les images

```bash
python -m src.prepare.extract_frames --source data/raw --sortie data/frames/vehicules \
       --intervalle 2 --seuil-similarite 0.93
```

L'extraction écarte automatiquement les images trop semblables à la
précédente : sur une caméra fixe, 90 % des images d'une nuit calme sont
identiques et n'apportent rien à l'entraînement.

### 2 bis. Ou récupérer des données publiques

Des pistes vérifiées et des correspondances de classes prêtes à l'emploi
sont listées dans [docs/sans_donnees.md](docs/sans_donnees.md#annexe--datasets-publics-repérés-août-2026) :

- **convoyeur** : [BeltCrack](https://github.com/UESTC-nnLab/BeltCrack) —
  **23 732 images industrielles réelles**, licence Apache 2.0, à utiliser en
  pré-entraînement (il annote des fissures, pas des déchirures) ;
- **véhicules** : Kaggle et les datasets de recherche SODA / MOCS / ACID ;
- **éclairage** : rien d'exploitable publiquement, à annoter soi-même.

```bash
python -m src.prepare.importer_dataset --source telechargements/convoyeur --inspecter
```

### 3. Annoter

Voir **[docs/annotation.md](docs/annotation.md)** : outils, définition
précise de chaque classe, règles de tracé et quantités visées.

Pour le convoyeur, gagnez du temps avec la pré-annotation automatique :

```bash
python -m src.detect.convoyeur_cv --source data/frames/convoyeur --pre-annoter
```

Le détecteur classique écrit des `.txt` YOLO que vous n'avez plus qu'à
corriger dans CVAT ou LabelImg, au lieu de tout tracer à la main.

### 4. Construire le dataset

```bash
python -m src.prepare.split_dataset --source data/annote_vehicules --modele vehicules
python -m src.prepare.check_dataset --modele vehicules
```

Le contrôle qualité détecte les erreurs qui font perdre le plus de temps :
classes hors plage, coordonnées non normalisées, boîtes vides, et surtout
**la même image présente en train et en val**, qui donne des scores
excellents et un modèle inutilisable sur le terrain.

### 5. Entraîner

```bash
# Validation de la chaîne en 2 minutes, même sur CPU
python -m src.train.train --modele vehicules --test-rapide

# Entraînement réel (GPU nécessaire)
python -m src.train.train --modele vehicules
python -m src.train.train --modele convoyeur
python -m src.train.train --modele eclairage
```

**Cette machine n'a pas de GPU** (`torch 2.6.0+cpu`, CUDA indisponible) : un
entraînement complet y prendrait plusieurs jours. Le script détecte le
matériel et vous prévient.

#### Où entraîner

```bash
python scripts/preparer_envoi.py        # archive code + datasets, sans les videos
```

| Plateforme | GPU | Limite | Notebook fourni |
|------------|-----|--------|-----------------|
| **Google Colab** | T4 16 Go | Session coupée après quelques heures | `notebooks/entrainement_colab.ipynb` |
| **Kaggle** | P100 ou T4 x2 | 30 h/semaine, session jusqu'à 12 h | `notebooks/entrainement_kaggle.ipynb` |
| **Poste GPU de l'entreprise** | variable | aucune | commandes ci-dessus, directement |

Kaggle est préférable pour le convoyeur (entraînement le plus long) car la
session ne se coupe pas en cours de route. Sur Colab, travaillez depuis
Google Drive : sinon les poids disparaissent à la fin de la session.

Sur un poste GPU de l'entreprise, rien de particulier : copiez le projet,
`pip install -r requirements.txt`, et lancez les mêmes commandes.

### 6. Évaluer, puis déployer

```bash
python -m src.train.evaluer --modele vehicules --exporter onnx
python -m src.pipeline.run_stream --camera cam_convoyeur_01
```

---

## Combien d'images faut-il annoter ?

Ordres de grandeur réalistes pour un stage, à partir d'un modèle
pré-entraîné COCO (le fine-tuning demande beaucoup moins de données qu'un
entraînement depuis zéro) :

| Modèle | Images | Instances par classe | Remarque |
|--------|--------|----------------------|----------|
| Véhicules | 800 – 1 500 | 150 – 300 | Les classes rares (bulldozer, excavatrice) sont les plus coûteuses |
| Éclairage | 300 – 600 | 200+ luminaires | Une seule classe : converge vite |
| Convoyeur | 200 – 400 | 100+ déchirures | **Le facteur limitant : les vraies déchirures sont rares** |

Ajoutez 5 à 10 % d'**images de fond** — sans aucun objet, donc sans fichier
`.txt`. Elles réduisent nettement les faux positifs, et beaucoup de projets
les oublient.

### Le vrai problème du convoyeur

Vous n'aurez pas 100 déchirures réelles pendant un stage : une usine bien
entretenue en connaît quelques-unes par an. Trois solutions, à combiner :

1. **La couche A (vision classique) est déjà opérationnelle** et ne demande
   aucune déchirure. Elle constitue votre livrable de base.
2. **Fabriquer des déchirures** : sur une chute de bande usagée (le service
   maintenance en a toujours), tracez des entailles et filmez-les sous
   l'éclairage réel. 30 minutes d'atelier donnent 50 images exploitables.
3. **Détection d'anomalie non supervisée** (phase 2) : on entraîne
   uniquement sur de la bande **saine**, dont vous avez des heures. Voir
   [docs/convoyeur_phase2.md](docs/convoyeur_phase2.md).

---

## Réglage sur le site réel

Les valeurs par défaut des configurations sont des points de départ, pas des
vérités. Trois choses sont à mesurer sur place — voir
**[docs/calibration.md](docs/calibration.md)** :

1. **Les zones et lignes** : tracez-les sur une vraie image de la caméra.
   ```bash
   python scripts/tracer_roi.py --video data/raw/portail.mp4 --frame 100
   ```
   Le script imprime le YAML à coller dans la configuration.

2. **Le sens de défilement de la bande** (`configs/convoyeur.yaml`,
   `roi.sens_defilement_deg`). C'est le paramètre le plus important du
   modèle 3 : mal réglé, les traces des rouleaux sont prises pour des
   déchirures et les vraies déchirures sont ignorées. 90 = bande verticale
   dans l'image, 0 = horizontale.

3. **L'échelle** (`cv.mm_par_pixel`) : mesurez la largeur réelle de la bande
   en millimètres, divisez par sa largeur en pixels. Sans cela, la gravité
   annoncée (mineure / majeure / critique) n'a aucun sens physique.

---

### Avant tout : votre caméra voit-elle les défauts ?

```bash
python scripts/tester_resolution_convoyeur.py --video data/raw/convoyeur.mp4        --largeur-bande-mm 1200
```

Le facteur limitant n'est presque jamais le modèle ni le jeu de données,
c'est ce que la caméra résout physiquement : résolution en mm/pixel, **flou
de mouvement** (une bande défile à 2–4 m/s), et contraste. Le script les
mesure sur vos images et rend un verdict.

## Anomalies du convoyeur : tout détecter, réagir selon la gravité

Les dix classes d'anomalies sont détectées et journalisées. Aucune n'est
ignorée. Ce qui varie, c'est le **niveau d'alarme** — donc la réaction
attendue.

| Gravité | Classes | Réaction |
|---|---|---|
| **critique** | `dechirure`, `jonction_defectueuse` | Arrêt machine, inspection immédiate |
| **majeure** | `perforation`, `cloque`, `corps_etranger`, `desalignement` | Intervention au prochain arrêt |
| **mineure / info** | `fissure`, `bord_effiloche`, `usure_surface`, `deversement` | Consigner, suivre l'évolution |

Le **journal garde tout**, quel que soit le niveau : c'est ce qui permet de
suivre l'évolution d'une fissure sur plusieurs semaines. Le seuil
`seuil_alerte_operateur` ne filtre que ce qui remonte à l'écran en temps
réel.

Sans cette graduation, l'opérateur reçoit vingt alarmes « critique » par
jour, cesse de les lire, et désactive le système. C'est le mode d'échec le
plus fréquent de ce type de projet, et il ne vient jamais d'un défaut de
détection.

Deux classes méritent une attention particulière : `corps_etranger` et
`desalignement` ne sont pas des dégradations, ce sont les **causes** des
déchirures. Les détecter évite l'incident au lieu de le constater.

Détail complet, mesures par classe et pièges rencontrés dans
[docs/anomalies_convoyeur.md](docs/anomalies_convoyeur.md).

---

## Lecture de plaque au portail

Greffée sur le modèle véhicules : la plaque n'est cherchée que dans la
boîte d'un véhicule détecté, ce qui divise par vingt les faux positifs par
rapport à une recherche sur l'image entière.

```bash
python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4  # faisabilite
python scripts/test_flux_plaque.py                                        # test logique
python -m src.pipeline.run_stream --camera cam_portail                    # production
```

Le module ne lance l'OCR **qu'une fois par véhicule**, sur la meilleure vue
de son passage, au franchissement de la ligne d'entrée. Lire chaque image
coûterait vingt à cent fois plus cher pour un résultat moins bon.

Résultat mesuré sur nos images d'essai, qui illustre pourquoi le test de
faisabilité passe avant tout le reste :

| Largeur de plaque | Attendu | Lu |
|---|---|---|
| 193 px | `12345 A 6` | `12345` |
| 117 px | `45678 B 12` | `45678 8 12` (B lu comme 8) |
| 59 px | `90123 C 3` | rien — écarté avant l'OCR |

Les chiffres passent, la lettre arabe rarement : les modèles OCR publics
sont entraînés sur des plaques latines. En pratique les chiffres suffisent
à rapprocher un camion d'un bon de livraison ; pour un contrôle d'accès
automatique, il faudrait entraîner un modèle de caractères marocains.

**Une immatriculation est une donnée personnelle** (loi 09-08). Finalité,
durée de conservation et accès sont à faire valider avant mise en service.
`anonymiser: true` dans `configs/plaque.yaml` floute les plaques au lieu de
les lire, sans rien retirer au comptage ni à la surveillance de zone.

---

## Traçabilité et suivi en production (MLOps)

Trois commandes couvrent le cycle de vie d'un modèle. Détail dans
[docs/mlops.md](docs/mlops.md).

```bash
python -m src.mlops.registre --lister                          # historique des entraînements
python -m src.mlops.registre --promouvoir convoyeur --version v1   # mise en production
python -m src.mlops.surveiller --jours 7                       # santé du système déployé
```

Chaque entraînement enregistre automatiquement l'**empreinte du dataset**
utilisé. C'est ce qui permet, trois semaines plus tard, de répondre à
« pourquoi ce modèle est-il moins bon que le précédent ? » : même empreinte
signifie que seule la configuration a changé, empreinte différente signifie
que c'est la donnée.

Le pipeline ne charge que `models/<modele>/production.pt`, promu
explicitement. Un entraînement en cours dans `runs/` ne peut donc pas
basculer en production par accident.

**Le modèle ne s'améliore pas tout seul en production** : YOLO n'apprend
rien de ce qu'il voit, et le réentraîner automatiquement sur ses propres
sorties lui ferait apprendre ses erreurs. La boucle d'amélioration passe par
une relecture humaine, mais l'étape la plus coûteuse — choisir quelles
images annoter — est outillée :

```bash
python scripts/collecter_pour_reannotation.py --source runs/alarmes --nombre 100
```

Il classe les images par utilité, le signal le plus fort étant le
**désaccord entre la vision classique et le modèle entraîné** : quand deux
méthodes indépendantes divergent, l'une se trompe, et c'est là que
l'annotation humaine rapporte le plus.

La surveillance analyse le journal d'alarmes et signale trois défaillances
qu'un modèle déployé ne signale jamais lui-même : la **dérive** du taux
d'alarme, le **silence** d'une caméra (presque toujours un flux coupé, pas
une usine devenue parfaite), et les **rafales** qui font désactiver le
système par les opérateurs.

---

## Ce qu'il faut demander à l'entreprise

À obtenir en début de stage, car chaque élément conditionne la suite :

- **Accès aux flux RTSP** (adresse IP, identifiants, numéro de canal) et
  autorisation écrite d'y accéder ;
- **Enregistrements historiques**, en particulier des incidents convoyeur ;
- **Un poste avec GPU**, ou l'autorisation d'utiliser Google Colab ;
- **Le plan des zones** : où sont les zones piétonnes, les voies de
  circulation, les quais — ce sont les polygones à tracer ;
- **Le seuil d'alarme attendu** : qui reçoit l'alerte, avec quel délai, et
  qui décide d'arrêter le convoyeur ;
- **Les règles internes sur les données** : ces caméras filment des
  personnes. Le comptage de véhicules est peu sensible, mais l'enregistrement
  d'images de personnel relève de la protection des données (loi 09-08 au
  Maroc). Faites valider l'usage et la durée de conservation des captures
  d'alarme par votre encadrant avant de mettre le système en production.

---

## État d'avancement

| Composant | État |
|-----------|------|
| Structure, configuration, utilitaires | Fait |
| Extraction d'images, découpage, contrôle qualité | Fait, testé |
| Import de datasets publics avec correspondance de classes | Fait, testé |
| Générateurs de datasets synthétiques (convoyeur, éclairage) | Fait — 1200 images annotées automatiquement |
| Convoyeur couche A (vision classique) | Fait, testé : 0 % de fausse alarme et 79 % de détection |
| Convoyeur couche B (YOLO segmentation) | **Entraîné** : déchirure mAP50 = 0,869 et rappel = 0,855 sur lot de test |
| Éclairage (photométrie + zones) | Fait, testé : 3/3 états de lampe corrects |
| Éclairage (détection de luminaires) | **Entraîné** : mAP50 = 0,931, précision 0,96, rappel 0,90 sur lot de test |
| Véhicules (suivi, comptage, zones) | Fait, 5/5 tests de règles métier. Modèle cimenterie non entraîné (repli COCO) |
| Plaque (localisation + OCR + meilleure vue) | Fait, testé — voir [docs/plaque.md](docs/plaque.md) |
| Pipeline temps réel + journal d'alarmes | Fait, testé à 15 img/s |
| Registre MLOps, promotion, surveillance | Fait, testé de bout en bout |
| Entraînement sur données **réelles** | En attente des vidéos de l'usine |
| Convoyeur couche C (anomalie non supervisée) | Phase 2, documentée |

Ce que ces chiffres disent et ne disent pas : la chaîne technique est
validée de bout en bout, de la génération de données à la mise en
production. Les performances mesurées portent sur des données synthétiques
et ne préjugent pas du comportement sur les caméras réelles.

---

## Documentation

| Guide | Contenu |
|---|---|
| [docs/sans_donnees.md](docs/sans_donnees.md) | Les quatre façons de démarrer sans données d'usine, datasets publics repérés |
| [docs/annotation.md](docs/annotation.md) | Outils, classes, règles de tracé, quantités visées |
| [docs/calibration.md](docs/calibration.md) | Les cinq réglages à mesurer sur site |
| [docs/anomalies_convoyeur.md](docs/anomalies_convoyeur.md) | Les 10 classes d'anomalies, gravité, ce que chaque couche détecte |
| [docs/plaque.md](docs/plaque.md) | Lecture de plaque : faisabilité, fonctionnement, cadre légal |
| [docs/mlops.md](docs/mlops.md) | Empreinte de dataset, registre, promotion, surveillance |
| [docs/convoyeur_phase2.md](docs/convoyeur_phase2.md) | Détection d'anomalie non supervisée |
