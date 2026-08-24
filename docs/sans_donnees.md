# Démarrer sans données de l'usine

C'est la situation réelle du projet : les caméras existent, mais vous n'avez
pas encore d'enregistrements exploitables. Voici les quatre voies possibles,
de la plus immédiate à la plus lente, avec ce que chacune permet vraiment.

---

## Voie 1 — Vision classique : aucune donnée requise

Le détecteur de déchirure fonctionne sans apprentissage.

```bash
python -m src.detect.convoyeur_cv --source data/raw/convoyeur_synthetique.mp4
```

C'est votre livrable de sécurité : même si aucune donnée n'arrive de tout le
stage, vous avez un système opérationnel. Il détecte les traces claires
allongées et rien d'autre, mais il le fait dès le premier jour.

---

## Voie 2 — Données synthétiques : entraîner dès aujourd'hui

```bash
python scripts/generer_dataset_convoyeur.py --nombre 600
python scripts/generer_dataset_eclairage.py --nombre 600
python -m src.prepare.check_dataset --modele convoyeur
python -m src.train.train --modele convoyeur
python -m src.train.train --modele eclairage
```

**Deux des trois modèles** ont un générateur synthétique. Le troisième, les
véhicules, n'en a pas : une voiture dessinée ne ressemble pas assez à une
vraie pour que l'apprentissage se transfère. Pour les véhicules, passez par
un dataset public (voie 3).

### Éclairage — 600 images, 2945 luminaires

Le générateur fait varier le moment de la journée (nuit, crépuscule, jour),
le nombre et la hauteur des mâts, l'état de chaque lampe, l'intensité des
halos, le brouillard et la poussière de ciment.

Un point de conception s'y joue : les lampes **éteintes** sont annotées au
même titre que les allumées, et le dataset en contient autant. Sans cela le
modèle n'apprendrait qu'à voir des taches lumineuses et deviendrait aveugle
exactement dans le cas qui intéresse — la panne. C'est aussi pour cela que
le dataset contient des images de jour : une lampe éteinte en plein soleil
est le cas le plus difficile, et le plus utile à apprendre.

Le générateur produit des images de bande avec perspective, rouleaux,
grains de clinker, éclairage non uniforme, poussière et bruit — puis y trace
des déchirures dont il **connaît le polygone exact**. Les annotations sont
donc parfaites par construction, ce qui n'arrive jamais avec des annotations
humaines.

Le principe s'appelle **randomisation de domaine** : on fait varier
fortement tout ce qui n'est pas le défaut, pour que le modèle apprenne la
forme du défaut et non les conditions de prise de vue.

### Résultat mesuré

**Convoyeur** — entraînement de démonstration à 3 époques sur processeur,
YOLO11n-seg, 600 images, évalué sur le **lot de test** (jamais vu pendant
l'entraînement) :

| Classe | mAP50 | Précision | Rappel |
|--------|-------|-----------|--------|
| dechirure | **0,869** | 0,699 | **0,855** |
| corps_etranger | 0,786 | 0,698 | 0,775 |
| perforation | 0,183 | 0,193 | 0,167 |
| **global** | 0,613 | 0,530 | 0,599 |

Lecture honnête de ces chiffres :

- la déchirure, seule classe qui compte vraiment, est bien apprise ;
- la perforation est faible parce qu'elle est rare dans le dataset (93
  instances) et petite : c'est le comportement attendu, pas un bug ;
- **ces chiffres mesurent la performance sur des images synthétiques**. Ils
  ne prédisent pas la performance sur les vraies caméras. Ils prouvent que
  la chaîne technique fonctionne, pas que le modèle est prêt.

### Ce que le modèle synthétique sert vraiment

1. valider la chaîne complète avant l'arrivée des données ;
2. servir de point de départ à affiner : partir de ce modèle demande environ
   dix fois moins d'images réelles que partir de zéro ;
3. donner des résultats chiffrés et des courbes pour le rapport.

**Il n'est pas déployable en production tel quel.** Écrivez-le dans le
rapport : un jury sanctionne bien plus durement une performance surestimée
qu'une limite assumée.

---

## Voie 2 bis — La meilleure : déchirures synthétiques sur images RÉELLES

C'est la réponse à l'objection la plus juste qu'on puisse faire à la voie 2 :
**mes images de convoyeur ne sont pas réelles**. Un trapèze noir avec du
bruit gaussien n'a ni la texture du caoutchouc, ni le flou de mouvement, ni
la poussière en suspension d'une vraie bande. Un modèle entraîné dessus
risque d'apprendre le rendu plutôt que le défaut, et de s'effondrer sur vos
caméras. C'est l'**écart de domaine** (sim-to-real gap).

```bash
# 1. Extraire des images de votre convoyeur, meme sans aucun defaut
python -m src.prepare.extract_frames --source data/raw/convoyeur.mp4        --sortie data/frames/convoyeur --intervalle 1

# 2. Y incruster des dechirures annotees automatiquement
python scripts/generer_dechirures_sur_reel.py --source data/frames/convoyeur        --par-image 3 --ratio-sain 0.25
```

Ici **tout est réel sauf la déchirure** : le fond, l'éclairage, la matière
transportée, le bruit du capteur, la compression du flux. Seuls quelques
milliers de pixels sont synthétiques. L'écart de domaine ne porte plus que
sur l'apparence de la déchirure au lieu de l'image entière.

Et cela retourne le problème du projet à votre avantage : **de la bande
saine, vous en avez des heures** — elle défile tous les jours. Ce sont les
déchirures qui manquent. Cette méthode transforme une ressource abondante
en dataset annoté.

### Trois précautions de réalisme

1. **Clarté relative, jamais absolue.** La déchirure est éclaircie par
   rapport à la luminosité *locale* de la bande, mesurée dans un anneau
   autour d'elle. Une trace à 220 sur une bande à 30 est crédible ; la même
   sur une bande à 180 ne l'est pas. Vérifié sur des images d'exposition
   variant du simple au triple : l'écart déchirure/bande reste toujours
   positif et du même ordre.
2. **Bords adoucis.** Une incrustation nette au pixel près se repère, et le
   modèle apprendrait à détecter le bord de collage.
3. **Flou et bruit réappliqués localement.** Sans cela la déchirure serait
   plus nette que le reste de l'image, et le modèle apprendrait à repérer
   la netteté au lieu du défaut.

### Ce que cela ne remplace pas

Une déchirure incrustée reste une déchirure dessinée. Elle n'a pas les
fibres arrachées, les bords irréguliers ni les effilochages d'une vraie.
Combinez donc avec la voie 4 : 50 vraies déchirures sur une chute de bande
valent mieux que 2000 incrustations, et les deux ensemble valent mieux que
chacune seule.

---

## Voie 3 — Datasets publics : pour les véhicules

Les engins de chantier sont bien couverts publiquement (Roboflow Universe,
Open Images). Les déchirures de convoyeur ne le sont pratiquement pas.

```bash
# 1. Telecharger un dataset au format YOLOv8/YOLOv11, puis inspecter ses classes
python -m src.prepare.importer_dataset --source telechargements/vehicules --inspecter

# 2. Remplir configs/correspondance_vehicules.yaml, puis importer
python -m src.prepare.importer_dataset --source telechargements/vehicules \
       --modele vehicules --correspondance configs/correspondance_vehicules.yaml
```

L'importateur convertit les classes du dataset vers votre vocabulaire et
**supprime** les annotations des classes mises à `null`. Règle de décision :
préférez `null` à une correspondance approximative. Une classe mal mappée
produit un modèle qui confond deux engins de façon systématique, et aucune
métrique ne vous le signalera.

Limite à connaître : un modèle entraîné sur des photos publiques de chantier
travaillera mal sur vos caméras — angle de vue plongeant, basse résolution,
poussière, éclairage nocturne. Utilisez-le comme pré-entraînement, pas comme
modèle final.

---

## Voie 4 — Fabriquer vos propres données réelles

La plus rentable pour le convoyeur, et souvent négligée.

1. demandez au service maintenance **une chute de bande usagée** — il y en a
   toujours ;
2. tracez-y des entailles de longueurs et d'orientations variées ;
3. filmez-la sous l'éclairage réel, avec la caméra réelle si possible.

Trente minutes d'atelier donnent 50 à 100 images de vraies déchirures, sur
le vrai matériau, sous le vrai éclairage. C'est plus utile que 2 000 images
synthétiques, et cela vaut une section entière dans le rapport.

---

## Ordre conseillé

| Semaine | Action | Livrable |
|---------|--------|----------|
| 1 | Voie 1 + Voie 2 | Système opérationnel, chaîne validée, premiers chiffres |
| 1 | Demander les accès RTSP et les enregistrements | — |
| 2 | Voie 4 (chute de bande) + extraction des vidéos reçues | Premières images réelles |
| 3-4 | Annotation, Voie 3 pour les véhicules | Datasets réels |
| 5 | Affinage des modèles synthétiques sur données réelles | Modèles déployables |
| 6 | Promotion, déploiement, surveillance | Système en production |

L'important est de ne jamais être bloqué en attendant des données : chaque
semaine produit un livrable, même si les vidéos arrivent tard.

---

## Annexe — datasets publics repérés (août 2026)

Ces pistes viennent d'une recherche web. **Je n'ai pas pu ouvrir les pages
moi-même** : Kaggle s'affiche en JavaScript et Roboflow renvoie une erreur
403 aux outils automatiques. Les chiffres ci-dessous proviennent donc des
résumés de résultats de recherche, pas d'une inspection directe.
**Vérifiez-les avant de fonder quoi que ce soit dessus.**

### Convoyeur — BeltCrack, le seul jeu de données réel sérieux

**C'est la meilleure ressource disponible pour ce projet**, et la seule que
j'aie pu vérifier directement (dépôt GitHub lisible, contrairement aux
pages Kaggle et Roboflow qui bloquent la lecture automatique).

| | |
|---|---|
| **Nom** | BeltCrack — *the First Sequential-image Industrial Conveyor Belt Crack Detection Dataset* |
| **Volume** | **23 732 images réelles** : BeltCrack14ks (14 087 images, 29 séquences) et BeltCrack9kd (9 645 images, 42 séquences) |
| **Conditions** | Environnements industriels réels : vues de dessus et de dessous, lumière forte du matin à faible éclairage du soir, temps ensoleillé/pluvieux/neigeux, vitesses de bande variables |
| **Format** | Pascal VOC (annotations XML, boîtes englobantes) |
| **Licence** | **Apache 2.0** — utilisation libre, y compris en entreprise, avec attribution |
| **Téléchargement** | <https://doi.org/10.57760/sciencedb.31181> — **mot de passe d'extraction : `cv205`** |
| **Code et description** | <https://github.com/UESTC-nnLab/BeltCrack> |
| **Article** | Pattern Recognition, 2026 — <https://doi.org/10.1016/j.patcog.2026.113598> (préprint : <https://arxiv.org/abs/2506.17892>) |

#### La réserve importante

Ce dataset annote des **fissures** (*crack*), pas des déchirures
longitudinales. Les deux se ressemblent sur du caoutchouc noir — trace
claire et allongée — mais ce ne sont pas les mêmes défauts :

- une **fissure** est une amorce de rupture, souvent transversale, qui
  évolue sur des semaines ;
- une **déchirure** est une rupture ouverte, longitudinale, qui impose
  l'arrêt immédiat.

Les confondre en production serait une erreur : toute fissure déclencherait
une alarme d'arrêt machine, et les opérateurs désactiveraient le système en
une semaine.

#### L'usage qui a du sens

**Pré-entraîner** sur BeltCrack, puis **affiner** sur vos propres images de
déchirure. Le gain ne vient pas des annotations mais des **textures de
caoutchouc réelles** sous éclairage industriel réel — exactement ce
qu'aucune génération synthétique ne peut fournir, et exactement ce qui
manquait à la voie 2.

```bash
# 1. Telecharger et extraire (mot de passe : cv205)
# 2. Inspecter les classes reelles du dataset
python -m src.prepare.importer_voc --source telechargements/BeltCrack14ks --inspecter

# 3. BeltCrack est en BOITES : basculez le modele en detection
#    configs/convoyeur.yaml -> tache: detect, modele_base: yolo11s.pt
#    (l'importateur refuse et vous le rappelle si vous l'oubliez)

# 4. Importer
python -m src.prepare.importer_voc --source telechargements/BeltCrack14ks        --modele convoyeur --correspondance configs/correspondance_beltcrack.yaml
python -m src.prepare.check_dataset --modele convoyeur
```

Le découpage train/val/test se fait **par séquence**, jamais par image :
deux images consécutives d'une même séquence vidéo se ressemblent trop, et
les séparer entre train et val gonflerait artificiellement les scores. Sur
un dataset séquentiel comme celui-ci, c'est l'erreur la plus facile à
commettre et la plus difficile à repérer après coup.

---

### Convoyeur — autres pistes, non vérifiées

Kaggle a des datasets de convoyeur, mais ils portent sur ce qui est *sur*
la bande (charbon, objets étrangers, minerai), pas sur l'*état* de la bande.
Après BeltCrack, Roboflow Universe est la source suivante — je n'ai pas pu
ouvrir ces pages (erreur 403), les chiffres viennent des résumés de
recherche :

| Dataset | Images annoncées | Classes annoncées | Intérêt |
|---|---|---|---|
| [Conveyor-belt-damage (segmentation)](https://universe.roboflow.com/sample-wy2mp/conveyor-belt-damage) | ~922 | Hole, Human, Conveyor, impact damage, Other Objects, patch work, Puncture, Roller Tear | **Le plus intéressant** : segmentation, donc directement compatible avec notre modèle 3 |
| [Conveyor Belt Damage](https://universe.roboflow.com/test-yfiry/conveyor-belt-damage-ucjlj) | ~325 | Belt Joint, Large Hole, Large Tear, Small Hole, Small Tear | Classes très proches de nos besoins |
| [conveyor belt tear](https://universe.roboflow.com/samruddhi-uxs8x/conveyor-belt-tear) | ~700 | déchirure | Ciblé sur la déchirure |
| [Conveyor Belt Damage Detection](https://universe.roboflow.com/cctv-tarjun/conveyor-belt-damage-detection-bvgsj-dk03r) | ~2353 | à vérifier | Le plus gros volume |

Sur Kaggle, le seul voisin est le [Coal Conveyor Belt Anomaly & Foreign
Object Dataset](https://www.kaggle.com/datasets/hanyv10086/coal-conveyor-belt-anomaly-and-foreign-object-dataset)
(objets étrangers en mine de charbon) : utile pour la classe
`corps_etranger`, qui est la **cause** des déchirures, pas pour la
déchirure elle-même.

Deux vérifications à faire avant de télécharger :

1. **Le format.** Notre modèle 3 est en segmentation. Un dataset en boîtes
   ne peut pas l'entraîner — l'importateur vous le dira, mais autant le
   savoir avant. Sur Roboflow, l'export « YOLOv8 » propose les deux ;
   choisissez celui qui correspond.
2. **La licence.** Beaucoup de datasets Roboflow sont en CC BY 4.0, certains
   sans licence claire. Un projet livré à une entreprise doit pouvoir
   justifier l'origine de ses données : notez la licence dans votre rapport.

Un piège de classe, déjà traité dans `configs/correspondance_convoyeur.yaml` :
**`Belt Joint` est une jonction normale de bande**, pas un défaut. Elle
ressemble beaucoup à une déchirure sur l'image. L'apprendre comme telle
déclencherait une alarme à chaque tour de bande. Elle est mise à `null`.

### Véhicules — plusieurs options sur Kaggle

| Dataset | Contenu annoncé |
|---|---|
| [PPE and Heavy Machinery detection (balanced)](https://www.kaggle.com/datasets/pablogarcher24/ppe-and-heavy-machinery-detection-balanced) | Excavatrices, tombereaux, bulldozers, chargeuses, compacteurs + EPI |
| [Vehicle Dataset for YOLO](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo) | ~3000 images, 6 classes, format YOLO |
| [Truck Tanker Image Dataset](https://www.kaggle.com/datasets/dataclusterlabs/truck-tanker-image-dataset-construction-vehicle) | Camions citernes — proche du `camion_citerne` de cimenterie |

Datasets de recherche, plus gros, cités dans la littérature : **SODA**
(19 846 images, 15 catégories), **MOCS** (41 668 images), **ACID** (Alberta
Construction Image Dataset). À chercher directement par leur nom.

Le dataset « PPE and Heavy Machinery » contient aussi des EPI : vos
camarades travaillent déjà dessus. Mettez ces classes à `null` dans la
correspondance pour ne pas dupliquer leur modèle.

### Éclairage — rien de directement exploitable

Aucun dataset public de détection de luminaires n'est ressorti. Les
datasets nocturnes connus (**NightOwls**, **ExDark**, **DarkFace**,
**CityPersons**) annotent des piétons et des véhicules, pas des lampes.

Ce n'est pas bloquant : le module éclairage fonctionne déjà par photométrie
sans aucun modèle. Si vous voulez la détection de luminaires, il faudra
annoter vous-même — c'est une classe unique et visuellement très marquée,
300 à 600 images suffisent, et c'est le plus rapide des trois à annoter.

### Import dans le projet

```bash
python -m src.prepare.importer_dataset --source telechargements/convoyeur --inspecter
python -m src.prepare.importer_dataset --source telechargements/convoyeur \
       --modele convoyeur --correspondance configs/correspondance_convoyeur.yaml
python -m src.prepare.check_dataset --modele convoyeur
```

Les correspondances de classes sont déjà préparées dans
`configs/correspondance_convoyeur.yaml` et
`configs/correspondance_vehicules.yaml`, avec les noms rencontrés dans ces
datasets. Corrigez-les d'après ce que renvoie `--inspecter` : les noms
varient d'un export à l'autre.
