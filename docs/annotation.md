# Guide d'annotation

L'annotation est l'étape qui détermine la performance finale. Un modèle
médiocre entraîné sur des annotations rigoureuses bat toujours un modèle
excellent entraîné sur des annotations approximatives.

## Outils

| Outil | Quand l'utiliser |
|-------|------------------|
| **LabelImg** | Boîtes simples, hors ligne, installation immédiate (`pip install labelImg`) |
| **CVAT** | Segmentation (convoyeur), travail à plusieurs, interpolation entre images d'une vidéo |
| **Roboflow** | Interface confortable, export YOLO direct — attention : les données partent sur un serveur externe, à faire valider par l'entreprise |
| **Label Studio** | Alternative libre à Roboflow, hébergée en interne |

Pour le convoyeur, CVAT est fortement conseillé : sa fonction
d'interpolation permet d'annoter une déchirure sur une image toutes les
vingt, et de laisser l'outil remplir les intermédiaires.

**Format de sortie : toujours « YOLO »**, jamais COCO JSON ni Pascal VOC.
Un fichier `.txt` par image, portant exactement le même nom.

---

## Modèle 1 — Éclairage

### Classe unique

```
0  luminaire
```

Un seul identifiant, quel que soit l'état de la lampe. L'état
(allumé / faible / éteint / défaillant) est calculé par photométrie et
**ne s'annote pas**. C'est un choix délibéré : annoter trois classes
exigerait des centaines d'exemples de lampes en panne, alors qu'une lampe
grillée se reconnaît à sa luminance en une ligne de code.

### Règles de tracé

- Encadrer **le globe ou la lentille**, pas le mât ni le support.
- Inclure le halo immédiat, mais pas la nappe de lumière au sol : le halo
  fait partie du signal photométrique, la nappe non.
- Annoter les luminaires **éteints comme allumés** : sinon le modèle
  n'apprend qu'à voir des taches lumineuses et devient aveugle exactement
  dans le cas qui vous intéresse — la panne.
- Ignorer les phares de véhicules et les fenêtres éclairées.
- Un luminaire partiellement masqué (par un camion) reste annoté si plus de
  la moitié est visible.

### Composition du lot

Environ moitié d'images de nuit, un quart au crépuscule, un quart de jour.
Les images de jour apprennent au modèle qu'une lampe éteinte est encore un
luminaire.

---

## Modèle 2 — Véhicules

### Classes

```
0 camion_benne        5 chargeuse
1 camion_citerne      6 chariot_elevateur
2 camion_leger        7 bulldozer
3 voiture             8 excavatrice
4 bus_navette         9 citerne_remorque
```

### Règles de tracé

- Boîte **serrée** sur le véhicule, remorque comprise s'il la tracte.
- Véhicule tronqué par le bord de l'image : annoter la partie visible, si
  elle dépasse 30 % du véhicule.
- Occlusion : annoter dès qu'un tiers du véhicule est visible et
  identifiable.
- Un camion citerne attelé = **une seule boîte** (`camion_citerne`). Une
  remorque dételée seule = `citerne_remorque`.
- En cas de doute entre deux classes, choisissez la plus fréquente sur le
  site plutôt que d'inventer une classe intermédiaire.

### Équilibrage

Le piège classique : 900 camions et 12 excavatrices. Le modèle apprend
alors à tout appeler « camion ». Deux parades :

1. rechercher activement les images contenant les engins rares (parcourez
   les enregistrements aux heures de travaux) ;
2. si une classe reste sous 50 instances, **fusionnez-la** avec une classe
   voisine plutôt que de la garder famélique — un modèle à 7 classes
   fiables vaut mieux qu'un modèle à 10 classes dont 3 ne marchent pas.

---

## Modèle 3 — Convoyeur

### Classes (segmentation, pas boîtes)

```
0 dechirure        3 corps_etranger
1 bord_effiloche   4 deversement
2 perforation      5 desalignement
```

### Pourquoi de la segmentation

Une déchirure longitudinale est longue, fine et diagonale. Sa boîte
englobante contiendrait environ 90 % de bande saine : le modèle apprendrait
« de la bande » plutôt que « une déchirure », et la mesure de longueur —
qui détermine la gravité — serait fausse. Le polygone épouse la trace et
donne une longueur exploitable.

### Règles de tracé

- Suivre la trace claire **au plus près**, sans marge de sécurité.
- Une déchirure interrompue en pointillés : **un seul polygone** couvrant
  toute la longueur, car physiquement c'est une seule déchirure.
- Ne pas annoter comme déchirure : les traces d'usure claires mais
  stables, les reflets sur les rouleaux, les résidus de matière blanche.
  Si vous hésitez, comparez avec l'image de la même zone de bande au tour
  précédent — une usure est permanente, un résidu passe.
- `corps_etranger` : tout bloc métallique ou pierre anguleuse sur la bande.
  C'est la **cause** la plus fréquente des déchirures ; le détecter permet
  d'agir avant l'incident, ce qui a plus de valeur que constater après.

### Méthode conseillée

1. Lancer la pré-annotation automatique :
   ```bash
   python -m src.detect.convoyeur_cv --source data/frames/convoyeur --pre-annoter
   ```
2. Ouvrir le dossier dans CVAT et **corriger** : supprimer les faux
   positifs, ajuster les contours, ajouter les défauts manqués.
3. Ne jamais entraîner directement sur les pré-annotations non relues : le
   modèle reproduirait exactement les erreurs du détecteur classique, sans
   rien apporter.

---

## Contrôle avant entraînement

```bash
python -m src.prepare.check_dataset --modele convoyeur
```

À corriger impérativement :

- **fuite train/val** : la même image dans deux lots ; les scores deviennent
  faux et le modèle échoue sur le terrain ;
- coordonnées hors de l'intervalle 0-1 (mauvais format d'export) ;
- identifiants de classe supérieurs au nombre de classes déclarées ;
- classes à zéro instance : retirez-les de `configs/data_*.yaml`.
