# Module plaque d'immatriculation (ANPR)

Lecture des immatriculations au portail, greffée sur le modèle véhicules.

---

## Avant tout : le test de faisabilité

**La lecture de plaque n'est pas d'abord un problème de modèle, c'est un
problème de pixels.** Si la plaque fait 50 px de large dans l'image,
l'information n'y est pas, et aucun entraînement ne la fera apparaître.

```bash
python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4 --images 30
```

Le script détecte les véhicules, localise leurs plaques, mesure leur
largeur et leur netteté, puis rend un verdict :

| Largeur médiane | Verdict | Ce que vous obtiendrez |
|---|---|---|
| ≥ 160 px | Lecture fiable | Chiffres et lettre, exploitable en contrôle d'accès |
| 120 – 160 px | Lecture partielle | Chiffres presque toujours, lettre rarement |
| 80 – 120 px | Peu fiable | Lectures erratiques, à ne pas automatiser |
| < 80 px | Impossible | Ce n'est pas un problème de modèle |

Si le verdict est négatif, par ordre de coût croissant :

1. **Vérifier la résolution du flux.** Beaucoup d'installations diffusent
   un sous-canal en 704×576 alors que la caméra filme en 1920×1080. C'est
   le gain le plus facile du projet, et il est gratuit.
2. **Zoomer** la caméra sur la voie d'entrée plutôt que sur tout le portail.
3. **Dédier une caméra** à la lecture, placée à 3–6 m de la voie, à hauteur
   de plaque, dans l'axe de circulation.

Un mode manuel existe si la détection automatique ne trouve rien :

```bash
python scripts/tester_lisibilite_plaque.py --image portail.jpg --manuel
```

Vous cliquez les deux bords d'une plaque, le script mesure.

---

## Comment le module fonctionne

Trois étages : **véhicule → plaque → caractères**.

### 1. Localisation

Une plaque est un **rectangle clair contenant du texte sombre**. Le module
cherche donc les zones claires et rectangulaires dans la partie basse du
véhicule (entre 40 % et 100 % de la hauteur de sa boîte), puis ne garde que
celles qui contiennent effectivement des caractères.

Ce choix mérite d'être expliqué, car la méthode classique fait l'inverse :
elle cherche directement le texte (chapeau noir + Sobel). Le problème est
qu'elle renvoie alors la position du **bloc de caractères**, pas celle de
la plaque. Sur nos essais, la largeur mesurée était sous-estimée de 23 à
33 %, ce qui fausse le verdict de lisibilité — une plaque de 198 px était
annoncée à 133 px. En mesurant le support plutôt que l'encre, l'erreur est
tombée à 2 %.

Si vous disposez d'un modèle YOLO dédié aux plaques, placez-le en
`models/plaque/production.pt` : il remplace automatiquement cette méthode.

### 2. Sélection de la meilleure vue

Un véhicule traverse le champ en 2 à 4 secondes, soit 20 à 100 images. Sa
plaque n'est grande et nette que sur quelques-unes. Le module **suit** le
véhicule, conserve la meilleure vue, et ne lance l'OCR **qu'une fois**, au
franchissement de la ligne d'entrée ou à la sortie du champ.

Sans cette sélection, il faudrait lancer 20 à 100 OCR par véhicule pour un
résultat moins bon, la meilleure vue n'étant jamais la première.

Un détail qui a demandé une correction : la netteté se mesure classiquement
par la variance du laplacien, mais **cette mesure dépend de l'échelle**.
Une petite plaque nette obtient un score bien plus élevé qu'une grande
plaque tout aussi nette. Utilisée telle quelle, la sélection retenait
systématiquement la plaque la plus **petite** du passage — l'inverse du but
recherché. Toute vignette est donc ramenée à 200 px de large avant mesure.

### 3. Lecture

Moteur détecté au démarrage : **EasyOCR** si disponible, sinon Tesseract,
sinon aucun. Sans moteur, le module enregistre quand même la vignette de la
plaque pour lecture par un opérateur — ce qui reste utile.

L'initialisation est paresseuse : EasyOCR charge plusieurs centaines de Mo
et met une dizaine de secondes à démarrer. Tant qu'aucun véhicule n'est
passé, ce coût n'est pas payé.

### Plaques marocaines

Format : **chiffres | lettre arabe | numéro de région**, par exemple
`12345 - أ - 6`.

Les modèles OCR publics sont entraînés sur des plaques latines. Résultat
mesuré sur nos images d'essai :

| Largeur | Attendu | Lu |
|---|---|---|
| 193 px | `12345 A 6` | `12345` |
| 117 px | `45678 B 12` | `45678 8 12` (le B lu comme un 8) |
| 59 px | `90123 C 3` | rien — écarté avant l'OCR |

Les chiffres passent, la lettre pas toujours. Dans la pratique, les
chiffres suffisent le plus souvent à rapprocher un camion d'un bon de
livraison. Pour un contrôle d'accès automatique, il faudrait entraîner un
modèle de caractères sur des plaques marocaines.

---

## Protection des données

Une immatriculation est une **donnée à caractère personnel**. Compter des
camions ne pose pas de difficulté particulière ; enregistrer et conserver
des immatriculations relève de la **loi 09-08**.

À faire valider par votre encadrant avant la mise en service :

- la **finalité** (contrôle d'accès ? rapprochement avec les livraisons ?) ;
- la **durée de conservation** des vignettes et du journal ;
- **qui** a accès aux enregistrements ;
- l'information des personnes concernées (panneau à l'entrée).

Le module prévoit une alternative : `anonymiser: true` dans
`configs/plaque.yaml` **floute** les plaques au lieu de les lire. Le
comptage des véhicules et la surveillance de zone continuent de
fonctionner. C'est la configuration à retenir si l'entreprise ne souhaite
pas conserver d'immatriculations.

---

## Utilisation

```bash
# 1. Faisabilite (a faire en premier)
python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4

# 2. Verification de la logique sur un passage simule
python scripts/test_flux_plaque.py

# 3. En production, via le pipeline (le module 'vehicules' est requis)
python -m src.pipeline.run_stream --camera cam_portail
```

Les passages sont journalisés dans `runs/plaques/passages.jsonl`, avec la
vignette de chaque plaque, sa largeur en pixels et le verdict de qualité —
ce qui permet de vérifier après coup si une lecture douteuse venait du
modèle ou de l'image.
