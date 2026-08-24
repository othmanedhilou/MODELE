# Calibration sur le site

Les valeurs livrées dans `configs/` sont des points de départ raisonnables.
Trois réglages doivent être mesurés sur les caméras réelles ; sans eux, le
système fonctionne mais ses diagnostics n'ont pas de sens physique.

---

## 1. Zones et lignes (les trois modèles)

```bash
python scripts/tracer_roi.py --video data/raw/portail.mp4 --frame 100
```

Cliquez les sommets, appuyez sur `n` pour valider une zone, `s` pour
terminer. Le script imprime le YAML normalisé à coller dans la
configuration.

Conseils :

- tracez sur une image **représentative**, pas sur la plus dégagée ;
- laissez une marge à l'intérieur des zones : un véhicule est repéré par le
  centre de sa boîte, qui atteint la zone avant ses roues ;
- une ligne de comptage se trace avec **deux points seulement**, placés
  perpendiculairement au sens de circulation, et suffisamment longue pour
  que personne ne puisse la contourner par les bords.

---

## 2. Sens de défilement de la bande (modèle 3)

`configs/convoyeur.yaml` → `roi.sens_defilement_deg`

C'est le paramètre le plus sensible du projet. Il vaut `90` si la bande
monte ou descend verticalement dans l'image, `0` si elle défile
horizontalement, une valeur intermédiaire si la caméra est de biais.

Il sert deux fois :

1. **Filtre d'orientation** : seuls les défauts alignés avec le sens de
   défilement (à `angle_max_deg` près) sont retenus. Une déchirure
   longitudinale suit la bande ; les traces de rouleaux, elles, sont
   perpendiculaires.
2. **Fermeture morphologique orientée** : le noyau est une ligne dans le
   sens de défilement, ce qui reconnecte une déchirure discontinue sans
   fusionner des taches perpendiculaires.

Mal réglé, le résultat s'inverse exactement : les rouleaux déclenchent des
alarmes et les vraies déchirures passent inaperçues. C'est un fait mesuré
sur la vidéo de test — avec un angle faux, 2 faux positifs sur bande saine
et 0 détection sur bande déchirée ; avec le bon angle, 0 et 1.

**Ne le laissez pas sur `auto`** sauf si la bande est nettement plus longue
que large dans le champ : sur une vue quasi carrée, l'axe principal est
ambigu et l'automatisme se trompe.

---

## 3. Échelle en millimètres (modèle 3)

`configs/convoyeur.yaml` → `cv.mm_par_pixel`

Méthode :

1. relevez la **largeur nominale de la bande** auprès de la maintenance
   (typiquement 800, 1000, 1200 ou 1400 mm) ;
2. ouvrez une image de la caméra et mesurez cette même largeur en pixels
   (`scripts/tracer_roi.py` affiche les coordonnées des clics) ;
3. `mm_par_pixel = largeur_mm / largeur_pixels`.

Exemple : bande de 1200 mm occupant 600 pixels → `mm_par_pixel: 2.0`.

Si la caméra est inclinée, l'échelle varie du haut au bas de l'image.
Mesurez alors au **milieu de la zone analysée** : l'erreur résiduelle reste
inférieure aux paliers de gravité, qui vont du simple au double.

Sans cette mesure, les seuils `mineure_mm` / `majeure_mm` / `critique_mm`
classent au hasard, et une alarme « critique » ne veut rien dire.

---

## 4. Seuils photométriques (modèle 1)

`configs/eclairage.yaml` → `photometrie` et `zones[].luminance_min`

Procédure, à faire **de nuit** :

1. capturez une image avec l'éclairage complet en fonctionnement ;
2. lancez le diagnostic :
   ```bash
   python -m src.detect.eclairage --source data/raw/parc_nuit.jpg
   ```
   Il affiche la luminance mesurée dans chaque zone ;
3. réglez `luminance_min` environ **25 % en dessous** de la valeur mesurée
   en fonctionnement normal. Trop près, vous aurez une alarme à chaque
   passage de camion qui masque un projecteur ; trop bas, une panne
   partielle passe inaperçue ;
4. si possible, refaites la mesure avec un luminaire volontairement éteint
   (demandez à la maintenance) : vous obtenez alors les deux bornes réelles
   et pouvez placer le seuil au milieu.

Le paramètre `delta_vs_scene` compare la lampe à son propre fond : il rend
le diagnostic insensible aux réglages automatiques de gain de la caméra,
qui font varier la luminance absolue d'une nuit à l'autre. Ne le mettez pas
à zéro.

---

## 4 bis. Lecture de plaque (module plaque)

`configs/plaque.yaml`

Deux réglages, dans cet ordre :

1. **Le verdict de faisabilité.** Lancez d'abord
   `python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4`.
   S'il annonce « lecture impossible », aucun réglage ne rattrapera :
   c'est la caméra qu'il faut changer. Voir [plaque.md](plaque.md).

2. **Le seuil de netteté** (`qualite.nettete_min`, par défaut 300). Il est
   mesuré après remise à l'échelle de la plaque à 200 px, ce qui rend la
   valeur comparable d'une distance à l'autre — mais elle dépend du capteur
   et de la compression du flux. Relevez la valeur affichée par le script
   de test sur une plaque nette et sur une plaque floue, et placez le seuil
   au milieu. Trop bas, des plaques floues seront envoyées à l'OCR et
   produiront des lectures fausses ; trop haut, des plaques lisibles seront
   écartées.

Les bornes de largeur (`largeur_illisible_px`, `largeur_limite_px`,
`largeur_bonne_px`) sont, elles, des constantes physiques liées à la taille
des caractères : ne les modifiez pas sans raison.

---

## 5. Cadence d'analyse (pipeline)

`configs/pipeline.yaml` → `cameras[].fps_analyse`

Réglez selon le temps de réaction utile, pas selon la capacité de la
machine :

| Usage | Cadence | Justification |
|-------|---------|---------------|
| Convoyeur | 10 – 15 img/s | Une déchirure s'aggrave en quelques secondes |
| Véhicules (portail) | 5 – 10 img/s | Un camion à 20 km/h parcourt 0,5 m entre deux images |
| Véhicules (parc) | 2 – 5 img/s | Mouvements lents |
| Éclairage | 1 image / 60 s | Une lampe grillée le reste |

Diviser la cadence par cinq divise la charge par cinq : c'est le levier le
plus efficace pour faire tenir plusieurs caméras sur une seule machine.
