# Taxonomie des anomalies de convoyeur

**Toute anomalie est détectée et journalisée. Aucune n'est ignorée.**
Ce qui varie d'une classe à l'autre, c'est la **gravité** de l'alarme, donc
la réaction attendue de l'opérateur.

---

## Les dix classes

| # | Classe | Nature | Gravité | Action attendue |
|---|--------|--------|---------|-----------------|
| 0 | `dechirure` | Rupture ouverte longitudinale | majeure → **critique** | Arrêt machine si > 500 mm, inspection immédiate |
| 8 | `jonction_defectueuse` | Jonction (splice) qui s'ouvre | majeure → **critique** | Arrêt et ré-agrafage |
| 2 | `perforation` | Trou traversant | mineure → majeure | Inspecter au prochain arrêt |
| 6 | `fissure` | Amorce de rupture, souvent transversale | info → majeure | Consigner et suivre l'évolution |
| 9 | `cloque` | Décollement, boursouflure | mineure → majeure | Planifier une réparation |
| 3 | `corps_etranger` | Bloc métallique, pierre | mineure → majeure | Retirer — **cause évitable** |
| 5 | `desalignement` | Bande décentrée | mineure → majeure | Régler les rouleaux |
| 1 | `bord_effiloche` | Bord abîmé, frange | info → mineure | Surveiller |
| 7 | `usure_surface` | Abrasion du revêtement | info → mineure | Consigner l'épaisseur |
| 4 | `deversement` | Matière hors bande | info → mineure | Vérifier le chargement amont |

### Pourquoi graduer plutôt que tout traiter pareil

Une fissure naissante et une bande en train de se déchirer ne peuvent pas
déclencher la même alerte. Sans graduation, l'opérateur reçoit vingt alarmes
« critique » par jour, cesse de les lire en une semaine, et finit par
désactiver le système. C'est le mode d'échec le plus fréquent des projets de
détection industrielle, et il ne vient jamais d'un défaut de détection.

Le **journal garde tout**, quel que soit le niveau : c'est ce qui permet de
suivre l'évolution d'une fissure sur plusieurs semaines. Le paramètre
`seuil_alerte_operateur` de `configs/convoyeur.yaml` ne filtre que ce qui
remonte en temps réel à l'écran.

### Les causes valent plus que les conséquences

`corps_etranger` et `desalignement` ne sont pas des dégradations de la
bande : ce sont ce qui **provoque** les déchirures. Un bloc métallique
détecté avant qu'il ne coince sous un racleur évite l'incident au lieu de le
constater. Ce sont les deux classes au meilleur rapport valeur/effort du
projet.

---

## Ce que chaque couche détecte réellement

Mesuré sur le lot de validation synthétique, en comparant la classe attribuée
par la règle géométrique à la vérité terrain :

| Classe | Couche A (vision classique) |
|---|---|
| `perforation` | **100 %** de classification correcte |
| `dechirure` | **92 %** |
| `fissure` | **88 %** |
| `jonction_defectueuse` | **71 %** |
| `cloque`, `corps_etranger` | échec (défauts trop peu contrastés) |
| `usure_surface`, `bord_effiloche`, `deversement` | jamais localisés |
| `desalignement` | traité à part, par suivi du centre de la bande |

La couche A localise environ **36 %** des anomalies annotées. C'est le
résultat attendu, et il est important de le dire clairement :

- elle couvre bien les anomalies **contrastées et critiques** — déchirure,
  fissure, perforation, jonction — celles qui imposent une réaction rapide ;
- elle rate les anomalies **diffuses** — usure, cloque, déversement — dont
  le contraste est trop faible pour un seuil statistique.

**La couche B (YOLO) n'est donc pas un raffinement optionnel : c'est elle qui
apporte la couverture complète.** La couche A reste le filet de sécurité qui
fonctionne sans entraînement, le jour où le modèle n'est pas encore prêt ou
qu'il échoue sur des conditions inédites.

---

## Deux pièges rencontrés, et comment ils sont traités

### Les rouleaux ressemblent à des fissures

Dès qu'on détecte les défauts **transversaux**, les traces de rouleaux
deviennent des faux positifs parfaits : claires, allongées, perpendiculaires
au défilement — exactement la signature d'une fissure.

Ce qui les distingue n'est pas leur forme mais leur **mouvement** : un
rouleau reste au même endroit, un défaut défile avec la bande. Le détecteur
construit donc une carte des pixels clairs en permanence et les retire. Cela
élimine du même coup les rayures sur l'objectif et les reflets fixes.

*Limite* : si la bande est à l'arrêt, un vrai défaut devient lui aussi fixe.
La carte n'est mise à jour que si du mouvement est détecté.

### Classer n'est pas filtrer

En passant d'un détecteur mono-classe à un classificateur multi-classes, on
perd le filtre implicite qui rejetait tout ce qui n'était pas allongé.
Résultat mesuré : chaque grain de clinker clair devenait une `perforation`,
et le taux de fausse alarme est passé de 0 à 23 détections parasites sur 76
images de bande saine.

Un filtre de crédibilité a été rétabli avant classification : un défaut est
retenu s'il est **allongé** (rupture) ou **suffisamment gros** (objet, trou).
Un petit blob compact est du bruit.

Après correction, sur la vidéo de test :

| Phase | Alarmes confirmées |
|---|---|
| Bande saine | **0 / 76 images (0 %)** |
| Bande déchirée | 94 / 119 images (79 %) |

---

## Datasets publics et anomalies

`configs/correspondance_beltcrack.yaml` mappe 23 noms de classes rencontrés
dans les datasets publics vers les dix classes du projet. **Seules les
classes qui ne sont pas des anomalies sont ignorées** : la bande elle-même,
le fond, les rouleaux sains, les personnes.

Les fissures de [BeltCrack](https://github.com/UESTC-nnLab/BeltCrack) sont
donc apprises comme `fissure` — ni confondues avec une déchirure, ni
écartées. Elles entrent au journal, leur évolution est suivie, et elles ne
déclenchent pas l'arrêt de la ligne.


---

## Le dataset n'est pas le facteur decisif : la camera l'est

Question legitime : quel jeu de donnees donnera de vrais resultats sur les
videos de l'usine ? La reponse honnete est qu'**aucun dataset ne repond a
cette question**, parce que le facteur limitant est ailleurs.

```bash
python scripts/tester_resolution_convoyeur.py --video data/raw/convoyeur.mp4        --largeur-bande-mm 1200
```

Le script mesure trois grandeurs sur **vos** images, par ordre d'importance :

| Grandeur | Ce qu'elle décide |
|---|---|
| **Résolution** (mm/pixel) | Une déchirure de 5 mm vue sur une bande de 150 px fait 0,6 pixel : elle n'existe pas dans l'image |
| **Flou de mouvement** | Le plus sous-estimé. Une bande défile à 2–4 m/s ; une caméra qui expose 1/30 s étale le défaut sur plusieurs centimètres |
| **Contraste** | La carcasse exposée doit ressortir du caoutchouc, malgré la poussière et la compression du flux |

### Pourquoi le flou de mouvement compte tant

C'est la différence entre une caméra de vidéosurveillance et une caméra
d'inspection industrielle. Les installations de détection de déchirure
utilisent des vitesses d'obturation élevées, voire des caméras linéaires,
précisément parce qu'une bande en mouvement efface les défauts fins sur une
exposition ordinaire.

Le script mesure ce flou de façon indépendante de la netteté générale : il
compare l'énergie de gradient **dans** le sens de défilement à celle
mesurée perpendiculairement. Sur une image nette les deux se valent ; si la
bande a bougé pendant l'exposition, la composante longitudinale s'effondre.

### Ce que dit le classement des datasets

| Source | Réalisme | Limite |
|---|---|---|
| Génération synthétique (`generer_dataset_convoyeur.py`) | faible | Le modèle apprend le rendu, pas le défaut |
| [BeltCrack](https://github.com/UESTC-nnLab/BeltCrack), 23 732 images réelles | élevé | Vues rapprochées, souvent dessus/dessous : la géométrie de prise de vue peut ne pas correspondre à vos caméras |
| Incrustation sur **vos** images (`generer_dechirures_sur_reel.py`) | très élevé | Le défaut reste dessiné |
| **Vos vraies déchirures** (chute de bande, incidents passés) | maximal | Rares, à collecter activement |

L'ordre de préférence est clair, mais il ne se joue qu'**après** le test de
résolution. Si la caméra ne résout pas le défaut, le meilleur dataset du
monde produira un modèle qui n'apprend rien d'utilisable.

C'est la même leçon que pour la lecture de plaque, et elle vaut d'être
écrite dans le rapport : en vision industrielle, la question de
l'acquisition passe avant celle du modèle.
