"""
Génère un dataset d'entraînement synthétique pour le modèle convoyeur,
annoté automatiquement au format YOLO segmentation.

Pourquoi c'est utile et pas un pis-aller
----------------------------------------
Les déchirures de bande n'existent pratiquement pas en jeu de données
public, et une usine bien entretenue en produit quelques-unes par an. Le
projet resterait bloqué des mois. La génération synthétique avec
RANDOMISATION DE DOMAINE (domain randomization) est la réponse habituelle
en vision industrielle : on fait varier fortement tout ce qui n'est pas le
défaut — éclairage, texture, poussière, perspective, rouleaux — pour que le
modèle apprenne la forme du défaut et non les conditions de prise de vue.

Le modèle obtenu n'est PAS prêt pour la production. Il sert à :
  - valider toute la chaîne technique avant l'arrivée des vraies données ;
  - fournir un point de départ à affiner (fine-tuning) sur 50 images
    réelles, ce qui demande dix fois moins d'images qu'un départ de zéro ;
  - produire des résultats chiffrés pour le rapport de stage.

Les annotations sont exactes par construction : on connaît le polygone de
la déchirure puisque c'est nous qui l'avons dessinée.

  python scripts/generer_dataset_convoyeur.py --nombre 600
  python scripts/generer_dataset_convoyeur.py --nombre 100 --sortie data/convoyeur_test
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, ecrire_image  # noqa: E402

configurer_console()

# Classes de configs/data_convoyeur.yaml
DECHIRURE, BORD_EFFILOCHE, PERFORATION, CORPS_ETRANGER = 0, 1, 2, 3
DEVERSEMENT, FISSURE, USURE, JONCTION, CLOQUE = 4, 6, 7, 8, 9

# Le desalignement n'est pas genere ici : c'est une propriete GLOBALE de
# l'image (bande decentree), pas un objet localisable. Il est traite par la
# couche vision classique, qui suit la position du centre de la bande.


def perspective_bande(largeur, hauteur, rng):
    """Quadrilatère de la bande, avec une perspective aléatoire plausible."""
    marge_haut = rng.uniform(0.22, 0.36)
    marge_bas = rng.uniform(0.04, 0.18)
    decalage = rng.uniform(-0.06, 0.06)
    return np.array([
        [(marge_haut + decalage) * largeur, 0],
        [(1 - marge_haut + decalage) * largeur, 0],
        [(1 - marge_bas + decalage) * largeur, hauteur],
        [(marge_bas + decalage) * largeur, hauteur],
    ], np.float32)


def bordures_bande(quadrilatere, y):
    """Abscisses des bords gauche et droit de la bande à la hauteur y."""
    (xhg, yh), (xhd, _), (xbd, yb), (xbg, _) = quadrilatere
    t = (y - yh) / max(yb - yh, 1e-6)
    return xhg + t * (xbg - xhg), xhd + t * (xbd - xhd)


def polygone_dechirure(quadrilatere, hauteur, rng):
    """
    Trace une déchirure longitudinale : une ligne médiane légèrement sinueuse
    dans le sens de défilement, épaissie de part et d'autre.
    Retourne le polygone du contour (Nx2) qui sert à la fois au dessin et à
    l'annotation.
    """
    y_debut = rng.uniform(-0.15, 0.55) * hauteur
    longueur = rng.uniform(0.25, 0.85) * hauteur
    y_fin = y_debut + longueur

    gauche, droite = bordures_bande(quadrilatere, np.clip(y_debut, 0, hauteur))
    x_debut = rng.uniform(gauche + 0.15 * (droite - gauche),
                          droite - 0.15 * (droite - gauche))
    derive = rng.uniform(-0.10, 0.10) * (droite - gauche)
    courbure = rng.uniform(-0.05, 0.05) * (droite - gauche)
    epaisseur = rng.uniform(2.5, 9.0)

    axe = []
    for pas in np.linspace(0.0, 1.0, 24):
        y = y_debut + pas * longueur
        x = x_debut + derive * pas + courbure * np.sin(pas * np.pi)
        axe.append((x, y))
    axe = np.array(axe, np.float32)

    # Décalage perpendiculaire pour donner une épaisseur variable
    tangentes = np.gradient(axe, axis=0)
    normales = np.stack([-tangentes[:, 1], tangentes[:, 0]], axis=1)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True) + 1e-6
    largeurs = epaisseur * (0.45 + 0.55 * np.sin(np.linspace(0.15, np.pi - 0.15, len(axe))))

    cote_a = axe + normales * largeurs[:, None]
    cote_b = axe - normales * largeurs[:, None]
    return np.concatenate([cote_a, cote_b[::-1]]), (y_debut, y_fin)


def polygone_transversal(quadrilatere, hauteur, rng, couverture):
    """
    Defaut allonge PERPENDICULAIRE au defilement : fissure si court,
    jonction de bande si elle traverse toute la largeur.
    """
    y = rng.uniform(0.1, 0.9) * hauteur
    gauche, droite = bordures_bande(quadrilatere, y)
    largeur_bande = droite - gauche
    longueur = couverture * largeur_bande
    x_debut = rng.uniform(gauche, droite - longueur) if longueur < largeur_bande         else gauche
    inclinaison = rng.uniform(-0.12, 0.12) * longueur
    epaisseur = rng.uniform(1.5, 5.0)

    axe = np.array([[x_debut + t * longueur,
                     y + inclinaison * (t - 0.5) + rng.uniform(-1.5, 1.5)]
                    for t in np.linspace(0, 1, 16)], np.float32)
    tangentes = np.gradient(axe, axis=0)
    normales = np.stack([-tangentes[:, 1], tangentes[:, 0]], axis=1)
    normales /= np.linalg.norm(normales, axis=1, keepdims=True) + 1e-6
    largeurs = epaisseur * (0.4 + 0.6 * np.sin(np.linspace(0.2, np.pi - 0.2, len(axe))))
    return np.concatenate([axe + normales * largeurs[:, None],
                           (axe - normales * largeurs[:, None])[::-1]])


def polygone_bord(quadrilatere, hauteur, rng):
    """Bord de bande effiloche : dentelure irreguliere le long d'un bord."""
    cote = rng.choice([0, 1])
    y0 = rng.uniform(0.1, 0.6) * hauteur
    longueur = rng.uniform(0.15, 0.4) * hauteur
    points_interieur, points_exterieur = [], []
    for t in np.linspace(0, 1, 14):
        y = y0 + t * longueur
        gauche, droite = bordures_bande(quadrilatere, y)
        bord = gauche if cote == 0 else droite
        sens = 1 if cote == 0 else -1
        profondeur = rng.uniform(3, 14)
        points_interieur.append([bord + sens * profondeur, y])
        points_exterieur.append([bord, y])
    return np.array(points_interieur + points_exterieur[::-1], np.float32)


def polygone_ellipse(centre, demi_l, demi_h, rng, sommets=14):
    """Contour elliptique bruite, pour cloque, usure ou deversement."""
    angles = np.linspace(0, 2 * np.pi, sommets, endpoint=False)
    rayons = 1.0 + rng.uniform(-0.18, 0.18, sommets)
    return np.stack([centre[0] + demi_l * rayons * np.cos(angles),
                     centre[1] + demi_h * rayons * np.sin(angles)], axis=1).astype(np.float32)


def gradient_eclairage(largeur, hauteur, rng):
    """
    Nappe d'éclairage non uniforme : un tunnel de convoyeur est éclairé par
    quelques lampes ponctuelles, jamais uniformément. Sans cela le modèle
    apprend un seuil global et s'effondre sur les vraies images.
    """
    nappe = np.zeros((hauteur, largeur), np.float32)
    y, x = np.mgrid[0:hauteur, 0:largeur].astype(np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, largeur), rng.uniform(0, hauteur)
        rayon = rng.uniform(0.25, 0.75) * max(largeur, hauteur)
        nappe += rng.uniform(12, 45) * np.exp(
            -((x - cx) ** 2 + (y - cy) ** 2) / (2 * rayon ** 2))
    return nappe


def generer_image(rng, largeur=960, hauteur=720, ratio_fond=0.15):
    """
    Produit une image et ses annotations YOLO segmentation.

    Une fraction des images est generee SANS aucune anomalie. Ces images de
    fond sont indispensables : sans elles, le modele n'a jamais vu de bande
    saine et finit par trouver un defaut partout. Avec dix familles
    d'anomalies possibles, presque chaque image en contiendrait une si on
    ne reservait pas explicitement ce quota.
    """
    saine = rng.random() < ratio_fond
    quadrilatere = perspective_bande(largeur, hauteur, rng)

    # Structure métallique de fond, claire et texturée
    fond = rng.uniform(75, 145)
    image = np.full((hauteur, largeur, 3), fond, np.float32)
    for _ in range(rng.integers(2, 6)):
        y0 = int(rng.uniform(0, hauteur))
        cv2.line(image, (0, y0), (largeur, y0 + int(rng.uniform(-40, 40))),
                 (fond * 1.25,) * 3, int(rng.integers(2, 9)))

    # Bande en caoutchouc : sombre, mais avec un niveau variable
    niveau_bande = rng.uniform(20, 52)
    cv2.fillPoly(image, [quadrilatere.astype(np.int32)], (niveau_bande,) * 3)

    masque_bande = np.zeros((hauteur, largeur), np.uint8)
    cv2.fillPoly(masque_bande, [quadrilatere.astype(np.int32)], 255)

    annotations = []

    # Rouleaux : traces claires PERPENDICULAIRES au défilement. Ce sont les
    # faux positifs les plus fréquents : il faut donc en mettre beaucoup.
    for _ in range(rng.integers(2, 7)):
        y0 = int(rng.uniform(0, hauteur))
        gauche, droite = bordures_bande(quadrilatere, y0)
        cv2.line(image, (int(gauche), y0), (int(droite), y0),
                 (niveau_bande + rng.uniform(8, 35),) * 3, int(rng.integers(1, 5)))

    # Matière transportée : grains de clinker plus clairs que la bande
    for _ in range(rng.integers(20, 160)):
        y0 = rng.uniform(0, hauteur)
        gauche, droite = bordures_bande(quadrilatere, y0)
        centre = (int(rng.uniform(gauche, droite)), int(y0))
        cv2.circle(image, centre, int(rng.integers(1, 7)),
                   (niveau_bande + rng.uniform(5, 40),) * 3, -1)

    # --- Défauts ---
    nb_dechirures = 0 if saine else rng.choice(
        [0, 1, 1, 1, 2], p=[0.18, 0.34, 0.20, 0.20, 0.08])
    for _ in range(int(nb_dechirures)):
        polygone, _ = polygone_dechirure(quadrilatere, hauteur, rng)
        clarte = rng.uniform(150, 250)
        cv2.fillPoly(image, [polygone.astype(np.int32)], (clarte,) * 3)
        annotations.append((DECHIRURE, polygone))

    # --- Fissure : trace claire fine, transversale ou oblique ---
    if not saine and rng.random() < 0.28:
        polygone = polygone_transversal(quadrilatere, hauteur, rng,
                                        rng.uniform(0.15, 0.55))
        cv2.fillPoly(image, [polygone.astype(np.int32)],
                     (niveau_bande + rng.uniform(70, 150),) * 3)
        annotations.append((FISSURE, polygone))

    # --- Jonction defectueuse : traverse toute la largeur de la bande ---
    if not saine and rng.random() < 0.25:
        polygone = polygone_transversal(quadrilatere, hauteur, rng,
                                        rng.uniform(0.88, 1.0))
        cv2.fillPoly(image, [polygone.astype(np.int32)],
                     (niveau_bande + rng.uniform(60, 130),) * 3)
        annotations.append((JONCTION, polygone))

    # --- Bord effiloche ---
    if not saine and rng.random() < 0.30:
        polygone = polygone_bord(quadrilatere, hauteur, rng)
        cv2.fillPoly(image, [polygone.astype(np.int32)],
                     (niveau_bande + rng.uniform(50, 120),) * 3)
        annotations.append((BORD_EFFILOCHE, polygone))

    # --- Usure de surface : zone claire diffuse, sans contour net ---
    if not saine and rng.random() < 0.16:
        y0 = rng.uniform(0.15, 0.85) * hauteur
        gauche, droite = bordures_bande(quadrilatere, y0)
        centre = (rng.uniform(gauche + 40, droite - 40), y0)
        demi_l = rng.uniform(30, 90)
        demi_h = rng.uniform(25, 80)
        polygone = polygone_ellipse(centre, demi_l, demi_h, rng)
        calque = image.copy()
        cv2.fillPoly(calque, [polygone.astype(np.int32)],
                     (niveau_bande + rng.uniform(18, 42),) * 3)
        # Melange doux : l'usure n'a pas de bord franc, contrairement a une
        # dechirure. C'est justement ce qui doit permettre de les distinguer.
        image = cv2.addWeighted(calque, 0.75, image, 0.25, 0)
        image = cv2.GaussianBlur(image, (7, 7), 0) if rng.random() < 0.4 else image
        annotations.append((USURE, polygone))

    # --- Cloque : boursouflure, claire d'un cote et ombree de l'autre ---
    if not saine and rng.random() < 0.22:
        y0 = rng.uniform(0.15, 0.85) * hauteur
        gauche, droite = bordures_bande(quadrilatere, y0)
        centre = (rng.uniform(gauche + 35, droite - 35), y0)
        demi_l, demi_h = rng.uniform(18, 45), rng.uniform(14, 38)
        polygone = polygone_ellipse(centre, demi_l, demi_h, rng)
        cv2.fillPoly(image, [polygone.astype(np.int32)],
                     (niveau_bande + rng.uniform(28, 60),) * 3)
        # Ombre portee du cote oppose a la lumiere : signature du relief
        decalage = int(demi_l * 0.35)
        ombre = polygone + np.array([decalage, decalage], np.float32)
        calque = image.copy()
        cv2.fillPoly(calque, [ombre.astype(np.int32)],
                     (max(niveau_bande - rng.uniform(8, 20), 5),) * 3)
        image = cv2.addWeighted(calque, 0.35, image, 0.65, 0)
        annotations.append((CLOQUE, polygone))

    # --- Deversement : matiere tombee HORS de la bande ---
    if not saine and rng.random() < 0.28:
        y0 = rng.uniform(0.3, 0.95) * hauteur
        gauche, droite = bordures_bande(quadrilatere, y0)
        cote = gauche if rng.random() < 0.5 else droite
        sens = -1 if cote == gauche else 1
        centre = (cote + sens * rng.uniform(20, 60), y0)
        polygone = polygone_ellipse(centre, rng.uniform(25, 70),
                                    rng.uniform(20, 55), rng)
        for _ in range(int(rng.integers(25, 90))):
            point = (int(centre[0] + rng.normal(0, 25)),
                     int(centre[1] + rng.normal(0, 20)))
            cv2.circle(image, point, int(rng.integers(2, 7)),
                       (rng.uniform(90, 150),) * 3, -1)
        annotations.append((DEVERSEMENT, polygone))

    if not saine and rng.random() < 0.26:      # perforation : trou ponctuel clair
        y0 = rng.uniform(0.1, 0.9) * hauteur
        gauche, droite = bordures_bande(quadrilatere, y0)
        centre = (rng.uniform(gauche + 20, droite - 20), y0)
        rayon = rng.uniform(5, 14)
        cv2.circle(image, (int(centre[0]), int(centre[1])), int(rayon),
                   (rng.uniform(170, 255),) * 3, -1)
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        polygone = np.stack([centre[0] + rayon * np.cos(angles),
                             centre[1] + rayon * np.sin(angles)], axis=1)
        annotations.append((PERFORATION, polygone))

    if not saine and rng.random() < 0.18:      # corps étranger : bloc clair et anguleux
        y0 = rng.uniform(0.1, 0.9) * hauteur
        gauche, droite = bordures_bande(quadrilatere, y0)
        cx = rng.uniform(gauche + 30, droite - 30)
        demi_l, demi_h = rng.uniform(12, 40), rng.uniform(10, 30)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        polygone = np.stack([cx + demi_l * np.cos(angles),
                             y0 + demi_h * np.sin(angles)], axis=1)
        cv2.fillPoly(image, [polygone.astype(np.int32)],
                     (rng.uniform(110, 200),) * 3)
        annotations.append((CORPS_ETRANGER, polygone))

    # --- Dégradations globales, appliquées après les défauts ---
    image += gradient_eclairage(largeur, hauteur, rng)[..., None]
    image += rng.normal(0, rng.uniform(2, 11), image.shape)   # bruit capteur
    if rng.random() < 0.35:                                   # voile de poussière
        image = image * rng.uniform(0.80, 0.97) + rng.uniform(8, 30)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (k, k), 0)

    # Annotations restreintes à la bande, puis normalisées
    lignes = []
    for classe, polygone in annotations:
        points = polygone.copy()
        points[:, 0] = np.clip(points[:, 0], 0, largeur - 1)
        points[:, 1] = np.clip(points[:, 1], 0, hauteur - 1)
        dedans = masque_bande[points[:, 1].astype(int), points[:, 0].astype(int)] > 0
        # Deux anomalies sont HORS bande par definition : le deversement
        # est de la matiere tombee a cote, le bord effiloche est sur le
        # bord. Leur appliquer la regle "doit etre sur la bande" les
        # supprimerait toutes, donc on ne les apprendrait jamais.
        couverture_min = 0.0 if classe in (DEVERSEMENT, BORD_EFFILOCHE) else 0.5
        if dedans.mean() < couverture_min:
            continue
        points[:, 0] /= largeur
        points[:, 1] /= hauteur
        coords = " ".join(f"{v:.6f}" for v in np.clip(points, 0, 1).flatten())
        lignes.append(f"{classe} {coords}")

    return image, lignes


def main():
    ap = argparse.ArgumentParser(
        description="Dataset synthétique annoté pour le modèle convoyeur")
    ap.add_argument("--nombre", type=int, default=600, help="Nombre total d'images")
    ap.add_argument("--sortie", default="data/convoyeur", help="Dossier du dataset")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--largeur", type=int, default=960)
    ap.add_argument("--hauteur", type=int, default=720)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    destination = Path(args.sortie)
    if not destination.is_absolute():
        destination = RACINE / args.sortie

    rng = np.random.default_rng(args.seed)
    bornes = np.cumsum([int(args.nombre * r) for r in args.ratios])
    compte = {"train": 0, "val": 0, "test": 0}
    instances = Counter()
    fonds = 0

    for index in range(args.nombre):
        lot = "train" if index < bornes[0] else "val" if index < bornes[1] else "test"
        image, lignes = generer_image(rng, args.largeur, args.hauteur)

        nom = f"synth_{index:05d}"
        ecrire_image(destination / "images" / lot / f"{nom}.jpg", image,
                     [cv2.IMWRITE_JPEG_QUALITY, 92])
        chemin_label = destination / "labels" / lot / f"{nom}.txt"
        chemin_label.parent.mkdir(parents=True, exist_ok=True)
        chemin_label.write_text("\n".join(lignes), encoding="utf-8")

        compte[lot] += 1
        if not lignes:
            fonds += 1
        for ligne in lignes:
            instances[int(ligne.split()[0])] += 1

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{args.nombre} images générées")

    print(f"\nDataset écrit dans {destination}")
    for lot, n in compte.items():
        print(f"  {lot:<6} {n:5d} images")
    noms = yaml.safe_load(
        (RACINE / "configs" / "data_convoyeur.yaml").read_text(
            encoding="utf-8"))["names"]
    print()
    print("Instances annotees :")
    for identifiant in sorted(noms):
        compte = instances.get(identifiant, 0)
        note = ""
        if identifiant == 5:
            note = "  (propriete globale : couche vision classique)"
        elif compte < 100:
            note = "  <-- peu representee"
        print(f"  {noms[identifiant]:<22} {compte:5d}{note}")
    print(f"  images de fond  {fonds:5d} (sans defaut, indispensables "
          f"pour limiter les faux positifs)")
    print("\nEtapes suivantes :")
    print("  python -m src.prepare.check_dataset --modele convoyeur")
    print("  python -m src.train.train --modele convoyeur --test-rapide")


if __name__ == "__main__":
    main()
