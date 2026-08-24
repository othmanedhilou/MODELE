"""
Genere un dataset synthetique annote pour le modele eclairage.

Rappel de conception : le modele n'apprend QU'UNE classe, 'luminaire'.
L'etat (allume / faible / eteint / defaillant) est deduit par photometrie,
pas appris. Le dataset doit donc contenir autant de lampes ETEINTES que
d'allumees : sinon le modele n'apprend qu'a voir des taches lumineuses et
devient aveugle exactement dans le cas qui interesse, la panne.

Randomisation de domaine appliquee :
  - heure : nuit, crepuscule, jour (une lampe eteinte de jour reste un
    luminaire, et c'est le cas le plus difficile) ;
  - nombre, taille et hauteur des luminaires ;
  - etat de chaque lampe, avec des halos d'intensites variees ;
  - brouillard, poussiere de ciment, bruit de capteur, flou ;
  - silhouettes de batiments et de silos en arriere-plan.

  python scripts/generer_dataset_eclairage.py --nombre 600
  python -m src.prepare.check_dataset --modele eclairage
  python -m src.train.train --modele eclairage
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, ecrire_image  # noqa: E402

configurer_console()
LUMINAIRE = 0        # unique classe de configs/data_eclairage.yaml


def halo(image, centre, rayon, intensite, teinte=(0.75, 0.9, 1.0)):
    """Nappe lumineuse gaussienne autour d'une source."""
    hauteur, largeur = image.shape[:2]
    y, x = np.ogrid[:hauteur, :largeur]
    distance2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    gain = intensite * np.exp(-distance2 / (2.0 * rayon ** 2))
    for canal in range(3):
        image[:, :, canal] += gain * teinte[canal]
    return image


def fond_scene(largeur, hauteur, moment, rng):
    """Ciel, sol et silhouettes, selon le moment de la journee."""
    if moment == "jour":
        ciel, sol = rng.uniform(150, 205), rng.uniform(90, 130)
    elif moment == "crepuscule":
        ciel, sol = rng.uniform(60, 105), rng.uniform(35, 60)
    else:
        ciel, sol = rng.uniform(12, 38), rng.uniform(8, 25)

    image = np.zeros((hauteur, largeur, 3), np.float32)
    horizon = int(hauteur * rng.uniform(0.35, 0.55))
    # Degrade vertical du ciel, plus clair pres de l'horizon
    for y in range(horizon):
        image[y] = ciel * (0.6 + 0.4 * y / max(horizon, 1))
    image[horizon:] = sol

    # Silhouettes : silos, batiments, tour de prechauffage
    for _ in range(rng.integers(2, 6)):
        x = int(rng.uniform(0, largeur))
        w = int(rng.uniform(0.05, 0.22) * largeur)
        h = int(rng.uniform(0.10, 0.45) * hauteur)
        couleur = sol * rng.uniform(0.55, 0.9)
        cv2.rectangle(image, (x, horizon - h), (x + w, horizon + 10),
                      (couleur,) * 3, -1)
    return image, horizon


def generer_image(rng, largeur=960, hauteur=640):
    """Une image de parc d'usine et les boites de ses luminaires."""
    moment = rng.choice(["nuit", "nuit", "nuit", "crepuscule", "jour"])
    image, horizon = fond_scene(largeur, hauteur, moment, rng)

    nb_lampes = int(rng.integers(2, 9))
    positions = np.sort(rng.uniform(0.05, 0.95, nb_lampes)) * largeur
    annotations = []
    lampes = []

    for x in positions:
        hauteur_mat = rng.uniform(0.10, 0.40) * hauteur
        y = horizon - hauteur_mat
        demi_l = rng.uniform(0.012, 0.035) * largeur
        demi_h = demi_l * rng.uniform(0.35, 0.7)

        # Etats equilibres : autant d'eteintes que d'allumees
        etat = rng.choice(["allume", "allume", "faible", "eteint", "eteint"])
        if moment == "jour":
            etat = rng.choice(["eteint", "eteint", "eteint", "allume"])

        # Mat
        cv2.line(image, (int(x), int(y + demi_h)), (int(x), int(horizon + 8)),
                 (float(rng.uniform(30, 70)),) * 3, int(rng.integers(2, 6)))
        lampes.append((x, y, demi_l, demi_h, etat))

    # Halos dessines avant les globes, pour que le globe reste net au centre
    for x, y, demi_l, demi_h, etat in lampes:
        if etat == "allume":
            image = halo(image, (x, y), demi_l * rng.uniform(4, 9),
                         rng.uniform(70, 170))
        elif etat == "faible":
            image = halo(image, (x, y), demi_l * rng.uniform(2, 4),
                         rng.uniform(18, 45))

    for x, y, demi_l, demi_h, etat in lampes:
        if etat == "allume":
            couleur = rng.uniform(220, 255)
        elif etat == "faible":
            couleur = rng.uniform(110, 165)
        else:
            couleur = rng.uniform(35, 75)
        cv2.ellipse(image, (int(x), int(y)), (int(demi_l), int(demi_h)),
                    0, 0, 360, (couleur,) * 3, -1)
        # Le luminaire est annote QUEL QUE SOIT son etat
        annotations.append((LUMINAIRE, x - demi_l, y - demi_h,
                            x + demi_l, y + demi_h))

    # Degradations globales
    if rng.random() < 0.35:                       # brouillard / poussiere
        image = image * rng.uniform(0.75, 0.95) + rng.uniform(6, 28)
    image += rng.normal(0, rng.uniform(2, 9), image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    lignes = []
    for classe, x1, y1, x2, y2 in annotations:
        cx = np.clip((x1 + x2) / 2 / largeur, 0, 1)
        cy = np.clip((y1 + y2) / 2 / hauteur, 0, 1)
        w = np.clip((x2 - x1) / largeur, 0, 1)
        h = np.clip((y2 - y1) / hauteur, 0, 1)
        if w <= 0 or h <= 0:
            continue
        lignes.append(f"{classe} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return image, lignes, moment


def main():
    ap = argparse.ArgumentParser(
        description="Dataset synthetique annote pour le modele eclairage")
    ap.add_argument("--nombre", type=int, default=600)
    ap.add_argument("--sortie", default="data/eclairage")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--largeur", type=int, default=960)
    ap.add_argument("--hauteur", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    destination = Path(args.sortie)
    if not destination.is_absolute():
        destination = RACINE / args.sortie

    rng = np.random.default_rng(args.seed)
    bornes = np.cumsum([int(args.nombre * r) for r in args.ratios])
    compte = {"train": 0, "val": 0, "test": 0}
    moments = {}
    total_luminaires = 0

    for index in range(args.nombre):
        lot = "train" if index < bornes[0] else "val" if index < bornes[1] else "test"
        image, lignes, moment = generer_image(rng, args.largeur, args.hauteur)

        nom = f"synth_{index:05d}"
        ecrire_image(destination / "images" / lot / f"{nom}.jpg", image,
                     [cv2.IMWRITE_JPEG_QUALITY, 92])
        chemin_label = destination / "labels" / lot / f"{nom}.txt"
        chemin_label.parent.mkdir(parents=True, exist_ok=True)
        chemin_label.write_text("\n".join(lignes), encoding="utf-8")

        compte[lot] += 1
        moments[moment] = moments.get(moment, 0) + 1
        total_luminaires += len(lignes)

        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{args.nombre} images generees")

    print(f"\nDataset ecrit dans {destination}")
    for lot, n in compte.items():
        print(f"  {lot:<6} {n:5d} images")
    print(f"\n  luminaires annotes : {total_luminaires}")
    print(f"  moments            : " + ", ".join(f"{m} : {c}"
                                                 for m, c in sorted(moments.items())))
    print("\nEtapes suivantes :")
    print("  python -m src.prepare.check_dataset --modele eclairage")
    print("  python -m src.train.train --modele eclairage")


if __name__ == "__main__":
    main()
