"""
Genere une scene de portail synthetique avec des plaques a trois distances,
pour valider le module plaque avant d'avoir les images reelles.

Les plaques imitent le format marocain : chiffres, lettre arabe, numero de
region, texte sombre sur fond clair.

  python scripts/generer_portail_synthetique.py
  python scripts/tester_lisibilite_plaque.py --image data/raw/portail_synthetique.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, ecrire_image  # noqa: E402

configurer_console()
LARGEUR, HAUTEUR = 1600, 900


def dessiner_plaque(image, centre, largeur_plaque, texte="12345 A 6"):
    """Dessine une plaque claire a texte sombre, au ratio marocain."""
    hauteur_plaque = max(8, int(largeur_plaque / 4.5))
    cx, cy = centre
    x1, y1 = int(cx - largeur_plaque / 2), int(cy - hauteur_plaque / 2)
    x2, y2 = x1 + largeur_plaque, y1 + hauteur_plaque

    cv2.rectangle(image, (x1, y1), (x2, y2), (235, 238, 240), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (60, 60, 60), max(1, largeur_plaque // 60))

    echelle = largeur_plaque / 190.0
    epaisseur = max(1, int(2 * echelle))
    cv2.putText(image, texte, (x1 + int(6 * echelle), y2 - int(8 * echelle)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75 * echelle, (25, 25, 28),
                epaisseur, cv2.LINE_AA)
    return (x1, y1, x2, y2)


def dessiner_vehicule(image, centre_bas, largeur_vehicule, couleur, texte_plaque):
    """Arriere de vehicule simplifie, avec sa plaque. Retourne (bbox, bbox_plaque)."""
    hauteur_vehicule = int(largeur_vehicule * 0.85)
    cx, ybas = centre_bas
    x1 = int(cx - largeur_vehicule / 2)
    y1 = int(ybas - hauteur_vehicule)
    x2, y2 = x1 + largeur_vehicule, ybas

    cv2.rectangle(image, (x1, y1), (x2, y2), couleur, -1)
    # Lunette arriere, plus sombre
    cv2.rectangle(image, (x1 + largeur_vehicule // 8, y1 + hauteur_vehicule // 10),
                  (x2 - largeur_vehicule // 8, y1 + hauteur_vehicule // 2),
                  tuple(int(c * 0.45) for c in couleur), -1)
    # Feux
    for decalage in (largeur_vehicule // 7, largeur_vehicule - largeur_vehicule // 7):
        cv2.rectangle(image, (x1 + decalage - largeur_vehicule // 14,
                              y1 + int(hauteur_vehicule * 0.58)),
                      (x1 + decalage + largeur_vehicule // 14,
                       y1 + int(hauteur_vehicule * 0.70)), (40, 40, 190), -1)

    largeur_plaque = int(largeur_vehicule * 0.30)
    bbox_plaque = dessiner_plaque(
        image, (cx, y1 + int(hauteur_vehicule * 0.86)), largeur_plaque, texte_plaque)
    return (x1, y1, x2, y2), bbox_plaque


def main():
    rng = np.random.default_rng(3)
    image = np.full((HAUTEUR, LARGEUR, 3), 150, np.uint8)
    image[:HAUTEUR // 2] = (185, 175, 160)                       # ciel
    cv2.rectangle(image, (0, HAUTEUR // 2), (LARGEUR, HAUTEUR), (95, 95, 100), -1)  # sol
    for y in range(HAUTEUR // 2, HAUTEUR, 60):                   # marquage au sol
        cv2.line(image, (0, y), (LARGEUR, y), (110, 110, 115), 2)

    # Trois vehicules a trois distances : plaques d'environ 60, 120 et 200 px
    vehicules = [
        ((330, 780), 660, (70, 90, 120), "12345 A 6"),    # proche
        ((900, 620), 400, (120, 70, 70), "45678 B 12"),   # moyen
        ((1330, 540), 200, (80, 110, 80), "90123 C 3"),   # loin
    ]
    verite = []
    for centre, largeur, couleur, texte in vehicules:
        bbox_vehicule, bbox_plaque = dessiner_vehicule(image, centre, largeur,
                                                       couleur, texte)
        verite.append({"vehicule": bbox_vehicule, "plaque": bbox_plaque,
                       "largeur_plaque": bbox_plaque[2] - bbox_plaque[0],
                       "texte": texte})

    image = np.clip(image.astype(np.int16)
                    + rng.normal(0, 4, image.shape).astype(np.int16),
                    0, 255).astype(np.uint8)

    chemin = RACINE / "data" / "raw" / "portail_synthetique.jpg"
    ecrire_image(chemin, image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Image : {chemin}")
    print("Verite terrain :")
    for element in verite:
        print(f"  plaque '{element['texte']}' : {element['largeur_plaque']} px "
              f"de large, vehicule {element['vehicule']}")
    return verite


if __name__ == "__main__":
    main()
