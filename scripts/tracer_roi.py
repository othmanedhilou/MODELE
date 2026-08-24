"""
Outil de tracé des zones (ROI) directement sur une image de la caméra.

Les polygones des fichiers configs/*.yaml sont en coordonnées normalisées
(0-1) : les saisir à la main est pénible et source d'erreurs. Ce script
affiche une image, vous cliquez les sommets, et il imprime le YAML à coller.

Commandes :
  clic gauche  : ajouter un sommet
  clic droit   : retirer le dernier sommet
  n            : valider la zone et en commencer une nouvelle
  s            : sauvegarder et quitter
  r            : tout effacer
  Echap        : quitter sans sauvegarder

  python scripts/tracer_roi.py --image data/frames/cam_portail/img_0001.jpg
  python scripts/tracer_roi.py --video data/raw/portail.mp4 --frame 100
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, lire_image  # noqa: E402

configurer_console()

sommets: list[tuple[int, int]] = []
zones_terminees: list[list[tuple[int, int]]] = []


def au_clic(evenement, x, y, drapeaux, parametres):
    if evenement == cv2.EVENT_LBUTTONDOWN:
        sommets.append((x, y))
    elif evenement == cv2.EVENT_RBUTTONDOWN and sommets:
        sommets.pop()


def dessiner(image):
    """Rendu de l'état courant : zones validées + polygone en cours."""
    rendu = image.copy()
    for index, zone in enumerate(zones_terminees):
        pts = np.array(zone, np.int32).reshape(-1, 1, 2)
        cv2.polylines(rendu, [pts], True, (0, 200, 0), 2)
        cv2.putText(rendu, f"zone_{index + 1}", zone[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if sommets:
        pts = np.array(sommets, np.int32).reshape(-1, 1, 2)
        cv2.polylines(rendu, [pts], False, (0, 165, 255), 2)
        for point in sommets:
            cv2.circle(rendu, point, 4, (0, 165, 255), -1)
    cv2.putText(rendu, "clic=sommet  n=zone suivante  s=sauver  r=effacer  ESC=quitter",
                (10, rendu.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return rendu


def imprimer_yaml(largeur, hauteur):
    """Affiche les zones au format attendu par les fichiers de configuration."""
    if not zones_terminees:
        print("Aucune zone validée (appuyez sur 'n' pour valider une zone).")
        return
    print("\n# ---- A coller dans configs/<modele>.yaml ----")
    print("zones:")
    for index, zone in enumerate(zones_terminees):
        points = ", ".join(f"[{x / largeur:.3f}, {y / hauteur:.3f}]" for x, y in zone)
        print(f"  - nom: zone_{index + 1}")
        print(f"    polygone: [{points}]")
        print(f"    luminance_min: 50")
    print("\n# ---- Pour une ligne de comptage (2 points seulement) ----")
    for zone in zones_terminees:
        if len(zone) == 2:
            (x1, y1), (x2, y2) = zone
            print(f"  - nom: ligne\n    p1: [{x1 / largeur:.3f}, {y1 / hauteur:.3f}]"
                  f"\n    p2: [{x2 / largeur:.3f}, {y2 / hauteur:.3f}]")


def main():
    ap = argparse.ArgumentParser(description="Tracé interactif des zones")
    ap.add_argument("--image", help="Image de référence")
    ap.add_argument("--video", help="Vidéo dont on extrait une image")
    ap.add_argument("--frame", type=int, default=0, help="Index d'image dans la vidéo")
    args = ap.parse_args()

    if args.image:
        chemin = Path(args.image)
        if not chemin.is_absolute():
            chemin = RACINE / args.image
        image = lire_image(chemin)
    elif args.video:
        chemin = Path(args.video)
        if not chemin.is_absolute():
            chemin = RACINE / args.video
        capture = cv2.VideoCapture(str(chemin))
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, image = capture.read()
        capture.release()
        image = image if ok else None
    else:
        print("Indiquez --image ou --video")
        return

    if image is None:
        print("Source illisible.")
        return

    hauteur, largeur = image.shape[:2]
    print(f"Image {largeur}x{hauteur}. Cliquez les sommets de la première zone.")

    cv2.namedWindow("tracer_roi")
    cv2.setMouseCallback("tracer_roi", au_clic)

    while True:
        cv2.imshow("tracer_roi", dessiner(image))
        touche = cv2.waitKey(20) & 0xFF
        if touche == 27:
            break
        if touche == ord("n") and len(sommets) >= 2:
            zones_terminees.append(list(sommets))
            sommets.clear()
            print(f"Zone {len(zones_terminees)} validée.")
        elif touche == ord("r"):
            sommets.clear()
            zones_terminees.clear()
            print("Effacé.")
        elif touche == ord("s"):
            if len(sommets) >= 2:
                zones_terminees.append(list(sommets))
                sommets.clear()
            imprimer_yaml(largeur, hauteur)
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
