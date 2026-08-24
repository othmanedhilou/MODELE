"""
Génère une scène nocturne synthétique de parc d'usine pour valider le
module d'éclairage avant l'arrivée des vidéos réelles.

Trois luminaires : un allumé, un très faible (en fin de vie), un éteint.
La zone de gauche est correctement éclairée, celle de droite ne l'est pas.

  python scripts/generer_eclairage_synthetique.py
  python -m src.detect.eclairage --source data/raw/parc_nuit_synthetique.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, ecrire_image  # noqa: E402

configurer_console()
LARGEUR, HAUTEUR = 1280, 720


def halo(image, centre, rayon, intensite):
    """Ajoute un halo lumineux gaussien, comme un projecteur dans la nuit."""
    y, x = np.ogrid[:HAUTEUR, :LARGEUR]
    distance2 = (x - centre[0]) ** 2 + (y - centre[1]) ** 2
    gain = intensite * np.exp(-distance2 / (2.0 * rayon ** 2))
    return np.clip(image.astype(np.float32) + gain[..., None], 0, 255).astype(np.uint8)


def main():
    # Fond nocturne : sol sombre, ciel un peu plus clair
    image = np.full((HAUTEUR, LARGEUR, 3), 18, np.uint8)
    image[:280] = (30, 26, 22)
    rng = np.random.default_rng(0)
    image = np.clip(image.astype(np.int16)
                    + rng.normal(0, 4, image.shape).astype(np.int16), 0, 255).astype(np.uint8)

    # Trois mâts d'éclairage
    lampes = [
        ((260, 200), 190, 150, "allume"),
        ((640, 200), 120,  38, "faible"),
        ((1020, 200),  0,   0, "eteint"),
    ]
    for (cx, cy), rayon, intensite, _ in lampes:
        cv2.line(image, (cx, cy + 20), (cx, HAUTEUR), (55, 55, 55), 6)   # mât
        if intensite > 0:
            image = halo(image, (cx, cy), rayon, intensite)

    # Globe du luminaire, dessiné après le halo
    for (cx, cy), _, intensite, _ in lampes:
        couleur = (255, 250, 225) if intensite > 120 else \
                  (150, 145, 120) if intensite > 0 else (48, 48, 50)
        cv2.ellipse(image, (cx, cy), (34, 16), 0, 0, 360, couleur, -1)

    ecrire_image(RACINE / "data" / "raw" / "parc_nuit_synthetique.jpg", image)
    print(f"Image : {RACINE / 'data' / 'raw' / 'parc_nuit_synthetique.jpg'}")
    print("Boîtes des luminaires (pour un test avec modèle) :")
    for (cx, cy), _, _, etat in lampes:
        print(f"  {etat:<8} bbox = [{cx-40}, {cy-24}, {cx+40}, {cy+24}]")


if __name__ == "__main__":
    main()
