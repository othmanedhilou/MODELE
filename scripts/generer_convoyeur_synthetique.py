"""
Génère une courte vidéo synthétique de convoyeur, avec et sans déchirure.

Sert à valider toute la chaîne de traitement AVANT d'avoir reçu les vidéos
réelles de l'usine, et à montrer une démonstration pendant la soutenance.

  python scripts/generer_convoyeur_synthetique.py
  python -m src.detect.convoyeur_cv --source data/raw/convoyeur_synthetique.mp4
"""
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console, ecrire_image  # noqa: E402

configurer_console()
LARGEUR, HAUTEUR, NB_IMAGES, FPS = 960, 720, 240, 25


def image_convoyeur(t: int, avec_dechirure: bool) -> np.ndarray:
    """Une image : structure métallique claire, bande noire, matière, défaut."""
    rng = np.random.default_rng(t)
    image = np.full((HAUTEUR, LARGEUR, 3), 110, np.uint8)      # structure claire
    cv2.rectangle(image, (0, 0), (LARGEUR, 40), (150, 150, 150), -1)

    # Bande noire légèrement trapézoïdale (perspective de la caméra)
    bande = np.array([[280, 40], [680, 40], [800, HAUTEUR], [160, HAUTEUR]])
    cv2.fillPoly(image, [bande], (28, 28, 30))

    # Texture caoutchouc + grains de clinker qui défilent
    bruit = rng.normal(0, 6, (HAUTEUR, LARGEUR, 1)).astype(np.int16)
    image = np.clip(image.astype(np.int16) + bruit, 0, 255).astype(np.uint8)
    for _ in range(60):
        x = int(rng.integers(200, 760))
        y = int((rng.integers(0, HAUTEUR) + t * 12) % HAUTEUR)
        cv2.circle(image, (x, y), int(rng.integers(2, 6)), (70, 68, 65), -1)

    # Rouleaux : bandes horizontales un peu plus claires (piège à faux positifs)
    for y in range(120, HAUTEUR, 180):
        cv2.line(image, (200, y), (780, y), (55, 55, 58), 3)

    if avec_dechirure:
        # Trace claire, fine, allongée, légèrement inclinée, qui défile
        y0 = int((t * 9) % (HAUTEUR + 300)) - 300
        p1 = (470, y0)
        p2 = (505, y0 + 260)
        cv2.line(image, p1, p2, (225, 222, 215), 5)
        cv2.line(image, p1, p2, (245, 244, 240), 2)

    return cv2.GaussianBlur(image, (3, 3), 0)


def main() -> None:
    dossier = RACINE / "data" / "raw"
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = dossier / "convoyeur_synthetique.mp4"

    ecrivain = cv2.VideoWriter(str(sortie), cv2.VideoWriter_fourcc(*"mp4v"),
                               FPS, (LARGEUR, HAUTEUR))
    for t in range(NB_IMAGES):
        # Bande saine sur la première moitié, déchirée sur la seconde
        ecrivain.write(image_convoyeur(t, avec_dechirure=t > NB_IMAGES // 2))
    ecrivain.release()

    # Deux images fixes pour les tests unitaires et le rapport
    ecrire_image(dossier / "convoyeur_sain.jpg", image_convoyeur(10, False))
    ecrire_image(dossier / "convoyeur_dechire.jpg", image_convoyeur(180, True))

    print(f"Vidéo   : {sortie}")
    print(f"Images  : {dossier / 'convoyeur_sain.jpg'}, "
          f"{dossier / 'convoyeur_dechire.jpg'}")


if __name__ == "__main__":
    main()
