"""
Extraction d'images à partir des vidéos de vidéosurveillance.

Deux pièges classiques que ce script évite :
  1. Extraire 25 images par seconde -> 24 h de vidéo = 2 millions d'images
     quasi identiques, inannotables. On échantillonne donc par intervalle.
  2. Une caméra fixe produit des images redondantes quand rien ne bouge.
     On filtre par différence d'histogramme : une image trop proche de la
     précédente retenue est écartée.

Exemple :
  python -m src.prepare.extract_frames --source data/raw/cam_portail.mp4 \
         --sortie data/frames/vehicules --intervalle 2 --seuil-similarite 0.93
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import ecrire_image  # noqa: E402


def histogramme(image: np.ndarray) -> np.ndarray:
    """Histogramme HSV normalisé, servant de signature visuelle de l'image."""
    petite = cv2.resize(image, (160, 120))
    hsv = cv2.cvtColor(petite, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def extraire_video(chemin_video: Path, dossier_sortie: Path, intervalle_s: float,
                   seuil_similarite: float, max_images: int, prefixe: str) -> int:
    """Extrait les images d'une vidéo. Retourne le nombre d'images écrites."""
    capture = cv2.VideoCapture(str(chemin_video))
    if not capture.isOpened():
        print(f"  [ERREUR] Impossible d'ouvrir {chemin_video}")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pas = max(1, int(round(fps * intervalle_s)))
    print(f"  {chemin_video.name} : {fps:.1f} fps, {total} frames, pas = {pas}")

    dossier_sortie.mkdir(parents=True, exist_ok=True)
    index_frame, ecrites, ignorees = 0, 0, 0
    hist_precedent = None

    while True:
        ok = capture.grab()          # grab() sans decode : bien plus rapide
        if not ok:
            break
        if index_frame % pas == 0:
            ok, image = capture.retrieve()
            if ok and image is not None:
                hist = histogramme(image)
                if hist_precedent is not None:
                    similarite = cv2.compareHist(hist_precedent, hist,
                                                 cv2.HISTCMP_CORREL)
                    if similarite > seuil_similarite:
                        ignorees += 1
                        index_frame += 1
                        continue
                nom = f"{prefixe}_{index_frame:07d}.jpg"
                ecrire_image(dossier_sortie / nom, image,
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
                hist_precedent = hist
                ecrites += 1
                if max_images and ecrites >= max_images:
                    break
        index_frame += 1

    capture.release()
    print(f"  -> {ecrites} images écrites, {ignorees} écartées (trop similaires)")
    return ecrites


def main() -> None:
    ap = argparse.ArgumentParser(description="Extraction d'images depuis des vidéos")
    ap.add_argument("--source", required=True,
                    help="Fichier vidéo ou dossier contenant des vidéos")
    ap.add_argument("--sortie", required=True, help="Dossier de destination")
    ap.add_argument("--intervalle", type=float, default=2.0,
                    help="Intervalle d'échantillonnage en secondes (défaut 2)")
    ap.add_argument("--seuil-similarite", type=float, default=0.93,
                    help="Au-dessus de ce score, l'image est jugée redondante")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Limite d'images par vidéo (0 = illimité)")
    args = ap.parse_args()

    source = Path(args.source)
    sortie = Path(args.sortie)
    extensions = {".mp4", ".avi", ".mkv", ".mov", ".dav", ".ts"}

    if source.is_dir():
        videos = sorted(p for p in source.rglob("*") if p.suffix.lower() in extensions)
    else:
        videos = [source]

    if not videos:
        print(f"Aucune vidéo trouvée dans {source}")
        return

    total = 0
    for video in videos:
        print(f"Traitement de {video} ...")
        total += extraire_video(video, sortie, args.intervalle,
                                args.seuil_similarite, args.max_images,
                                video.stem)
    print(f"\nTerminé : {total} images dans {sortie}")


if __name__ == "__main__":
    main()
