"""
Test de faisabilite de la detection de defauts sur VOTRE camera de convoyeur.

Meme logique que tester_lisibilite_plaque.py, et meme conclusion possible :
le facteur limitant n'est presque jamais le modele ni le jeu de donnees,
c'est ce que la camera est physiquement capable de resoudre.

Trois grandeurs sont mesurees, dans cet ordre d'importance :

1. RESOLUTION. Combien de pixels la largeur de bande occupe-t-elle ? Une
   dechirure de 5 mm de large sur une bande de 1200 mm vue sur 150 pixels
   fait 0,6 pixel : elle n'existe pas dans l'image, aucun entrainement ne
   la fera apparaitre.

2. FLOU DE MOUVEMENT. C'est le point le plus souvent oublie. Une bande
   defile a 2 a 4 m/s. Une camera de videosurveillance ordinaire expose
   pendant 1/30 s : le defaut se retrouve etale sur plusieurs centimetres
   et disparait dans le fond. Les installations industrielles de detection
   de dechirure utilisent pour cette raison des vitesses d'obturation
   elevees, voire des cameras lineaires.

3. CONTRASTE. Une carcasse textile exposee doit ressortir du caoutchouc.
   Sous un eclairage insuffisant ou dans la poussiere, l'ecart s'efface.

  python scripts/tester_resolution_convoyeur.py --video data/raw/convoyeur.mp4 \
         --largeur-bande-mm 1200
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console, lire_image  # noqa: E402
from src.detect.convoyeur_cv import DetecteurDechirureCV  # noqa: E402

configurer_console()


def largeur_bande_px(masque):
    """Largeur mediane de la bande, en pixels."""
    largeurs = []
    for y in range(0, masque.shape[0], max(1, masque.shape[0] // 40)):
        colonnes = np.nonzero(masque[y])[0]
        if colonnes.size >= 2:
            largeurs.append(float(colonnes[-1] - colonnes[0]))
    return float(np.median(largeurs)) if largeurs else 0.0


def flou_directionnel(gris, masque, angle_bande):
    """
    Estime le flou de mouvement le long du sens de defilement.

    On compare la nettete mesuree DANS le sens du defilement a celle mesuree
    perpendiculairement. Sur une image nette, les deux sont comparables. Si
    la bande a bouge pendant l'exposition, les details sont etales dans le
    sens du mouvement : la nettete longitudinale s'effondre, la nettete
    transversale reste. Le rapport des deux mesure donc le flou de
    mouvement, independamment de la nettete generale de l'image.
    """
    zone = cv2.bitwise_and(gris, gris, mask=masque)
    radians = np.radians(angle_bande)
    dx, dy = np.cos(radians), np.sin(radians)

    sobel_x = cv2.Sobel(zone, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(zone, cv2.CV_32F, 0, 1, ksize=3)
    longitudinal = np.abs(sobel_x * dx + sobel_y * dy)
    transversal = np.abs(-sobel_x * dy + sobel_y * dx)

    pixels = masque > 0
    energie_long = float(longitudinal[pixels].mean()) if pixels.any() else 0.0
    energie_trans = float(transversal[pixels].mean()) if pixels.any() else 0.0
    if energie_trans < 1e-6:
        return 1.0
    return energie_long / energie_trans


def contraste_bande(gris, masque):
    """Ecart type robuste des niveaux de gris de la bande."""
    pixels = gris[masque > 0]
    if pixels.size < 100:
        return 0.0
    mediane = float(np.median(pixels))
    return float(np.median(np.abs(pixels - mediane)) * 1.4826)


def verdict(mesures, largeur_bande_mm):
    """Synthese et recommandation concrete."""
    largeur_px = mesures["largeur_px"]
    print()
    print("=" * 66)
    if largeur_px < 20:
        print("VERDICT : bande non detectee dans l'image.")
        print("  Renseignez le polygone 'roi' de configs/convoyeur.yaml, ou")
        print("  desactivez roi.auto_detection si le cadrage est atypique.")
        return 1

    mm_par_px = largeur_bande_mm / largeur_px
    print(f"Largeur de bande      : {largeur_px:.0f} px pour {largeur_bande_mm:.0f} mm")
    print(f"Echelle               : {mm_par_px:.2f} mm par pixel")
    print(f"Flou de mouvement     : {mesures['flou']:.2f} "
          f"(1.0 = net, < 0.6 = etale dans le sens de defilement)")
    print(f"Contraste de la bande : {mesures['contraste']:.1f} niveaux de gris")
    print()

    # Un defaut doit couvrir au moins 3 pixels de large pour survivre au
    # seuillage, a la compression video et au filtrage morphologique.
    defaut_min_mm = 3 * mm_par_px
    print(f"Plus petit defaut detectable : environ {defaut_min_mm:.0f} mm de large")
    print(f"  (il faut au moins 3 pixels : en dessous, le defaut disparait")
    print(f"   au seuillage et dans la compression du flux)")
    print()

    problemes = []
    if mm_par_px > 4.0:
        problemes.append(
            f"RESOLUTION INSUFFISANTE : {mm_par_px:.1f} mm/px. Une dechirure "
            f"naissante de 5 a 10 mm de large est invisible.")
    elif mm_par_px > 2.5:
        problemes.append(
            f"Resolution limite : {mm_par_px:.1f} mm/px. Seules les dechirures "
            f"deja franches seront vues, pas les amorces.")

    if mesures["flou"] < 0.6:
        problemes.append(
            "FLOU DE MOUVEMENT IMPORTANT : les details sont etales dans le "
            "sens de defilement. C'est le probleme le plus frequent et le "
            "plus sous-estime sur une camera de videosurveillance ordinaire.")

    if mesures["contraste"] < 6:
        problemes.append(
            "CONTRASTE TRES FAIBLE : la bande est presque uniforme. Verifiez "
            "l'eclairage, et que le flux n'est pas trop compresse.")

    if not problemes:
        print("VERDICT : CETTE CAMERA CONVIENT.")
        print("  Entrainez sur vos propres images : c'est de loin le meilleur")
        print("  jeu de donnees possible. Extrayez des images de bande saine,")
        print("  puis incrustez-y des defauts annotes automatiquement :")
        print("    python -m src.prepare.extract_frames --source <video> \\")
        print("           --sortie data/frames/convoyeur --intervalle 1")
        print("    python scripts/generer_dechirures_sur_reel.py \\")
        print("           --source data/frames/convoyeur --par-image 3")
        return 0

    print("VERDICT : LIMITES IDENTIFIEES")
    for probleme in problemes:
        print(f"  - {probleme}")
    print()
    print("  Par ordre de cout croissant :")
    print("   1. verifier la resolution du flux : un sous-canal 704x576 est")
    print("      souvent diffuse alors que la camera filme en 1920x1080 ;")
    print("   2. augmenter la vitesse d'obturation de la camera, quitte a")
    print("      renforcer l'eclairage : c'est ce qui supprime le flou ;")
    print("   3. rapprocher ou zoomer la camera sur la bande plutot que sur")
    print("      toute la zone ;")
    print("   4. dedier une camera a l'inspection, montee pres de la bande.")
    print("      C'est la solution industrielle usuelle pour la detection de")
    print("      dechirure, et elle rend le probleme facile.")
    print()
    print("  En attendant, la couche vision classique reste utile pour les")
    print("  gros defauts et les corps etrangers, qui eux sont visibles.")
    return 1


def main():
    ap = argparse.ArgumentParser(
        description="Faisabilite de la detection de defauts de convoyeur")
    ap.add_argument("--video", help="Video du convoyeur")
    ap.add_argument("--image", help="Image du convoyeur")
    ap.add_argument("--largeur-bande-mm", type=float, required=True,
                    help="Largeur reelle de la bande, en mm (demandez a la "
                         "maintenance : typiquement 800, 1000, 1200 ou 1400)")
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--config", default="configs/convoyeur.yaml")
    args = ap.parse_args()

    config = charger_config(args.config)
    detecteur = DetecteurDechirureCV(config)

    images = []
    if args.image:
        chemin = Path(args.image)
        if not chemin.is_absolute():
            chemin = RACINE / args.image
        image = lire_image(chemin)
        if image is None:
            print(f"Image illisible : {chemin}")
            return 1
        images = [image]
    elif args.video:
        chemin = Path(args.video)
        if not chemin.is_absolute():
            chemin = RACINE / args.video
        capture = cv2.VideoCapture(str(chemin))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pas = max(1, total // max(args.images, 1))
        for index in range(0, total, pas):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok:
                images.append(image)
            if len(images) >= args.images:
                break
        capture.release()
    else:
        print("Indiquez --video ou --image")
        return 1

    if not images:
        print("Aucune image lisible.")
        return 1
    print(f"{len(images)} image(s) analysee(s), {images[0].shape[1]}x{images[0].shape[0]}")

    largeurs, flous, contrastes = [], [], []
    for image in images:
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detecteur._masque_roi = None
        masque = detecteur._obtenir_masque(detecteur.clahe.apply(gris))
        if masque is None or masque.sum() == 0:
            continue
        largeurs.append(largeur_bande_px(masque))
        flous.append(flou_directionnel(gris, masque, detecteur._angle_bande))
        contrastes.append(contraste_bande(gris, masque))

    if not largeurs:
        print("Bande non detectee sur les images analysees.")
        return 1

    return verdict({"largeur_px": float(np.median(largeurs)),
                    "flou": float(np.median(flous)),
                    "contraste": float(np.median(contrastes))},
                   args.largeur_bande_mm)


if __name__ == "__main__":
    sys.exit(main())
