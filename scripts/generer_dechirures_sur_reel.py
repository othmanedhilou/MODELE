"""
Composite des dechirures synthetiques sur de VRAIES images de convoyeur.

Pourquoi cette methode plutot que la generation entierement synthetique
--------------------------------------------------------------------
Une image entierement dessinee ne ressemble pas a une vraie : texture du
caoutchouc, matiere transportee, flou de mouvement, poussiere en
suspension, vapeur, structures metalliques. Un modele entraine dessus
apprend le rendu, pas le defaut, et s'effondre sur les vraies cameras.
C'est l'ecart de domaine (sim-to-real gap).

Ici, TOUT est reel sauf la dechirure elle-meme : le fond, l'eclairage, la
matiere, le bruit du capteur, la compression du flux. Seuls quelques
milliers de pixels sont synthetiques. L'ecart de domaine se reduit a la
seule apparence de la dechirure, au lieu de porter sur l'image entiere.

Et surtout, cela resout le vrai probleme du projet : la bande SAINE, vous
en avez des heures, elle defile tous les jours. Ce sont les dechirures qui
manquent. Cette methode transforme donc une ressource abondante en dataset
d'entrainement annote.

Trois precautions de realisme sont appliquees :
  1. la clarte de la dechirure est calculee par rapport a la luminosite
     LOCALE de la bande, jamais en valeur absolue : une trace de 220 sur
     une bande a 30 est credible, la meme sur une bande a 180 ne l'est pas ;
  2. les bords sont adoucis, une dechirure collee au pixel pres se voit ;
  3. le flou et le bruit de l'image d'origine sont reappliques par-dessus,
     sinon la dechirure est plus nette que le reste et le modele apprend
     a reperer la nettete au lieu du defaut.

  python scripts/generer_dechirures_sur_reel.py --source data/frames/convoyeur
  python scripts/generer_dechirures_sur_reel.py --source data/frames/convoyeur \
         --par-image 3 --ratio-sain 0.25
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import (charger_config, configurer_console,  # noqa: E402
                              ecrire_image, lire_image)
from src.detect.convoyeur_cv import DetecteurDechirureCV  # noqa: E402

configurer_console()
DECHIRURE = 0
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


def masque_bande(image, detecteur):
    """Masque de la bande, via la detection de ROI du module convoyeur."""
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detecteur._masque_roi = None
    return detecteur._obtenir_masque(detecteur.clahe.apply(gris))


def polygone_dechirure(masque, angle_bande, rng):
    """
    Trace une dechirure a l'interieur de la bande, orientee dans le sens de
    defilement. Retourne le polygone du contour, ou None si le placement
    echoue (bande trop petite ou trop etroite).
    """
    ys, xs = np.nonzero(masque)
    if ys.size < 500:
        return None

    # Point de depart tire au sort dans la bande, en evitant les bords
    for _ in range(20):
        index = int(rng.integers(0, ys.size))
        x0, y0 = float(xs[index]), float(ys[index])
        marge = 0.08 * masque.shape[1]
        if (masque[:, max(0, int(x0 - marge)):int(x0 + marge)] > 0).all(axis=None):
            break

    etendue_y = ys.max() - ys.min()
    longueur = rng.uniform(0.12, 0.6) * max(etendue_y, 50)
    epaisseur = rng.uniform(1.5, 7.0)
    courbure = rng.uniform(-0.06, 0.06) * longueur
    derive = rng.uniform(-0.12, 0.12) * longueur

    radians = np.radians(angle_bande)
    direction = np.array([np.cos(radians), np.sin(radians)], np.float32)
    normale = np.array([-direction[1], direction[0]], np.float32)

    axe = []
    for pas in np.linspace(-0.5, 0.5, 24):
        decalage = (courbure * np.cos(pas * np.pi) + derive * pas)
        point = (np.array([x0, y0], np.float32)
                 + direction * (pas * longueur) + normale * decalage)
        axe.append(point)
    axe = np.array(axe, np.float32)

    largeurs = epaisseur * (0.35 + 0.65 * np.sin(np.linspace(0.15, np.pi - 0.15, len(axe))))
    cote_a = axe + normale * largeurs[:, None]
    cote_b = axe - normale * largeurs[:, None]
    polygone = np.concatenate([cote_a, cote_b[::-1]])

    # La dechirure doit rester majoritairement sur la bande
    points = np.clip(polygone, [0, 0],
                     [masque.shape[1] - 1, masque.shape[0] - 1]).astype(int)
    dedans = masque[points[:, 1], points[:, 0]] > 0
    if dedans.mean() < 0.85:
        return None
    return polygone


def incruster(image, polygone, masque_bande_image, rng):
    """
    Incruste une dechirure dans l'image, en respectant l'eclairage local.

    Retourne l'image modifiee. La clarte est calculee a partir de la
    luminosite mediane de la bande AUTOUR de la dechirure : c'est ce qui
    rend l'incrustation credible sur une image sombre comme sur une image
    surexposee, et ce qui empeche le modele d'apprendre un seuil absolu.
    """
    hauteur, largeur = image.shape[:2]
    masque = np.zeros((hauteur, largeur), np.uint8)
    cv2.fillPoly(masque, [polygone.astype(np.int32)], 255)
    if masque.sum() == 0:
        return image, None

    # Luminosite locale de la bande : anneau autour de la dechirure
    dilate = cv2.dilate(masque, np.ones((41, 41), np.uint8))
    anneau = cv2.bitwise_and(cv2.subtract(dilate, masque), masque_bande_image)
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valeurs = gris[anneau > 0]
    if valeurs.size < 50:
        return image, None
    fond_local = float(np.median(valeurs))

    # Une carcasse textile exposee est nettement plus claire que le
    # caoutchouc, sans jamais etre parfaitement blanche.
    ecart = rng.uniform(55, 165)
    clarte = float(np.clip(fond_local + ecart, 0, 245))

    # Texture interne : la carcasse n'est pas uniforme
    texture = np.full((hauteur, largeur), clarte, np.float32)
    texture += rng.normal(0, clarte * 0.10, (hauteur, largeur)).astype(np.float32)

    # Bords adoucis : une incrustation nette au pixel se repere a l'oeil
    alpha = cv2.GaussianBlur(masque.astype(np.float32) / 255.0, (5, 5), 0)
    alpha = np.clip(alpha * rng.uniform(0.85, 1.0), 0, 1)[..., None]

    resultat = image.astype(np.float32) * (1 - alpha) + texture[..., None] * alpha

    # Le flou et le bruit de l'image d'origine sont reappliques localement,
    # sinon la dechirure est plus nette que le reste de l'image et le modele
    # apprend a detecter la nettete plutot que le defaut.
    x, y, w, h = cv2.boundingRect(masque)
    marge = 12
    x1, y1 = max(0, x - marge), max(0, y - marge)
    x2, y2 = min(largeur, x + w + marge), min(hauteur, y + h + marge)
    zone = resultat[y1:y2, x1:x2]
    if zone.size:
        k = int(rng.choice([3, 3, 5]))
        zone = cv2.GaussianBlur(zone, (k, k), 0)
        zone += rng.normal(0, rng.uniform(2, 6), zone.shape)
        resultat[y1:y2, x1:x2] = zone

    return np.clip(resultat, 0, 255).astype(np.uint8), masque


def ligne_yolo(polygone, forme):
    """Polygone normalise au format YOLO segmentation."""
    hauteur, largeur = forme[:2]
    points = polygone.astype(np.float32).copy()
    points[:, 0] = np.clip(points[:, 0] / largeur, 0, 1)
    points[:, 1] = np.clip(points[:, 1] / hauteur, 0, 1)
    coords = " ".join(f"{v:.6f}" for v in points.flatten())
    return f"{DECHIRURE} {coords}"


def main():
    ap = argparse.ArgumentParser(
        description="Dechirures synthetiques sur images reelles de convoyeur")
    ap.add_argument("--source", required=True,
                    help="Dossier d'images REELLES de bande (saine)")
    ap.add_argument("--sortie", default="data/convoyeur",
                    help="Dataset de destination")
    ap.add_argument("--par-image", type=int, default=2,
                    help="Nombre de variantes generees par image source")
    ap.add_argument("--ratio-sain", type=float, default=0.2,
                    help="Part d'images conservees sans defaut (images de fond)")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--config", default="configs/convoyeur.yaml")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = RACINE / args.source
    destination = Path(args.sortie)
    if not destination.is_absolute():
        destination = RACINE / args.sortie

    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXT_IMAGES)
    if not images:
        print(f"Aucune image dans {source}")
        print()
        print("Ce script a besoin d'images REELLES de votre convoyeur, meme")
        print("sans aucun defaut. Pour les obtenir :")
        print("  python -m src.prepare.extract_frames --source data/raw/convoyeur.mp4 \\")
        print("         --sortie data/frames/convoyeur --intervalle 1")
        return 1

    config = charger_config(args.config)
    detecteur = DetecteurDechirureCV(config)
    rng = np.random.default_rng(args.seed)

    print(f"{len(images)} image(s) reelle(s) trouvee(s) dans {source}")
    total = len(images) * args.par_image
    bornes = np.cumsum([int(total * r) for r in args.ratios])

    index, compte, avec_defaut, echecs = 0, {"train": 0, "val": 0, "test": 0}, 0, 0

    for chemin in images:
        origine = lire_image(chemin)
        if origine is None:
            continue
        try:
            bande = masque_bande(origine, detecteur)
        except Exception:
            echecs += 1
            continue
        if bande is None or bande.sum() == 0:
            echecs += 1
            continue

        for _ in range(args.par_image):
            lot = ("train" if index < bornes[0]
                   else "val" if index < bornes[1] else "test")
            image = origine.copy()
            lignes = []

            if rng.random() >= args.ratio_sain:
                for _ in range(int(rng.integers(1, 3))):
                    polygone = polygone_dechirure(bande, detecteur._angle_bande, rng)
                    if polygone is None:
                        continue
                    image, masque = incruster(image, polygone, bande, rng)
                    if masque is not None:
                        lignes.append(ligne_yolo(polygone, image.shape))
                if lignes:
                    avec_defaut += 1

            nom = f"reel_{chemin.stem}_{index:05d}"
            ecrire_image(destination / "images" / lot / f"{nom}.jpg", image,
                         [cv2.IMWRITE_JPEG_QUALITY, 92])
            chemin_label = destination / "labels" / lot / f"{nom}.txt"
            chemin_label.parent.mkdir(parents=True, exist_ok=True)
            chemin_label.write_text("\n".join(lignes), encoding="utf-8")

            compte[lot] += 1
            index += 1

    print(f"\n{index} image(s) generee(s) dans {destination}")
    for lot, n in compte.items():
        print(f"  {lot:<6} {n:5d}")
    print(f"  avec dechirure  {avec_defaut}")
    print(f"  images de fond  {index - avec_defaut}")
    if echecs:
        print(f"\n[ATTENTION] {echecs} image(s) ecartee(s) : bande non detectee.")
        print("  Verifiez le polygone 'roi' de configs/convoyeur.yaml, ou")
        print("  desactivez roi.auto_detection si le cadrage est atypique.")

    print("\nEtapes suivantes :")
    print("  python -m src.prepare.check_dataset --modele convoyeur")
    print("  python -m src.mlops.registre --empreinte convoyeur")
    print("  python -m src.train.train --modele convoyeur")
    return 0


if __name__ == "__main__":
    sys.exit(main())
