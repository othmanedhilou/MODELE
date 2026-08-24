"""
Test de faisabilite de la lecture de plaque sur VOTRE camera.

A lancer AVANT d'investir du temps dans le module plaque. La lecture
d'immatriculation n'est pas d'abord un probleme de modele, c'est un
probleme de pixels : si la plaque fait 40 px de large dans l'image,
l'information n'y est pas, et aucun entrainement ne la fera apparaitre.

Trois modes :
  --image / --video : mesure automatique sur les vehicules detectes
  --manuel          : vous cliquez les deux extremites d'une plaque, le
                      script mesure. A utiliser si la detection automatique
                      ne trouve rien, ou pour verifier son resultat.

  python scripts/tester_lisibilite_plaque.py --image data/frames/portail/x.jpg
  python scripts/tester_lisibilite_plaque.py --video data/raw/portail.mp4 --images 30
  python scripts/tester_lisibilite_plaque.py --image x.jpg --manuel
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console, lire_image  # noqa: E402
from src.detect.plaque import DetecteurPlaque, nettete  # noqa: E402

configurer_console()
CLASSES_VEHICULES = {"car", "truck", "bus", "motorcycle"}
points_manuels = []


def au_clic(evenement, x, y, drapeaux, parametres):
    if evenement == cv2.EVENT_LBUTTONDOWN:
        points_manuels.append((x, y))


def mesure_manuelle(image, config):
    """Mesure la largeur d'une plaque designee a la souris."""
    detecteur = DetecteurPlaque(config)
    affichage = image.copy()
    cv2.namedWindow("mesure")
    cv2.setMouseCallback("mesure", au_clic)
    print("Cliquez le bord GAUCHE puis le bord DROIT d'une plaque. "
          "Echap pour terminer.")

    while True:
        rendu = affichage.copy()
        for point in points_manuels:
            cv2.circle(rendu, point, 4, (0, 0, 255), -1)
        if len(points_manuels) >= 2:
            cv2.line(rendu, points_manuels[-2], points_manuels[-1], (0, 0, 255), 2)
        cv2.imshow("mesure", rendu)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        if len(points_manuels) >= 2 and len(points_manuels) % 2 == 0:
            p1, p2 = points_manuels[-2], points_manuels[-1]
            largeur = abs(p2[0] - p1[0])
            hauteur = max(int(largeur / 4.5), 6)      # ratio plaque marocaine
            haut = min(p1[1], p2[1]) - hauteur // 2
            vignette = image[max(0, haut):haut + hauteur,
                             min(p1[0], p2[0]):max(p1[0], p2[0])]
            valeur = nettete(vignette) if vignette.size else 0.0
            verdict = detecteur.verdict_qualite(largeur, valeur)
            print(f"  largeur = {largeur:4d} px | nettete = {valeur:7.1f} "
                  f"-> {verdict.upper()}")
            points_manuels.clear()
    cv2.destroyAllWindows()


def analyser_image(image, detecteur, modele_vehicules, mesures):
    """Detecte les vehicules puis mesure leurs plaques."""
    predictions = modele_vehicules.predict(image, conf=0.35, verbose=False)[0]
    if predictions.boxes is None:
        return 0

    noms = modele_vehicules.names
    trouvees = 0
    for boite, classe in zip(predictions.boxes.xyxy.cpu().numpy(),
                             predictions.boxes.cls.cpu().numpy()):
        if noms[int(classe)] not in CLASSES_VEHICULES:
            continue
        largeur_vehicule = boite[2] - boite[0]
        candidats = detecteur.localiser(image, boite)
        if not candidats:
            mesures.append({"largeur_plaque": 0, "nettete": 0.0,
                            "verdict": "non trouvee",
                            "largeur_vehicule": largeur_vehicule})
            continue
        meilleur = candidats[0]
        mesures.append({"largeur_plaque": meilleur["largeur_px"],
                        "nettete": meilleur["nettete"],
                        "verdict": meilleur["verdict"],
                        "largeur_vehicule": largeur_vehicule})
        trouvees += 1
    return trouvees


def verdict_global(mesures, config):
    """Synthese et recommandation concrete."""
    print()
    print("=" * 64)
    if not mesures:
        print("Aucun vehicule detecte sur les images analysees.")
        print("Verifiez que les images contiennent bien des vehicules, ou")
        print("utilisez le mode --manuel pour mesurer vous-meme.")
        return 1

    trouvees = [m for m in mesures if m["largeur_plaque"] > 0]
    print(f"{len(mesures)} vehicule(s) analyse(s), "
          f"{len(trouvees)} plaque(s) localisee(s)")

    if not trouvees:
        print()
        print("VERDICT : aucune plaque localisee.")
        print("Deux causes possibles, dans cet ordre de probabilite :")
        print("  1. les plaques sont trop petites pour etre distinguees ;")
        print("  2. les vehicules sont vus de cote ou de dos trop incline.")
        print("Verifiez avec --manuel avant de conclure.")
        return 1

    largeurs = np.array([m["largeur_plaque"] for m in trouvees])
    nettetes = np.array([m["nettete"] for m in trouvees])
    ratio = np.array([m["largeur_plaque"] / max(m["largeur_vehicule"], 1)
                      for m in trouvees])

    print(f"  largeur de plaque : mediane {np.median(largeurs):.0f} px, "
          f"min {largeurs.min():.0f}, max {largeurs.max():.0f}")
    print(f"  nettete           : mediane {np.median(nettetes):.0f}")
    print(f"  plaque / vehicule : {np.median(ratio):.1%} de la largeur")

    comptes = {}
    for mesure in mesures:
        comptes[mesure["verdict"]] = comptes.get(mesure["verdict"], 0) + 1
    print("  verdicts          : " + ", ".join(f"{v} : {c}"
                                               for v, c in sorted(comptes.items())))

    mediane = float(np.median(largeurs))
    seuils = config["qualite"]
    print()
    if mediane >= seuils["largeur_bonne_px"]:
        print("VERDICT : LECTURE FIABLE POSSIBLE.")
        print("  Activez le module plaque et entrainez un detecteur dedie")
        print("  pour gagner en precision de localisation.")
        return 0
    if mediane >= seuils["largeur_limite_px"]:
        print("VERDICT : LECTURE PARTIELLE POSSIBLE.")
        print("  Les chiffres seront lus la plupart du temps, la lettre arabe")
        print("  rarement. Suffisant pour rapprocher un camion d'un bon de")
        print("  livraison, insuffisant pour un controle d'acces automatique.")
        print("  Gain facile : augmenter la resolution du flux RTSP si la")
        print("  camera le permet (souvent regle en 720p alors qu'elle fait 4K).")
        return 0
    if mediane >= seuils["largeur_illisible_px"]:
        print("VERDICT : A LA LIMITE, resultats peu fiables.")
    else:
        print("VERDICT : LECTURE IMPOSSIBLE avec cette camera.")
    print()
    print("  Ce n'est PAS un probleme de modele : l'information n'est pas")
    print("  dans l'image. Par ordre de cout croissant :")
    print("   1. verifier la resolution du flux (un flux 'sous-canal' 704x576")
    print("      est parfois utilise alors que la camera filme en 1920x1080) ;")
    print("   2. zoomer la camera sur la voie d'entree plutot que sur tout le")
    print("      portail ;")
    print("   3. dedier une camera a la lecture de plaque, placee a 3-6 m de")
    print("      la voie, a hauteur de plaque, dans l'axe de circulation.")
    print()
    print("  En attendant, le comptage des vehicules et la surveillance de")
    print("  zone fonctionnent : ils n'ont pas besoin de cette resolution.")
    return 1


def main():
    ap = argparse.ArgumentParser(description="Faisabilite de la lecture de plaque")
    ap.add_argument("--image", help="Image du portail")
    ap.add_argument("--video", help="Video du portail")
    ap.add_argument("--images", type=int, default=20,
                    help="Nombre d'images a analyser dans la video")
    ap.add_argument("--manuel", action="store_true",
                    help="Mesure a la souris au lieu de la detection automatique")
    ap.add_argument("--config", default="configs/plaque.yaml")
    args = ap.parse_args()

    config = charger_config(args.config)

    if args.image:
        chemin = Path(args.image)
        if not chemin.is_absolute():
            chemin = RACINE / args.image
        image = lire_image(chemin)
        if image is None:
            print(f"Image illisible : {chemin}")
            return 1
        print(f"Image : {chemin.name} ({image.shape[1]}x{image.shape[0]})")
        if args.manuel:
            mesure_manuelle(image, config)
            return 0
        images = [image]
    elif args.video:
        chemin = Path(args.video)
        if not chemin.is_absolute():
            chemin = RACINE / args.video
        capture = cv2.VideoCapture(str(chemin))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pas = max(1, total // max(args.images, 1))
        images = []
        for index in range(0, total, pas):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok:
                images.append(image)
            if len(images) >= args.images:
                break
        capture.release()
        print(f"Video : {chemin.name}, {len(images)} image(s) echantillonnee(s)")
    else:
        print("Indiquez --image ou --video")
        return 1

    from ultralytics import YOLO
    modele_vehicules = YOLO("yolo11s.pt")
    detecteur = DetecteurPlaque(config)

    mesures = []
    for image in images:
        analyser_image(image, detecteur, modele_vehicules, mesures)

    return verdict_global(mesures, config)


if __name__ == "__main__":
    sys.exit(main())
