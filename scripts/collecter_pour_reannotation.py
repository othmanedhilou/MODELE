"""
Collecte les images de production les plus utiles a annoter, pour ameliorer
le modele au fil du temps.

Ce que ce script est, et ce qu'il n'est pas
-------------------------------------------
Un modele YOLO ne s'entraine PAS tout seul sur ce qu'il voit. Il applique
ce qu'il a appris et n'en garde aucune trace. Un modele deploye ne
s'ameliore jamais de lui-meme.

Et il ne FAUT pas le reentrainer automatiquement sur ses propres sorties :
il apprendrait ses erreurs. Une fausse alarme reinjectee comme verite
devient une regle apprise, puis se reproduit plus souvent, puis est
reapprise. Le modele derive et personne ne s'en apercoit, parce que ses
metriques internes s'ameliorent pendant que ses resultats reels se
degradent.

La boucle correcte comporte une relecture humaine :

    production -> collecte des cas UTILES -> correction humaine
                       -> reentrainement -> evaluation -> promotion

Ce script automatise la premiere etape, celle qui prend le plus de temps :
choisir QUELLES images meritent d'etre annotees. Annoter au hasard 500
images de bande saine n'apporte rien ; annoter 50 images bien choisies
change le modele.

Comment les images sont choisies
--------------------------------
Trois signaux, du plus fort au plus faible :

1. DESACCORD entre la couche A (vision classique) et la couche B (YOLO).
   Quand deux methodes independantes ne disent pas la meme chose, l'une des
   deux se trompe : c'est exactement la ou l'annotation humaine tranche.
   C'est le signal le plus rentable et il ne coute rien.

2. CONFIANCE FAIBLE du modele. Une detection a 0,30 est une hesitation.

3. CLASSE RARE. Une detection de 'jonction_defectueuse' vaut plus qu'une
   millieme dechirure, parce que le modele en a vu peu.

  python scripts/collecter_pour_reannotation.py --source data/frames/convoyeur
  python scripts/collecter_pour_reannotation.py --source runs/alarmes --nombre 80
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import (charger_config, configurer_console,  # noqa: E402
                              ecrire_image, lire_image)
from src.detect.convoyeur_cv import DetecteurDechirureCV  # noqa: E402
from src.mlops.registre import poids_production  # noqa: E402

configurer_console()
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


def recouvrement(boite_a, boite_b):
    """Intersection sur union de deux boites (x1, y1, x2, y2)."""
    x1 = max(boite_a[0], boite_b[0]); y1 = max(boite_a[1], boite_b[1])
    x2 = min(boite_a[2], boite_b[2]); y2 = min(boite_a[3], boite_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aire_a = max(0, boite_a[2] - boite_a[0]) * max(0, boite_a[3] - boite_a[1])
    aire_b = max(0, boite_b[2] - boite_b[0]) * max(0, boite_b[3] - boite_b[1])
    union = aire_a + aire_b - inter
    return inter / union if union > 0 else 0.0


def score_interet(candidats_cv, detections_yolo, frequences, couche_b):
    """
    Note l'interet d'annoter cette image. Plus haut = plus utile.

    Le desaccord entre les deux couches domine volontairement les autres
    signaux : c'est le seul qui identifie une erreur certaine de l'une des
    deux methodes, sans qu'on sache encore laquelle.
    """
    score, raisons = 0.0, []

    # Le desaccord n'a de sens que si les DEUX couches se sont prononcees.
    # Sans modele promu, compter les detections de la couche A comme un
    # desaccord reviendrait a noter toutes les images de la meme facon.
    if couche_b:
        boites_cv = [c["bbox"] for c in candidats_cv]
        boites_yolo = [d["bbox"] for d in detections_yolo]

        apparies = sum(1 for boite in boites_cv
                       if any(recouvrement(boite, autre) > 0.2
                              for autre in boites_yolo))
        orphelins_cv = len(boites_cv) - apparies
        orphelins_yolo = len(boites_yolo) - apparies

        if orphelins_cv or orphelins_yolo:
            score += 10.0 * (orphelins_cv + orphelins_yolo)
            raisons.append(f"desaccord A/B ({orphelins_cv} vus par A seule, "
                           f"{orphelins_yolo} par B seule)")
    elif candidats_cv:
        # Faute de mieux : les images ou la couche A voit quelque chose sont
        # plus informatives que les images vides.
        score += 2.0 * len(candidats_cv)
        raisons.append(f"{len(candidats_cv)} detection(s) couche A")

    faibles = [d for d in detections_yolo if d["score"] < 0.5]
    if faibles:
        score += 3.0 * len(faibles)
        raisons.append(f"{len(faibles)} detection(s) peu sure(s)")

    for detection in detections_yolo:
        rarete = 1.0 / (1.0 + frequences.get(detection["classe"], 0))
        score += 4.0 * rarete
    for candidat in candidats_cv:
        rarete = 1.0 / (1.0 + frequences.get(candidat["classe"], 0))
        score += 2.0 * rarete

    classes_rares = {d["classe"] for d in detections_yolo
                     if frequences.get(d["classe"], 0) < 30}
    if classes_rares:
        raisons.append(f"classe(s) peu vue(s) : {sorted(classes_rares)}")

    if not candidats_cv and not detections_yolo:
        # Une image sans rien reste utile en petite quantite : elle apprend
        # au modele a quoi ressemble une bande saine.
        score += 0.5
        raisons.append("image de fond")

    return score, raisons


def main():
    ap = argparse.ArgumentParser(
        description="Selection des images de production a annoter en priorite")
    ap.add_argument("--source", required=True,
                    help="Dossier d'images de production, ou runs/alarmes")
    ap.add_argument("--sortie", default="data/a_annoter",
                    help="Dossier de sortie, a ouvrir dans CVAT ou LabelImg")
    ap.add_argument("--nombre", type=int, default=100,
                    help="Nombre d'images retenues")
    ap.add_argument("--modele", default="convoyeur",
                    choices=["convoyeur", "eclairage", "vehicules"])
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = RACINE / args.source
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in EXT_IMAGES)
    if not images:
        print(f"Aucune image dans {source}")
        print()
        print("Deux sources possibles :")
        print("  - les captures d'alarme du pipeline : runs/alarmes/")
        print("  - des images extraites d'une video :")
        print("      python -m src.prepare.extract_frames --source <video> \\")
        print("             --sortie data/frames/convoyeur --intervalle 1")
        return 1

    config = charger_config(f"configs/{args.modele}.yaml")
    detecteur = DetecteurDechirureCV(config) if args.modele == "convoyeur" else None

    modele_yolo = None
    poids = poids_production(args.modele)
    if poids is not None:
        from ultralytics import YOLO
        modele_yolo = YOLO(str(poids))
        print(f"Couche B chargee : {poids}")
    else:
        print("Aucun modele en production : selection basee sur la seule")
        print("couche A. Le signal de desaccord, le plus utile, est indisponible")
        print("tant qu'un modele n'est pas promu.")

    # Frequences deja vues : ce qui est rare vaut plus cher a annoter
    frequences = Counter()
    racine_data = RACINE / "data" / args.modele / "labels"
    noms = yaml.safe_load(
        (RACINE / "configs" / f"data_{args.modele}.yaml").read_text(
            encoding="utf-8"))["names"]
    for label in racine_data.rglob("*.txt"):
        for ligne in label.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                frequences[noms.get(int(float(ligne.split()[0])), "?")] += 1

    print(f"{len(images)} image(s) a examiner, {sum(frequences.values())} "
          f"instances deja annotees")

    classees = []
    for chemin in images:
        image = lire_image(chemin)
        if image is None:
            continue

        candidats_cv = []
        if detecteur is not None:
            detecteur._masque_roi = None
            candidats_cv = detecteur.analyser(image)["candidats"]

        detections = []
        if modele_yolo is not None:
            resultat = modele_yolo.predict(image, conf=0.20, verbose=False)[0]
            if resultat.boxes is not None:
                for boite, classe, score in zip(
                        resultat.boxes.xyxy.cpu().numpy(),
                        resultat.boxes.cls.cpu().numpy(),
                        resultat.boxes.conf.cpu().numpy()):
                    detections.append({"bbox": tuple(float(v) for v in boite),
                                       "classe": modele_yolo.names[int(classe)],
                                       "score": float(score)})

        score, raisons = score_interet(candidats_cv, detections,
                                       frequences, modele_yolo is not None)
        classees.append((score, chemin, candidats_cv, detections, raisons))

    classees.sort(key=lambda x: x[0], reverse=True)
    retenues = classees[:args.nombre]

    destination = Path(args.sortie)
    if not destination.is_absolute():
        destination = RACINE / args.sortie
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "images").mkdir(parents=True, exist_ok=True)
    (destination / "labels").mkdir(parents=True, exist_ok=True)

    fiche = []
    for rang, (score, chemin, candidats_cv, detections, raisons) in enumerate(retenues, 1):
        image = lire_image(chemin)
        nom = f"{rang:04d}_{chemin.stem}"
        ecrire_image(destination / "images" / f"{nom}.jpg", image,
                     [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Pre-annotations : celles de la couche A, a corriger a la main
        lignes = []
        if detecteur is not None and candidats_cv:
            lignes = detecteur.label_yolo({"candidats": candidats_cv}, image.shape)
        (destination / "labels" / f"{nom}.txt").write_text(
            chr(10).join(lignes), encoding="utf-8")

        fiche.append({"rang": rang, "fichier": f"{nom}.jpg",
                      "score": round(score, 2), "origine": str(chemin.name),
                      "raisons": raisons,
                      "pre_annotations": len(lignes)})

    (destination / "a_verifier.json").write_text(
        json.dumps(fiche, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{len(retenues)} image(s) retenue(s) dans {destination}")
    print()
    print("Les dix plus utiles a annoter :")
    for element in fiche[:10]:
        raisons = " | ".join(element["raisons"]) or "-"
        print(f"  {element['rang']:3d}. score {element['score']:6.1f}  {raisons[:62]}")
    print()
    print("Suite du travail :")
    print(f"  1. ouvrir {destination}/images dans CVAT ou LabelImg ;")
    print("  2. CORRIGER les pre-annotations : elles viennent de la vision")
    print("     classique et contiennent des erreurs. Les entrainer telles")
    print("     quelles reviendrait a apprendre au modele les erreurs de la")
    print("     couche A, ce qui n'apporterait rien ;")
    print(f"  3. python -m src.prepare.split_dataset --source {args.sortie} "
          f"--modele {args.modele}")
    print(f"  4. python -m src.prepare.check_dataset --modele {args.modele}")
    print(f"  5. reentrainer, evaluer, puis promouvoir si les metriques")
    print(f"     s'ameliorent SUR LE LOT DE TEST.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
