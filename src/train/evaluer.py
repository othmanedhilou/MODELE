"""
Évaluation d'un modèle entraîné sur le lot de test, et export au format
déploiement (ONNX) pour l'intégration au système de vidéosurveillance.

  python -m src.train.evaluer --modele vehicules
  python -m src.train.evaluer --modele convoyeur --exporter onnx

Le lot de test n'est jamais vu pendant l'entraînement : c'est le seul
chiffre que vous pouvez présenter comme performance réelle dans le rapport.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (RACINE, charger_config,  # noqa: E402
                              resoudre_data_yaml)
from src.mlops.registre import enregistrer_evaluation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Évaluation et export d'un modèle")
    ap.add_argument("--modele", required=True,
                    choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--poids", help="Chemin explicite vers un .pt")
    ap.add_argument("--lot", default="test", choices=["val", "test"])
    ap.add_argument("--exporter", choices=["onnx", "openvino", "engine"],
                    help="Format d'export pour le déploiement")
    args = ap.parse_args()

    config = charger_config(RACINE / "configs" / f"{args.modele}.yaml")
    poids = Path(args.poids) if args.poids else (
        RACINE / "runs" / args.modele / "train" / "weights" / "best.pt")

    if not poids.exists():
        print(f"Poids introuvables : {poids}")
        print(f"Entraînez d'abord :  python -m src.train.train --modele {args.modele}")
        return

    modele = YOLO(str(poids))

    # La resolution d'evaluation doit etre CELLE DE L'ENTRAINEMENT. Un modele
    # entraine a 320 puis evalue a 1024 s'effondre : les objets n'ont plus la
    # taille apprise. Le cas se produit des qu'on entraine avec --test-rapide,
    # ou qu'on modifie imgsz dans la configuration apres coup. On lit donc la
    # valeur enregistree dans le point de sauvegarde du modele.
    imgsz = config["entrainement"]["imgsz"]
    try:
        imgsz_entrainement = (modele.ckpt or {}).get("train_args", {}).get("imgsz")
        if imgsz_entrainement and int(imgsz_entrainement) != int(imgsz):
            print(f"[RESOLUTION] Modele entraine a {imgsz_entrainement}, "
                  f"configuration a {imgsz}. On evalue a {imgsz_entrainement}.")
            imgsz = int(imgsz_entrainement)
    except Exception:
        pass

    metriques = modele.val(
        data=str(resoudre_data_yaml(config["data_yaml"])),
        split=args.lot,
        imgsz=imgsz,
        conf=0.001,      # conf basse pour une courbe PR complète
        iou=0.6,
        plots=True,
        project=str(RACINE / "runs" / args.modele),
        name=f"eval_{args.lot}",
        exist_ok=True,
    )

    print(f"\n=== Résultats sur le lot '{args.lot}' ===")
    print(f"mAP50    : {metriques.box.map50:.4f}")
    print(f"mAP50-95 : {metriques.box.map:.4f}")
    print(f"Précision: {metriques.box.mp:.4f}")
    print(f"Rappel   : {metriques.box.mr:.4f}")

    print("\n--- Détail par classe (utile pour le rapport de stage) ---")
    noms = modele.names
    for index, identifiant in enumerate(metriques.box.ap_class_index):
        print(f"  {noms[int(identifiant)]:<22} "
              f"mAP50={metriques.box.ap50[index]:.3f}  "
              f"P={metriques.box.p[index]:.3f}  "
              f"R={metriques.box.r[index]:.3f}")

    # Un rappel faible sur la déchirure est le seul chiffre qui compte vraiment
    if args.modele == "convoyeur":
        print("\n[LECTURE] Pour le convoyeur, privilégiez le RAPPEL sur la classe")
        print("          'dechirure' : rater une déchirure coûte une bande")
        print("          (plusieurs dizaines de milliers de dirhams), alors qu'une")
        print("          fausse alerte ne coûte qu'une vérification opérateur.")

    par_classe = {}
    for index, identifiant in enumerate(metriques.box.ap_class_index):
        par_classe[noms[int(identifiant)]] = {
            "mAP50": metriques.box.ap50[index],
            "precision": metriques.box.p[index],
            "rappel": metriques.box.r[index],
        }
    enregistrer_evaluation(args.modele, poids, args.lot, {
        "mAP50": metriques.box.map50,
        "mAP50-95": metriques.box.map,
        "precision": metriques.box.mp,
        "rappel": metriques.box.mr,
    }, par_classe)
    print("Resultat enregistre au registre "
          "(python -m src.mlops.registre --lister)")

    if args.exporter:
        print(f"\nExport au format {args.exporter} ...")
        chemin = modele.export(format=args.exporter, imgsz=config["entrainement"]["imgsz"],
                               half=False, simplify=True)
        print(f"Modèle exporté : {chemin}")


if __name__ == "__main__":
    main()
