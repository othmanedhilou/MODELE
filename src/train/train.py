"""
Entraînement unifié des trois modèles, piloté par les fichiers configs/.

  python -m src.train.train --modele vehicules
  python -m src.train.train --modele convoyeur --epochs 300 --batch 8
  python -m src.train.train --modele eclairage --reprendre

Le script détecte automatiquement le matériel : GPU CUDA, Apple MPS, sinon
CPU. Sur CPU il réduit la charge et prévient que l'entraînement complet
n'est pas réaliste (utilisez Google Colab, voir notebooks/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (RACINE, charger_config,  # noqa: E402
                              resoudre_data_yaml)
from src.mlops.registre import enregistrer_experience  # noqa: E402


def detecter_materiel() -> str:
    """Retourne l'identifiant de device attendu par Ultralytics."""
    if torch.cuda.is_available():
        nom = torch.cuda.get_device_name(0)
        memoire = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[MATERIEL] GPU CUDA : {nom} ({memoire:.1f} Go)")
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[MATERIEL] Apple MPS")
        return "mps"
    print("[MATERIEL] CPU uniquement.")
    print("           Un entraînement YOLO complet sur CPU prend plusieurs jours.")
    print("           Utilisez Google Colab (notebooks/entrainement_colab.ipynb)")
    print("           ou le GPU d'un poste de l'entreprise.")
    return "cpu"


def construire_arguments(config: dict, device: str, surcharges: dict) -> dict:
    """Fusionne configuration YAML, matériel détecté et options de ligne de commande."""
    entrainement = dict(config.get("entrainement", {}))
    augmentation = dict(config.get("augmentation", {}))

    arguments = {
        "data": str(resoudre_data_yaml(config["data_yaml"])),
        "project": str(RACINE / "runs" / config["nom"]),
        "name": "train",
        "exist_ok": True,
        "device": device,
        "plots": True,
        "val": True,
        **entrainement,
        **augmentation,
    }

    # Sur CPU on allège pour permettre au moins un test de bout en bout
    if device == "cpu":
        arguments["workers"] = 0
        arguments["batch"] = min(arguments.get("batch", 8), 4)
        arguments["amp"] = False

    arguments.update({c: v for c, v in surcharges.items() if v is not None})
    return arguments


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement des modèles de détection")
    ap.add_argument("--modele", required=True,
                    choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--modele-base", dest="modele_base",
                    help="Écrase le poids de départ (ex. yolo11m.pt)")
    ap.add_argument("--reprendre", action="store_true",
                    help="Reprend l'entraînement interrompu (last.pt)")
    ap.add_argument("--test-rapide", action="store_true",
                    help="3 epochs sur petite résolution, pour valider la chaîne")
    args = ap.parse_args()

    config = charger_config(RACINE / "configs" / f"{args.modele}.yaml")
    device = detecter_materiel()

    surcharges = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz}
    if args.test_rapide:
        surcharges.update({"epochs": 3, "imgsz": 320, "batch": 2, "patience": 0})
        print("[MODE] Test rapide : on valide la chaîne, pas la performance.")

    arguments = construire_arguments(config, device, surcharges)

    dossier_run = RACINE / "runs" / config["nom"] / "train"
    dernier = dossier_run / "weights" / "last.pt"
    if args.reprendre and dernier.exists():
        print(f"[REPRISE] depuis {dernier}")
        modele = YOLO(str(dernier))
        arguments["resume"] = True
    else:
        poids_depart = args.modele_base or config["modele_base"]
        print(f"[DEPART] poids pré-entraînés : {poids_depart}")
        modele = YOLO(poids_depart)

    print(f"[DATASET] {arguments['data']}")
    print(f"[SORTIE ] {dossier_run}\n")

    resultats = modele.train(**arguments)

    poids = dossier_run / "weights" / "best.pt"
    print()
    print("=== Entrainement termine ===")
    print(f"Meilleurs poids : {poids}")

    metriques = {}
    try:
        metriques = resultats.results_dict
        print(f"mAP50    : {metriques.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP50-95 : {metriques.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"Precision: {metriques.get('metrics/precision(B)', 0):.4f}")
        print(f"Rappel   : {metriques.get('metrics/recall(B)', 0):.4f}")
    except Exception:
        pass

    # Tracabilite : on inscrit l'entrainement au registre avec l'empreinte du
    # dataset utilise. Sans cela, impossible de savoir plus tard quelles
    # donnees ont produit quel modele, ni pourquoi deux runs different.
    try:
        fiche = enregistrer_experience(args.modele, arguments, metriques, poids)
        print()
        print(f"Registre : {args.modele} {fiche['version']} "
              f"(dataset {fiche['dataset']['empreinte']}, "
              f"{fiche['dataset']['images']} images)")
    except Exception as erreur:
        print(f"[AVERTISSEMENT] Enregistrement au registre impossible : {erreur}")

    print()
    print("Etapes suivantes :")
    print(f"  python -m src.train.evaluer --modele {args.modele}")
    print("  python -m src.mlops.registre --lister")


if __name__ == "__main__":
    main()
