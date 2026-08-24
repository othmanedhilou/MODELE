"""
Contrôle qualité d'un dataset YOLO avant de lancer l'entraînement.

Vérifie : appariement image/label, identifiants de classe valides,
coordonnées normalisées, boîtes dégénérées, doublons, équilibre des classes
et fuite de données entre train et val (même image dans deux lots).

Un entraînement lancé sur un dataset non vérifié fait perdre des heures :
ce script coûte 10 secondes.

Exemple :
  python -m src.prepare.check_dataset --modele convoyeur
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}
MIN_INSTANCES_CONSEILLE = 100


def empreinte(chemin: Path) -> str:
    """Empreinte MD5 du contenu, pour repérer les doublons exacts."""
    return hashlib.md5(chemin.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrôle qualité dataset YOLO")
    ap.add_argument("--modele", required=True,
                    choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--segmentation", action="store_true",
                    help="Le dataset contient des polygones (convoyeur)")
    args = ap.parse_args()

    racine_data = RACINE / "data" / args.modele
    data_yaml = RACINE / "configs" / f"data_{args.modele}.yaml"
    noms = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    nb_classes = len(noms)
    est_segmentation = args.segmentation or args.modele == "convoyeur"

    erreurs, avertissements = [], []
    compte_global = Counter()
    empreintes = defaultdict(list)

    print(f"=== Dataset {args.modele} ({nb_classes} classes) ===\n")

    for lot in ("train", "val", "test"):
        dossier_img = racine_data / "images" / lot
        dossier_lbl = racine_data / "labels" / lot
        if not dossier_img.exists():
            continue

        images = sorted(p for p in dossier_img.iterdir()
                        if p.suffix.lower() in EXT_IMAGES)
        compte_lot = Counter()
        sans_label, boites = 0, 0

        for image in images:
            empreintes[empreinte(image)].append(f"{lot}/{image.name}")
            label = dossier_lbl / (image.stem + ".txt")
            if not label.exists():
                sans_label += 1
                continue

            for num, ligne in enumerate(
                    label.read_text(encoding="utf-8").splitlines(), start=1):
                ligne = ligne.strip()
                if not ligne:
                    continue
                champs = ligne.split()
                try:
                    classe = int(float(champs[0]))
                    valeurs = [float(v) for v in champs[1:]]
                except ValueError:
                    erreurs.append(f"{label.name}:{num} ligne illisible")
                    continue

                if not 0 <= classe < nb_classes:
                    erreurs.append(f"{label.name}:{num} classe {classe} hors "
                                   f"plage 0-{nb_classes - 1}")
                    continue

                attendu = "polygone (>=6 valeurs)" if est_segmentation else "4 valeurs"
                if est_segmentation:
                    valide = len(valeurs) >= 6 and len(valeurs) % 2 == 0
                else:
                    valide = len(valeurs) == 4
                if not valide:
                    erreurs.append(f"{label.name}:{num} format invalide, "
                                   f"attendu {attendu}")
                    continue

                if any(v < -0.001 or v > 1.001 for v in valeurs):
                    erreurs.append(f"{label.name}:{num} coordonnées non "
                                   f"normalisées (hors 0-1)")
                if not est_segmentation and (valeurs[2] <= 0 or valeurs[3] <= 0):
                    erreurs.append(f"{label.name}:{num} boîte de taille nulle")

                compte_lot[classe] += 1
                boites += 1

        compte_global.update(compte_lot)
        print(f"[{lot}] {len(images):5d} images | {boites:6d} instances | "
              f"{sans_label} sans annotation")
        for classe in sorted(compte_lot):
            print(f"        {noms[classe]:<20} {compte_lot[classe]:6d}")
        print()

    # Doublons et fuite entre lots
    for liste in empreintes.values():
        if len(liste) > 1:
            lots = {chemin.split("/")[0] for chemin in liste}
            if len(lots) > 1:
                erreurs.append(f"FUITE train/val : image identique dans {liste}")
            else:
                avertissements.append(f"Doublon exact : {liste}")

    # Classes trop rares ou absentes
    for identifiant, nom in noms.items():
        total = compte_global.get(identifiant, 0)
        if total == 0:
            avertissements.append(f"Classe '{nom}' : aucune instance annotée")
        elif total < MIN_INSTANCES_CONSEILLE:
            avertissements.append(
                f"Classe '{nom}' : {total} instances seulement "
                f"(visez {MIN_INSTANCES_CONSEILLE}+ pour un résultat fiable)")

    print("=" * 60)
    if erreurs:
        print(f"\n{len(erreurs)} ERREUR(S) - a corriger avant d'entrainer :")
        for e in erreurs[:30]:
            print(f"  - {e}")
        if len(erreurs) > 30:
            print(f"  ... et {len(erreurs) - 30} autres")
    if avertissements:
        print(f"\n{len(avertissements)} AVERTISSEMENT(S) :")
        for a in avertissements[:30]:
            print(f"  - {a}")
    if not erreurs and not avertissements:
        print("\nDataset conforme. Vous pouvez lancer l'entrainement.")


if __name__ == "__main__":
    main()
