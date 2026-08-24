"""
Répartition d'un dossier annoté en train / val / test au format YOLO.

Entrée attendue : un dossier plat contenant les images ET les .txt YOLO
(sortie typique de LabelImg, CVAT ou Roboflow), par exemple :
    data/annote_vehicules/img001.jpg
    data/annote_vehicules/img001.txt

Sortie : data/<modele>/images/{train,val,test} + labels/{train,val,test}

Le tirage est stratifié par classe majoritaire de chaque image : sans cela,
avec un dataset déséquilibré (peu de chariots élévateurs par exemple),
la validation peut ne contenir aucune instance de la classe rare.

Exemple :
  python -m src.prepare.split_dataset --source data/annote_vehicules \
         --modele vehicules --ratios 0.7 0.2 0.1
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


def classe_majoritaire(chemin_label: Path) -> int:
    """Classe la plus représentée dans un fichier d'annotation (-1 si vide)."""
    if not chemin_label.exists():
        return -1
    classes = []
    for ligne in chemin_label.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if ligne:
            classes.append(int(float(ligne.split()[0])))
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Découpage train/val/test YOLO")
    ap.add_argument("--source", required=True, help="Dossier plat images + labels")
    ap.add_argument("--modele", required=True,
                    choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copier", action="store_true",
                    help="Copier au lieu de déplacer (garde la source intacte)")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = RACINE / source
    destination = RACINE / "data" / args.modele

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in EXT_IMAGES)
    if not images:
        print(f"Aucune image trouvée dans {source}")
        return

    # Regroupement par classe majoritaire pour un tirage stratifié
    groupes = defaultdict(list)
    sans_annotation = 0
    for image in images:
        label = image.with_suffix(".txt")
        if not label.exists():
            sans_annotation += 1
        groupes[classe_majoritaire(label)].append(image)

    if sans_annotation:
        print(f"[INFO] {sans_annotation} image(s) sans .txt : traitées comme "
              f"images de fond (utile et normal, gardez-en 5 a 10 %).")

    aleatoire = random.Random(args.seed)
    repartition = {"train": [], "val": [], "test": []}
    r_train, r_val, _ = args.ratios

    for classe, fichiers in groupes.items():
        aleatoire.shuffle(fichiers)
        n = len(fichiers)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        repartition["train"] += fichiers[:n_train]
        repartition["val"] += fichiers[n_train:n_train + n_val]
        repartition["test"] += fichiers[n_train + n_val:]

    operation = shutil.copy2 if args.copier else shutil.move
    for lot, fichiers in repartition.items():
        dossier_img = destination / "images" / lot
        dossier_lbl = destination / "labels" / lot
        dossier_img.mkdir(parents=True, exist_ok=True)
        dossier_lbl.mkdir(parents=True, exist_ok=True)
        for image in fichiers:
            operation(str(image), str(dossier_img / image.name))
            label = image.with_suffix(".txt")
            if label.exists():
                operation(str(label), str(dossier_lbl / label.name))
        print(f"{lot:>5} : {len(fichiers):5d} images -> {dossier_img}")

    print(f"\nTerminé. Vérifiez maintenant avec :\n"
          f"  python -m src.prepare.check_dataset --modele {args.modele}")


if __name__ == "__main__":
    main()
