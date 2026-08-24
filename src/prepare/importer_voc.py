"""
Import d'un dataset au format Pascal VOC (annotations XML) vers le format
YOLO du projet.

Beaucoup de datasets industriels publies avec des articles scientifiques
sont livres en VOC : une arborescence VOCdevkit avec JPEGImages/,
Annotations/*.xml et ImageSets/Main/*.txt. C'est notamment le cas de
BeltCrack, le seul jeu de donnees REEL de fissures de bande transporteuse
publie a ce jour (23 732 images industrielles, licence Apache 2.0).

Le decoupage train/val/test d'origine est conserve s'il existe dans
ImageSets/Main : reutiliser le decoupage des auteurs permet de comparer vos
resultats aux leurs. Sinon un decoupage est fait par sequence, jamais par
image : deux images consecutives d'une meme sequence video se ressemblent
trop, les separer entre train et val gonfle artificiellement les scores.

  python -m src.prepare.importer_voc --source telechargements/BeltCrack14ks --inspecter
  python -m src.prepare.importer_voc --source telechargements/BeltCrack14ks \
         --modele convoyeur --correspondance configs/correspondance_beltcrack.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console  # noqa: E402

configurer_console()
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


def lire_xml(chemin: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Retourne (largeur, hauteur, [(classe, xmin, ymin, xmax, ymax), ...])."""
    racine = ET.parse(chemin).getroot()
    taille = racine.find("size")
    largeur = int(float(taille.find("width").text))
    hauteur = int(float(taille.find("height").text))

    objets = []
    for objet in racine.iter("object"):
        difficile = objet.find("difficult")
        if difficile is not None and difficile.text == "1":
            continue
        nom = objet.find("name").text.strip()
        boite = objet.find("bndbox")
        objets.append((
            nom,
            float(boite.find("xmin").text), float(boite.find("ymin").text),
            float(boite.find("xmax").text), float(boite.find("ymax").text),
        ))
    return largeur, hauteur, objets


def trouver_annotations(source: Path) -> list[Path]:
    """Localise les XML, quelle que soit la profondeur de l'arborescence."""
    return sorted(source.rglob("*.xml"))


def image_associee(chemin_xml: Path, source: Path) -> Path | None:
    """
    Retrouve l'image correspondant a une annotation.

    On tente d'abord le remplacement direct Annotations -> JPEGImages, qui
    est la convention VOC, puis une recherche par nom si l'arborescence
    differe.
    """
    for extension in EXT_IMAGES:
        candidat = Path(str(chemin_xml.parent).replace("Annotations", "JPEGImages"))
        candidat = candidat / (chemin_xml.stem + extension)
        if candidat.exists():
            return candidat
    for extension in EXT_IMAGES:
        trouves = list(source.rglob(chemin_xml.stem + extension))
        if trouves:
            return trouves[0]
    return None


def sequence_de(chemin: Path, source: Path) -> str:
    """
    Identifiant de sequence : le dossier parent de l'image dans BeltCrack,
    ou le prefixe du nom de fichier a defaut. Sert a decouper le dataset
    sans separer deux images consecutives entre train et val.
    """
    relatif = chemin.relative_to(source)
    if len(relatif.parts) > 2:
        return relatif.parts[-2]
    return chemin.stem.rsplit("_", 1)[0]


def inspecter(source: Path) -> None:
    """Affiche les classes du dataset VOC et un modele de correspondance."""
    annotations = trouver_annotations(source)
    if not annotations:
        print(f"Aucun fichier .xml dans {source}")
        print("Verifiez que le dataset est bien au format Pascal VOC "
              "(dossier Annotations/ contenant des .xml).")
        return

    comptes = Counter()
    sequences = set()
    sans_image = 0
    for chemin in annotations:
        try:
            _, _, objets = lire_xml(chemin)
        except ET.ParseError:
            continue
        for nom, *_ in objets:
            comptes[nom] += 1
        image = image_associee(chemin, source)
        if image is None:
            sans_image += 1
        else:
            sequences.add(sequence_de(image, source))

    print(f"Dataset VOC : {source}")
    print(f"  {len(annotations)} annotation(s), {sum(comptes.values())} objet(s)")
    print(f"  {len(sequences)} sequence(s) detectee(s)")
    if sans_image:
        print(f"  [ATTENTION] {sans_image} annotation(s) sans image associee")
    print()
    for nom, compte in comptes.most_common():
        print(f"  {nom:<30} {compte:7d} objets")

    print("\n# --- Modele de correspondance a enregistrer dans")
    print("#     configs/correspondance_<nom>.yaml ---")
    print("correspondance:")
    for nom in sorted(comptes):
        print(f"  {nom}: null")


def importer(source: Path, modele: str, chemin_correspondance: Path,
             prefixe: str, ratios, seed: int, copier: bool) -> None:
    """Convertit un dataset VOC vers data/<modele>/ au format YOLO."""
    correspondance = yaml.safe_load(
        chemin_correspondance.read_text(encoding="utf-8"))["correspondance"]
    cibles = yaml.safe_load(
        (RACINE / "configs" / f"data_{modele}.yaml").read_text(encoding="utf-8"))["names"]
    identifiant_par_nom = {nom: identifiant for identifiant, nom in cibles.items()}

    tache = yaml.safe_load(
        (RACINE / "configs" / f"{modele}.yaml").read_text(encoding="utf-8"))["tache"]
    if tache == "segment":
        print(f"[INCOMPATIBLE] Le modele '{modele}' est configure en segmentation,")
        print("  mais un dataset VOC ne contient que des BOITES, pas des polygones.")
        print(f"  Corrigez configs/{modele}.yaml :")
        print("    tache: detect")
        print("    modele_base: yolo11s.pt")
        print("  Puis relancez. L'import est interrompu.")
        return

    annotations = trouver_annotations(source)
    if not annotations:
        print(f"Aucun .xml dans {source}")
        return

    # Regroupement par sequence, pour un decoupage sans fuite train/val
    par_sequence = defaultdict(list)
    for chemin_xml in annotations:
        image = image_associee(chemin_xml, source)
        if image is not None:
            par_sequence[sequence_de(image, source)].append((chemin_xml, image))

    aleatoire = random.Random(seed)
    sequences = sorted(par_sequence)
    aleatoire.shuffle(sequences)
    n = len(sequences)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n > 2 else 0
    lots = {}
    for index, sequence in enumerate(sequences):
        lots[sequence] = ("train" if index < n_train
                          else "val" if index < n_train + n_val else "test")
    destination = RACINE / "data" / modele
    operation = shutil.copy2 if copier else shutil.move
    statistiques = Counter()
    compte_lot = Counter()
    inconnues = set()
    sans_objet = 0

    for sequence, elements in par_sequence.items():
        lot = lots[sequence]
        for chemin_xml, image in elements:
            try:
                largeur, hauteur, objets = lire_xml(chemin_xml)
            except ET.ParseError:
                continue
            if largeur <= 0 or hauteur <= 0:
                continue

            lignes = []
            for nom, xmin, ymin, xmax, ymax in objets:
                cible = correspondance.get(nom, "__absent__")
                if cible == "__absent__":
                    inconnues.add(nom)
                    continue
                if cible is None:
                    continue
                if cible not in identifiant_par_nom:
                    raise ValueError(
                        f"La correspondance vise '{cible}', absent de "
                        f"configs/data_{modele}.yaml. Classes disponibles : "
                        f"{sorted(identifiant_par_nom)}")

                cx = (xmin + xmax) / 2 / largeur
                cy = (ymin + ymax) / 2 / hauteur
                w = (xmax - xmin) / largeur
                h = (ymax - ymin) / hauteur
                if w <= 0 or h <= 0:
                    continue
                cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                w, h = min(w, 1.0), min(h, 1.0)
                lignes.append(f"{identifiant_par_nom[cible]} "
                              f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                statistiques[cible] += 1

            # Les images sans objet retenu deviennent des images de fond.
            # Elles sont utiles, mais pas en quantite illimitee : au dela de
            # 15 pourcent elles desequilibrent l'entrainement.
            if not lignes:
                sans_objet += 1
                if sans_objet > 0.15 * max(sum(compte_lot.values()), 1) + 30:
                    continue

            nom_final = f"{prefixe}_{sequence}_{image.stem}"
            dossier_img = destination / "images" / lot
            dossier_lbl = destination / "labels" / lot
            dossier_img.mkdir(parents=True, exist_ok=True)
            dossier_lbl.mkdir(parents=True, exist_ok=True)
            operation(str(image), str(dossier_img / f"{nom_final}{image.suffix}"))
            (dossier_lbl / f"{nom_final}.txt").write_text(
                chr(10).join(lignes), encoding="utf-8")
            compte_lot[lot] += 1

    if inconnues:
        print(f"[ATTENTION] Classes absentes de la correspondance, ignorees : "
              f"{sorted(inconnues)}")

    print()
    print(f"{sum(compte_lot.values())} image(s) importee(s) dans {destination}")
    for lot in ("train", "val", "test"):
        print(f"  {lot:<6} {compte_lot[lot]:6d}")
    print(f"  decoupage par sequence : {n} sequence(s). Aucune image d'une meme")
    print("  sequence n'est repartie entre deux lots, ce qui evite de gonfler")
    print("  artificiellement les scores avec des images quasi identiques.")
    print()
    print("Instances par classe du projet :")
    for nom, compte in statistiques.most_common():
        print(f"  {nom:<22} {compte:7d}")
    print()
    print("Controlez le resultat :")
    print(f"  python -m src.prepare.check_dataset --modele {modele}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Import d'un dataset Pascal VOC")
    ap.add_argument("--source", required=True)
    ap.add_argument("--inspecter", action="store_true")
    ap.add_argument("--modele", choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--correspondance")
    ap.add_argument("--prefixe", default="voc")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deplacer", action="store_true",
                    help="Deplacer au lieu de copier (economise l'espace disque)")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = RACINE / args.source
    if not source.exists():
        print(f"Dossier introuvable : {source}")
        return 1

    if args.inspecter:
        inspecter(source)
        return 0
    if not args.modele or not args.correspondance:
        print("--modele et --correspondance sont requis (ou utilisez --inspecter)")
        return 1

    chemin = Path(args.correspondance)
    if not chemin.is_absolute():
        chemin = RACINE / args.correspondance
    importer(source, args.modele, chemin, args.prefixe, args.ratios,
             args.seed, not args.deplacer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
