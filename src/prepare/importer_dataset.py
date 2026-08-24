"""
Importe un dataset public au format YOLO et convertit ses classes vers le
vocabulaire du projet.

Pourquoi ce script
------------------
Aucun dataset public n'a exactement vos classes. Un export Roboflow
« construction vehicles » contiendra `dump truck`, `excavator`, `wheel
loader` ; un export trafic contiendra `car`, `truck`, `bus`. Renuméroter
des milliers de fichiers `.txt` à la main est impraticable et source
d'erreurs silencieuses : une classe mal mappée donne un modèle qui confond
systématiquement deux engins, sans qu'aucune métrique ne le signale.

Ce script lit le `data.yaml` du dataset importé, affiche ses classes, et
applique une table de correspondance vers `configs/data_<modele>.yaml`. Les
classes non mappées sont ignorées (leurs annotations sont supprimées), ce
qui est le comportement voulu : mieux vaut ne pas apprendre une classe que
l'apprendre sous une mauvaise étiquette.

Utilisation
-----------
1. Inspecter les classes du dataset téléchargé :
     python -m src.prepare.importer_dataset --source telechargements/vehicules --inspecter

2. Créer le fichier de correspondance, puis importer :
     python -m src.prepare.importer_dataset --source telechargements/vehicules \
            --modele vehicules --correspondance configs/correspondance_vehicules.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console  # noqa: E402

configurer_console()
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


def trouver_data_yaml(source: Path) -> Path | None:
    """Localise le data.yaml d'un export Roboflow / Ultralytics."""
    for motif in ("data.yaml", "data.yml", "dataset.yaml"):
        trouves = list(source.rglob(motif))
        if trouves:
            return trouves[0]
    return None


def classes_source(source: Path) -> dict[int, str]:
    """Classes du dataset importé, depuis son data.yaml."""
    chemin = trouver_data_yaml(source)
    if chemin is None:
        raise FileNotFoundError(
            f"Aucun data.yaml dans {source}. Le dataset doit être exporté au "
            f"format « YOLOv8 / YOLOv11 » et non COCO ou Pascal VOC.")
    donnees = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    noms = donnees.get("names", {})
    if isinstance(noms, list):
        noms = dict(enumerate(noms))
    return {int(k): v for k, v in noms.items()}


def inspecter(source: Path) -> None:
    """Affiche les classes du dataset et un modèle de correspondance à remplir."""
    noms = classes_source(source)
    comptes = Counter()
    for label in source.rglob("*.txt"):
        if label.name in {"classes.txt", "requirements.txt"}:
            continue
        for ligne in label.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                comptes[int(float(ligne.split()[0]))] += 1

    print(f"Dataset : {source}")
    print(f"{len(noms)} classe(s), {sum(comptes.values())} instance(s)\n")
    for identifiant, nom in sorted(noms.items()):
        print(f"  {identifiant:3d}  {nom:<30} {comptes.get(identifiant, 0):7d} instances")

    print("\n# --- Modèle de correspondance à enregistrer dans")
    print("#     configs/correspondance_<modele>.yaml ---")
    print("# Mettez `null` pour ignorer une classe (ses annotations seront")
    print("# supprimées), sinon le nom exact d'une classe de votre projet.")
    print("correspondance:")
    for identifiant, nom in sorted(noms.items()):
        print(f"  {nom}: null")


def importer(source: Path, modele: str, chemin_correspondance: Path,
             prefixe: str, copier: bool) -> None:
    """Convertit et copie le dataset importé dans data/<modele>/."""
    noms_source = classes_source(source)
    correspondance = yaml.safe_load(
        chemin_correspondance.read_text(encoding="utf-8"))["correspondance"]

    cibles = yaml.safe_load(
        (RACINE / "configs" / f"data_{modele}.yaml").read_text(encoding="utf-8"))["names"]
    identifiant_par_nom = {nom: identifiant for identifiant, nom in cibles.items()}

    # Table finale : identifiant source -> identifiant cible (ou None = ignorer)
    table, inconnues = {}, []
    for identifiant, nom in noms_source.items():
        cible = correspondance.get(nom, "__absent__")
        if cible == "__absent__":
            inconnues.append(nom)
            table[identifiant] = None
        elif cible is None:
            table[identifiant] = None
        elif cible in identifiant_par_nom:
            table[identifiant] = identifiant_par_nom[cible]
        else:
            raise ValueError(
                f"La correspondance vise '{cible}', qui n'existe pas dans "
                f"configs/data_{modele}.yaml. Classes disponibles : "
                f"{sorted(identifiant_par_nom)}")

    if inconnues:
        print(f"[ATTENTION] Classes absentes du fichier de correspondance, "
              f"donc ignorées : {inconnues}")

    # Le format d'annotation du dataset importé doit correspondre à la tâche
    # du modèle. Un dataset en boîtes ne peut pas entraîner un modèle de
    # segmentation : autant le dire ici plutôt qu'après une heure de copie.
    tache = yaml.safe_load(
        (RACINE / "configs" / f"{modele}.yaml").read_text(encoding="utf-8"))["tache"]
    exemple = next((l for l in source.rglob("*.txt")
                    if l.name not in {"classes.txt", "requirements.txt"}
                    and l.read_text(encoding="utf-8").strip()), None)
    if exemple is not None:
        premiere = exemple.read_text(encoding="utf-8").splitlines()[0]
        nb_valeurs = len(premiere.split()) - 1
        format_source = "segmentation" if nb_valeurs >= 6 else "boites"
        format_attendu = "segmentation" if tache == "segment" else "boites"
        if format_source != format_attendu:
            print()
            print(f"[INCOMPATIBLE] Le dataset est en {format_source}, alors "
                  f"que le modele '{modele}' est configure en {format_attendu}.")
            if format_source == "boites":
                print(f"  Corrigez configs/{modele}.yaml :")
                print(f"    tache: detect")
                print(f"    modele_base: yolo11s.pt")
                print("  ... ou cherchez un dataset en segmentation.")
            else:
                print(f"  Les polygones seront conserves ; passez "
                      f"configs/{modele}.yaml en tache: segment.")
            print()

    destination = RACINE / "data" / modele
    operation = shutil.copy2 if copier else shutil.move
    statistiques = Counter()
    images_traitees = images_vides = 0

    for image in sorted(p for p in source.rglob("*") if p.suffix.lower() in EXT_IMAGES):
        # Le lot d'origine (train/valid/test) est conservé s'il est lisible
        parties = {p.lower() for p in image.parts}
        lot = "val" if parties & {"valid", "val"} else \
              "test" if "test" in parties else "train"

        label = image.parent.parent / "labels" / (image.stem + ".txt")
        if not label.exists():
            label = image.with_suffix(".txt")

        lignes_converties = []
        if label.exists():
            for ligne in label.read_text(encoding="utf-8").splitlines():
                ligne = ligne.strip()
                if not ligne:
                    continue
                champs = ligne.split()
                nouvelle = table.get(int(float(champs[0])))
                if nouvelle is None:
                    continue
                lignes_converties.append(" ".join([str(nouvelle)] + champs[1:]))
                statistiques[cibles[nouvelle]] += 1

        # Une image dont toutes les annotations ont été supprimées devient une
        # image de fond : c'est utile, mais pas en quantité illimitée.
        if not lignes_converties:
            images_vides += 1
            if images_vides > 0.10 * max(images_traitees, 1) + 20:
                continue

        nom_final = f"{prefixe}_{image.stem}{image.suffix}"
        dossier_img = destination / "images" / lot
        dossier_lbl = destination / "labels" / lot
        dossier_img.mkdir(parents=True, exist_ok=True)
        dossier_lbl.mkdir(parents=True, exist_ok=True)

        operation(str(image), str(dossier_img / nom_final))
        (dossier_lbl / f"{prefixe}_{image.stem}.txt").write_text(
            "\n".join(lignes_converties), encoding="utf-8")
        images_traitees += 1

    print(f"\n{images_traitees} image(s) importée(s) dans {destination}")
    print("Instances par classe du projet :")
    for nom, compte in sorted(statistiques.items(), key=lambda kv: -kv[1]):
        print(f"  {nom:<22} {compte:7d}")
    print(f"\nContrôlez le résultat :\n"
          f"  python -m src.prepare.check_dataset --modele {modele}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Import d'un dataset public YOLO")
    ap.add_argument("--source", required=True, help="Dossier du dataset téléchargé")
    ap.add_argument("--inspecter", action="store_true",
                    help="Affiche les classes et un modèle de correspondance")
    ap.add_argument("--modele", choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--correspondance", help="YAML de correspondance des classes")
    ap.add_argument("--prefixe", default="pub",
                    help="Préfixe des fichiers importés, pour les distinguer "
                         "de vos images d'usine")
    ap.add_argument("--copier", action="store_true", default=True,
                    help="Copier plutôt que déplacer (défaut)")
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
    importer(source, args.modele, chemin, args.prefixe, args.copier)
    return 0


if __name__ == "__main__":
    sys.exit(main())
