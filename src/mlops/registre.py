"""
Registre des datasets, des expériences et des modèles.

Le problème que cela résout
---------------------------
Au bout de trois semaines d'entraînements, on se retrouve avec
`best.pt`, `best2.pt`, `best_final.pt`, `best_final_v2.pt`, et plus personne
ne sait lequel tourne en production ni sur quelles données il a été
entraîné. Quand une régression apparaît, il devient impossible de savoir ce
qui a changé.

Trois mécanismes le corrigent :

1. EMPREINTE DE DATASET. Une signature calculée sur le contenu des
   annotations et la liste des images. Deux entraînements donnant des
   résultats différents avec la même empreinte viennent d'un changement de
   configuration ; avec des empreintes différentes, c'est la donnée qui a
   changé. Sans cette distinction, on cherche à l'aveugle.

2. REGISTRE D'EXPERIENCES. Chaque entraînement inscrit une ligne :
   empreinte du dataset, configuration, métriques, chemin des poids.

3. PROMOTION EN PRODUCTION. Un modèle validé est copié vers
   `models/<modele>/production.pt`, et le pipeline ne charge que celui-là.
   Le passage en production devient un acte explicite et tracé.

  python -m src.mlops.registre --empreinte convoyeur
  python -m src.mlops.registre --lister
  python -m src.mlops.registre --promouvoir convoyeur --version v2
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console  # noqa: E402

configurer_console()

DOSSIER_MODELES = RACINE / "models"
FICHIER_REGISTRE = DOSSIER_MODELES / "registre.json"
EXT_IMAGES = {".jpg", ".jpeg", ".png", ".bmp"}


# ------------------------------------------------------- empreinte dataset
def empreinte_dataset(modele: str) -> dict:
    """
    Signature reproductible du dataset d'un modèle.

    Le hachage porte sur les NOMS d'images et le CONTENU des annotations,
    pas sur les pixels : recompresser une image ne change pas l'empreinte,
    alors que corriger une annotation la change. C'est bien ce qu'on veut
    suivre — la vérité terrain, pas l'encodage.
    """
    racine = RACINE / "data" / modele
    condensat = hashlib.sha256()
    statistiques = {"images": 0, "instances": 0, "par_lot": {}, "par_classe": {}}

    noms = yaml.safe_load(
        (RACINE / "configs" / f"data_{modele}.yaml").read_text(encoding="utf-8"))["names"]

    for lot in ("train", "val", "test"):
        dossier_images = racine / "images" / lot
        dossier_labels = racine / "labels" / lot
        if not dossier_images.exists():
            continue
        images = sorted(p for p in dossier_images.iterdir()
                        if p.suffix.lower() in EXT_IMAGES)
        statistiques["par_lot"][lot] = len(images)
        statistiques["images"] += len(images)

        for image in images:
            condensat.update(f"{lot}/{image.name}".encode("utf-8"))
            label = dossier_labels / (image.stem + ".txt")
            if not label.exists():
                continue
            contenu = label.read_text(encoding="utf-8").strip()
            condensat.update(contenu.encode("utf-8"))
            for ligne in contenu.splitlines():
                if ligne.strip():
                    classe = int(float(ligne.split()[0]))
                    nom = noms.get(classe, str(classe))
                    statistiques["par_classe"][nom] = \
                        statistiques["par_classe"].get(nom, 0) + 1
                    statistiques["instances"] += 1

    statistiques["empreinte"] = condensat.hexdigest()[:16]
    return statistiques


# ------------------------------------------------------------- persistance
def charger_registre() -> dict:
    if FICHIER_REGISTRE.exists():
        return json.loads(FICHIER_REGISTRE.read_text(encoding="utf-8"))
    return {"experiences": [], "production": {}}


def sauver_registre(registre: dict) -> None:
    DOSSIER_MODELES.mkdir(parents=True, exist_ok=True)
    FICHIER_REGISTRE.write_text(
        json.dumps(registre, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------ expériences
def enregistrer_experience(modele: str, arguments: dict, metriques: dict,
                           chemin_poids) -> dict:
    """Inscrit un entraînement terminé dans le registre et retourne sa fiche."""
    registre = charger_registre()
    dataset = empreinte_dataset(modele)

    versions = [e["version"] for e in registre["experiences"] if e["modele"] == modele]
    numero = max((int(v.lstrip("v")) for v in versions), default=0) + 1

    # On ne garde que les paramètres qui influencent réellement le résultat
    interessants = ("epochs", "imgsz", "batch", "lr0", "optimizer", "patience",
                    "mosaic", "hsv_v", "copy_paste", "degrees", "scale")
    fiche = {
        "modele": modele,
        "version": f"v{numero}",
        "date": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "empreinte": dataset["empreinte"],
            "images": dataset["images"],
            "instances": dataset["instances"],
            "par_lot": dataset["par_lot"],
            "par_classe": dataset["par_classe"],
        },
        "modele_base": arguments.get("model", arguments.get("modele_base", "")),
        "hyperparametres": {c: arguments[c] for c in interessants if c in arguments},
        "metriques": {c: round(float(v), 4) for c, v in metriques.items()
                      if isinstance(v, (int, float))},
        "poids": str(Path(chemin_poids).relative_to(RACINE))
        if Path(chemin_poids).is_relative_to(RACINE) else str(chemin_poids),
    }
    registre["experiences"].append(fiche)
    sauver_registre(registre)
    return fiche


def lister(modele: str | None = None) -> None:
    """Affiche l'historique des entraînements, le plus récent en dernier."""
    registre = charger_registre()
    experiences = [e for e in registre["experiences"]
                   if modele is None or e["modele"] == modele]
    if not experiences:
        print("Aucune expérience enregistrée. Lancez un entraînement "
              "(python -m src.train.train --modele <nom>).")
        return

    print(f"{'modele':<11} {'ver':<5} {'date':<17} {'dataset':<17} "
          f"{'imgs':>5} {'mAP50':>7} {'mAP50-95':>9}  prod")
    print("-" * 88)
    for e in experiences:
        production = registre["production"].get(e["modele"], {})
        marque = " <=" if production.get("version") == e["version"] else ""
        metriques = e["metriques"]
        map50 = metriques.get("metrics/mAP50(B)", metriques.get("mAP50", 0))
        map5095 = metriques.get("metrics/mAP50-95(B)", metriques.get("mAP50-95", 0))
        print(f"{e['modele']:<11} {e['version']:<5} {e['date'][:16]:<17} "
              f"{e['dataset']['empreinte']:<17} {e['dataset']['images']:>5} "
              f"{map50:>7.3f} {map5095:>9.3f}{marque}")

    print("\nLes lignes marquées <= sont en production.")
    empreintes = {e["dataset"]["empreinte"] for e in experiences}
    if len(empreintes) > 1:
        print(f"{len(empreintes)} versions de dataset differentes : comparez les "
              f"metriques a empreinte EGALE, sinon vous comparez deux choses.")


def promouvoir(modele: str, version: str) -> int:
    """Copie les poids d'une expérience vers models/<modele>/production.pt."""
    registre = charger_registre()
    fiche = next((e for e in registre["experiences"]
                  if e["modele"] == modele and e["version"] == version), None)
    if fiche is None:
        print(f"Expérience introuvable : {modele} {version}")
        return 1

    source = RACINE / fiche["poids"]
    if not source.exists():
        print(f"Poids absents : {source}")
        return 1

    coherent, message = verifier_coherence(modele, source)
    if not coherent:
        print(f"[REFUS] Ce modele ne correspond pas aux classes declarees dans")
        print(f"        configs/data_{modele}.yaml : {message}")
        print("        Reentrainez-le sur la configuration actuelle avant de")
        print("        le promouvoir. Un modele decale ne plante pas : il")
        print("        renvoie de mauvais noms de defauts, ce qui est pire.")
        return 1

    destination = DOSSIER_MODELES / modele / "production.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    registre["production"][modele] = {
        "version": version,
        "promu_le": datetime.now().isoformat(timespec="seconds"),
        "empreinte_dataset": fiche["dataset"]["empreinte"],
        "metriques": fiche["metriques"],
        "poids": str(destination.relative_to(RACINE)),
    }
    sauver_registre(registre)
    print(f"{modele} {version} promu en production -> {destination}")
    print("Le pipeline chargera automatiquement ces poids au prochain démarrage.")
    return 0


def enregistrer_evaluation(modele: str, chemin_poids, lot: str,
                           metriques: dict, par_classe: dict | None = None) -> None:
    """
    Rattache le résultat d'une évaluation à l'expérience qui a produit ces
    poids. Les métriques de validation servent à choisir le meilleur epoch ;
    seules celles du lot de test sont présentables comme performance réelle,
    et il faut donc les distinguer explicitement.
    """
    registre = charger_registre()
    cible = str(Path(chemin_poids))
    fiche = None
    for experience in reversed(registre["experiences"]):
        if experience["modele"] != modele:
            continue
        if Path(RACINE / experience["poids"]).resolve() == Path(cible).resolve():
            fiche = experience
            break
    if fiche is None:
        return

    fiche.setdefault("evaluations", {})[lot] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "global": {c: round(float(v), 4) for c, v in metriques.items()},
        "par_classe": {c: {k: round(float(x), 4) for k, x in v.items()}
                       for c, v in (par_classe or {}).items()},
    }
    sauver_registre(registre)


def verifier_coherence(modele: str, chemin_poids) -> tuple[bool, str]:
    """
    Verifie qu'un modele entraine correspond encore aux classes declarees.

    Le cas se produit des qu'on ajoute une classe au projet : le modele
    deja promu continue de se charger, mais ses sorties sont decalees.
    Un modele a 6 classes charge avec une configuration a 10 classes ne
    plante pas - il renvoie simplement de MAUVAIS noms de defauts, ce qui
    est bien pire qu'une erreur franche.

    Retourne (coherent, message).
    """
    try:
        from ultralytics import YOLO
        modele_charge = YOLO(str(chemin_poids))
        classes_modele = set(modele_charge.names.values())
    except Exception as erreur:
        return False, f"poids illisibles : {erreur}"

    chemin_data = RACINE / "configs" / f"data_{modele}.yaml"
    classes_config = set(
        yaml.safe_load(chemin_data.read_text(encoding="utf-8"))["names"].values())

    if classes_modele == classes_config:
        return True, "coherent"

    manquantes = sorted(classes_config - classes_modele)
    en_trop = sorted(classes_modele - classes_config)
    details = []
    if manquantes:
        details.append(f"classes absentes du modele : {manquantes}")
    if en_trop:
        details.append(f"classes inconnues de la configuration : {en_trop}")
    return False, "; ".join(details)


def deposer_production(modele: str) -> None:
    """Retire un modele de la production (sans effacer son experience)."""
    registre = charger_registre()
    chemin = DOSSIER_MODELES / modele / "production.pt"
    if chemin.exists():
        chemin.unlink()
    registre["production"].pop(modele, None)
    sauver_registre(registre)


def poids_production(modele: str) -> Path | None:
    """Chemin des poids de production d'un modèle, ou None s'il n'y en a pas."""
    chemin = DOSSIER_MODELES / modele / "production.pt"
    return chemin if chemin.exists() else None


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Registre datasets / expériences / modèles")
    ap.add_argument("--empreinte", metavar="MODELE",
                    choices=["eclairage", "vehicules", "convoyeur"],
                    help="Affiche l'empreinte et la composition du dataset")
    ap.add_argument("--lister", action="store_true", help="Historique des entraînements")
    ap.add_argument("--modele", choices=["eclairage", "vehicules", "convoyeur"],
                    help="Filtre pour --lister, ou modèle à promouvoir")
    ap.add_argument("--promouvoir", metavar="MODELE",
                    choices=["eclairage", "vehicules", "convoyeur"])
    ap.add_argument("--version", help="Version à promouvoir (ex. v2)")
    args = ap.parse_args()

    if args.empreinte:
        statistiques = empreinte_dataset(args.empreinte)
        print(f"Dataset '{args.empreinte}'")
        print(f"  empreinte : {statistiques['empreinte']}")
        print(f"  images    : {statistiques['images']}  {statistiques['par_lot']}")
        print(f"  instances : {statistiques['instances']}")
        for nom, compte in sorted(statistiques["par_classe"].items(),
                                  key=lambda kv: -kv[1]):
            print(f"      {nom:<20} {compte:6d}")
        if statistiques["images"] == 0:
            print("\n  Dataset vide : rien a entrainer pour l'instant.")
        return 0

    if args.promouvoir:
        if not args.version:
            print("--version est requis avec --promouvoir")
            return 1
        return promouvoir(args.promouvoir, args.version)

    if args.lister:
        lister(args.modele)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
