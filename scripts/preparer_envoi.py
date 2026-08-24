"""
Prépare une archive du projet à envoyer sur une plateforme d'entraînement
(Colab, Kaggle, poste GPU de l'entreprise).

L'archive contient le code, les configurations et les datasets annotés, mais
PAS les vidéos brutes ni les poids : ce sont les fichiers les plus lourds et
ils n'ont rien à faire dans un envoi. Une archive de 8 Go met une heure à
téléverser et fait échouer la session.

  python scripts/preparer_envoi.py
  python scripts/preparer_envoi.py --modele convoyeur   # un seul dataset
  python scripts/preparer_envoi.py --avec-videos        # inclut data/raw
"""
import argparse
import sys
import zipfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import configurer_console  # noqa: E402

configurer_console()

DOSSIERS_CODE = ["configs", "src", "scripts", "docs", "notebooks"]
EXCLUS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}


def a_exclure(chemin: Path) -> bool:
    return any(partie in EXCLUS for partie in chemin.parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive du projet pour entraînement distant")
    ap.add_argument("--modele", choices=["eclairage", "vehicules", "convoyeur"],
                    help="N'inclure que le dataset de ce modèle")
    ap.add_argument("--avec-videos", action="store_true",
                    help="Inclure data/raw (vidéos brutes, souvent très lourd)")
    ap.add_argument("--avec-synthetiques", action="store_true",
                    help="Inclure les images synthétiques (par défaut exclues : "
                         "elles se régénèrent en une minute sur le GPU distant)")
    ap.add_argument("--sortie", default="MODELE_envoi.zip")
    args = ap.parse_args()

    archive = RACINE / args.sortie
    modeles = [args.modele] if args.modele else ["eclairage", "vehicules", "convoyeur"]

    fichiers, images_par_modele, synthetiques = [], {}, {}

    for dossier in DOSSIERS_CODE:
        for chemin in (RACINE / dossier).rglob("*"):
            if chemin.is_file() and not a_exclure(chemin):
                fichiers.append(chemin)

    for nom in ("README.md", "requirements.txt", ".gitignore"):
        if (RACINE / nom).exists():
            fichiers.append(RACINE / nom)

    for modele in modeles:
        compte = 0
        for sous in ("images", "labels"):
            base = RACINE / "data" / modele / sous
            if not base.exists():
                continue
            for chemin in base.rglob("*"):
                if not chemin.is_file() or chemin.name == ".gitkeep":
                    continue
                # Les images synthetiques portent le prefixe 'synth_'. Elles
                # sont exclues par defaut : les regenerer sur le GPU distant
                # prend une minute, les televerser prend une heure. Les images
                # reelles, elles, sont toujours incluses.
                if chemin.name.startswith("synth_") and not args.avec_synthetiques:
                    synthetiques[modele] = synthetiques.get(modele, 0) + (sous == "images")
                    continue
                fichiers.append(chemin)
                compte += sous == "images"
        images_par_modele[modele] = compte

    if args.avec_videos:
        for chemin in (RACINE / "data" / "raw").rglob("*"):
            if chemin.is_file() and chemin.name != ".gitkeep":
                fichiers.append(chemin)

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for chemin in fichiers:
            zf.write(chemin, Path("MODELE") / chemin.relative_to(RACINE))

    taille_mo = archive.stat().st_size / 1e6
    print(f"Archive : {archive}")
    print(f"Taille  : {taille_mo:.1f} Mo ({len(fichiers)} fichiers)\n")

    print("Images d'entraînement incluses :")
    vide = True
    for modele, compte in images_par_modele.items():
        exclues = synthetiques.get(modele, 0)
        note = f"   ({exclues} synthetiques exclues, regenerables)" if exclues else ""
        print(f"  {modele:<12} {compte:5d} image(s) reelle(s){note}")
        vide = vide and compte == 0

    if vide:
        print()
        print("Aucune image REELLE dans l archive. Ce n est pas bloquant :")
        print("les datasets synthetiques se regenerent sur place en une")
        print("minute, avec les commandes prevues dans les notebooks :")
        print("  python scripts/generer_dataset_convoyeur.py --nombre 1200")
        print("  python scripts/generer_dataset_eclairage.py --nombre 800")
        print()
        print("Televerser 1 Go d images regenerables prendrait une heure")
        print("pour rien. En revanche, vos images REELLES d usine doivent")
        print("etre incluses : placez-les dans data/<modele>/images|labels,")
        print("ou joignez les videos avec --avec-videos.")
        return 0

    print("\nEtapes suivantes :")
    print("  Colab  : deposez le zip dans Google Drive, puis ouvrez")
    print("           notebooks/entrainement_colab.ipynb")
    print("  Kaggle : creez un Dataset a partir du zip, puis ouvrez")
    print("           notebooks/entrainement_kaggle.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
