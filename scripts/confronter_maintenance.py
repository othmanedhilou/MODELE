"""
Confronte les alarmes du systeme au registre de maintenance de l'usine.

Le seul signal de retour qui existe reellement
----------------------------------------------
On aimerait que le modele apprenne seul, par renforcement. C'est
impossible, et pas faute d'outils : l'apprentissage par renforcement exige
une RECOMPENSE, c'est-a-dire un retour de l'environnement disant si
l'action etait bonne. Or la bande ne dit rien. Quand le modele annonce une
dechirure, rien dans l'image suivante ne confirme ni n'infirme.

Sauf un endroit : le REGISTRE DE MAINTENANCE. Si une bande a ete reparee
le 12 mars, alors une alarme du 11 mars etait probablement vraie, et une
semaine sans intervention apres une alarme la rend douteuse. C'est un
signal faible, differe et imparfait - mais c'est le seul qui soit produit
automatiquement par l'usine, sans travail d'annotation supplementaire.

Ce script en tire trois choses que rien d'autre ne donne :

  1. une estimation de la PRECISION reelle en production, sur vos
     installations et non sur un lot de test ;
  2. les DETECTIONS MANQUEES : une reparation sans alarme prealable est un
     defaut que le systeme n'a pas vu. C'est la seule facon de mesurer le
     rappel en exploitation ;
  3. un lot d'images a reannoter en priorite, deja triees en confirmees et
     douteuses.

Format attendu du registre (CSV, separateur virgule ou point-virgule) :

    date,equipement,type,description
    2026-03-12,convoyeur_01,reparation,Reparation dechirure longitudinale
    2026-03-20,convoyeur_01,inspection,Controle mensuel RAS

  python scripts/confronter_maintenance.py --maintenance maintenance.csv
  python scripts/confronter_maintenance.py --maintenance maintenance.csv \
         --fenetre-avant 7 --exporter data/a_annoter_confirmees
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2] if False else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console  # noqa: E402

configurer_console()

# Mots indiquant une intervention CORRECTIVE, donc un defaut avere.
# Une inspection de routine ne confirme rien.
MOTS_REPARATION = ("reparation", "réparation", "remplacement", "changement",
                   "agrafage", "jonction", "vulcanisation", "rustine",
                   "arret", "arrêt", "incident", "casse", "rupture",
                   "dechirure", "déchirure", "fissure", "trou")


def lire_maintenance(chemin):
    """Lit le registre de maintenance, en acceptant , ou ; comme separateur."""
    texte = Path(chemin).read_text(encoding="utf-8-sig")
    separateur = ";" if texte.count(";") > texte.count(",") else ","
    interventions = []
    for ligne in csv.DictReader(texte.splitlines(), delimiter=separateur):
        ligne = {(c or "").strip().lower(): (v or "").strip()
                 for c, v in ligne.items()}
        brut = ligne.get("date", "")
        date = None
        for format_date in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                date = datetime.strptime(brut[:10], format_date)
                break
            except ValueError:
                continue
        if date is None:
            continue
        description = " ".join([ligne.get("type", ""), ligne.get("description", "")])
        interventions.append({
            "date": date,
            "equipement": ligne.get("equipement", ligne.get("équipement", "")),
            "description": description.strip(),
            "corrective": any(mot in description.lower() for mot in MOTS_REPARATION),
        })
    return sorted(interventions, key=lambda i: i["date"])


def lire_alarmes(chemin):
    """Charge le journal d'alarmes du pipeline."""
    if not Path(chemin).exists():
        return []
    alarmes = []
    for ligne in Path(chemin).read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            entree = json.loads(ligne)
            entree["_date"] = datetime.fromisoformat(entree["horodatage"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        alarmes.append(entree)
    return sorted(alarmes, key=lambda a: a["_date"])


def confronter(alarmes, interventions, fenetre_avant, fenetre_apres):
    """
    Rapproche alarmes et interventions correctives, UN POUR UN.

    L'appariement un pour un est essentiel. Si une meme reparation pouvait
    confirmer toutes les alarmes de la semaine qui la precede, la precision
    afficherait 100 pour cent des que les interventions sont frequentes -
    ce qui reviendrait a mesurer la densite du planning de maintenance,
    pas la qualite du modele.

    Chaque intervention corrective ne confirme donc qu'UNE alarme : la plus
    proche dans le temps qui ne soit pas deja appariee.

    Une alarme sans intervention est DOUTEUSE. Une intervention sans alarme
    est une DETECTION MANQUEE. Le rapprochement reste temporel, donc
    imparfait : deux defauts proches peuvent etre confondus, et une
    reparation planifiee de longue date peut suivre une alarme sans lien
    avec elle. A lire comme une tendance, pas comme une verite terrain.
    """
    correctives = [i for i in interventions if i["corrective"]]
    alarmes_appariees = {}
    manquees = []

    for intervention in correctives:
        candidates = [
            (intervention["date"] - alarme["_date"]).days
            for alarme in alarmes
        ]
        meilleure, ecart_min = None, None
        for index, alarme in enumerate(alarmes):
            if index in alarmes_appariees:
                continue
            ecart = (intervention["date"] - alarme["_date"]).days
            if 0 <= ecart <= fenetre_avant and (ecart_min is None or ecart < ecart_min):
                meilleure, ecart_min = index, ecart
        if meilleure is not None:
            alarmes_appariees[meilleure] = intervention
        else:
            # Aucune alarme libre : y en avait-il une, meme deja appariee ?
            couverte = any(0 <= (intervention["date"] - a["_date"]).days <= fenetre_apres
                           for a in alarmes)
            if not couverte:
                manquees.append(intervention)

    confirmees, douteuses = [], []
    for index, alarme in enumerate(alarmes):
        if index in alarmes_appariees:
            alarme["_intervention"] = alarmes_appariees[index]
            confirmees.append(alarme)
        else:
            douteuses.append(alarme)

    return confirmees, douteuses, manquees


def main():
    ap = argparse.ArgumentParser(
        description="Confrontation alarmes / registre de maintenance")
    ap.add_argument("--maintenance", required=True, help="Fichier CSV du registre")
    ap.add_argument("--journal", help="Journal d'alarmes (defaut : configs/pipeline.yaml)")
    ap.add_argument("--fenetre-avant", type=int, default=7,
                    help="Jours entre une alarme et l'intervention qui la confirme")
    ap.add_argument("--fenetre-apres", type=int, default=14,
                    help="Anteriorite max d'une alarme pour couvrir une intervention")
    ap.add_argument("--exporter", help="Dossier ou copier les captures a reannoter")
    args = ap.parse_args()

    chemin_maintenance = Path(args.maintenance)
    if not chemin_maintenance.is_absolute():
        chemin_maintenance = RACINE / args.maintenance
    if not chemin_maintenance.exists():
        print(f"Registre introuvable : {chemin_maintenance}")
        print()
        print("Demandez au service maintenance un export de ses interventions")
        print("sur les convoyeurs. Quatre colonnes suffisent :")
        print("  date,equipement,type,description")
        print()
        print("C'est le seul signal de retour que l'usine produit toute seule.")
        return 1

    journal = args.journal or charger_config("configs/pipeline.yaml")["sorties"]["journal"]
    chemin_journal = Path(journal)
    if not chemin_journal.is_absolute():
        chemin_journal = RACINE / journal

    interventions = lire_maintenance(chemin_maintenance)
    alarmes = lire_alarmes(chemin_journal)
    correctives = [i for i in interventions if i["corrective"]]

    print(f"Registre : {len(interventions)} intervention(s), "
          f"dont {len(correctives)} corrective(s)")
    print(f"Journal  : {len(alarmes)} alarme(s)")
    if not alarmes:
        print()
        print("Aucune alarme enregistree : faites tourner le pipeline avant")
        print("de confronter. python -m src.pipeline.run_stream --camera <nom>")
        return 1
    if not correctives:
        print()
        print("Aucune intervention CORRECTIVE reconnue dans le registre.")
        print("Le script repere les interventions par mots-cles (reparation,")
        print("agrafage, vulcanisation, rupture...). Verifiez la colonne")
        print("'description', ou completez MOTS_REPARATION dans ce script.")
        return 1

    confirmees, douteuses, manquees = confronter(
        alarmes, interventions, args.fenetre_avant, args.fenetre_apres)

    total = len(confirmees) + len(douteuses)
    precision = len(confirmees) / max(total, 1)
    rappel = 1 - len(manquees) / max(len(correctives), 1)

    print()
    print("=" * 62)
    print("ESTIMATION DES PERFORMANCES EN EXPLOITATION")
    print(f"  alarmes confirmees par une intervention : {len(confirmees):4d}")
    print(f"  alarmes sans suite                      : {len(douteuses):4d}")
    print(f"  interventions sans alarme prealable     : {len(manquees):4d}")
    print()
    print(f"  precision estimee : {precision:5.1%}  "
          f"(part des alarmes suivies d'une reparation)")
    print(f"  rappel estime     : {rappel:5.1%}  "
          f"(part des reparations precedees d'une alarme)")
    print()
    print("  Ces chiffres sont des ESTIMATIONS obtenues par rapprochement")
    print("  temporel, pas une verite terrain annotee. Lisez-les comme une")
    print("  tendance : un rappel qui chute d'un mois sur l'autre est un")
    print("  signal fiable, sa valeur absolue l'est moins.")

    if manquees:
        print()
        print("DETECTIONS MANQUEES - les plus instructives du lot :")
        for intervention in manquees[:8]:
            print(f"  {intervention['date']:%Y-%m-%d}  {intervention['equipement']:<16} "
                  f"{intervention['description'][:42]}")
        print()
        print("  Chacune est un defaut reel que le systeme n'a pas vu.")
        print("  Recuperez les enregistrements des jours precedents : ce sont")
        print("  les images les plus precieuses du projet.")

    par_type = Counter(a.get("type", "?") for a in confirmees)
    if par_type:
        print()
        print("Types d'alarme les plus souvent confirmes :")
        for nom, compte in par_type.most_common(6):
            print(f"  {nom:<28} {compte:4d}")

    if args.exporter:
        destination = Path(args.exporter)
        if not destination.is_absolute():
            destination = RACINE / args.exporter
        for lot, alarmes_lot in (("confirmees", confirmees), ("douteuses", douteuses)):
            dossier = destination / lot
            if dossier.exists():
                shutil.rmtree(dossier)
            dossier.mkdir(parents=True, exist_ok=True)
            copiees = 0
            for alarme in alarmes_lot:
                capture = alarme.get("snapshot")
                if capture and (RACINE / capture).exists():
                    shutil.copy2(RACINE / capture, dossier / Path(capture).name)
                    copiees += 1
            print(f"\n{copiees} capture(s) copiee(s) dans {dossier}")
        print()
        print("Annotez d'abord le lot 'confirmees' : ce sont des defauts reels,")
        print("donc de la verite terrain presque gratuite. Le lot 'douteuses'")
        print("contient vos faux positifs, tout aussi utiles a corriger.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
