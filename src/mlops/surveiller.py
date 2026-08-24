"""
Surveillance du système en production, à partir du journal d'alarmes.

Ce que cela detecte, et pourquoi c'est indispensable
----------------------------------------------------
Un modele deploye ne previent pas quand il se degrade. Il continue de
tourner et de produire des sorties d'apparence normale. Les trois modes de
defaillance reels sont :

1. DERIVE DU TAUX D'ALARME. Le nombre d'alarmes par heure change nettement
   sans qu'aucune cause metier ne l'explique. Une camera a bouge, un
   projecteur a ete ajoute, la bande a ete remplacee par un modele plus
   clair : les conditions ne sont plus celles de l'entrainement.

2. SILENCE. Une camera n'a plus emis la moindre alarme depuis plusieurs
   jours. C'est presque toujours une panne de flux, pas une usine devenue
   parfaite. C'est le mode de defaillance le plus dangereux, parce qu'il
   ressemble au bon fonctionnement.

3. RAFALE. Des centaines d'alarmes en quelques minutes : l'operateur cesse
   de les lire et finit par desactiver le systeme. Une rafale doit remonter
   comme une alarme sur le systeme lui-meme.

  python -m src.mlops.surveiller
  python -m src.mlops.surveiller --jours 7 --exporter runs/rapport_surveillance.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console  # noqa: E402

configurer_console()


def charger_journal(chemin: Path, depuis: datetime | None = None) -> list[dict]:
    """Lit le journal JSONL en ignorant les lignes corrompues."""
    if not chemin.exists():
        return []
    entrees = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        try:
            entree = json.loads(ligne)
            entree["_date"] = datetime.fromisoformat(entree["horodatage"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
        if depuis is None or entree["_date"] >= depuis:
            entrees.append(entree)
    return entrees


def taux_par_heure(entrees: list[dict]) -> dict[str, float]:
    """Alarmes par heure et par caméra sur la période couverte."""
    if not entrees:
        return {}
    par_camera = defaultdict(list)
    for entree in entrees:
        par_camera[entree["camera"]].append(entree["_date"])
    taux = {}
    for camera, dates in par_camera.items():
        duree_h = max((max(dates) - min(dates)).total_seconds() / 3600, 1.0)
        taux[camera] = len(dates) / duree_h
    return taux


def detecter_rafales(entrees: list[dict], seuil: int, fenetre_min: int) -> list[dict]:
    """Repère les périodes où le nombre d'alarmes explose sur une courte durée."""
    rafales = []
    par_camera = defaultdict(list)
    for entree in entrees:
        par_camera[entree["camera"]].append(entree)

    for camera, liste in par_camera.items():
        liste.sort(key=lambda e: e["_date"])
        debut = 0
        for fin in range(len(liste)):
            while (liste[fin]["_date"] - liste[debut]["_date"]) > timedelta(minutes=fenetre_min):
                debut += 1
            nombre = fin - debut + 1
            if nombre >= seuil:
                if rafales and rafales[-1]["camera"] == camera and \
                        liste[fin]["_date"] - rafales[-1]["fin"] < timedelta(minutes=fenetre_min):
                    rafales[-1]["fin"] = liste[fin]["_date"]
                    rafales[-1]["nombre"] = max(rafales[-1]["nombre"], nombre)
                else:
                    rafales.append({"camera": camera, "debut": liste[debut]["_date"],
                                    "fin": liste[fin]["_date"], "nombre": nombre})
    return rafales


def rapport(jours: int, seuil_derive: float, seuil_rafale: int,
            fenetre_rafale: int, silence_jours: float,
            exporter: str | None) -> int:
    pipeline = charger_config("configs/pipeline.yaml")
    chemin = RACINE / pipeline["sorties"]["journal"]

    maintenant = datetime.now()
    recentes = charger_journal(chemin, maintenant - timedelta(days=jours))
    reference = charger_journal(chemin, maintenant - timedelta(days=jours * 4))
    reference = [e for e in reference if e["_date"] < maintenant - timedelta(days=jours)]

    print(f"=== Surveillance du systeme ===")
    print(f"Journal   : {chemin}")
    print(f"Periode   : {jours} derniers jours")
    print(f"Alarmes   : {len(recentes)} (reference anterieure : {len(reference)})\n")

    if not recentes and not reference:
        print("Journal vide : le systeme n'a encore rien enregistre.")
        print("Lancez le pipeline (python -m src.pipeline.run_stream --camera <nom>).")
        return 0

    anomalies = []

    # --- Repartition ---
    print("--- Repartition des alarmes ---")
    for intitule, cle in (("Par camera", "camera"), ("Par modele", "modele"),
                          ("Par type", "type"), ("Par gravite", "gravite")):
        comptes = Counter(e.get(cle, "?") for e in recentes)
        print(f"{intitule} :")
        for valeur, compte in comptes.most_common():
            part = 100 * compte / max(len(recentes), 1)
            print(f"    {valeur:<28} {compte:6d}  ({part:5.1f} %)")
    print()

    # --- Derive du taux d'alarme ---
    taux_recent = taux_par_heure(recentes)
    taux_reference = taux_par_heure(reference)
    print("--- Taux d'alarme par heure ---")
    for camera in sorted(set(taux_recent) | set(taux_reference)):
        actuel = taux_recent.get(camera, 0.0)
        avant = taux_reference.get(camera)
        if avant is None:
            print(f"  {camera:<22} {actuel:6.2f}/h   (pas de reference)")
            continue
        variation = (actuel - avant) / max(avant, 1e-6)
        marque = ""
        if abs(variation) >= seuil_derive:
            marque = "  <== DERIVE"
            anomalies.append(
                f"Derive sur '{camera}' : {avant:.2f}/h -> {actuel:.2f}/h "
                f"({variation:+.0%})")
        print(f"  {camera:<22} {actuel:6.2f}/h   avant {avant:6.2f}/h "
              f"({variation:+6.0%}){marque}")
    print()

    # --- Silence ---
    derniere_vue = {}
    for entree in charger_journal(chemin):
        camera = entree["camera"]
        if camera not in derniere_vue or entree["_date"] > derniere_vue[camera]:
            derniere_vue[camera] = entree["_date"]

    cameras_config = {c["nom"] for c in pipeline.get("cameras", [])}
    print("--- Derniere alarme par camera ---")
    for camera in sorted(cameras_config | set(derniere_vue)):
        date = derniere_vue.get(camera)
        if date is None:
            print(f"  {camera:<22} jamais")
            anomalies.append(f"'{camera}' n'a jamais emis d'alarme : flux "
                             f"probablement inaccessible")
            continue
        anciennete = (maintenant - date).total_seconds() / 86400
        marque = ""
        if anciennete > silence_jours:
            marque = "  <== SILENCE"
            anomalies.append(f"'{camera}' muette depuis {anciennete:.1f} jours : "
                             f"verifiez le flux avant de conclure a une "
                             f"absence d'incident")
        print(f"  {camera:<22} {date:%Y-%m-%d %H:%M}  "
              f"(il y a {anciennete:5.1f} j){marque}")
    print()

    # --- Rafales ---
    rafales = detecter_rafales(recentes, seuil_rafale, fenetre_rafale)
    if rafales:
        print("--- Rafales d'alarmes ---")
        for rafale in rafales:
            print(f"  {rafale['camera']:<22} {rafale['nombre']:4d} alarmes en "
                  f"{fenetre_rafale} min a partir de {rafale['debut']:%Y-%m-%d %H:%M}")
            anomalies.append(
                f"Rafale sur '{rafale['camera']}' : {rafale['nombre']} alarmes en "
                f"{fenetre_rafale} min, le {rafale['debut']:%Y-%m-%d %H:%M}")
        print()

    if exporter:
        import csv
        chemin_csv = Path(exporter)
        if not chemin_csv.is_absolute():
            chemin_csv = RACINE / exporter
        chemin_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
            ecrivain = csv.writer(f)
            ecrivain.writerow(["horodatage", "camera", "modele", "type", "gravite"])
            for e in recentes:
                ecrivain.writerow([e["horodatage"], e["camera"], e["modele"],
                                   e["type"], e["gravite"]])
        print(f"Export CSV : {chemin_csv}\n")

    print("=" * 60)
    if anomalies:
        print(f"{len(anomalies)} POINT(S) A VERIFIER :")
        for anomalie in anomalies:
            print(f"  - {anomalie}")
        return 1
    print("Aucune anomalie de fonctionnement detectee sur la periode.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Surveillance du systeme en production")
    ap.add_argument("--jours", type=int, default=7,
                    help="Fenetre d'analyse en jours (defaut 7)")
    ap.add_argument("--seuil-derive", type=float, default=0.5,
                    help="Variation relative du taux d'alarme signalee (defaut 0.5)")
    ap.add_argument("--seuil-rafale", type=int, default=20,
                    help="Nombre d'alarmes constituant une rafale")
    ap.add_argument("--fenetre-rafale", type=int, default=10,
                    help="Duree de la fenetre de rafale, en minutes")
    ap.add_argument("--silence-jours", type=float, default=2.0,
                    help="Au dela, une camera muette est signalee")
    ap.add_argument("--exporter", help="Chemin d'un export CSV des alarmes")
    args = ap.parse_args()
    return rapport(args.jours, args.seuil_derive, args.seuil_rafale,
                   args.fenetre_rafale, args.silence_jours, args.exporter)


if __name__ == "__main__":
    sys.exit(main())
