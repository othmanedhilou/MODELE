"""
MODELE 1 - Contrôle de l'éclairage de l'usine.

Deux diagnostics complémentaires, qui ne répondent pas à la même question :

  A. ETAT DES LUMINAIRES  ("telle lampe est-elle en panne ?")
     YOLO détecte les luminaires ; leur état (allumé / faible / éteint) est
     déduit par PHOTOMETRIE plutôt qu'appris. Raison : entraîner trois
     classes exigerait des centaines d'exemples de lampes en panne, qui par
     définition sont rares. La luminance de la boîte donne la réponse
     immédiatement et se calibre en deux minutes.
     Un luminaire qui bascule sans arrêt entre allumé et éteint sur la
     fenêtre d'observation est classé DEFAILLANT (amorçage, ballast usé).

  B. NIVEAU D'ECLAIREMENT PAR ZONE  ("cette zone est-elle assez éclairée ?")
     Mesure de la luminance médiane dans des polygones définis en
     configuration. C'est le diagnostic qui intéresse la sécurité : une
     zone de circulation sous-éclairée la nuit est un risque, que la cause
     soit une lampe grillée, un projecteur mal orienté ou un obstacle.

Les deux analyses ne sont valides que la nuit (ou en intérieur) : en plein
soleil, une lampe éteinte est normale. Le module détecte donc jour/nuit
avant de conclure.

  python -m src.detect.eclairage --source data/raw/parc_nuit.mp4
  python -m src.detect.eclairage --source 0 --poids runs/eclairage/train/weights/best.pt
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (COULEURS, bandeau_etat, dessiner_boite,  # noqa: E402
                              dessiner_polygone, est_la_nuit, lire_image,
                              masque_polygone)

ETATS = ("allume", "faible", "eteint")


class Luminaire:
    """Un luminaire suivi dans le temps, avec son historique d'état."""

    _compteur = 0

    def __init__(self, bbox, fenetre: int):
        Luminaire._compteur += 1
        self.identifiant = Luminaire._compteur
        self.bbox = bbox
        self.historique = deque(maxlen=fenetre)
        self.vu_depuis = 0

    @property
    def centre(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def etat_stable(self, ratio_confirmation: float) -> str:
        """État majoritaire sur la fenêtre, s'il dépasse le ratio de confirmation."""
        if not self.historique:
            return "inconnu"
        compte = {etat: self.historique.count(etat) for etat in ETATS}
        etat, occurrences = max(compte.items(), key=lambda kv: kv[1])
        if occurrences / len(self.historique) >= ratio_confirmation:
            return etat
        return "instable"

    def taux_basculement(self) -> float:
        """Proportion de changements d'état d'une image à l'autre (clignotement)."""
        if len(self.historique) < 3:
            return 0.0
        changements = sum(1 for a, b in zip(self.historique, list(self.historique)[1:])
                          if a != b)
        return changements / (len(self.historique) - 1)


class AnalyseurEclairage:
    """Contrôle photométrique de l'éclairage, avec ou sans modèle YOLO."""

    def __init__(self, config: dict, poids: str | Path | None = None):
        self.cfg = config
        self.photo = config["photometrie"]
        self.zones = config.get("zones", [])
        self.jour_nuit = config.get("jour_nuit", {})

        self.modele = None
        if poids and Path(poids).exists():
            from ultralytics import YOLO
            self.modele = YOLO(str(poids))
            print(f"[ECLAIRAGE] Modèle chargé : {poids}")
        else:
            print("[ECLAIRAGE] Aucun modèle de luminaire : analyse par zones "
                  "uniquement (le diagnostic de sécurité reste opérationnel).")

        self._luminaires: list[Luminaire] = []
        self._masques_zones: dict[str, np.ndarray] = {}

    # ------------------------------------------------------- photométrie
    @staticmethod
    def _luminance_boite(canal_v: np.ndarray, bbox) -> float:
        """
        Luminance représentative d'un luminaire : 90e centile de sa boîte.

        On ne prend pas la moyenne car la boîte contient toujours du fond
        sombre autour du globe ; la moyenne d'une lampe allumée serait tirée
        vers le bas et la ferait passer pour éteinte.
        """
        h, w = canal_v.shape
        x1 = max(0, int(bbox[0]));  y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]));  y2 = min(h, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(np.percentile(canal_v[y1:y2, x1:x2], 90))

    def _etat_instantane(self, luminance_boite: float, luminance_scene: float) -> str:
        """Compare la lampe à son propre fond : robuste au réglage de la caméra."""
        ecart = luminance_boite - luminance_scene
        if (luminance_boite >= self.photo["seuil_allume"]
                and ecart >= self.photo["delta_vs_scene"]):
            return "allume"
        if luminance_boite >= self.photo["seuil_faible"]:
            return "faible"
        return "eteint"

    # ------------------------------------------------- suivi des luminaires
    def _associer(self, boites) -> list[Luminaire]:
        """
        Associe les détections courantes aux luminaires déjà connus.

        La caméra est fixe et les luminaires ne bougent pas : une simple
        association par distance des centres suffit, inutile de sortir un
        tracker complet.
        """
        fenetre = self.photo["fenetre_frames"]
        for luminaire in self._luminaires:
            luminaire.vu_depuis += 1

        for bbox in boites:
            centre = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            diagonale = np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
            seuil = max(20.0, diagonale)

            meilleur, distance_min = None, float("inf")
            for luminaire in self._luminaires:
                distance = np.hypot(centre[0] - luminaire.centre[0],
                                    centre[1] - luminaire.centre[1])
                if distance < distance_min:
                    meilleur, distance_min = luminaire, distance

            if meilleur is not None and distance_min <= seuil:
                meilleur.bbox = bbox
                meilleur.vu_depuis = 0
            else:
                nouveau = Luminaire(bbox, fenetre)
                nouveau.vu_depuis = 0
                self._luminaires.append(nouveau)

        # Oubli des luminaires disparus depuis longtemps (masqués par un camion)
        self._luminaires = [l for l in self._luminaires if l.vu_depuis < 90]
        return [l for l in self._luminaires if l.vu_depuis == 0]

    # ------------------------------------------------------------ analyse
    def analyser(self, image: np.ndarray) -> dict:
        """Analyse une image et retourne le diagnostic complet d'éclairage."""
        canal_v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        luminance_scene = float(np.median(canal_v))
        nuit = est_la_nuit(image, self.jour_nuit)

        # --- A. état des luminaires ---
        resultats_luminaires = []
        if self.modele is not None:
            predictions = self.modele.predict(image, conf=0.35, verbose=False)[0]
            boites = predictions.boxes.xyxy.cpu().numpy().tolist() if predictions.boxes else []
            actifs = self._associer(boites)

            for luminaire in actifs:
                valeur = self._luminance_boite(canal_v, luminaire.bbox)
                luminaire.historique.append(
                    self._etat_instantane(valeur, luminance_scene))

                etat = luminaire.etat_stable(self.photo["ratio_confirmation"])
                if luminaire.taux_basculement() >= self.photo["seuil_clignotement"]:
                    etat = "defaillant"

                resultats_luminaires.append({
                    "id": luminaire.identifiant,
                    "bbox": luminaire.bbox,
                    "luminance": round(valeur, 1),
                    "etat": etat,
                    "clignotement": round(luminaire.taux_basculement(), 2),
                })

        # --- B. niveau d'éclairement par zone ---
        resultats_zones = []
        for zone in self.zones:
            nom = zone["nom"]
            if nom not in self._masques_zones:
                self._masques_zones[nom] = masque_polygone(image.shape, zone["polygone"])
            masque = self._masques_zones[nom]
            valeurs = canal_v[masque > 0]
            mesure = float(np.median(valeurs)) if valeurs.size else 0.0

            applicable = nuit or not zone.get("actif_la_nuit_seulement", True)
            conforme = (not applicable) or mesure >= zone["luminance_min"]
            resultats_zones.append({
                "nom": nom,
                "luminance": round(mesure, 1),
                "seuil": zone["luminance_min"],
                "applicable": applicable,
                "conforme": conforme,
            })

        # --- Synthèse des alarmes ---
        alarmes = []
        if nuit:
            for r in resultats_luminaires:
                if r["etat"] == "eteint":
                    alarmes.append({"type": "luminaire_eteint", "gravite": "majeure",
                                    "id": r["id"], "luminance": r["luminance"]})
                elif r["etat"] in ("faible", "defaillant"):
                    alarmes.append({"type": f"luminaire_{r['etat']}", "gravite": "mineure",
                                    "id": r["id"], "luminance": r["luminance"]})
        for z in resultats_zones:
            if not z["conforme"]:
                alarmes.append({"type": "zone_sous_eclairee", "gravite": "majeure",
                                "id": z["nom"], "luminance": z["luminance"],
                                "seuil": z["seuil"]})

        return {
            "nuit": nuit,
            "luminance_scene": round(luminance_scene, 1),
            "luminaires": resultats_luminaires,
            "zones": resultats_zones,
            "alarmes": alarmes,
        }

    # ---------------------------------------------------------- affichage
    def annoter(self, image: np.ndarray, resultat: dict) -> np.ndarray:
        """Rend le diagnostic lisible sur l'image."""
        gravite_par_etat = {"allume": "ok", "faible": "mineure", "eteint": "critique",
                            "defaillant": "majeure", "instable": "info",
                            "inconnu": "neutre"}
        sortie = image.copy()

        for zone in resultat["zones"]:
            config_zone = next(z for z in self.zones if z["nom"] == zone["nom"])
            couleur = COULEURS["ok"] if zone["conforme"] else COULEURS["critique"]
            dessiner_polygone(sortie, config_zone["polygone"],
                              f"{zone['nom']} {zone['luminance']:.0f}/{zone['seuil']}",
                              couleur)

        for luminaire in resultat["luminaires"]:
            dessiner_boite(sortie, luminaire["bbox"],
                           f"L{luminaire['id']} {luminaire['etat']} "
                           f"({luminaire['luminance']:.0f})",
                           gravite_par_etat.get(luminaire["etat"], "neutre"))

        bandeau_etat(sortie, [
            f"Eclairage - {'NUIT' if resultat['nuit'] else 'JOUR'} "
            f"(luminance scene {resultat['luminance_scene']:.0f})",
            f"Luminaires : {len(resultat['luminaires'])}",
            f"Alarmes    : {len(resultat['alarmes'])}",
        ])
        return sortie


# --------------------------------------------------------------- autonome
def main() -> None:
    import argparse
    from src.utils.common import RACINE, charger_config

    ap = argparse.ArgumentParser(description="Contrôle de l'éclairage")
    ap.add_argument("--source", required=True, help="Image, vidéo ou index webcam")
    ap.add_argument("--config", default="configs/eclairage.yaml")
    ap.add_argument("--poids", help="Poids YOLO de détection des luminaires")
    ap.add_argument("--sans-affichage", action="store_true")
    args = ap.parse_args()

    config = charger_config(args.config)
    poids = args.poids
    if poids is None:
        defaut = RACINE / "runs" / "eclairage" / "train" / "weights" / "best.pt"
        poids = defaut if defaut.exists() else None

    analyseur = AnalyseurEclairage(config, poids)
    source = args.source
    chemin = Path(source) if source.isdigit() is False else None
    if chemin is not None and not chemin.is_absolute():
        chemin = RACINE / source

    # Image fixe
    if chemin is not None and chemin.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = lire_image(chemin)
        if image is None:
            print(f"Image illisible : {chemin}")
            return
        resultat = analyseur.analyser(image)
        for cle in ("nuit", "luminance_scene"):
            print(f"{cle} : {resultat[cle]}")
        for zone in resultat["zones"]:
            etat = "conforme" if zone["conforme"] else "SOUS-ECLAIREE"
            print(f"  zone {zone['nom']:<20} {zone['luminance']:6.1f} "
                  f"(seuil {zone['seuil']}) -> {etat}")
        for luminaire in resultat["luminaires"]:
            print(f"  luminaire L{luminaire['id']:<3} {luminaire['etat']:<11} "
                  f"luminance {luminaire['luminance']:.0f}")
        if not args.sans_affichage:
            cv2.imshow("eclairage", analyseur.annoter(image, resultat))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # Flux vidéo
    capture = cv2.VideoCapture(int(source) if source.isdigit() else str(chemin))
    if not capture.isOpened():
        print(f"Impossible d'ouvrir la source : {source}")
        return
    while True:
        ok, image = capture.read()
        if not ok:
            break
        resultat = analyseur.analyser(image)
        if not args.sans_affichage:
            cv2.imshow("eclairage", analyseur.annoter(image, resultat))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
