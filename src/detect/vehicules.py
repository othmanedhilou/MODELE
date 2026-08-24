"""
MODELE 2 - Détection, suivi et surveillance des véhicules.

Fonctions couvertes :
  - détection des engins (YOLO fine-tuné cimenterie, ou COCO en attendant) ;
  - suivi multi-objets par ByteTrack, qui donne un identifiant stable :
    sans identifiant, on ne peut ni compter ni mesurer un stationnement ;
  - comptage directionnel de franchissement de ligne (entrées / sorties) ;
  - intrusion d'engin dans une zone interdite (zone piétonne) ;
  - véhicule à l'arrêt trop longtemps dans une zone de circulation.

Repli sans entraînement
-----------------------
Tant que le modèle cimenterie n'est pas entraîné, on utilise yolo11s.pt
(COCO) qui connaît déjà car, truck, bus et motorcycle. La correspondance
COCO -> classes du projet est faite par CORRESPONDANCE_COCO : le comptage
et les zones fonctionnent donc dès le premier jour, avec une granularité
plus grossière (pas de distinction chargeuse / chariot élévateur).

  python -m src.detect.vehicules --source data/raw/portail.mp4
  python -m src.detect.vehicules --source 0 --poids runs/vehicules/train/weights/best.pt
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (bandeau_etat, cote_de_ligne,  # noqa: E402
                              dessiner_boite, dessiner_polygone,
                              point_dans_polygone)

# Correspondance des classes COCO vers le vocabulaire du projet, utilisée
# tant que le modèle spécifique n'est pas entraîné.
CORRESPONDANCE_COCO = {
    "car": "voiture",
    "truck": "camion_benne",
    "bus": "bus_navette",
    "motorcycle": "deux_roues",
    "person": "personne",
}


class SuiviVehicule:
    """Historique d'un véhicule suivi : trajectoire, immobilité, franchissements."""

    def __init__(self, identifiant: int, classe: str, fps: float):
        self.identifiant = identifiant
        self.classe = classe
        self.trajectoire = deque(maxlen=60)
        self.premiere_vue = 0.0
        self.derniere_vue = 0.0
        self.immobile_depuis: float | None = None
        self.cotes_lignes: dict[str, float] = {}
        self.zones_actuelles: set[str] = set()
        self.entree_zone: dict[str, float] = {}
        self.fps = fps

    def deplacement_recent(self, nb_points: int = 15) -> float:
        """Amplitude du déplacement du centre sur les dernières observations."""
        if len(self.trajectoire) < 2:
            return 0.0
        points = np.array(list(self.trajectoire)[-nb_points:])
        return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


class DetecteurVehicules:
    """Détection + suivi + règles métier (comptage, zones, stationnement)."""

    def __init__(self, config: dict, poids: str | Path | None = None, fps: float = 25.0):
        from ultralytics import YOLO

        self.cfg = config
        self.inf = config.get("inference", {})
        self.lignes = config.get("lignes_comptage", [])
        self.zones = config.get("zones", [])
        self.stationnement = config.get("stationnement", {})
        self.fps = fps

        chemin = Path(poids) if poids else None
        if chemin and chemin.exists():
            self.modele = YOLO(str(chemin))
            self.mode_coco = False
            print(f"[VEHICULES] Modèle cimenterie : {chemin}")
        else:
            self.modele = YOLO(config.get("modele_base", "yolo11s.pt"))
            self.mode_coco = True
            print("[VEHICULES] Repli COCO (yolo11s.pt) : classes génériques "
                  "car/truck/bus. Comptage et zones opérationnels.")

        self.suivis: dict[int, SuiviVehicule] = {}
        self.compteurs = defaultdict(lambda: {"entree": 0, "sortie": 0})
        self._horloge = 0.0

    def _nom_classe(self, identifiant: int) -> str:
        """Nom de classe dans le vocabulaire du projet."""
        brut = self.modele.names[int(identifiant)]
        if self.mode_coco:
            return CORRESPONDANCE_COCO.get(brut, brut)
        return brut

    def _classes_utiles(self) -> list[int] | None:
        """En mode COCO, on ne demande au modèle que les classes pertinentes."""
        if not self.mode_coco:
            return None
        inverse = {nom: i for i, nom in self.modele.names.items()}
        return [inverse[nom] for nom in CORRESPONDANCE_COCO if nom in inverse]

    # ------------------------------------------------------------ analyse
    def analyser(self, image: np.ndarray) -> dict:
        """Traite une image : détections, suivi, puis application des règles."""
        self._horloge += 1.0 / max(self.fps, 1.0)
        hauteur, largeur = image.shape[:2]

        predictions = self.modele.track(
            image,
            persist=True,
            conf=self.inf.get("conf", 0.35),
            iou=self.inf.get("iou", 0.5),
            tracker=self.inf.get("tracker", "bytetrack.yaml"),
            classes=self._classes_utiles(),
            verbose=False,
        )[0]

        detections, evenements = [], []
        if predictions.boxes is None or predictions.boxes.id is None:
            return {"detections": [], "evenements": [], "compteurs": dict(self.compteurs)}

        boites = predictions.boxes.xyxy.cpu().numpy()
        identifiants = predictions.boxes.id.cpu().numpy().astype(int)
        classes = predictions.boxes.cls.cpu().numpy().astype(int)
        scores = predictions.boxes.conf.cpu().numpy()

        vus = set()
        for bbox, identifiant, classe, score in zip(boites, identifiants, classes, scores):
            nom = self._nom_classe(classe)
            centre = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            vus.add(int(identifiant))

            suivi = self.suivis.get(int(identifiant))
            if suivi is None:
                suivi = SuiviVehicule(int(identifiant), nom, self.fps)
                suivi.premiere_vue = self._horloge
                self.suivis[int(identifiant)] = suivi
            suivi.classe = nom
            suivi.derniere_vue = self._horloge
            suivi.trajectoire.append(centre)

            evenements += self._regle_lignes(suivi, centre, largeur, hauteur)
            evenements += self._regle_zones(suivi, centre, largeur, hauteur, nom)
            evenements += self._regle_stationnement(suivi, nom)

            detections.append({
                "id": int(identifiant), "classe": nom, "score": float(score),
                "bbox": [float(v) for v in bbox], "centre": centre,
            })

        # Nettoyage des objets sortis du champ depuis plus de 10 secondes
        for identifiant in [i for i, s in self.suivis.items()
                            if self._horloge - s.derniere_vue > 10.0]:
            del self.suivis[identifiant]

        return {"detections": detections, "evenements": evenements,
                "compteurs": {k: dict(v) for k, v in self.compteurs.items()}}

    # -------------------------------------------------------- règles métier
    def _regle_lignes(self, suivi, centre, largeur, hauteur) -> list[dict]:
        """
        Comptage directionnel : on mémorise de quel côté de la ligne se
        trouvait le véhicule, et on déclenche au changement de signe.
        Compter une simple proximité produirait des doublons à chaque image.
        """
        evenements = []
        for ligne in self.lignes:
            p1 = (ligne["p1"][0] * largeur, ligne["p1"][1] * hauteur)
            p2 = (ligne["p2"][0] * largeur, ligne["p2"][1] * hauteur)
            cote = cote_de_ligne(centre, p1, p2)
            precedent = suivi.cotes_lignes.get(ligne["nom"])
            suivi.cotes_lignes[ligne["nom"]] = cote

            if precedent is None or precedent == 0 or np.sign(precedent) == np.sign(cote):
                continue

            sens_positif = ligne.get("sens_positif", "entree")
            sens = sens_positif if cote > 0 else (
                "sortie" if sens_positif == "entree" else "entree")
            self.compteurs[ligne["nom"]][sens] += 1
            evenements.append({
                "type": "franchissement", "gravite": "info",
                "ligne": ligne["nom"], "sens": sens,
                "id": suivi.identifiant, "classe": suivi.classe,
                "total": dict(self.compteurs[ligne["nom"]]),
            })
        return evenements

    def _regle_zones(self, suivi, centre, largeur, hauteur, nom_classe) -> list[dict]:
        """Intrusion en zone interdite et mesure du temps de stationnement."""
        evenements = []
        for zone in self.zones:
            nom = zone["nom"]
            dedans = point_dans_polygone(centre, zone["polygone"], largeur, hauteur)

            if dedans and nom not in suivi.zones_actuelles:
                suivi.zones_actuelles.add(nom)
                suivi.entree_zone[nom] = self._horloge
            elif not dedans and nom in suivi.zones_actuelles:
                suivi.zones_actuelles.discard(nom)
                duree = self._horloge - suivi.entree_zone.pop(nom, self._horloge)
                if zone.get("mesurer_temps_stationnement"):
                    evenements.append({
                        "type": "sortie_zone", "gravite": "info", "zone": nom,
                        "id": suivi.identifiant, "classe": suivi.classe,
                        "duree_s": round(duree, 1),
                    })
                continue

            if not dedans:
                continue

            interdites = zone.get("classes_interdites") or []
            if nom_classe in interdites:
                duree = self._horloge - suivi.entree_zone.get(nom, self._horloge)
                if duree >= zone.get("alarme_apres_s", 0):
                    evenements.append({
                        "type": "intrusion_zone", "gravite": "majeure", "zone": nom,
                        "id": suivi.identifiant, "classe": nom_classe,
                        "duree_s": round(duree, 1),
                    })
        return evenements

    def _regle_stationnement(self, suivi, nom_classe) -> list[dict]:
        """Détecte un véhicule immobile depuis trop longtemps."""
        if not self.stationnement.get("actif", False):
            return []

        seuil_px = self.stationnement.get("deplacement_max_px", 25)
        if suivi.deplacement_recent() <= seuil_px:
            if suivi.immobile_depuis is None:
                suivi.immobile_depuis = self._horloge
            duree = self._horloge - suivi.immobile_depuis
            if duree >= self.stationnement.get("duree_alarme_s", 300):
                return [{"type": "vehicule_immobile", "gravite": "mineure",
                         "id": suivi.identifiant, "classe": nom_classe,
                         "duree_s": round(duree, 1)}]
        else:
            suivi.immobile_depuis = None
        return []

    # ---------------------------------------------------------- affichage
    def annoter(self, image: np.ndarray, resultat: dict) -> np.ndarray:
        """Dessine détections, trajectoires, lignes, zones et compteurs."""
        sortie = image.copy()
        hauteur, largeur = image.shape[:2]

        for zone in self.zones:
            dessiner_polygone(sortie, zone["polygone"], zone["nom"], (0, 165, 255))

        for ligne in self.lignes:
            p1 = (int(ligne["p1"][0] * largeur), int(ligne["p1"][1] * hauteur))
            p2 = (int(ligne["p2"][0] * largeur), int(ligne["p2"][1] * hauteur))
            cv2.line(sortie, p1, p2, (255, 255, 0), 2)
            cv2.putText(sortie, ligne["nom"], (p1[0], max(12, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        graves = {e.get("id") for e in resultat["evenements"]
                  if e["gravite"] in ("majeure", "critique")}
        for detection in resultat["detections"]:
            gravite = "critique" if detection["id"] in graves else "ok"
            dessiner_boite(sortie, detection["bbox"],
                           f"#{detection['id']} {detection['classe']} "
                           f"{detection['score']:.2f}", gravite)
            suivi = self.suivis.get(detection["id"])
            if suivi and len(suivi.trajectoire) > 1:
                points = np.array(suivi.trajectoire, np.int32).reshape(-1, 1, 2)
                cv2.polylines(sortie, [points], False, (255, 200, 0), 1)

        lignes_texte = [f"Vehicules suivis : {len(resultat['detections'])}"]
        for nom, compte in resultat["compteurs"].items():
            lignes_texte.append(f"{nom} : {compte['entree']} entrees / "
                                f"{compte['sortie']} sorties")
        bandeau_etat(sortie, lignes_texte)
        return sortie


# --------------------------------------------------------------- autonome
def main() -> None:
    import argparse
    from src.utils.common import RACINE, charger_config

    ap = argparse.ArgumentParser(description="Détection et suivi des véhicules")
    ap.add_argument("--source", required=True, help="Vidéo ou index webcam")
    ap.add_argument("--config", default="configs/vehicules.yaml")
    ap.add_argument("--poids", help="Poids du modèle cimenterie entraîné")
    ap.add_argument("--sauver-video", help="Chemin de sortie .mp4 annoté")
    ap.add_argument("--sans-affichage", action="store_true")
    args = ap.parse_args()

    config = charger_config(args.config)
    poids = args.poids
    if poids is None:
        defaut = RACINE / "runs" / "vehicules" / "train" / "weights" / "best.pt"
        poids = defaut if defaut.exists() else None

    source = args.source
    chemin = source if source.isdigit() else str(
        Path(source) if Path(source).is_absolute() else RACINE / source)
    capture = cv2.VideoCapture(int(source) if source.isdigit() else chemin)
    if not capture.isOpened():
        print(f"Impossible d'ouvrir la source : {source}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    detecteur = DetecteurVehicules(config, poids, fps=fps)

    enregistreur = None
    if args.sauver_video:
        enregistreur = cv2.VideoWriter(
            args.sauver_video, cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ok, image = capture.read()
        if not ok:
            break
        resultat = detecteur.analyser(image)
        for evenement in resultat["evenements"]:
            if evenement["gravite"] != "info" or evenement["type"] == "franchissement":
                print(f"  {evenement}")
        annotee = detecteur.annoter(image, resultat)
        if enregistreur is not None:
            enregistreur.write(annotee)
        if not args.sans_affichage:
            cv2.imshow("vehicules", annotee)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    if enregistreur is not None:
        enregistreur.release()
    cv2.destroyAllWindows()
    print("\nComptages finaux :")
    for nom, compte in detecteur.compteurs.items():
        print(f"  {nom} : {compte['entree']} entrées, {compte['sortie']} sorties")


if __name__ == "__main__":
    main()
