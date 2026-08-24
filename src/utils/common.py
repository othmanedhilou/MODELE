"""
Utilitaires partagés par les trois modèles.

Contient : chargement de configuration, gestion des régions d'intérêt (ROI)
en coordonnées normalisées, journal d'alarmes et primitives d'affichage.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def configurer_console() -> None:
    """
    Force la sortie console en UTF-8.

    Sous Windows, le terminal utilise cp1252 par défaut : le moindre accent
    dans un print() fait planter le script avec UnicodeEncodeError. Appelé
    à l'import, ce correctif rend tous les scripts du projet utilisables
    depuis l'invite de commandes Windows.
    """
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


configurer_console()


# Racine du projet = deux niveaux au-dessus de ce fichier (src/utils/ -> .)
RACINE = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- config
def charger_config(chemin: str | Path) -> dict:
    """Charge un YAML de configuration, en résolvant le chemin depuis la racine."""
    chemin = Path(chemin)
    if not chemin.is_absolute():
        chemin = RACINE / chemin
    if not chemin.exists():
        raise FileNotFoundError(f"Configuration introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def chemin_projet(*parties: str) -> Path:
    """Construit un chemin absolu à partir de la racine du projet."""
    return RACINE.joinpath(*parties)


# ------------------------------------------------------------------- ROI
def polygone_en_pixels(polygone_norm, largeur: int, hauteur: int) -> np.ndarray:
    """Convertit un polygone normalisé (0-1) en coordonnées pixels entières."""
    pts = np.array(polygone_norm, dtype=np.float32)
    pts[:, 0] *= largeur
    pts[:, 1] *= hauteur
    return pts.astype(np.int32)


def masque_polygone(forme, polygone_norm) -> np.ndarray:
    """Crée un masque binaire (uint8, 0/255) de la zone décrite par le polygone."""
    hauteur, largeur = forme[:2]
    masque = np.zeros((hauteur, largeur), dtype=np.uint8)
    pts = polygone_en_pixels(polygone_norm, largeur, hauteur)
    cv2.fillPoly(masque, [pts], 255)
    return masque


def point_dans_polygone(point_xy, polygone_norm, largeur: int, hauteur: int) -> bool:
    """Teste si un point pixel (x, y) est à l'intérieur du polygone normalisé."""
    pts = polygone_en_pixels(polygone_norm, largeur, hauteur)
    return cv2.pointPolygonTest(pts, (float(point_xy[0]), float(point_xy[1])), False) >= 0


def cote_de_ligne(point_xy, p1, p2) -> float:
    """
    Signe du produit vectoriel : indique de quel côté du segment p1->p2 se
    trouve le point. Sert au comptage directionnel de franchissement de ligne.
    """
    return ((p2[0] - p1[0]) * (point_xy[1] - p1[1])
            - (p2[1] - p1[1]) * (point_xy[0] - p1[0]))


# -------------------------------------------------------------- luminance
def luminance(image: np.ndarray, masque: np.ndarray | None = None) -> float:
    """Luminance médiane (canal V du HSV), éventuellement restreinte à un masque."""
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    if masque is not None:
        valeurs = v[masque > 0]
        if valeurs.size == 0:
            return 0.0
        return float(np.median(valeurs))
    return float(np.median(v))


def est_la_nuit(image: np.ndarray, config_jour_nuit: dict) -> bool:
    """
    Détermine si la scène est nocturne, soit par horaire système, soit par
    la luminance globale de l'image (plus robuste en intérieur / tunnel).
    """
    if config_jour_nuit.get("utiliser_horaire", False):
        heure = datetime.now().hour
        debut = config_jour_nuit.get("heure_nuit_debut", 19)
        fin = config_jour_nuit.get("heure_nuit_fin", 6)
        return heure >= debut or heure < fin
    return luminance(image) < config_jour_nuit.get("seuil_luminance_nuit", 85)


# -------------------------------------------------------------- alarmes
class JournalAlarmes:
    """
    Journal d'alarmes au format JSONL, avec anti-rebond : une même alarme
    (même caméra, même type, même objet) n'est réécrite qu'après un délai.
    """

    def __init__(self, chemin_journal, dossier_snapshots,
                 delai_repetition_s: int = 60, sauver_snapshot: bool = True):
        self.chemin = Path(chemin_journal)
        if not self.chemin.is_absolute():
            self.chemin = RACINE / self.chemin
        self.chemin.parent.mkdir(parents=True, exist_ok=True)

        self.dossier_snapshots = Path(dossier_snapshots)
        if not self.dossier_snapshots.is_absolute():
            self.dossier_snapshots = RACINE / self.dossier_snapshots
        self.dossier_snapshots.mkdir(parents=True, exist_ok=True)

        self.delai_repetition_s = delai_repetition_s
        self.sauver_snapshot = sauver_snapshot
        self._derniere_emission = {}

    def emettre(self, camera: str, modele: str, type_alarme: str,
                gravite: str = "info", details: dict | None = None,
                image: np.ndarray | None = None) -> bool:
        """
        Enregistre une alarme. Retourne True si elle a réellement été écrite,
        False si elle a été absorbée par l'anti-rebond.
        """
        cle = f"{camera}|{modele}|{type_alarme}|{(details or {}).get('id', '')}"
        maintenant = time.time()
        if maintenant - self._derniere_emission.get(cle, 0.0) < self.delai_repetition_s:
            return False
        self._derniere_emission[cle] = maintenant

        horodatage = datetime.now()
        entree = {
            "horodatage": horodatage.isoformat(timespec="seconds"),
            "camera": camera,
            "modele": modele,
            "type": type_alarme,
            "gravite": gravite,
            "details": details or {},
        }

        if image is not None and self.sauver_snapshot:
            nom = f"{horodatage:%Y%m%d_%H%M%S}_{camera}_{type_alarme}.jpg"
            chemin_img = self.dossier_snapshots / nom
            ecrire_image(chemin_img, image)
            entree["snapshot"] = str(chemin_img.relative_to(RACINE))

        with open(self.chemin, "a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")

        print(f"[ALARME/{gravite.upper()}] {camera} - {type_alarme} - {details}")
        return True


# -------------------------------------------------------------- affichage
COULEURS = {
    "ok": (80, 200, 80),
    "info": (200, 200, 80),
    "mineure": (0, 200, 255),
    "majeure": (0, 120, 255),
    "critique": (0, 0, 255),
    "neutre": (200, 200, 200),
}


def dessiner_boite(image, xyxy, etiquette: str, gravite: str = "neutre") -> None:
    """Dessine une boîte annotée avec un fond lisible sous le texte."""
    couleur = COULEURS.get(gravite, COULEURS["neutre"])
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), couleur, 2)
    (tw, th), _ = cv2.getTextSize(etiquette, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), couleur, -1)
    cv2.putText(image, etiquette, (x1 + 3, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def dessiner_polygone(image, polygone_norm, etiquette: str = "",
                      couleur=(255, 180, 0), epaisseur: int = 2) -> None:
    """Trace un polygone de zone (coordonnées normalisées) sur l'image."""
    h, w = image.shape[:2]
    pts = polygone_en_pixels(polygone_norm, w, h)
    cv2.polylines(image, [pts], True, couleur, epaisseur)
    if etiquette:
        origine = (int(pts[0][0]) + 4, max(12, int(pts[0][1]) - 6))
        cv2.putText(image, etiquette, origine,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, couleur, 1, cv2.LINE_AA)


def bandeau_etat(image, lignes) -> None:
    """Affiche un bandeau semi-transparent d'état en haut à gauche."""
    if not lignes:
        return
    hauteur = 18 * len(lignes) + 10
    calque = image.copy()
    cv2.rectangle(calque, (0, 0), (360, hauteur), (0, 0, 0), -1)
    cv2.addWeighted(calque, 0.55, image, 0.45, 0, image)
    for i, texte in enumerate(lignes):
        cv2.putText(image, texte, (8, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------- entrées/sorties images
# ATTENTION : cv2.imread et cv2.imwrite passent par l'API ANSI de Windows et
# ECHOUENT SILENCIEUSEMENT si le chemin contient des caractères non latins
# (accents, arabe, chinois...). Le chemin de ce projet en contient
# ("سطح المكتب"), donc tout le code doit passer par ces deux fonctions.
def lire_image(chemin, drapeaux: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Lecture d'image robuste aux chemins Unicode."""
    try:
        octets = np.fromfile(str(chemin), dtype=np.uint8)
    except (FileNotFoundError, OSError):
        return None
    if octets.size == 0:
        return None
    return cv2.imdecode(octets, drapeaux)


def ecrire_image(chemin, image: np.ndarray, params: list | None = None) -> bool:
    """Écriture d'image robuste aux chemins Unicode."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    try:
        ok, tampon = cv2.imencode(chemin.suffix, image, params or [])
        if not ok:
            return False
        tampon.tofile(str(chemin))
        return True
    except Exception:
        return False


def resoudre_data_yaml(chemin_relatif) -> Path:
    """
    Réécrit un data_*.yaml avec un chemin `path` ABSOLU et retourne le
    fichier résolu.

    Ultralytics ne résout pas `path` par rapport au fichier YAML mais par
    rapport à son réglage global `datasets_dir` (dans
    %APPDATA%/Ultralytics/settings.json). Sur un poste ayant déjà servi à un
    autre projet, ce réglage pointe ailleurs et l'entraînement échoue avec un
    « images not found » qui désigne un dossier totalement étranger.
    Écrire le chemin en absolu supprime cette dépendance à un état global.
    """
    source = Path(chemin_relatif)
    if not source.is_absolute():
        source = RACINE / source
    donnees = yaml.safe_load(source.read_text(encoding="utf-8"))

    chemin = Path(donnees.get("path", "."))
    if not chemin.is_absolute():
        chemin = (RACINE / chemin).resolve()
    donnees["path"] = str(chemin)

    destination = RACINE / "runs" / "_data" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        yaml.safe_dump(donnees, f, allow_unicode=True, sort_keys=False)
    return destination
