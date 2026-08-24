"""
MODULE PLAQUE - lecture des immatriculations au portail.

Chaine a trois etages, greffee sur le modele vehicules :
    vehicule detecte  ->  plaque localisee  ->  caracteres lus

Le point critique n'est pas le modele, c'est la RESOLUTION
--------------------------------------------------------
Une plaque se lit a partir d'environ 100 a 150 pixels de largeur. En
dessous de 80 px, aucun modele au monde n'y arrive : l'information n'est
pas dans l'image. Mesurez d'abord avec :

    python scripts/tester_lisibilite_plaque.py --image data/frames/portail/x.jpg

Si le verdict est "illisible", ce n'est pas un probleme d'entrainement mais
de camera : il faut zoomer, rapprocher, ou dedier une camera a la voie
d'entree. Le savoir avant coute cinq minutes, l'ignorer coute trois
semaines.

Selection de la meilleure image
-------------------------------
Un vehicule traverse le champ en quelques secondes ; sa plaque n'est nette
et de face que sur quelques images. On ne lance donc pas l'OCR a chaque
image : on suit le vehicule, on conserve la meilleure vue de sa plaque
(largeur x nettete), et on ne lit qu'une fois, a sa sortie du champ. C'est
dix a quarante fois moins de calcul, et un resultat meilleur.

Avertissement : une immatriculation est une donnee personnelle. Voir
l'entete de configs/plaque.yaml.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import RACINE, ecrire_image  # noqa: E402

# Caracteres arabes utilises sur les plaques marocaines
LETTRES_ARABES = "\u0623\u0628\u062c\u062f\u0647\u0648\u0632\u062d\u0637\u064a\u0643\u0644\u0645\u0646\u0633\u0639\u0641\u0635\u0642\u0631\u0634\u062a\u062b\u062e\u0630\u0636\u0638\u063a"


LARGEUR_REFERENCE = 200      # largeur a laquelle toute plaque est ramenee


def nettete(image: np.ndarray) -> float:
    """
    Nettete d'une plaque, comparable d'une echelle a l'autre.

    La variance du laplacien est la mesure de flou classique, mais elle
    depend fortement de la taille de l'image : une petite plaque nette
    obtient un score bien plus eleve qu'une grande plaque tout aussi nette,
    parce que ses details occupent plus de pixels relatifs. Utilisee telle
    quelle pour choisir la meilleure vue d'un passage, elle selectionne
    systematiquement la plaque la PLUS PETITE, c'est-a-dire exactement
    l'inverse de ce qu'on veut.

    On ramene donc toute vignette a une largeur de reference avant de
    mesurer. Une petite plaque agrandie reste lisse, donc peu nette : c'est
    le comportement correct, l'agrandissement ne cree pas de detail.
    """
    if image.size == 0:
        return 0.0
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gris.shape[1] < 8:
        return 0.0
    facteur = LARGEUR_REFERENCE / gris.shape[1]
    interpolation = cv2.INTER_AREA if facteur < 1 else cv2.INTER_CUBIC
    gris = cv2.resize(gris, None, fx=facteur, fy=facteur,
                      interpolation=interpolation)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


class LecteurOCR:
    """
    Enveloppe autour du moteur OCR disponible.

    L'initialisation est PARESSEUSE : easyocr charge plusieurs centaines de
    Mo de modeles et met dix secondes a demarrer. Tant qu'aucune plaque
    n'est reellement lue, on ne paie pas ce cout. Sur un pipeline qui
    tourne des heures sans voir un vehicule, la difference est nette.
    """

    def __init__(self, moteur: str = "auto", langues=("ar", "fr")):
        self.moteur_demande = moteur
        self.langues = list(langues)
        self._lecteur = None
        self._moteur = None

    @property
    def moteur(self) -> str:
        if self._moteur is None:
            self._initialiser()
        return self._moteur

    def _initialiser(self) -> None:
        candidats = ([self.moteur_demande] if self.moteur_demande != "auto"
                     else ["easyocr", "tesseract"])
        for nom in candidats:
            if nom == "easyocr":
                try:
                    import easyocr
                    # easyocr n'accepte pas 'ar' et 'fr' ensemble : l'arabe
                    # ne se combine qu'avec l'anglais. On privilegie donc
                    # l'arabe si demande, sinon le latin.
                    langues = ["ar", "en"] if "ar" in self.langues else ["en"]
                    self._lecteur = easyocr.Reader(langues, gpu=False, verbose=False)
                    self._moteur = "easyocr"
                    return
                except Exception as erreur:
                    print(f"[OCR] easyocr indisponible : {erreur}")
            elif nom == "tesseract":
                try:
                    import pytesseract
                    pytesseract.get_tesseract_version()
                    self._lecteur = pytesseract
                    self._moteur = "tesseract"
                    return
                except Exception as erreur:
                    print(f"[OCR] tesseract indisponible : {erreur}")
        self._moteur = "aucun"
        print("[OCR] Aucun moteur disponible : les plaques seront enregistrees "
              "en vignette pour lecture par un operateur.")

    def lire(self, vignette: np.ndarray) -> tuple[str, float]:
        """Retourne (texte brut, confiance). Chaine vide si aucune lecture."""
        if self.moteur == "aucun":
            return "", 0.0
        try:
            if self._moteur == "easyocr":
                resultats = self._lecteur.readtext(vignette, detail=1, paragraph=False)
                if not resultats:
                    return "", 0.0
                meilleur = max(resultats, key=lambda r: r[2])
                return str(meilleur[1]), float(meilleur[2])
            texte = self._lecteur.image_to_string(
                vignette, config="--psm 7 --oem 3")
            return texte.strip(), 0.5 if texte.strip() else 0.0
        except Exception:
            return "", 0.0


def normaliser_plaque(texte_brut: str, config_format: dict) -> tuple[str, bool]:
    """
    Nettoie une lecture OCR et verifie qu'elle ressemble a une plaque
    marocaine (chiffres, lettre, numero de region).

    Retourne (texte normalise, conforme). Un texte non conforme n'est pas
    jete : il est conserve avec un indicateur, car une lecture partielle
    reste exploitable pour rapprocher un camion d'un bon de livraison.
    """
    texte = texte_brut.upper()
    texte = re.sub(r"[^0-9A-Z" + LETTRES_ARABES + r"]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()

    chiffres = re.sub(r"[^0-9]", "", texte)
    if not config_format.get("verifier", True):
        return texte, bool(texte)

    for prefixe in config_format.get("prefixes_speciaux", []):
        if texte.startswith(prefixe):
            return texte, len(chiffres) >= 3

    conforme = (config_format.get("chiffres_min", 4)
                <= len(chiffres)
                <= config_format.get("chiffres_max", 6) + 2)
    return texte, conforme


class DetecteurPlaque:
    """Localise et lit les plaques dans les boites de vehicules detectees."""

    def __init__(self, config: dict):
        self.cfg = config
        self.loc = config["localisation"]
        self.qualite = config["qualite"]
        self.format = config["format"]
        self.anonymiser = config.get("anonymiser", False)

        self.modele = None
        chemin = RACINE / self.loc.get("poids", "")
        if chemin.suffix == ".pt" and chemin.exists():
            from ultralytics import YOLO
            self.modele = YOLO(str(chemin))
            print(f"[PLAQUE] Modele de localisation : {chemin}")
        else:
            print("[PLAQUE] Localisation par vision classique "
                  "(aucun modele de plaque entraine).")

        self.ocr = LecteurOCR(config["lecture"]["moteur"],
                              config["lecture"]["langues"])
        self._meilleures = {}      # id de suivi -> meilleure vue de la plaque

    # ------------------------------------------------------- localisation
    def _energie_texte(self, plaque: np.ndarray) -> float:
        """
        Quantite de texte sombre presente dans une zone claire.

        Sert a distinguer une vraie plaque d'un simple reflet blanc ou d'un
        autocollant : la plaque contient des caracteres, pas le reflet.
        """
        if plaque.size == 0:
            return 0.0
        gris = cv2.cvtColor(plaque, cv2.COLOR_BGR2GRAY)
        noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        chapeau = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, noyau)
        return float((chapeau > 40).mean())

    def _localiser_cv(self, vignette_vehicule: np.ndarray) -> list[dict]:
        """
        Localisation sans apprentissage.

        Une plaque est un RECTANGLE CLAIR CONTENANT DU TEXTE SOMBRE. On
        cherche donc d'abord les zones claires et rectangulaires dans la
        partie basse du vehicule, puis on ne garde que celles qui contiennent
        effectivement des caracteres.

        Chercher directement le texte (chapeau noir + Sobel, methode
        classique) donne la position du bloc de caracteres et non celle de la
        plaque : la largeur mesuree est alors sous-estimee de 20 a 35 pour
        cent, ce qui fausse le verdict de lisibilite. On mesure donc le
        support, pas l'encre.
        """
        hauteur, largeur = vignette_vehicule.shape[:2]
        if hauteur < 20 or largeur < 40:
            return []

        haut = int(self.loc["bande_verticale"][0] * hauteur)
        bas = int(self.loc["bande_verticale"][1] * hauteur)
        bande = vignette_vehicule[haut:bas]
        if bande.size == 0:
            return []

        gris = cv2.cvtColor(bande, cv2.COLOR_BGR2GRAY)
        gris = cv2.bilateralFilter(gris, 5, 30, 30)

        # Zones nettement plus claires que la carrosserie environnante.
        # Seuil adaptatif : une plaque blanche sur voiture blanche reste
        # detectee par sa bordure et son texte, une plaque sur carrosserie
        # sombre ressort tres nettement.
        seuil = max(np.percentile(gris, 88), gris.mean() + 25)
        claires = (gris >= seuil).astype(np.uint8) * 255
        claires = cv2.morphologyEx(
            claires, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)))
        claires = cv2.morphologyEx(
            claires, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        contours, _ = cv2.findContours(claires, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        aire_vehicule = float(hauteur * largeur)
        candidats = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w < self.loc["largeur_min_px"]:
                continue
            ratio = w / h
            if not self.loc["ratio_min"] <= ratio <= self.loc["ratio_max"]:
                continue
            aire = w * h
            if not (self.loc["aire_min_ratio"] * aire_vehicule <= aire
                    <= self.loc["aire_max_ratio"] * aire_vehicule):
                continue
            # Une plaque est pleine : son contour remplit sa boite englobante.
            # Un reflet diffus ou une grille de calandre ne le font pas.
            remplissage = cv2.contourArea(contour) / max(aire, 1)
            if remplissage < 0.55:
                continue

            plaque = bande[y:y + h, x:x + w]
            if plaque.size == 0:
                continue
            energie = self._energie_texte(plaque)
            if energie < 0.02:          # zone claire sans aucun caractere
                continue

            clarte = float(cv2.cvtColor(plaque, cv2.COLOR_BGR2GRAY).mean())
            candidats.append({
                "bbox_locale": (x, y + haut, x + w, y + h + haut),
                "largeur_px": w,
                "nettete": nettete(plaque),
                "clarte": clarte,
                "energie_texte": round(energie, 3),
                "score_forme": w * (0.5 + energie),
            })

        candidats.sort(key=lambda c: c["score_forme"], reverse=True)
        return candidats[:3]

    def localiser(self, image: np.ndarray, bbox_vehicule) -> list[dict]:
        """Candidats plaque dans la boite d'un vehicule, en coordonnees image."""
        x1, y1, x2, y2 = (int(v) for v in bbox_vehicule)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return []
        vignette = image[y1:y2, x1:x2]

        if self.modele is not None:
            resultats = self.modele.predict(vignette, conf=0.25, verbose=False)[0]
            candidats = []
            if resultats.boxes is not None:
                for boite, score in zip(resultats.boxes.xyxy.cpu().numpy(),
                                        resultats.boxes.conf.cpu().numpy()):
                    bx1, by1, bx2, by2 = (int(v) for v in boite)
                    plaque = vignette[by1:by2, bx1:bx2]
                    if plaque.size == 0:
                        continue
                    candidats.append({
                        "bbox_locale": (bx1, by1, bx2, by2),
                        "largeur_px": bx2 - bx1,
                        "nettete": nettete(plaque),
                        "clarte": float(cv2.cvtColor(plaque,
                                                     cv2.COLOR_BGR2GRAY).mean()),
                        "score_forme": float(score) * (bx2 - bx1),
                    })
        else:
            candidats = self._localiser_cv(vignette)

        for candidat in candidats:
            bx1, by1, bx2, by2 = candidat["bbox_locale"]
            candidat["bbox"] = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
            candidat["verdict"] = self.verdict_qualite(candidat["largeur_px"],
                                                       candidat["nettete"])
        return candidats

    # ---------------------------------------------------------- qualite
    def verdict_qualite(self, largeur_px: float, valeur_nettete: float) -> str:
        """Classe la lisibilite d'une plaque d'apres sa taille et sa nettete."""
        if largeur_px < self.qualite["largeur_illisible_px"]:
            return "illisible"
        if valeur_nettete < self.qualite["nettete_min"]:
            return "flou"
        if largeur_px < self.qualite["largeur_limite_px"]:
            return "limite"
        if largeur_px < self.qualite["largeur_bonne_px"]:
            return "correct"
        return "bon"

    # ------------------------------------------- suivi et meilleure vue
    def observer(self, image: np.ndarray, identifiant: int, bbox_vehicule) -> dict | None:
        """
        Enregistre la meilleure vue de la plaque d'un vehicule suivi.

        Appelee a chaque image. Ne fait AUCUN OCR : elle ne fait que
        conserver le meilleur recadrage vu jusqu'ici. La lecture n'a lieu
        qu'au passage par conclure().
        """
        candidats = self.localiser(image, bbox_vehicule)
        if not candidats:
            return None
        # La largeur prime : plus la plaque est grande, plus l'information y
        # est. La nettete n'intervient que comme penalite, plafonnee, pour
        # ne pas retenir une grande plaque completement floue.
        def score_vue(candidat):
            penalite = min(candidat["nettete"] / max(self.qualite["nettete_min"], 1.0),
                           1.5)
            return candidat["largeur_px"] * max(penalite, 0.15)

        meilleur = max(candidats, key=score_vue)
        score = score_vue(meilleur)

        precedent = self._meilleures.get(identifiant)
        if precedent is None or score > precedent["score"]:
            bx1, by1, bx2, by2 = meilleur["bbox"]
            marge = max(2, int(0.06 * (bx2 - bx1)))
            recadrage = image[max(0, by1 - marge):by2 + marge,
                              max(0, bx1 - marge):bx2 + marge].copy()
            self._meilleures[identifiant] = {
                "score": score,
                "vignette": recadrage,
                "largeur_px": meilleur["largeur_px"],
                "nettete": meilleur["nettete"],
                "verdict": meilleur["verdict"],
                "bbox": meilleur["bbox"],
            }
        return meilleur

    def conclure(self, identifiant: int) -> dict | None:
        """
        Lit la plaque du vehicule qui quitte le champ, a partir de sa
        meilleure vue, puis oublie le suivi. C'est le seul endroit ou l'OCR
        est appele : une fois par vehicule, pas une fois par image.
        """
        donnees = self._meilleures.pop(identifiant, None)
        if donnees is None:
            return None

        resultat = {
            "id": identifiant,
            "largeur_px": int(donnees["largeur_px"]),
            "nettete": round(donnees["nettete"], 1),
            "verdict": donnees["verdict"],
            "texte": "",
            "conforme": False,
            "confiance": 0.0,
            "vignette": donnees["vignette"],
        }

        if self.anonymiser or donnees["verdict"] == "illisible":
            return resultat

        vignette = donnees["vignette"]
        # Agrandissement : l'OCR travaille mieux sur une plaque d'au moins
        # 200 px de large, meme si l'information n'est pas augmentee.
        if vignette.shape[1] < 200 and vignette.shape[1] > 0:
            facteur = 200 / vignette.shape[1]
            vignette = cv2.resize(vignette, None, fx=facteur, fy=facteur,
                                  interpolation=cv2.INTER_CUBIC)
        gris = cv2.cvtColor(vignette, cv2.COLOR_BGR2GRAY)
        gris = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)

        brut, confiance = self.ocr.lire(gris)
        if confiance >= self.cfg["lecture"]["confiance_min"]:
            texte, conforme = normaliser_plaque(brut, self.format)
            resultat.update({"texte": texte, "conforme": conforme,
                             "confiance": round(confiance, 3)})
        return resultat

    def flouter(self, image: np.ndarray, bbox) -> None:
        """Anonymise une plaque en place, si la conservation est interdite."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        zone = image[y1:y2, x1:x2]
        if zone.size:
            image[y1:y2, x1:x2] = cv2.GaussianBlur(zone, (31, 31), 0)
