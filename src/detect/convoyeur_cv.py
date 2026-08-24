"""
COUCHE A - Détection de déchirure de bande transporteuse par vision
classique, sans aucune donnée d'entraînement.

Principe physique
-----------------
La bande est en caoutchouc noir : c'est de très loin la surface la plus
sombre et la plus uniforme du champ. Une déchirure expose la carcasse
textile interne (blanche/beige) ou laisse passer la lumière : elle apparaît
donc comme une trace CLAIRE, FINE et ALLONGÉE dans le sens de défilement.

Chaîne de traitement
--------------------
1. ROI      : on isole la bande (détection automatique de la zone sombre,
              ou polygone de secours défini dans configs/convoyeur.yaml).
2. Seuillage: seuil ADAPTATIF médiane + k * MAD calculé sur la bande elle-même.
              Insensible aux variations d'éclairage du tunnel, contrairement
              à un seuil fixe qui déclencherait à chaque passage de nuage.
3. Morpho   : ouverture (retire poussière et bruit), puis fermeture avec un
              noyau orienté dans le sens de défilement (reconnecte une
              déchirure apparaissant en pointillés).
4. Formes   : on ne garde que les blobs allongés et alignés sur le sens de
              défilement. Un déversement de clinker est large et non aligné,
              un reflet de rouleau est court : tous deux sont écartés.
5. Temporel : confirmation sur une fenêtre glissante. Une poussière traverse
              une seule image, une déchirure est présente sur plusieurs.

Utilisation autonome
--------------------
  python -m src.detect.convoyeur_cv --source data/raw/convoyeur.mp4
  python -m src.detect.convoyeur_cv --source data/frames/convoyeur --pre-annoter
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (lire_image, masque_polygone,  # noqa: E402
                              polygone_en_pixels)


class DetecteurDechirureCV:
    """Détecteur de déchirure par vision classique, prêt à l'emploi."""

    def __init__(self, config: dict):
        self.cfg = config
        self.cv = config["cv"]
        self.roi_cfg = config["roi"]
        self.gravite_cfg = config["gravite"]
        self.mm_par_pixel = self.cv.get("mm_par_pixel", 1.0)

        self.clahe = cv2.createCLAHE(
            clipLimit=self.cv.get("clahe_clip", 2.0),
            tileGridSize=(self.cv.get("clahe_grid", 8),) * 2)

        self._masque_roi = None
        # Sens de défilement de la bande dans l'image (degrés, 0 = horizontal).
        # Fixé par l'installation : on ne le devine que si la config dit "auto".
        sens = self.roi_cfg.get("sens_defilement_deg", 90)
        self._sens_auto = str(sens).lower() == "auto"
        self._angle_bande = 90.0 if self._sens_auto else float(sens)
        self._historique = deque(maxlen=self.cv.get("fenetre_frames", 12))
        self._centres_bande = deque(maxlen=120)
        self._reference_centre = None
        self._carte_fixe = None      # frequence de clarte par pixel
        self._frames_vues = 0
        self._image_precedente = None

    # ------------------------------------------------------------- ROI
    def _detecter_bande(self, gris: np.ndarray) -> np.ndarray | None:
        """
        Trouve automatiquement la bande : plus grande composante sombre et
        compacte de l'image. Retourne None si le résultat n'est pas crédible,
        auquel cas on retombe sur le polygone de configuration.
        """
        flou = cv2.GaussianBlur(gris, (9, 9), 0)
        seuil = np.percentile(flou, 40)          # la bande occupe le bas de l'histogramme
        sombre = (flou < seuil).astype(np.uint8) * 255
        noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        sombre = cv2.morphologyEx(sombre, cv2.MORPH_CLOSE, noyau)
        sombre = cv2.morphologyEx(sombre, cv2.MORPH_OPEN, noyau)

        contours, _ = cv2.findContours(sombre, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        plus_grand = max(contours, key=cv2.contourArea)
        ratio = cv2.contourArea(plus_grand) / (gris.shape[0] * gris.shape[1])
        # Une bande crédible occupe entre 10 % et 85 % du champ
        if not 0.10 <= ratio <= 0.85:
            return None

        masque = np.zeros_like(gris)
        cv2.drawContours(masque, [plus_grand], -1, 255, -1)
        # On érode pour ne pas inclure les bords métalliques clairs
        masque = cv2.erode(masque, np.ones((9, 9), np.uint8))
        if self._sens_auto:
            self._angle_bande = self._angle_principal(plus_grand)
        return masque

    @staticmethod
    def _angle_principal(contour: np.ndarray) -> float:
        """
        Orientation du grand côté du rectangle englobant minimal, en degrés
        (0 = horizontal, 90 = vertical). Plus stable que l'ACP des points du
        contour, qui est déséquilibrée par la densité de points du périmètre.
        """
        (_, _), (largeur, hauteur), angle = cv2.minAreaRect(contour)
        return float((angle if largeur >= hauteur else angle + 90) % 180)

    def _obtenir_masque(self, gris: np.ndarray) -> np.ndarray:
        """Masque de la bande, calculé une seule fois (caméra et bande fixes)."""
        if self._masque_roi is not None:
            return self._masque_roi

        masque = None
        if self.roi_cfg.get("auto_detection", True):
            masque = self._detecter_bande(gris)

        if masque is None:
            masque = masque_polygone(gris.shape, self.roi_cfg["polygone"])
            if self._sens_auto:
                pts = polygone_en_pixels(self.roi_cfg["polygone"], *gris.shape[::-1])
                self._angle_bande = self._angle_principal(pts.reshape(-1, 1, 2))

        self._masque_roi = masque
        return masque

    # ------------------------------------------- segmentation des anomalies
    def _seuils(self, gris, masque):
        """
        Seuils statistiques clair et sombre, calcules sur la bande.

        On utilise la mediane et la MAD (ecart absolu median) plutot que la
        moyenne et l'ecart-type : quelques pixels tres clairs (le defaut
        lui-meme) ne deplacent pas ces estimateurs, alors qu'ils tireraient
        un ecart-type vers le haut et finiraient par masquer le defaut.
        """
        pixels = gris[masque > 0]
        if pixels.size < 500:
            return None
        mediane = float(np.median(pixels))
        mad = float(np.median(np.abs(pixels - mediane)))

        ecart_clair = max(self.cv["k_mad"] * 1.4826 * mad,
                          float(self.cv["seuil_absolu_min"]))
        ecart_sombre = max(self.cv.get("k_mad_sombre", 5.0) * 1.4826 * mad,
                           float(self.cv.get("seuil_absolu_min_sombre", 25)))
        return (mediane,
                min(mediane + ecart_clair, 250.0),
                max(mediane - ecart_sombre, 3.0))

    def _noyau_oriente(self, longueur):
        """Noyau lineaire oriente dans le sens de defilement de la bande."""
        noyau = np.zeros((longueur, longueur), np.uint8)
        cv2.line(noyau, (longueur // 2, 0), (longueur // 2, longueur - 1), 1, 1)
        matrice = cv2.getRotationMatrix2D(
            (longueur / 2, longueur / 2), 90.0 - self._angle_bande, 1.0)
        noyau = cv2.warpAffine(noyau, matrice, (longueur, longueur))
        return (noyau > 0).astype(np.uint8)

    def _binariser(self, gris, masque):
        """
        Deux masques binaires : anomalies CLAIRES et anomalies SOMBRES.

        Ne chercher que les zones claires reviendrait a ignorer toute une
        famille d'anomalies : une perforation laisse voir l'ombre ou la
        structure sous la bande, donc une tache plus SOMBRE que le
        caoutchouc, jamais plus claire.
        """
        seuils = self._seuils(gris, masque)
        if seuils is None:
            vide = np.zeros_like(gris)
            return vide, vide
        _, seuil_clair, seuil_sombre = seuils

        clair = ((gris > seuil_clair) & (masque > 0)).astype(np.uint8) * 255
        k = self.cv.get("ouverture", 3)
        if k > 1:
            clair = cv2.morphologyEx(
                clair, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        longueur = self.cv.get("fermeture_longitudinale", 21)
        if longueur > 1:
            noyau = self._noyau_oriente(longueur)
            if noyau.sum() > 0:
                clair = cv2.morphologyEx(clair, cv2.MORPH_CLOSE, noyau)

        sombre = np.zeros_like(gris)
        if self.cv.get("detecter_sombre", True):
            # On erode la ROI : le pourtour de la bande est naturellement
            # sombre et produirait un lisere de faux positifs.
            interieur = cv2.erode(masque, np.ones((15, 15), np.uint8))
            sombre = ((gris < seuil_sombre) & (interieur > 0)).astype(np.uint8) * 255
            ks = self.cv.get("ouverture_sombre", 5)
            noyau_compact = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
            sombre = cv2.morphologyEx(sombre, cv2.MORPH_OPEN, noyau_compact)
            sombre = cv2.morphologyEx(sombre, cv2.MORPH_CLOSE, noyau_compact)

        clair = self._retirer_structures_fixes(clair, gris)
        return clair, sombre

    def _retirer_structures_fixes(self, clair, gris):
        """
        Retire du masque clair les pixels qui sont clairs en permanence.

        Un rouleau, une rayure sur l'objectif ou un reflet fixe de
        l'eclairage occupent toujours les memes pixels ; un defaut defile
        avec la bande. La frequence de clarte par pixel separe donc les deux
        sans rien connaitre de leur forme, la ou un filtre geometrique
        confondrait une trace de rouleau avec une fissure.
        """
        config = self.cv.get("suppression_structures", {})
        if not config.get("actif", True):
            return clair

        if self._carte_fixe is None or self._carte_fixe.shape != clair.shape:
            self._carte_fixe = np.zeros(clair.shape, np.float32)
            self._frames_vues = 0

        # La carte n'est mise a jour que si la bande bouge : a l'arret, un
        # vrai defaut deviendrait fixe et finirait par etre supprime.
        en_mouvement = True
        if self._image_precedente is not None:
            difference = cv2.absdiff(gris, self._image_precedente)
            en_mouvement = float(difference.mean()) > 1.0
        self._image_precedente = gris.copy()

        if en_mouvement:
            inertie = config.get("inertie", 0.02)
            cv2.accumulateWeighted((clair > 0).astype(np.float32),
                                   self._carte_fixe, inertie)
            self._frames_vues += 1

        if self._frames_vues < config.get("frames_apprentissage", 40):
            return clair

        structures = (self._carte_fixe > config.get("seuil_frequence", 0.65))
        structures = cv2.dilate(structures.astype(np.uint8),
                                np.ones((3, 3), np.uint8))
        return cv2.bitwise_and(clair, clair, mask=(1 - structures))

    # -------------------------------------------- classification des defauts
    ORDRE_GRAVITE = ("info", "mineure", "majeure", "critique")

    def _distance_au_bord(self, contour, masque):
        """Distance du defaut au bord de la bande, en fraction de sa largeur."""
        x, y, w, h = cv2.boundingRect(contour)
        ligne = masque[min(y + h // 2, masque.shape[0] - 1)]
        colonnes = np.nonzero(ligne)[0]
        if colonnes.size < 2:
            return 1.0
        gauche, droite = float(colonnes[0]), float(colonnes[-1])
        largeur_bande = max(droite - gauche, 1.0)
        centre_x = x + w / 2.0
        return min(centre_x - gauche, droite - centre_x) / largeur_bande

    def _est_credible(self, mesures, origine):
        """
        Ecarte le bruit avant toute tentative de classification.

        En passant d'un detecteur mono-classe a un classificateur multi-
        classes, on perd le filtre implicite qui rejetait tout ce qui
        n'etait pas allonge. Sans ce garde-fou, chaque grain de clinker
        clair devient une "perforation" et chaque ombre une anomalie : la
        couverture augmente, mais au prix d'un taux de fausse alarme qui
        rend le systeme inutilisable.

        Regle : un defaut est credible s'il est ALLONGE (rupture) ou
        SUFFISAMMENT GROS (objet, trou). Un petit blob compact est du bruit.
        """
        classification = self.cv.get("classification", {})
        compacte = classification.get("elongation_compacte", 2.0)
        aire_objet = classification.get("aire_objet_min_px", 800)

        allonge = mesures["elongation"] >= self.cv["elongation_min"]
        gros = mesures["aire_px"] >= aire_objet

        if origine == "sombre":
            # Un trou est compact et bien rempli ; une ombre diffuse ne l'est pas.
            return gros and mesures["remplissage"] >= 0.45

        if allonge:
            return True
        if gros and mesures["elongation"] < compacte:
            return True
        return False

    def _classer(self, mesures, origine):
        """
        Attribue un type d'anomalie a partir de la geometrie du defaut.

        Sans modele entraine, on ne dispose que de la forme, de
        l'orientation et de la position. La regle est volontairement simple
        et lisible ; c'est la couche B (YOLO) qui affine ensuite. L'objectif
        ici n'est pas la finesse mais la COUVERTURE : aucune anomalie ne
        doit sortir de la chaine sans etre nommee.
        """
        classification = self.cv.get("classification", {})
        compacte = classification.get("elongation_compacte", 2.0)
        transversal = classification.get("angle_transversal_deg", 55)
        bord = classification.get("distance_bord_ratio", 0.12)
        aire_objet = classification.get("aire_objet_min_px", 800)

        elongation = mesures["elongation"]
        ecart = mesures["ecart_angle"]
        aire = mesures["aire_px"]

        if origine == "sombre":
            # Une zone sombre au milieu du caoutchouc est un trou ; allongee,
            # c'est plutot un objet sombre pose sur la bande.
            if elongation < compacte:
                return "perforation"
            return "corps_etranger"

        if mesures["distance_bord"] <= bord and elongation >= compacte:
            return "bord_effiloche"
        if elongation < compacte:
            return "corps_etranger" if aire >= aire_objet else "perforation"
        if ecart <= self.cv["angle_max_deg"]:
            return "dechirure"
        if ecart >= transversal:
            # Allongee mais perpendiculaire au defilement : jonction de bande
            # si elle traverse toute la largeur, fissure sinon.
            return ("jonction_defectueuse"
                    if mesures["couverture_largeur"] > 0.7 else "fissure")
        return "fissure"

    def _gravite(self, classe, longueur_mm):
        """
        Gravite d'une anomalie : niveau deduit de sa taille, puis borne par
        le plancher et le plafond propres a sa classe.

        Toutes les anomalies sont detectees et journalisees ; ce sont leurs
        NIVEAUX d'alarme qui different, donc la reaction attendue. Une
        fissure naissante et une bande en train de se dechirer ne peuvent
        pas declencher la meme alerte.
        """
        if longueur_mm >= self.gravite_cfg["critique_mm"]:
            niveau = "critique"
        elif longueur_mm >= self.gravite_cfg["majeure_mm"]:
            niveau = "majeure"
        elif longueur_mm >= self.gravite_cfg["mineure_mm"]:
            niveau = "mineure"
        else:
            niveau = "info"

        regles = self.cfg.get("gravite_par_classe", {}).get(classe)
        if regles is None:
            return niveau
        index = self.ORDRE_GRAVITE.index(niveau)
        plancher = self.ORDRE_GRAVITE.index(regles.get("plancher", "info"))
        plafond = self.ORDRE_GRAVITE.index(regles.get("plafond", "critique"))
        return self.ORDRE_GRAVITE[min(max(index, plancher), plafond)]

    def _action(self, classe):
        """Consigne operateur associee a la classe, inscrite dans l'alarme."""
        return self.cfg.get("gravite_par_classe", {}).get(classe, {}).get(
            "action", "Inspecter au prochain arret.")

    def _candidats(self, binaire, masque, aire_image, origine):
        """Extrait et classe les anomalies d'un masque binaire."""
        contours, _ = cv2.findContours(binaire, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        aire_min = (self.cv["aire_min_px"] if origine == "clair"
                    else self.cv["aire_min_px"] * 0.6)
        candidats = []
        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire < aire_min or aire > self.cv["aire_max_ratio"] * aire_image:
                continue

            (cx, cy), (largeur, hauteur), angle = cv2.minAreaRect(contour)
            grand = max(largeur, hauteur)
            petit = max(min(largeur, hauteur), 1.0)
            angle_long = (angle if largeur >= hauteur else angle + 90) % 180
            ecart = abs(angle_long - self._angle_bande)
            ecart = min(ecart, 180 - ecart)

            x, y, w, h = cv2.boundingRect(contour)
            ligne = masque[min(y + h // 2, masque.shape[0] - 1)]
            colonnes = np.nonzero(ligne)[0]
            largeur_bande = max(float(colonnes[-1] - colonnes[0]), 1.0)                 if colonnes.size >= 2 else float(masque.shape[1])

            mesures = {
                "aire_px": float(aire),
                "elongation": grand / petit,
                "ecart_angle": float(ecart),
                "distance_bord": self._distance_au_bord(contour, masque),
                "couverture_largeur": w / largeur_bande,
                "remplissage": float(aire) / max(grand * petit, 1.0),
            }
            if not self._est_credible(mesures, origine):
                continue
            classe = self._classer(mesures, origine)
            longueur_mm = grand * self.mm_par_pixel

            candidats.append({
                "classe": classe,
                "origine": origine,
                "bbox": (x, y, x + w, y + h),
                "centre": (float(cx), float(cy)),
                "contour": contour,
                "longueur_px": float(grand),
                "longueur_mm": float(longueur_mm),
                "gravite": self._gravite(classe, longueur_mm),
                "action": self._action(classe),
                **{c: round(v, 3) for c, v in mesures.items()},
            })

        candidats.sort(key=lambda c: (self.ORDRE_GRAVITE.index(c["gravite"]),
                                      c["longueur_px"]), reverse=True)
        return candidats

    # ------------------------------------------------------------ analyse
    def _desalignement(self, masque):
        """
        Suit la position du centre de la bande et signale une derive.

        Une bande decentree use ses bords et finit par se dechirer : c'est
        une CAUSE, donc l'anomalie la plus rentable a detecter tot. La
        position de reference s'etablit sur les premieres images, en
        supposant le reglage initial correct.
        """
        config = self.cv.get("desalignement", {})
        if not config.get("actif", True):
            return None

        colonnes = np.nonzero(masque.any(axis=0))[0]
        if colonnes.size < 2:
            return None
        centre = float(colonnes.mean())
        largeur_bande = float(colonnes[-1] - colonnes[0])

        self._centres_bande.append(centre)
        if self._reference_centre is None:
            if len(self._centres_bande) < config.get("frames_reference", 60):
                return None
            self._reference_centre = float(np.median(self._centres_bande))
            return None

        derive = (centre - self._reference_centre) / max(largeur_bande, 1.0)
        if abs(derive) < config.get("derive_max_ratio", 0.04):
            return None

        longueur_mm = abs(derive) * largeur_bande * self.mm_par_pixel
        return {
            "classe": "desalignement",
            "origine": "geometrie",
            "bbox": (int(colonnes[0]), 0, int(colonnes[-1]), masque.shape[0]),
            "centre": (centre, masque.shape[0] / 2.0),
            "contour": None,
            "longueur_px": abs(derive) * largeur_bande,
            "longueur_mm": longueur_mm,
            "derive_ratio": round(derive, 4),
            "gravite": self._gravite("desalignement", longueur_mm),
            "action": self._action("desalignement"),
        }

    def analyser(self, image: np.ndarray) -> dict:
        """
        Analyse une image et retourne toutes les anomalies detectees.

        La confirmation temporelle compte le nombre d'IMAGES contenant au
        moins une anomalie sur la fenetre glissante, et non la persistance
        d'une meme position : la bande defile, un defaut ne reste pas au
        meme endroit d'une image a l'autre.
        """
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gris = self.clahe.apply(gris)

        masque = self._obtenir_masque(gris)
        clair, sombre = self._binariser(gris, masque)
        aire_image = float(gris.shape[0] * gris.shape[1])

        candidats = (self._candidats(clair, masque, aire_image, "clair")
                     + self._candidats(sombre, masque, aire_image, "sombre"))

        derive = self._desalignement(masque)
        if derive is not None:
            candidats.append(derive)

        candidats.sort(key=lambda c: (self.ORDRE_GRAVITE.index(c["gravite"]),
                                      c["longueur_px"]), reverse=True)

        self._historique.append(len(candidats) > 0)
        presence = sum(self._historique)
        confirme = presence >= self.cv["detections_min"] and bool(candidats)

        gravite = "ok"
        par_classe = {}
        if candidats:
            gravite = max((c["gravite"] for c in candidats),
                          key=self.ORDRE_GRAVITE.index)
            for candidat in candidats:
                par_classe[candidat["classe"]] = par_classe.get(candidat["classe"], 0) + 1

        return {
            "candidats": candidats,
            "par_classe": par_classe,
            "confirme": confirme,
            "gravite": gravite if confirme else "info",
            "binaire": clair,
            "binaire_sombre": sombre,
            "presence_fenetre": presence,
            "taille_fenetre": self._historique.maxlen,
        }

    def annoter(self, image: np.ndarray, resultat: dict) -> np.ndarray:
        """Dessine le diagnostic : chaque anomalie avec son type et sa gravite."""
        from src.utils.common import COULEURS, bandeau_etat

        sortie = image.copy()
        if self._masque_roi is not None:
            contours, _ = cv2.findContours(self._masque_roi, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sortie, contours, -1, (255, 180, 0), 1)

        for candidat in resultat["candidats"]:
            couleur = COULEURS.get(candidat["gravite"], COULEURS["neutre"])
            if candidat["contour"] is not None:
                cv2.drawContours(sortie, [candidat["contour"]], -1, couleur, 2)
            else:
                x1, y1, x2, y2 = candidat["bbox"]
                cv2.rectangle(sortie, (x1, y1), (x2, y2), couleur, 2)
            x1, y1 = candidat["bbox"][0], candidat["bbox"][1]
            etiquette = f"{candidat['classe']} {candidat['longueur_mm']:.0f}mm"
            cv2.putText(sortie, etiquette, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, couleur, 1, cv2.LINE_AA)

        resume = ", ".join(f"{nom} x{n}" for nom, n
                           in sorted(resultat["par_classe"].items()))
        bandeau_etat(sortie, [
            f"Convoyeur : {'ANOMALIE CONFIRMEE' if resultat['confirme'] else 'surveillance'}",
            f"Anomalies : {len(resultat['candidats'])}",
            resume[:44] if resume else "aucune",
            f"Presence  : {resultat['presence_fenetre']}/{resultat['taille_fenetre']}",
            f"Gravite   : {resultat['gravite']}",
        ])
        return sortie

    def label_yolo(self, resultat: dict, forme) -> list[str]:
        """
        Convertit les anomalies en annotations YOLO segmentation, avec la
        CLASSE deduite par le classificateur geometrique.

        Sert a PRE-ANNOTER le dataset de la couche B. A relire et corriger
        a la main avant entrainement : ces annotations sont une aide a la
        saisie, pas une verite terrain. Le classificateur se trompe
        notamment entre fissure et dechirure sur les defauts obliques.
        """
        import yaml
        chemin = RACINE / self.cfg["data_yaml"]
        noms = yaml.safe_load(chemin.read_text(encoding="utf-8"))["names"]
        identifiant_par_nom = {nom: i for i, nom in noms.items()}

        hauteur, largeur = forme[:2]
        lignes = []
        for candidat in resultat["candidats"]:
            if candidat["contour"] is None:
                continue
            identifiant = identifiant_par_nom.get(candidat["classe"])
            if identifiant is None:
                continue
            approx = cv2.approxPolyDP(
                candidat["contour"],
                0.005 * cv2.arcLength(candidat["contour"], True), True)
            points = approx.reshape(-1, 2).astype(np.float32)
            if len(points) < 3:
                continue
            points[:, 0] /= largeur
            points[:, 1] /= hauteur
            points = np.clip(points, 0.0, 1.0)
            coords = " ".join(f"{v:.6f}" for v in points.flatten())
            lignes.append(f"{identifiant} {coords}")
        return lignes


# --------------------------------------------------------------- autonome
def main() -> None:
    import argparse
    from src.utils.common import RACINE, charger_config

    ap = argparse.ArgumentParser(
        description="Détection de déchirure de convoyeur (vision classique)")
    ap.add_argument("--source", required=True,
                    help="Vidéo, image, dossier d'images, ou index de webcam")
    ap.add_argument("--config", default="configs/convoyeur.yaml")
    ap.add_argument("--pre-annoter", action="store_true",
                    help="Écrit les .txt YOLO à côté des images (dossier seulement)")
    ap.add_argument("--sauver-video", help="Chemin de sortie .mp4 annoté")
    ap.add_argument("--sans-affichage", action="store_true")
    args = ap.parse_args()

    config = charger_config(args.config)
    detecteur = DetecteurDechirureCV(config)

    source = args.source
    chemin = Path(source)
    if not chemin.is_absolute() and not source.isdigit():
        chemin = RACINE / source

    # --- Mode dossier d'images (pré-annotation) ---
    if chemin.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted(p for p in chemin.iterdir() if p.suffix.lower() in extensions)
        nb_defauts = 0
        for fichier in images:
            image = lire_image(fichier)
            if image is None:
                continue
            detecteur._masque_roi = None      # ROI recalculée pour chaque image
            resultat = detecteur.analyser(image)
            if resultat["candidats"]:
                nb_defauts += 1
                if args.pre_annoter:
                    lignes = detecteur.label_yolo(resultat, image.shape)
                    fichier.with_suffix(".txt").write_text(
                        "\n".join(lignes), encoding="utf-8")
            if not args.sans_affichage:
                cv2.imshow("convoyeur", detecteur.annoter(image, resultat))
                if cv2.waitKey(200) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()
        print(f"{len(images)} images analysées, {nb_defauts} avec candidat(s).")
        if args.pre_annoter:
            print("Pré-annotations .txt écrites. RELISEZ-LES dans LabelImg/CVAT "
                  "avant d'entraîner : elles contiennent des faux positifs.")
        return

    # --- Mode vidéo / flux ---
    capture = cv2.VideoCapture(int(source) if source.isdigit() else str(chemin))
    if not capture.isOpened():
        print(f"Impossible d'ouvrir la source : {source}")
        return

    enregistreur = None
    if args.sauver_video:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        enregistreur = cv2.VideoWriter(args.sauver_video,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       fps, (largeur, hauteur))

    nb_confirmations = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        resultat = detecteur.analyser(image)
        if resultat["confirme"]:
            nb_confirmations += 1
        annotee = detecteur.annoter(image, resultat)
        if enregistreur is not None:
            enregistreur.write(annotee)
        if not args.sans_affichage:
            cv2.imshow("convoyeur", annotee)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    if enregistreur is not None:
        enregistreur.release()
        print(f"Vidéo annotée : {args.sauver_video}")
    cv2.destroyAllWindows()
    print(f"{nb_confirmations} image(s) avec déchirure confirmée.")


if __name__ == "__main__":
    main()
