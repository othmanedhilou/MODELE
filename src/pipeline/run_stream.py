"""
Pipeline temps réel : une caméra, un ou plusieurs modèles, un journal
d'alarmes commun.

Principes de conception
-----------------------
1. CADENCE ADAPTEE AU RISQUE. On n'analyse pas tout à 25 images/s. Une
   déchirure de bande doit être vue en quelques dixièmes de seconde
   (15 img/s), un véhicule en une seconde (10 img/s), une lampe grillée
   peut attendre une minute. La configuration fixe fps_analyse par caméra,
   ce qui divise la charge CPU/GPU par 5 à 20.

2. LECTURE NON BLOQUANTE. Un flux RTSP accumule les images dans un tampon :
   si le traitement est plus lent que la caméra, on prend du retard qui
   grandit sans fin. Le lecteur tourne donc dans un fil séparé et ne garde
   que la DERNIERE image. Mieux vaut sauter des images que diagnostiquer
   une bande déchirée avec dix minutes de retard.

3. RECONNEXION AUTOMATIQUE. Une caméra industrielle coupe régulièrement.
   Le pipeline reconnecte seul au lieu de s'arrêter.

  python -m src.pipeline.run_stream --camera cam_convoyeur_01
  python -m src.pipeline.run_stream --source data/raw/convoyeur_synthetique.mp4 \
         --modeles convoyeur --sans-affichage
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.common import (RACINE, JournalAlarmes,  # noqa: E402
                              charger_config, configurer_console)
from src.mlops.registre import (poids_production,  # noqa: E402
                                verifier_coherence)

configurer_console()


class LecteurFlux:
    """Lecteur de flux à fil séparé, qui ne conserve que l'image la plus récente."""

    def __init__(self, source, reconnexion_s: float = 5.0):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.reconnexion_s = reconnexion_s
        self.image = None
        self.actif = True
        self.fps_source = 25.0
        # Une source fichier est lue à sa cadence réelle : sinon le fil de
        # lecture vide la vidéo en une seconde et la démonstration se termine
        # avant que le traitement ait vu la moitié des images.
        self.est_fichier = (isinstance(self.source, str)
                            and not self.source.lower().startswith("rtsp")
                            and Path(self.source).exists())
        self._verrou = threading.Lock()
        self._fil = threading.Thread(target=self._boucle, daemon=True)

    def demarrer(self) -> "LecteurFlux":
        self._fil.start()
        # On attend la première image (au plus 10 s) pour ne pas démarrer à vide
        for _ in range(100):
            if self.image is not None:
                break
            time.sleep(0.1)
        return self

    def _boucle(self) -> None:
        capture = None
        while self.actif:
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(self.source)
                # Tampon minimal : on veut le temps réel, pas l'historique
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    print(f"[FLUX] Connexion impossible à {self.source}, "
                          f"nouvel essai dans {self.reconnexion_s:.0f} s")
                    time.sleep(self.reconnexion_s)
                    continue
                self.fps_source = capture.get(cv2.CAP_PROP_FPS) or 25.0
                print(f"[FLUX] Connecté à {self.source} ({self.fps_source:.0f} fps)")

            ok, image = capture.read()
            if not ok:
                capture.release()
                capture = None
                # Fichier vidéo terminé : on arrête au lieu de boucler sans fin
                if self.est_fichier:
                    self.actif = False
                    break
                continue

            with self._verrou:
                self.image = image

            if self.est_fichier:
                time.sleep(1.0 / max(self.fps_source, 1.0))

        if capture is not None:
            capture.release()

    def lire(self):
        with self._verrou:
            return None if self.image is None else self.image.copy()

    def arreter(self) -> None:
        self.actif = False


class Surveillance:
    """Assemble les modèles demandés pour une caméra et applique les règles."""

    def __init__(self, nom_camera: str, modeles: list[str], journal: JournalAlarmes,
                 fps: float = 25.0):
        self.camera = nom_camera
        self.journal = journal
        self.modules = {}
        pipeline = charger_config("configs/pipeline.yaml")
        poids_cfg = pipeline.get("poids", {})

        def resoudre_poids(nom_modele):
            """
            Poids a charger pour un modele, par ordre de priorite :
              1. models/<modele>/production.pt, promu explicitement via le
                 registre MLOps. C'est le seul chemin utilise en exploitation :
                 on ne veut pas qu'un entrainement en cours dans runs/ passe
                 en production sans decision humaine.
              2. le chemin declare dans configs/pipeline.yaml (mise au point).
              3. rien : le module bascule en mode de repli.
            """
            promu = poids_production(nom_modele)
            if promu is not None:
                coherent, message = verifier_coherence(nom_modele, promu)
                if not coherent:
                    print(f"[{nom_modele.upper()}] POIDS PERIMES, IGNORES : {message}")
                    print(f"    Le modele en production a ete entraine sur une autre")
                    print(f"    liste de classes. Le charger produirait de mauvais")
                    print(f"    noms de defauts. Bascule en mode de repli.")
                    return None
                print(f"[{nom_modele.upper()}] poids de production : {promu}")
                return promu
            candidat = RACINE / poids_cfg.get(nom_modele, "")
            if candidat.exists():
                print(f"[{nom_modele.upper()}] poids de mise au point : {candidat}")
                print(f"    (non promu en production - voir "
                      f"python -m src.mlops.registre --lister)")
                return candidat
            return None

        if "convoyeur" in modeles:
            from src.detect.convoyeur_cv import DetecteurDechirureCV
            config = charger_config("configs/convoyeur.yaml")
            self.modules["convoyeur"] = DetecteurDechirureCV(config)
            self.cfg_convoyeur = config
            chemin = resoudre_poids("convoyeur")
            if chemin is not None:
                from ultralytics import YOLO
                self.modules["convoyeur_yolo"] = YOLO(str(chemin))
                print(f"[CONVOYEUR] Couche B active : {chemin}")
            else:
                print("[CONVOYEUR] Couche A seule (vision classique). "
                      "Entraînez la couche B pour gagner en précision.")

        if "eclairage" in modeles:
            from src.detect.eclairage import AnalyseurEclairage
            config = charger_config("configs/eclairage.yaml")
            self.modules["eclairage"] = AnalyseurEclairage(
                config, resoudre_poids("eclairage"))

        if "vehicules" in modeles:
            from src.detect.vehicules import DetecteurVehicules
            config = charger_config("configs/vehicules.yaml")
            self.modules["vehicules"] = DetecteurVehicules(
                config, resoudre_poids("vehicules"), fps=fps)

        # La plaque se greffe sur les vehicules : sans detection de vehicule,
        # il n'y a pas de boite dans laquelle chercher une plaque.
        if "plaque" in modeles:
            if "vehicules" not in self.modules:
                print("[PLAQUE] Ignore : le module 'vehicules' est requis.")
            else:
                from src.detect.plaque import DetecteurPlaque
                config_plaque = charger_config("configs/plaque.yaml")
                if config_plaque.get("actif", True):
                    self.modules["plaque"] = DetecteurPlaque(config_plaque)
                    self.cfg_plaque = config_plaque
                    self._vus_plaque = {}
                    dossier = RACINE / config_plaque["sorties"]["dossier_plaques"]
                    dossier.mkdir(parents=True, exist_ok=True)
                    self._journal_plaques = (
                        RACINE / config_plaque["sorties"]["journal"])

        # L'éclairage est coûteux et lent à évoluer : on l'espace dans le temps
        self._dernier_eclairage = 0.0
        self._intervalle_eclairage = charger_config(
            "configs/eclairage.yaml")["alarme"]["intervalle_analyse_s"] \
            if "eclairage" in modeles else 0.0
        self._resultat_eclairage = None

    def traiter(self, image):
        """Analyse une image avec tous les modules et retourne l'image annotée."""
        annotee = image

        if "convoyeur" in self.modules:
            detecteur = self.modules["convoyeur"]
            resultat = detecteur.analyser(image)
            if resultat["confirme"]:
                # Une alarme par TYPE d'anomalie, pas une seule pour l'image.
                # Regrouper fissure et dechirure sous une alarme unique
                # reviendrait a perdre l'information la plus utile : ce que
                # l'operateur doit faire, qui n'est pas le meme selon le type.
                ordre = ("info", "mineure", "majeure", "critique")
                par_type = {}
                for candidat in resultat["candidats"]:
                    actuel = par_type.get(candidat["classe"])
                    if actuel is None or (ordre.index(candidat["gravite"])
                                          > ordre.index(actuel["gravite"])):
                        par_type[candidat["classe"]] = candidat

                seuil = self.cfg_convoyeur.get("seuil_alerte_operateur", "info")
                for classe, candidat in par_type.items():
                    # Le journal garde TOUT : c'est ce qui permet de suivre
                    # l'evolution d'une fissure sur plusieurs semaines. Le
                    # seuil ne filtre que ce qui remonte a l'operateur.
                    alerter = ordre.index(candidat["gravite"]) >= ordre.index(seuil)
                    self.journal.emettre(
                        self.camera, "convoyeur", f"anomalie_{classe}",
                        gravite=candidat["gravite"],
                        details={"id": classe,
                                 "longueur_mm": round(candidat["longueur_mm"]),
                                 "action": candidat.get("action", ""),
                                 "alerte_operateur": alerter,
                                 "nombre": sum(1 for c in resultat["candidats"]
                                               if c["classe"] == classe)},
                        image=image if alerter else None)
            annotee = detecteur.annoter(annotee, resultat)

        if "eclairage" in self.modules:
            maintenant = time.time()
            if maintenant - self._dernier_eclairage >= self._intervalle_eclairage:
                self._dernier_eclairage = maintenant
                self._resultat_eclairage = self.modules["eclairage"].analyser(image)
                for alarme in self._resultat_eclairage["alarmes"]:
                    self.journal.emettre(
                        self.camera, "eclairage", alarme["type"],
                        gravite=alarme["gravite"], details=alarme, image=image)
            if self._resultat_eclairage is not None:
                annotee = self.modules["eclairage"].annoter(
                    annotee, self._resultat_eclairage)

        if "vehicules" in self.modules:
            detecteur = self.modules["vehicules"]
            resultat = detecteur.analyser(image)
            for evenement in resultat["evenements"]:
                if evenement["gravite"] == "info" and \
                        evenement["type"] != "franchissement":
                    continue
                self.journal.emettre(
                    self.camera, "vehicules", evenement["type"],
                    gravite=evenement["gravite"], details=evenement,
                    image=image if evenement["gravite"] != "info" else None)
            if "plaque" in self.modules:
                self._traiter_plaques(image, resultat)

            annotee = detecteur.annoter(annotee, resultat)

        return annotee

    def _traiter_plaques(self, image, resultat_vehicules) -> None:
        """
        Observe la plaque de chaque vehicule suivi, et ne declenche la lecture
        qu'une fois : au franchissement de la ligne d'entree, ou lorsque le
        vehicule quitte le champ. Lire a chaque image couterait dix a
        quarante fois plus cher pour un resultat moins bon, la meilleure vue
        n'etant pas forcement la premiere.
        """
        import json
        from datetime import datetime
        from src.utils.common import ecrire_image

        detecteur = self.modules["plaque"]
        horloge = self.modules["vehicules"]._horloge
        presents = set()
        for detection in resultat_vehicules["detections"]:
            identifiant = detection["id"]
            presents.add(identifiant)
            self._vus_plaque[identifiant] = horloge
            detecteur.observer(image, identifiant, detection["bbox"])

        # Vehicules ayant franchi la ligne, ou disparus depuis 3 secondes
        a_conclure = {e["id"] for e in resultat_vehicules["evenements"]
                      if e["type"] == "franchissement"}
        a_conclure |= {identifiant for identifiant, vu in self._vus_plaque.items()
                       if identifiant not in presents
                       and horloge - vu > 3.0}

        for identifiant in a_conclure:
            self._vus_plaque.pop(identifiant, None)
            lecture = detecteur.conclure(identifiant)
            if lecture is None:
                continue

            horodatage = datetime.now()
            entree = {
                "horodatage": horodatage.isoformat(timespec="seconds"),
                "camera": self.camera,
                "id_vehicule": lecture["id"],
                "plaque": lecture["texte"],
                "conforme": lecture["conforme"],
                "confiance": lecture["confiance"],
                "largeur_px": lecture["largeur_px"],
                "verdict_qualite": lecture["verdict"],
            }
            if self.cfg_plaque["sorties"].get("sauver_vignette", True)                     and lecture["vignette"].size:
                nom = f"{horodatage:%Y%m%d_%H%M%S}_{self.camera}_{lecture['id']}.jpg"
                chemin = (RACINE / self.cfg_plaque["sorties"]["dossier_plaques"] / nom)
                ecrire_image(chemin, lecture["vignette"])
                entree["vignette"] = str(chemin.relative_to(RACINE))

            with open(self._journal_plaques, "a", encoding="utf-8") as f:
                f.write(json.dumps(entree, ensure_ascii=False))
                f.write(chr(10))

            etiquette = lecture["texte"] or "(illisible)"
            print(f"[PLAQUE] {self.camera} - vehicule #{lecture['id']} - "
                  f"{etiquette} ({lecture['largeur_px']} px, {lecture['verdict']})")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pipeline de surveillance temps réel")
    ap.add_argument("--camera", help="Nom d'une caméra définie dans configs/pipeline.yaml")
    ap.add_argument("--source", help="Source directe (fichier, RTSP ou index webcam)")
    ap.add_argument("--modeles", nargs="+",
                    choices=["eclairage", "vehicules", "convoyeur", "plaque"],
                    help="Modèles à activer (obligatoire avec --source)")
    ap.add_argument("--fps-analyse", type=float, default=10.0)
    ap.add_argument("--sans-affichage", action="store_true")
    ap.add_argument("--duree", type=float, default=0,
                    help="Arrêt automatique après N secondes (0 = illimité)")
    args = ap.parse_args()

    pipeline = charger_config("configs/pipeline.yaml")
    sorties = pipeline["sorties"]

    if args.camera:
        camera = next((c for c in pipeline["cameras"] if c["nom"] == args.camera), None)
        if camera is None:
            noms = ", ".join(c["nom"] for c in pipeline["cameras"])
            print(f"Caméra inconnue : {args.camera}. Disponibles : {noms}")
            return
        nom, source = camera["nom"], camera["source"]
        modeles, fps_analyse = camera["modeles"], camera.get("fps_analyse", 10)
    elif args.source:
        if not args.modeles:
            print("--modeles est obligatoire avec --source")
            return
        nom, source = "manuelle", args.source
        modeles, fps_analyse = args.modeles, args.fps_analyse
    else:
        print("Indiquez --camera ou --source. Caméras configurées :")
        for c in pipeline["cameras"]:
            print(f"  {c['nom']:<20} modeles={c['modeles']}")
        return

    chemin = Path(str(source))
    if not str(source).isdigit() and not str(source).startswith("rtsp") \
            and not chemin.is_absolute():
        candidat = RACINE / str(source)
        if candidat.exists():
            source = candidat

    journal = JournalAlarmes(sorties["journal"], sorties["dossier_alarmes"],
                             delai_repetition_s=60,
                             sauver_snapshot=sorties.get("sauver_snapshot", True))

    print(f"\n=== Surveillance '{nom}' ===")
    print(f"Source  : {source}")
    print(f"Modeles : {', '.join(modeles)}")
    print(f"Cadence : {fps_analyse} img/s analysees")
    print(f"Journal : {journal.chemin}\n")

    # Les modeles sont charges AVANT d'ouvrir le flux. Le chargement de YOLO
    # prend plusieurs secondes ; sur une source fichier lue en temps reel, la
    # video defilerait pendant ce temps et on manquerait le debut. Sur un flux
    # RTSP, on accumulerait un retard equivalent des le demarrage.
    surveillance = Surveillance(nom, modeles, journal, fps=fps_analyse)

    lecteur = LecteurFlux(source).demarrer()
    if lecteur.lire() is None:
        print("Aucune image reçue : vérifiez la source.")
        lecteur.arreter()
        return
    afficher = not args.sans_affichage and sorties.get("afficher_fenetre", True)

    periode = 1.0 / max(fps_analyse, 0.1)
    debut, prochaine, images, mesure_debut = time.time(), 0.0, 0, time.time()

    try:
        while lecteur.actif:
            maintenant = time.time()
            if maintenant < prochaine:
                time.sleep(min(0.005, prochaine - maintenant))
                continue
            prochaine = maintenant + periode

            image = lecteur.lire()
            if image is None:
                continue

            annotee = surveillance.traiter(image)
            images += 1

            if images % 50 == 0:
                debit = images / max(time.time() - mesure_debut, 1e-6)
                print(f"  [{nom}] {images} images analysées, {debit:.1f} img/s réel")

            if afficher:
                cv2.imshow(f"surveillance - {nom}", annotee)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if args.duree and time.time() - debut >= args.duree:
                break
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'opérateur.")
    finally:
        lecteur.arreter()
        cv2.destroyAllWindows()
        duree = time.time() - debut
        print(f"\n{images} images analysées en {duree:.1f} s "
              f"({images / max(duree, 1e-6):.1f} img/s)")
        print(f"Journal des alarmes : {journal.chemin}")


if __name__ == "__main__":
    main()
