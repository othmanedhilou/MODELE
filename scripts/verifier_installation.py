"""
Vérification complète de l'installation et de la chaîne de traitement.

Exécute tous les contrôles d'un coup : dépendances, matériel, configurations,
génération des données de test, et les trois détecteurs. À lancer après
l'installation et après chaque modification importante.

  python scripts/verifier_installation.py
"""
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console  # noqa: E402

configurer_console()
resultats = []


def verifier(intitule, fonction):
    try:
        detail = fonction()
        resultats.append((True, intitule, detail))
        print(f"  [OK]    {intitule}" + (f" - {detail}" if detail else ""))
    except Exception as erreur:
        resultats.append((False, intitule, str(erreur)))
        print(f"  [ECHEC] {intitule} - {erreur}")


def dependances():
    import cv2, numpy, torch, ultralytics, yaml  # noqa: F401
    return (f"opencv {cv2.__version__}, torch {torch.__version__}, "
            f"ultralytics {ultralytics.__version__}")


def materiel():
    import torch
    if torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(0)}"
    return "CPU seulement - entrainez sur Colab (notebooks/)"


def configurations():
    noms = ["eclairage", "vehicules", "convoyeur", "plaque", "pipeline",
            "data_eclairage", "data_vehicules", "data_convoyeur",
            "correspondance_vehicules", "correspondance_convoyeur"]
    for nom in noms:
        charger_config(f"configs/{nom}.yaml")
    return f"{len(noms)} fichiers valides"


def chemin_unicode():
    from src.utils.common import ecrire_image, lire_image
    import numpy as np
    test = RACINE / "runs" / "_verification.jpg"
    image = np.full((32, 32, 3), 128, np.uint8)
    if not ecrire_image(test, image):
        raise RuntimeError("ecriture impossible")
    if lire_image(test) is None:
        raise RuntimeError("lecture impossible")
    test.unlink()
    ascii_ok = str(RACINE).isascii()
    return "chemin ASCII" if ascii_ok else \
        "chemin NON ASCII - contourne par lire_image/ecrire_image"


def donnees_test():
    for script in ("generer_convoyeur_synthetique.py",
                   "generer_eclairage_synthetique.py",
                   "generer_portail_synthetique.py"):
        sortie = subprocess.run([sys.executable, str(RACINE / "scripts" / script)],
                                capture_output=True, text=True, cwd=RACINE)
        if sortie.returncode != 0:
            raise RuntimeError(f"{script} : {sortie.stderr.strip()[:200]}")
    return "video, scene nocturne et scene de portail generees"


def detecteur_convoyeur():
    from src.detect.convoyeur_cv import DetecteurDechirureCV
    from src.utils.common import lire_image
    config = charger_config("configs/convoyeur.yaml")

    sain = DetecteurDechirureCV(config).analyser(
        lire_image(RACINE / "data/raw/convoyeur_sain.jpg"))
    dechire = DetecteurDechirureCV(config).analyser(
        lire_image(RACINE / "data/raw/convoyeur_dechire.jpg"))

    if sain["candidats"]:
        raise RuntimeError(f"{len(sain['candidats'])} faux positif(s) sur bande saine")
    if not dechire["candidats"]:
        raise RuntimeError("dechirure non detectee")
    return (f"0 faux positif, dechirure detectee "
            f"({dechire['candidats'][0]['longueur_mm']:.0f} mm)")


def analyseur_eclairage():
    import cv2, numpy as np
    from src.detect.eclairage import AnalyseurEclairage
    from src.utils.common import lire_image
    config = charger_config("configs/eclairage.yaml")
    analyseur = AnalyseurEclairage(config, poids=None)
    image = lire_image(RACINE / "data/raw/parc_nuit_synthetique.jpg")
    canal = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    scene = float(np.median(canal))

    attendus = {"allume": [220, 176, 300, 224], "faible": [600, 176, 680, 224],
                "eteint": [980, 176, 1060, 224]}
    for etat, bbox in attendus.items():
        obtenu = analyseur._etat_instantane(
            analyseur._luminance_boite(canal, bbox), scene)
        if obtenu != etat:
            raise RuntimeError(f"lampe '{etat}' diagnostiquee '{obtenu}'")
    return "3/3 etats de lampe corrects, detection nuit operationnelle"


def regles_vehicules():
    sortie = subprocess.run(
        [sys.executable, str(RACINE / "scripts" / "test_regles_vehicules.py")],
        capture_output=True, text=True, cwd=RACINE)
    if sortie.returncode != 0:
        raise RuntimeError(sortie.stdout.strip()[-300:])
    return "5/5 tests de regles metier"


def registre_mlops():
    from src.mlops.registre import charger_registre, empreinte_dataset
    statistiques = empreinte_dataset("convoyeur")
    registre = charger_registre()
    production = registre["production"].get("convoyeur")
    detail = (f"dataset convoyeur {statistiques['images']} images "
              f"(empreinte {statistiques['empreinte']}), "
              f"{len(registre['experiences'])} experience(s)")
    if production:
        detail += f", production {production['version']}"
    return detail


def modele_entraine():
    from src.mlops.registre import poids_production
    chemin = poids_production("convoyeur")
    if chemin is None:
        return "aucun modele promu (mode repli vision classique)"
    from ultralytics import YOLO
    from src.utils.common import lire_image
    modele = YOLO(str(chemin))
    image = lire_image(RACINE / "data/raw/convoyeur_dechire.jpg")
    resultat = modele.predict(image, conf=0.25, verbose=False)[0]
    nombre = 0 if resultat.boxes is None else len(resultat.boxes)
    return f"convoyeur promu, {nombre} detection(s) sur l'image de test"


def surveillance():
    from src.mlops.surveiller import charger_journal
    from src.utils.common import charger_config
    chemin = RACINE / charger_config("configs/pipeline.yaml")["sorties"]["journal"]
    entrees = charger_journal(chemin)
    return f"{len(entrees)} alarme(s) au journal"


def module_plaque():
    import subprocess
    sortie = subprocess.run(
        [sys.executable, str(RACINE / "scripts" / "test_flux_plaque.py")],
        capture_output=True, text=True, cwd=RACINE)
    if sortie.returncode != 0:
        raise RuntimeError(sortie.stdout.strip()[-300:])
    return "selection de la meilleure vue et controle de format conformes"


def localisation_plaque():
    from src.detect.plaque import DetecteurPlaque
    from src.utils.common import lire_image
    config = charger_config("configs/plaque.yaml")
    image = lire_image(RACINE / "data/raw/portail_synthetique.jpg")
    if image is None:
        raise RuntimeError("image de test absente : lancez "
                           "scripts/generer_portail_synthetique.py")
    detecteur = DetecteurPlaque(config)
    attendus = [((0, 219, 660, 780), 198), ((700, 280, 1100, 620), 120),
                ((1230, 370, 1430, 540), 60)]
    ecarts = []
    for bbox, largeur_attendue in attendus:
        candidats = detecteur.localiser(image, bbox)
        if not candidats:
            raise RuntimeError(f"plaque de {largeur_attendue} px non localisee")
        mesure = candidats[0]["largeur_px"]
        ecarts.append(100 * abs(mesure - largeur_attendue) / largeur_attendue)
    if max(ecarts) > 10:
        raise RuntimeError(f"erreur de largeur de {max(ecarts):.0f} pourcent")
    return f"3/3 plaques localisees, erreur max {max(ecarts):.1f} pourcent"


def principal():
    print("=== Verification de l'installation ===\n")
    verifier("Dependances Python", dependances)
    verifier("Materiel de calcul", materiel)
    verifier("Fichiers de configuration", configurations)
    verifier("Lecture/ecriture images (chemin)", chemin_unicode)
    verifier("Generation des donnees de test", donnees_test)
    verifier("Modele 3 - convoyeur (vision classique)", detecteur_convoyeur)
    verifier("Modele 1 - eclairage (photometrie)", analyseur_eclairage)
    verifier("Modele 2 - vehicules (regles metier)", regles_vehicules)
    verifier("MLOps - registre et empreinte dataset", registre_mlops)
    verifier("MLOps - modele promu en production", modele_entraine)
    verifier("Module plaque - localisation", localisation_plaque)
    verifier("Module plaque - logique de passage", module_plaque)
    verifier("MLOps - journal de surveillance", surveillance)

    echecs = [r for r in resultats if not r[0]]
    print(f"\n{len(resultats) - len(echecs)}/{len(resultats)} verifications reussies")
    if echecs:
        print("\nA corriger :")
        for _, intitule, detail in echecs:
            print(f"  - {intitule} : {detail}")
        return 1
    print()
    print("Installation conforme. Trois pistes selon ce dont vous disposez :")
    print("  - sans aucune donnee   : docs/sans_donnees.md")
    print("  - videos de l usine    : README.md, chaine de travail")
    print("  - lecture de plaque    : docs/plaque.md, test de faisabilite")
    return 0


if __name__ == "__main__":
    sys.exit(principal())
