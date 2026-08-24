"""
Test de la logique de lecture de plaque sur un passage simule.

Un vehicule s'approche du portail : sa plaque passe de 55 a 200 px de
large. Le module doit conserver la MEILLEURE vue du passage, pas la
premiere ni la derniere, et ne lancer l'OCR qu'une seule fois.

C'est la propriete qui rend le module utilisable en temps reel : sans
elle, il faudrait lire chaque image (quarante OCR par vehicule au lieu
d'un) et le resultat serait souvent tire d'une image floue.

  python scripts/test_flux_plaque.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console  # noqa: E402
from src.detect.plaque import DetecteurPlaque, normaliser_plaque  # noqa: E402
from scripts.generer_portail_synthetique import dessiner_vehicule  # noqa: E402

configurer_console()


def image_passage(largeur_vehicule, flou=0):
    """Une image du passage : vehicule de la largeur demandee, centre."""
    image = np.full((900, 1600, 3), 150, np.uint8)
    image[:450] = (185, 175, 160)
    cv2.rectangle(image, (0, 450), (1600, 900), (95, 95, 100), -1)
    bbox, bbox_plaque = dessiner_vehicule(
        image, (800, 870), int(largeur_vehicule), (70, 90, 120), "12345 A 6")
    if flou > 0:
        k = 2 * flou + 1
        image = cv2.GaussianBlur(image, (k, k), 0)
    return image, bbox, bbox_plaque[2] - bbox_plaque[0]


def executer():
    config = charger_config("configs/plaque.yaml")
    config["lecture"]["moteur"] = "aucun"     # on teste la selection, pas l'OCR
    detecteur = DetecteurPlaque(config)
    echecs = []

    # --- Test 1 : le vehicule s'approche, la plaque grandit ---
    # Le flou est maximal a mi-parcours, pour verifier que la selection
    # tient compte de la nettete et pas seulement de la taille.
    largeurs_vues = []
    etapes = [(200, 0), (350, 2), (500, 3), (650, 1), (700, 0), (400, 0)]
    for largeur_vehicule, flou in etapes:
        image, bbox, largeur_plaque = image_passage(largeur_vehicule, flou)
        largeurs_vues.append(largeur_plaque)
        detecteur.observer(image, 1, bbox)

    retenue = detecteur._meilleures.get(1)
    print(f"Test 1 - plaques vues : {largeurs_vues}")
    if retenue is None:
        echecs.append("Test 1 : aucune vue conservee")
    else:
        print(f"         vue retenue : {retenue['largeur_px']} px, "
              f"nettete {retenue['nettete']:.0f}, verdict {retenue['verdict']}")
        # La meilleure vue doit etre une des plus grandes ET nettes,
        # donc l'etape (700, 0) : environ 210 px de plaque.
        if retenue["largeur_px"] < 0.8 * max(largeurs_vues):
            echecs.append(f"Test 1 : vue retenue trop petite "
                          f"({retenue['largeur_px']} px pour un maximum de "
                          f"{max(largeurs_vues)} px)")

    # --- Test 2 : conclure ne rend la plaque qu'une fois ---
    premiere = detecteur.conclure(1)
    seconde = detecteur.conclure(1)
    print(f"Test 2 - premiere conclusion : "
          f"{'obtenue' if premiere else 'absente'} ; "
          f"seconde : {'obtenue' if seconde else 'absente'}")
    if premiere is None:
        echecs.append("Test 2 : premiere conclusion vide")
    if seconde is not None:
        echecs.append("Test 2 : le vehicule a ete conclu deux fois")

    # --- Test 3 : une plaque trop petite est ecartee avant l'OCR ---
    detecteur2 = DetecteurPlaque(config)
    image, bbox, largeur_plaque = image_passage(180)
    detecteur2.observer(image, 2, bbox)
    resultat = detecteur2.conclure(2)
    verdict = resultat["verdict"] if resultat else "aucune"
    print(f"Test 3 - plaque de {largeur_plaque} px -> verdict '{verdict}'")
    if resultat and resultat["verdict"] != "illisible":
        echecs.append(f"Test 3 : plaque de {largeur_plaque} px jugee '{verdict}' "
                      f"au lieu d'illisible")

    # --- Test 4 : normalisation du format marocain ---
    cas = [
        ("12345 A 6", True), ("45678 8 12", True), ("1 2 3", False),
        ("WW 4521", True), ("", False), ("123456789012", False),
    ]
    print("Test 4 - controle de format :")
    for brut, attendu in cas:
        texte, conforme = normaliser_plaque(brut, config["format"])
        marque = "ok " if conforme == attendu else "ECHEC"
        print(f"         [{marque}] '{brut:12}' -> '{texte:12}' conforme={conforme}")
        if conforme != attendu:
            echecs.append(f"Test 4 : '{brut}' conforme={conforme}, attendu {attendu}")

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) :")
        for e in echecs:
            print(f"  - {e}")
        return 1
    print("Logique de lecture de plaque conforme.")
    return 0


if __name__ == "__main__":
    sys.exit(executer())
